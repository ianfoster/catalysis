#!/usr/bin/env python3
"""Run catalyst discovery pipeline.

This script orchestrates the full discovery pipeline:
1. Creates a GeneratorAgent on the local machine (Mac) using OpenAI/Argonne LLM
2. Runs the discovery loop with ShepherdAgents on Spark via Globus Compute
3. Shepherds use the vLLM server on Spark for their LLM reasoning

Architecture:
    Mac (local)                          Spark (remote)
    ┌─────────────────┐                  ┌─────────────────────────────┐
    │ GeneratorAgent  │ ──GC tasks──►    │ vLLM server (port 8000)     │
    │ (OpenAI/Argonne)│                  │                             │
    │ - propose       │                  │ ShepherdAgent (GC task 1)   │
    │ - collect       │                  │ ShepherdAgent (GC task 2)   │
    │ - converge      │                  │ ...                         │
    └─────────────────┘                  └─────────────────────────────┘

Usage:
    # Use OpenAI for Generator, Spark vLLM for Shepherds
    python scripts/run_discovery.py --endpoint $GC_ENDPOINT \\
        --generator-llm openai --shepherd-llm-url http://spark:8000/v1

    # Use Argonne inference for Generator
    python scripts/run_discovery.py --endpoint $GC_ENDPOINT \\
        --generator-llm argonne --shepherd-llm-url http://spark:8000/v1

    # Test mode with fewer iterations
    python scripts/run_discovery.py --endpoint $GC_ENDPOINT \\
        --generator-llm openai --shepherd-llm-url http://spark:8000/v1 \\
        --max-iterations 2 --candidates-per-iteration 2
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def start_llm_server_on_spark(
    endpoint: str,
    model_path: str,
    port: int = 8080,
    run_duration: int = 0,
) -> dict:
    """Start llama-cpp server on Spark via Globus Compute.

    Args:
        endpoint: GC endpoint ID
        model_path: Path to GGUF model on Spark
        port: Server port
        run_duration: How long to run (0 = test only, then keep running)

    Returns:
        Server info dict with status and connection URL
    """
    from globus_compute_sdk import Client, Executor
    from scripts.start_llm_server import start_llm_server_gc

    logging.info("Starting LLM server on Spark...")
    logging.info(f"  Endpoint: {endpoint}")
    logging.info(f"  Model: {model_path}")
    logging.info(f"  Port: {port}")

    client = Client()
    func_id = client.register_function(start_llm_server_gc)
    logging.info(f"  Function ID: {func_id}")

    config = {
        "model_path": model_path,
        "host": "0.0.0.0",
        "port": port,
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "run_duration": run_duration,
        "startup_timeout": 180,  # Give it time to load model
    }

    with Executor(endpoint_id=endpoint) as ex:
        future = ex.submit_to_registered_function(func_id, args=(config,))
        try:
            # Wait for server to start (not the full run duration)
            result = future.result(timeout=300)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return result


def check_llm_server(url: str, timeout: int = 10) -> bool:
    """Check if LLM server is reachable.

    Args:
        url: LLM server URL (e.g., http://spark:8080/v1)
        timeout: Connection timeout

    Returns:
        True if server is responding
    """
    import urllib.request
    import urllib.error

    try:
        health_url = url.rstrip("/v1").rstrip("/") + "/health"
        req = urllib.request.urlopen(health_url, timeout=timeout)
        return req.status == 200
    except Exception:
        pass

    # Try models endpoint
    try:
        models_url = url.rstrip("/") + "/models"
        req = urllib.request.urlopen(models_url, timeout=timeout)
        return req.status == 200
    except Exception:
        return False


async def run_discovery(
    config: dict,
    gc_endpoint: str,
    shepherd_llm_url: str,
    gc_function_map: dict | None = None,
) -> dict:
    """Run the catalyst discovery loop.

    Args:
        config: Full configuration dict (generator.llm has Generator's LLM config)
        gc_endpoint: Globus Compute endpoint ID
        shepherd_llm_url: URL to LLM server on Spark (for Shepherds)
        gc_function_map: Optional GC function IDs for simulations

    Returns:
        Final results summary
    """
    from skills.generator import GeneratorAgent

    generator_config = config.get("generator", {})
    shepherd_config = config.get("shepherd", {})

    # Generator uses config-based LLM (OpenAI/Argonne), not Spark
    gen_llm = generator_config.get("llm", {})
    logging.info("Creating GeneratorAgent...")
    logging.info(f"  Generator LLM: {gen_llm.get('model', 'default')} via {gen_llm.get('base_url', gen_llm.get('shared_url', 'config'))}")
    logging.info(f"  Shepherd LLM: {shepherd_llm_url}")
    logging.info(f"  GC Endpoint: {gc_endpoint}")

    # Pass shepherd_llm_url to shepherd_config so GC tasks use Spark LLM
    shepherd_config["llm_url"] = shepherd_llm_url

    generator = GeneratorAgent(
        config=generator_config,
        shepherd_config=shepherd_config,
        gc_function_map=gc_function_map or {},
        gc_endpoint=gc_endpoint,
        llm_url=None,  # Generator uses config, not passed URL
    )

    logging.info("Starting discovery loop...")
    start_time = time.time()

    try:
        results = await generator.run()
    except KeyboardInterrupt:
        logging.info("Discovery interrupted by user")
        results = {"interrupted": True, "error": "KeyboardInterrupt"}
    except Exception as e:
        logging.error(f"Discovery failed: {e}")
        results = {"error": str(e)}

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed

    logging.info(f"Discovery complete in {elapsed:.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run catalyst discovery pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use OpenAI for Generator, Spark vLLM for Shepherds
    python scripts/run_discovery.py --endpoint $GC_ENDPOINT \\
        --generator-llm openai --shepherd-llm-url http://spark:8000/v1

    # Use Argonne inference for Generator
    python scripts/run_discovery.py --endpoint $GC_ENDPOINT \\
        --generator-llm argonne --shepherd-llm-url http://spark:8000/v1

    # Quick test run
    python scripts/run_discovery.py --endpoint $GC_ENDPOINT \\
        --generator-llm openai --shepherd-llm-url http://spark:8000/v1 \\
        --max-iterations 1
        """,
    )

    # Required args
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Globus Compute endpoint ID for Spark",
    )

    # Generator LLM options (runs on Mac)
    parser.add_argument(
        "--generator-llm",
        choices=["openai", "argonne"],
        default="openai",
        help="LLM provider for GeneratorAgent (default: openai)",
    )
    parser.add_argument(
        "--generator-model",
        help="Model name for Generator (default: gpt-4o for openai, Meta-Llama-3.1-70B-Instruct for argonne)",
    )

    # Shepherd LLM options (runs on Spark)
    parser.add_argument(
        "--shepherd-llm-url",
        required=True,
        help="URL to vLLM server on Spark for Shepherds (e.g., http://spark:8000/v1)",
    )
    parser.add_argument(
        "--shepherd-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name for Shepherds on Spark",
    )

    # Generation options
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum generation iterations (overrides config)",
    )
    parser.add_argument(
        "--candidates-per-iteration",
        type=int,
        help="Candidates per iteration (overrides config)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        help="Budget per candidate evaluation (overrides config)",
    )
    parser.add_argument(
        "--num-shepherds",
        type=int,
        help="Max concurrent shepherd agents (overrides config)",
    )

    # Config and output
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        default="data/discovery_results.json",
        help="Output file for results (default: data/discovery_results.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config = {}
        logging.warning(f"Config file {args.config} not found, using defaults")
    else:
        config = load_config(args.config)

    # Setup Generator LLM config based on --generator-llm
    gen_config = config.setdefault("generator", {})
    gen_llm = gen_config.setdefault("llm", {})

    if args.generator_llm == "openai":
        gen_llm["base_url"] = "https://api.openai.com/v1"
        gen_llm["model"] = args.generator_model or "gpt-4o"
        gen_llm["api_key_env"] = "OPENAI_API_KEY"
        # Verify API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            logging.error("OPENAI_API_KEY environment variable not set")
            sys.exit(1)
    elif args.generator_llm == "argonne":
        gen_llm["base_url"] = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
        gen_llm["model"] = args.generator_model or "meta-llama/Meta-Llama-3.1-70B-Instruct"
        gen_llm["api_key_env"] = "ARGONNE_ACCESS_TOKEN"
        # Verify API key is set
        if not os.environ.get("ARGONNE_ACCESS_TOKEN"):
            logging.warning("ARGONNE_ACCESS_TOKEN not set. Run: python scripts/argonne_auth.py")

    # Setup other config sections
    generation = gen_config.setdefault("generation", {
        "candidates_per_iteration": 6,
        "max_iterations": 20,
    })
    gen_config.setdefault("convergence", {
        "patience": 3,
        "min_improvement": 0.01,
        "llm_judgment": True,
    })
    shepherd = gen_config.setdefault("shepherd", {
        "budget_per_candidate": 100.0,
        "num_concurrent": 4,
        "timeout": 3600,
    })
    gen_config.setdefault("state", {
        "checkpoint_path": "data/generator_state.json",
        "results_path": "data/generator_results.jsonl",
    })

    # Shepherd config (passed to GC tasks)
    shepherd_config = config.setdefault("shepherd", {})
    shepherd_config["llm_model"] = args.shepherd_model

    # Apply command-line overrides
    if args.max_iterations:
        generation["max_iterations"] = args.max_iterations
    if args.candidates_per_iteration:
        generation["candidates_per_iteration"] = args.candidates_per_iteration
    if args.budget:
        shepherd["budget_per_candidate"] = args.budget
    if args.num_shepherds:
        shepherd["num_concurrent"] = args.num_shepherds

    # Check Shepherd LLM server is reachable
    logging.info(f"Checking Shepherd LLM server at {args.shepherd_llm_url}...")
    if check_llm_server(args.shepherd_llm_url):
        logging.info("Shepherd LLM server is responding")
    else:
        logging.warning(
            f"Could not reach Shepherd LLM server at {args.shepherd_llm_url}. "
            "Discovery may fail if server is not running on Spark."
        )

    # Get GC function map from config
    gc_config = config.get("globus_compute", {})
    gc_function_map = gc_config.get("functions", {})

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run discovery
    logging.info("=" * 60)
    logging.info("Starting Catalyst Discovery")
    logging.info("=" * 60)
    logging.info(f"Generator LLM: {args.generator_llm} ({gen_llm.get('model')})")
    logging.info(f"Shepherd LLM: {args.shepherd_llm_url} ({args.shepherd_model})")
    logging.info(f"GC Endpoint: {args.endpoint}")
    logging.info(f"Max iterations: {generation.get('max_iterations', 20)}")
    logging.info(f"Candidates/iteration: {generation.get('candidates_per_iteration', 6)}")
    logging.info(f"Budget/candidate: {shepherd.get('budget_per_candidate', 100.0)}")
    logging.info(f"GC simulation functions: {len(gc_function_map)} registered")
    if gc_function_map:
        logging.info(f"  Available: {', '.join(gc_function_map.keys())}")
    logging.info("=" * 60)

    try:
        results = asyncio.run(
            run_discovery(
                config=config,
                gc_endpoint=args.endpoint,
                shepherd_llm_url=args.shepherd_llm_url,
                gc_function_map=gc_function_map,
            )
        )
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        results = {"error": "Interrupted", "ok": False}

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Discovery Summary")
    print("=" * 60)

    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Iterations completed: {results.get('iteration', 0)}")
        print(f"Candidates evaluated: {results.get('total_evaluated', 0)}")
        print(f"Best score: {results.get('best_score', 0):.1f}")

        best = results.get("best_candidate")
        if best:
            metals = best.get("metals", [])
            metal_str = "+".join(f"{m['element']}{m['wt_pct']}%" for m in metals[:3])
            support = best.get("support", "?")
            print(f"Best candidate: {metal_str}/{support}")

        print(f"Stop reason: {results.get('stop_reason', 'unknown')}")
        print(f"Elapsed time: {results.get('elapsed_seconds', 0):.1f}s")

    print("=" * 60)

    return 0 if results.get("ok", True) and "error" not in results else 1


if __name__ == "__main__":
    sys.exit(main())
