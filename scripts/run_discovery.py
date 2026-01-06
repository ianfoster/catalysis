#!/usr/bin/env python3
"""Run catalyst discovery pipeline.

This script orchestrates the full discovery pipeline:
1. Optionally starts llama-cpp LLM server on Spark via Globus Compute
2. Creates a GeneratorAgent on the local machine (Mac)
3. Runs the discovery loop with ShepherdAgents on Spark

Architecture:
    Mac (local)                          Spark (remote)
    ┌─────────────────┐                  ┌─────────────────────────────┐
    │ GeneratorAgent  │ ──GC tasks──►    │ llama-cpp server (port 8080)│
    │ - propose       │                  │                             │
    │ - collect       │                  │ ShepherdAgent (GC task 1)   │
    │ - converge      │                  │ ShepherdAgent (GC task 2)   │
    └─────────────────┘                  │ ...                         │
                                         └─────────────────────────────┘

Usage:
    # Basic: Use existing LLM server on Spark
    python scripts/run_discovery.py --endpoint $GC_ENDPOINT --llm-url http://spark:8080/v1

    # Full: Start LLM server, then run discovery
    python scripts/run_discovery.py --endpoint $GC_ENDPOINT \\
        --start-llm --model /path/to/model.gguf

    # Test mode with fewer iterations
    python scripts/run_discovery.py --endpoint $GC_ENDPOINT --llm-url http://spark:8080/v1 \\
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
    llm_url: str,
    gc_function_map: dict | None = None,
) -> dict:
    """Run the catalyst discovery loop.

    Args:
        config: Full configuration dict
        gc_endpoint: Globus Compute endpoint ID
        llm_url: URL to LLM server on Spark
        gc_function_map: Optional GC function IDs for simulations

    Returns:
        Final results summary
    """
    from skills.generator import GeneratorAgent

    generator_config = config.get("generator", {})
    shepherd_config = config.get("shepherd", {})

    logging.info("Creating GeneratorAgent...")
    logging.info(f"  GC Endpoint: {gc_endpoint}")
    logging.info(f"  LLM URL: {llm_url}")

    generator = GeneratorAgent(
        config=generator_config,
        shepherd_config=shepherd_config,
        gc_function_map=gc_function_map or {},
        gc_endpoint=gc_endpoint,
        llm_url=llm_url,
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
    # Use existing LLM server
    python scripts/run_discovery.py --endpoint $GC_ENDPOINT \\
        --llm-url http://192.168.1.100:8080/v1

    # Start LLM server first
    python scripts/run_discovery.py --endpoint $GC_ENDPOINT \\
        --start-llm --model /home/user/models/llama-3.gguf

    # Quick test run
    python scripts/run_discovery.py --endpoint $GC_ENDPOINT \\
        --llm-url http://spark:8080/v1 --max-iterations 1
        """,
    )

    # Required args
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Globus Compute endpoint ID for Spark",
    )

    # LLM server options (mutually exclusive)
    llm_group = parser.add_mutually_exclusive_group(required=True)
    llm_group.add_argument(
        "--llm-url",
        help="URL to existing LLM server (e.g., http://spark:8080/v1)",
    )
    llm_group.add_argument(
        "--start-llm",
        action="store_true",
        help="Start LLM server on Spark via GC",
    )

    # LLM server options (when starting)
    parser.add_argument(
        "--model",
        help="Path to GGUF model file on Spark (required with --start-llm)",
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        default=8000,
        help="LLM server port (default: 8000 for vLLM)",
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

    # Validate args
    if args.start_llm and not args.model:
        parser.error("--model is required when using --start-llm")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Create default config
        config = {
            "generator": {
                "llm": {
                    "mode": "shared",
                    "model": "gpt-3.5-turbo",
                },
                "generation": {
                    "candidates_per_iteration": 6,
                    "max_iterations": 20,
                },
                "convergence": {
                    "patience": 3,
                    "min_improvement": 0.01,
                    "llm_judgment": True,
                },
                "shepherd": {
                    "budget_per_candidate": 100.0,
                    "num_concurrent": 4,
                    "timeout": 3600,
                },
                "state": {
                    "checkpoint_path": "data/generator_state.json",
                    "results_path": "data/generator_results.jsonl",
                },
            },
            "shepherd": {},
            "globus_compute": {
                "endpoint_id": args.endpoint,
                "functions": {},
            },
        }
        logging.warning(f"Config file {args.config} not found, using defaults")
    else:
        config = load_config(args.config)

    # Apply command-line overrides
    gen_config = config.setdefault("generator", {})
    generation = gen_config.setdefault("generation", {})
    shepherd = gen_config.setdefault("shepherd", {})

    if args.max_iterations:
        generation["max_iterations"] = args.max_iterations
    if args.candidates_per_iteration:
        generation["candidates_per_iteration"] = args.candidates_per_iteration
    if args.budget:
        shepherd["budget_per_candidate"] = args.budget
    if args.num_shepherds:
        shepherd["num_concurrent"] = args.num_shepherds

    # Determine LLM URL
    llm_url = args.llm_url

    if args.start_llm:
        logging.info("Starting LLM server on Spark...")
        server_result = start_llm_server_on_spark(
            endpoint=args.endpoint,
            model_path=args.model,
            port=args.llm_port,
            run_duration=0,  # Just test, will run separately
        )

        if not server_result.get("ok"):
            logging.error(f"Failed to start LLM server: {server_result.get('error')}")
            sys.exit(1)

        # Server started successfully - construct URL
        # Note: We need the actual IP/hostname of Spark
        llm_url = server_result.get("server_url", f"http://localhost:{args.llm_port}/v1")
        logging.info(f"LLM server started: {llm_url}")

        # User needs to ensure server keeps running
        logging.warning(
            "Note: LLM server was tested but may have stopped. "
            "For persistent server, use start_llm_server.py with --run-duration"
        )

    # Check LLM server is reachable
    logging.info(f"Checking LLM server at {llm_url}...")
    if check_llm_server(llm_url):
        logging.info("LLM server is responding")
    else:
        logging.warning(
            f"Could not reach LLM server at {llm_url}. "
            "Discovery may fail if server is not running."
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
    logging.info(f"Endpoint: {args.endpoint}")
    logging.info(f"LLM URL: {llm_url}")
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
                llm_url=llm_url,
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
