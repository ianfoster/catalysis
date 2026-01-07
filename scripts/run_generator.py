#!/usr/bin/env python3
"""Run GeneratorAgent on Mac, connecting to agents already running on Spark.

This script runs the GeneratorAgent locally (Mac) which:
1. Uses OpenAI or Argonne for LLM reasoning (candidate proposal, convergence)
2. Connects to Redis exchange to discover ShepherdAgents on Spark
3. Dispatches candidate evaluations to Shepherds via Academy

Prerequisites:
    1. Start agents on Spark first:
       ssh spark
       python scripts/run_spark_agents.py --llm-url http://localhost:8000/v1 \
           --redis-host localhost --redis-port 6379

    2. Redis must be accessible from Mac (either local or tunneled from Spark)

Architecture:
    Mac (local)                          Spark (remote)
    ┌─────────────────┐                  ┌─────────────────────────────────┐
    │ GeneratorAgent  │ ◄──Redis──►      │ vLLM Server (port 8000)         │
    │ (OpenAI/Argonne)│                  │                                 │
    │ - propose       │                  │ ShepherdAgent(s)                │
    │ - collect       │                  │   └─► SimAgents (MACE, etc.)    │
    │ - converge      │                  │                                 │
    └─────────────────┘                  └─────────────────────────────────┘

Usage:
    # Use OpenAI for Generator, connect to Spark via Redis
    python scripts/run_generator.py --redis-host localhost --generator-llm openai

    # Use Argonne inference for Generator
    python scripts/run_generator.py --redis-host spark-ip --generator-llm argonne

    # Quick test
    python scripts/run_generator.py --redis-host localhost --generator-llm openai \
        --max-iterations 1 --candidates-per-iteration 3
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


async def run_generator(
    config: dict,
    redis_host: str | None = None,
    redis_port: int = 6379,
    exchange_url: str | None = None,
) -> dict:
    """Run the GeneratorAgent with Academy exchange.

    Args:
        config: Full configuration dict
        redis_host: Redis hostname for Academy exchange (if using Redis)
        redis_port: Redis port
        exchange_url: HTTP exchange URL (if using cloud exchange)

    Returns:
        Final results summary
    """
    from academy.manager import Manager

    from skills.generator import GeneratorAgent

    generator_config = config.get("generator", {})
    shepherd_config = config.get("shepherd", {})

    gen_llm = generator_config.get("llm", {})
    logging.info("Starting GeneratorAgent...")
    logging.info(f"  Generator LLM: {gen_llm.get('model')} via {gen_llm.get('base_url', 'config')}")

    # Create exchange factory based on connection type
    if exchange_url:
        from academy.exchange import HttpExchangeFactory
        exchange_factory = HttpExchangeFactory(url=exchange_url)
        logging.info(f"  Exchange: HTTP ({exchange_url})")
    else:
        from academy.exchange import RedisExchangeFactory
        exchange_factory = RedisExchangeFactory(hostname=redis_host, port=redis_port)
        logging.info(f"  Exchange: Redis ({redis_host}:{redis_port})")

    start_time = time.time()

    async with await Manager.from_exchange_factory(exchange_factory) as manager:
        # Launch GeneratorAgent within Academy context
        generator = await manager.launch(
            GeneratorAgent,
            kwargs={
                "config": generator_config,
                "shepherd_config": shepherd_config,
                "redis_host": redis_host,
                "redis_port": redis_port,
            },
        )

        logging.info("GeneratorAgent launched, starting discovery loop...")

        # The generator will discover ShepherdAgents via the exchange
        # and dispatch evaluations to them
        try:
            # Use the run action to start the generation loop
            results = await generator.run_discovery({})
        except Exception as e:
            logging.error(f"Discovery failed: {e}")
            results = {"error": str(e), "ok": False}

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed

    logging.info(f"Discovery complete in {elapsed:.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run GeneratorAgent on Mac with Academy exchange",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Connect to local Redis (tunnel from Spark)
    python scripts/run_generator.py --redis-host localhost --generator-llm openai

    # Connect directly to Spark's Redis
    python scripts/run_generator.py --redis-host spark-hostname --generator-llm argonne

    # Quick test run
    python scripts/run_generator.py --redis-host localhost --generator-llm openai \\
        --max-iterations 1 --candidates-per-iteration 3
        """,
    )

    # Exchange connection (one required)
    exchange_group = parser.add_mutually_exclusive_group(required=True)
    exchange_group.add_argument(
        "--redis-host",
        help="Redis hostname (where Spark agents are connected)",
    )
    exchange_group.add_argument(
        "--exchange-url",
        help="Academy HTTP exchange URL (authenticated cloud exchange)",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )

    # Generator LLM options
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

    # Setup Generator LLM config
    gen_config = config.setdefault("generator", {})
    gen_llm = gen_config.setdefault("llm", {})

    if args.generator_llm == "openai":
        gen_llm["base_url"] = "https://api.openai.com/v1"
        gen_llm["model"] = args.generator_model or "gpt-4o"
        gen_llm["api_key_env"] = "OPENAI_API_KEY"
        if not os.environ.get("OPENAI_API_KEY"):
            logging.error("OPENAI_API_KEY environment variable not set")
            sys.exit(1)
    elif args.generator_llm == "argonne":
        gen_llm["base_url"] = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
        gen_llm["model"] = args.generator_model or "meta-llama/Meta-Llama-3.1-70B-Instruct"
        gen_llm["api_key_env"] = "ARGONNE_ACCESS_TOKEN"
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
        "num_concurrent": 8,
        "timeout": 3600,
    })
    gen_config.setdefault("state", {
        "checkpoint_path": "data/generator_state.json",
        "results_path": "data/generator_results.jsonl",
    })

    # Apply command-line overrides
    if args.max_iterations:
        generation["max_iterations"] = args.max_iterations
    if args.candidates_per_iteration:
        generation["candidates_per_iteration"] = args.candidates_per_iteration
    if args.budget:
        shepherd["budget_per_candidate"] = args.budget

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run discovery
    logging.info("=" * 60)
    logging.info("Starting Catalyst Discovery (Academy Mode)")
    logging.info("=" * 60)
    logging.info(f"Generator LLM: {args.generator_llm} ({gen_llm.get('model')})")
    if args.exchange_url:
        logging.info(f"Exchange: HTTP ({args.exchange_url})")
    else:
        logging.info(f"Exchange: Redis ({args.redis_host}:{args.redis_port})")
    logging.info(f"Max iterations: {generation.get('max_iterations', 20)}")
    logging.info(f"Candidates/iteration: {generation.get('candidates_per_iteration', 6)}")
    logging.info(f"Budget/candidate: {shepherd.get('budget_per_candidate', 100.0)}")
    logging.info("=" * 60)

    try:
        results = asyncio.run(
            run_generator(
                config=config,
                redis_host=args.redis_host,
                redis_port=args.redis_port,
                exchange_url=args.exchange_url,
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
