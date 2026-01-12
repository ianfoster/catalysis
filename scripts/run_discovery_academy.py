#!/usr/bin/env python3
"""Run catalyst discovery with Academy agents via Redis.

This script runs the GeneratorAgent locally, connecting to ShepherdAgents
running on Polaris via Redis exchange.

Architecture:
    Mac (local)                          Polaris (PBS job)
    ┌─────────────────┐                  ┌─────────────────────────────┐
    │ GeneratorAgent  │◄──── Redis ────► │ ShepherdAgent(s)            │
    │                 │     Exchange     │ QEAgent, GPAWAgent, etc.    │
    └─────────────────┘                  └─────────────────────────────┘

Usage:
    # Connect to agents on Polaris via Redis
    python scripts/run_discovery_academy.py --redis-host polaris.alcf.anl.gov

    # Quick test
    python scripts/run_discovery_academy.py --redis-host polaris.alcf.anl.gov \\
        --max-iterations 2 --candidates-per-iteration 3
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def run_discovery(
    redis_host: str,
    redis_port: int,
    max_iterations: int,
    candidates_per_iteration: int,
    budget: float,
    num_shepherds: int,
    config: dict,
):
    """Run discovery with Academy agents."""
    from academy.manager import Manager
    from academy.exchange import RedisExchangeFactory

    from skills.generator import GeneratorAgent

    logging.info("Connecting to Redis exchange...")
    logging.info(f"  Redis: {redis_host}:{redis_port}")

    exchange_factory = RedisExchangeFactory(hostname=redis_host, port=redis_port)

    async with await Manager.from_exchange_factory(exchange_factory) as manager:
        logging.info("Connected to Academy exchange")

        # Configure generator
        generator_config = config.get("generator", {})
        generator_config.setdefault("generation", {})
        generator_config["generation"]["max_iterations"] = max_iterations
        generator_config["generation"]["candidates_per_iteration"] = candidates_per_iteration
        generator_config.setdefault("shepherd", {})
        generator_config["shepherd"]["budget_per_candidate"] = budget
        generator_config["shepherd"]["max_concurrent_shepherds"] = num_shepherds

        # LLM config - use Argonne inference API
        generator_config.setdefault("llm", {})
        generator_config["llm"]["mode"] = "remote"
        generator_config["llm"]["model"] = config.get("generator", {}).get("llm", {}).get(
            "model", "meta-llama/Meta-Llama-3.1-70B-Instruct"
        )

        logging.info("Launching GeneratorAgent...")
        generator = await manager.launch(
            GeneratorAgent,
            kwargs={
                "config": generator_config,
                "use_academy": True,  # Use Academy agents instead of GC
            },
        )

        logging.info("Starting discovery loop...")
        start_time = time.time()

        try:
            # Run the generator
            results = await generator.run({
                "max_iterations": max_iterations,
                "candidates_per_iteration": candidates_per_iteration,
            })
        except KeyboardInterrupt:
            logging.info("Discovery interrupted by user")
            results = {"interrupted": True, "error": "KeyboardInterrupt"}
        except Exception as e:
            logging.error(f"Discovery failed: {e}")
            import traceback
            traceback.print_exc()
            results = {"error": str(e)}

        elapsed = time.time() - start_time
        results["elapsed_seconds"] = elapsed

        logging.info(f"Discovery complete in {elapsed:.1f}s")
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Run catalyst discovery with Academy agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Connect to Polaris agents
    python scripts/run_discovery_academy.py --redis-host polaris.alcf.anl.gov

    # Quick test
    python scripts/run_discovery_academy.py --redis-host polaris.alcf.anl.gov \\
        --max-iterations 2 --candidates-per-iteration 3

    # Local testing (if agents running locally)
    python scripts/run_discovery_academy.py --redis-host localhost
        """,
    )

    parser.add_argument(
        "--redis-host",
        required=True,
        help="Redis server hostname (e.g., polaris.alcf.anl.gov)",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum discovery iterations (default: 10)",
    )
    parser.add_argument(
        "--candidates-per-iteration",
        type=int,
        default=5,
        help="Candidates per iteration (default: 5)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=200.0,
        help="Budget per candidate (default: 200)",
    )
    parser.add_argument(
        "--num-shepherds",
        type=int,
        default=2,
        help="Max concurrent shepherds (default: 2)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        default="data/discovery_results.json",
        help="Output file (default: data/discovery_results.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Print banner
    logging.info("=" * 60)
    logging.info("Catalyst Discovery (Academy Mode)")
    logging.info("=" * 60)
    logging.info(f"Redis: {args.redis_host}:{args.redis_port}")
    logging.info(f"Max iterations: {args.max_iterations}")
    logging.info(f"Candidates/iteration: {args.candidates_per_iteration}")
    logging.info(f"Budget/candidate: {args.budget}")
    logging.info("=" * 60)

    # Test Redis connection first
    try:
        import redis
        r = redis.Redis(host=args.redis_host, port=args.redis_port)
        r.ping()
        logging.info("Redis connection OK")
    except Exception as e:
        logging.error(f"Cannot connect to Redis at {args.redis_host}:{args.redis_port}")
        logging.error(f"Error: {e}")
        logging.error("Make sure Redis is running on Polaris login node")
        sys.exit(1)

    # Run discovery
    try:
        results = asyncio.run(run_discovery(
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            max_iterations=args.max_iterations,
            candidates_per_iteration=args.candidates_per_iteration,
            budget=args.budget,
            num_shepherds=args.num_shepherds,
            config=config,
        ))
    except KeyboardInterrupt:
        logging.info("Interrupted")
        results = {"error": "Interrupted"}

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        print(f"Iterations: {results.get('iteration', 0)}")
        print(f"Candidates evaluated: {results.get('total_evaluated', 0)}")
        print(f"Best score: {results.get('best_score', 0):.3f}")
        print(f"Elapsed: {results.get('elapsed_seconds', 0):.1f}s")

    print("=" * 60)


if __name__ == "__main__":
    main()
