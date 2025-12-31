#!/usr/bin/env python3
"""Entry point for GeneratorAgent-based catalyst discovery.

This script launches a GeneratorAgent that:
1. Uses LLM to propose catalyst candidates
2. Spawns ShepherdAgents to evaluate each candidate
3. Iterates until convergence or max iterations

Usage:
    python app_generator.py [options]

Examples:
    # Basic run with defaults (local)
    python app_generator.py

    # Connect to Academy agents on Spark
    python app_generator.py --exchange-url http://spark:8080 --llm-url http://spark:8000/v1

    # Custom iterations and candidates per batch
    python app_generator.py --max-iterations 10 --candidates-per-iteration 8

    # Resume from checkpoint
    python app_generator.py --resume

    # With custom LLM endpoint
    python app_generator.py --llm-url http://localhost:8000/v1 --llm-model llama3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml

from academy.manager import Manager
from academy.exchange import LocalExchangeFactory

from skills.generator import GeneratorAgent
from orchestration.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Run catalyst discovery with GeneratorAgent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config files
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Base configuration file",
    )
    p.add_argument(
        "--config-local",
        default="config.local.yaml",
        help="Local configuration overrides",
    )

    # Generation parameters
    p.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum generation iterations",
    )
    p.add_argument(
        "--candidates-per-iteration",
        type=int,
        default=None,
        help="Candidates to propose per iteration",
    )

    # Convergence parameters
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Iterations without improvement before stopping",
    )
    p.add_argument(
        "--no-llm-judgment",
        action="store_true",
        help="Disable LLM convergence judgment",
    )

    # Shepherd parameters
    p.add_argument(
        "--shepherd-budget",
        type=float,
        default=None,
        help="Budget per candidate evaluation",
    )
    p.add_argument(
        "--max-concurrent-shepherds",
        type=int,
        default=None,
        help="Maximum concurrent ShepherdAgents",
    )

    # LLM parameters
    p.add_argument(
        "--llm-url",
        default=None,
        help="LLM endpoint URL (overrides config)",
    )
    p.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name (overrides config)",
    )

    # State management
    p.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path for state checkpoint",
    )
    p.add_argument(
        "--results-path",
        default=None,
        help="Path for results JSONL",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if available",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignore existing checkpoint",
    )

    # Academy exchange (for connecting to remote agents)
    p.add_argument(
        "--exchange-url",
        default=None,
        help="Academy HTTP exchange URL (e.g., http://spark:8080)",
    )
    p.add_argument(
        "--redis-host",
        default=None,
        help="Redis hostname for distributed exchange (e.g., spark)",
    )
    p.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )

    # Globus Compute (optional, alternative to Academy)
    p.add_argument(
        "--gc-endpoint-cheap",
        default=None,
        help="Globus Compute endpoint ID for cheap tests",
    )
    p.add_argument(
        "--gc-endpoint-gpu",
        default=None,
        help="Globus Compute endpoint ID for GPU tests",
    )
    p.add_argument(
        "--gc-functions",
        default=None,
        help="JSON file with Globus Compute function map (from register_gc_functions.py)",
    )

    return p.parse_args()


def build_config(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """Build configuration from files and CLI args.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (generator_config, shepherd_config, gc_function_map)
    """
    # Load base config
    base_config = {}
    if Path(args.config).exists():
        base_config = load_config(args.config, args.config_local)
    else:
        logger.warning("Config file %s not found, using defaults", args.config)

    # Extract sections
    generator_config = base_config.get("generator", {})
    shepherd_config = base_config.get("shepherd", {})

    # Apply CLI overrides to generator config
    if args.max_iterations is not None:
        generator_config.setdefault("generation", {})["max_iterations"] = args.max_iterations
    if args.candidates_per_iteration is not None:
        generator_config.setdefault("generation", {})["candidates_per_iteration"] = args.candidates_per_iteration
    if args.patience is not None:
        generator_config.setdefault("convergence", {})["patience"] = args.patience
    if args.no_llm_judgment:
        generator_config.setdefault("convergence", {})["llm_judgment"] = False
    if args.shepherd_budget is not None:
        generator_config.setdefault("shepherd", {})["budget_per_candidate"] = args.shepherd_budget
    if args.max_concurrent_shepherds is not None:
        generator_config.setdefault("shepherd", {})["max_concurrent_shepherds"] = args.max_concurrent_shepherds
    if args.checkpoint_path:
        generator_config.setdefault("state", {})["checkpoint_path"] = args.checkpoint_path
    if args.results_path:
        generator_config.setdefault("state", {})["results_path"] = args.results_path

    # Apply LLM overrides
    if args.llm_url:
        generator_config.setdefault("llm", {})["shared_url"] = args.llm_url
        shepherd_config.setdefault("llm", {})["shared_url"] = args.llm_url
    if args.llm_model:
        generator_config.setdefault("llm", {})["model"] = args.llm_model
        shepherd_config.setdefault("llm", {})["model"] = args.llm_model

    # Apply GC endpoint overrides
    if args.gc_endpoint_cheap:
        shepherd_config.setdefault("endpoints", {})["cheap"] = args.gc_endpoint_cheap
    if args.gc_endpoint_gpu:
        shepherd_config.setdefault("endpoints", {})["gpu"] = args.gc_endpoint_gpu

    # Handle checkpoint
    if args.fresh:
        checkpoint_path = generator_config.get("state", {}).get("checkpoint_path", "data/generator_state.json")
        if Path(checkpoint_path).exists():
            logger.info("Removing existing checkpoint: %s", checkpoint_path)
            Path(checkpoint_path).unlink()

    # Build GC function map from config or CLI
    gc_function_map = {}

    # First check CLI option for function map JSON
    if args.gc_functions:
        import json as json_module
        with open(args.gc_functions) as f:
            gc_data = json_module.load(f)
        gc_function_map = gc_data.get("functions", {})
        # Also use endpoint from JSON if not overridden
        if not args.gc_endpoint_gpu and gc_data.get("endpoint_id"):
            shepherd_config.setdefault("endpoints", {})["gpu"] = gc_data["endpoint_id"]
            shepherd_config.setdefault("endpoints", {})["cheap"] = gc_data["endpoint_id"]
        logger.info("Loaded %d GC functions from %s", len(gc_function_map), args.gc_functions)
    else:
        # Fall back to config file
        gc_config = base_config.get("globus_compute", {})
        if gc_config.get("enabled"):
            functions = gc_config.get("functions", {})
            for name, func_id in functions.items():
                if func_id:
                    gc_function_map[name] = func_id

    return generator_config, shepherd_config, gc_function_map


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.info("Building configuration...")
    generator_config, shepherd_config, gc_function_map = build_config(args)

    logger.info("Generator config: %s", generator_config)
    logger.info("Shepherd config: %s", shepherd_config)

    # Create exchange factory (local, Redis, or HTTP)
    if args.redis_host:
        from academy.exchange import RedisExchangeFactory
        exchange_factory = RedisExchangeFactory(hostname=args.redis_host, port=args.redis_port)
        logger.info("Using Redis exchange: %s:%d", args.redis_host, args.redis_port)
    elif args.exchange_url:
        from academy.exchange import HttpExchangeFactory
        exchange_factory = HttpExchangeFactory(url=args.exchange_url)
        logger.info("Using HTTP exchange: %s", args.exchange_url)
    else:
        exchange_factory = LocalExchangeFactory()
        logger.info("Using local exchange")

    # Create Manager and launch GeneratorAgent
    logger.info("Starting GeneratorAgent...")

    try:
        async with await Manager.from_exchange_factory(exchange_factory) as manager:
            generator = await manager.launch(
                GeneratorAgent,
                kwargs={
                    "config": generator_config,
                    "shepherd_config": shepherd_config,
                    "gc_function_map": gc_function_map,
                },
            )

            logger.info("GeneratorAgent launched, waiting for completion...")

            # The generator has a @loop that runs autonomously
            # We can monitor status periodically or just wait
            consecutive_failures = 0
            while True:
                await asyncio.sleep(10)

                try:
                    status = await generator.get_status({})
                    consecutive_failures = 0  # Reset on success

                    if not status.get("running", True):
                        logger.info("Generator stopped: %s", status.get("stop_reason"))
                        break

                    logger.info(
                        "Status: iteration=%d, best_score=%.1f, total=%d",
                        status.get("iteration", 0),
                        status.get("best_score", 0),
                        status.get("total_candidates", 0),
                    )
                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures >= 2:
                        # Agent likely terminated - exit polling loop
                        logger.info("Generator agent terminated")
                        break
                    logger.warning("Failed to get status: %s", e)

            # Get final results (may fail if agent already terminated)
            try:
                results = await generator.get_results({"top_n": 10})
                logger.info("=== Final Results ===")
                for i, c in enumerate(results.get("candidates", []), 1):
                    candidate = c.get("candidate", {})
                    assessment = c.get("final_assessment", {})
                    logger.info(
                        "%d. Score=%d %s - %s",
                        i,
                        assessment.get("viability_score", 0),
                        assessment.get("recommendation", "?"),
                        _format_candidate(candidate),
                    )
            except Exception:
                logger.info("=== Results saved to data/generator_results.jsonl ===")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        return 1

    return 0


def _format_candidate(candidate: dict) -> str:
    """Format candidate for display."""
    support = candidate.get("support", "?")
    metals = candidate.get("metals", [])
    metal_str = "/".join(
        f"{m.get('element', '?')}{m.get('wt_pct', 0):.0f}"
        for m in metals
    )
    return f"{metal_str} on {support}"


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
