#!/usr/bin/env python3
"""Launch Spark agents from Mac via Globus Compute.

This script runs on Mac and uses Globus Compute to:
1. Start the vLLM server on Spark
2. Start all Academy agents (Shepherds + SimAgents) on Spark
3. Optionally run the Generator locally

Architecture:
    Mac (local)                          Spark (remote via GC)
    ┌─────────────────┐                  ┌─────────────────────────────────┐
    │ launch_spark.py │ ──GC task──►     │ vLLM Server (port 8000)         │
    │                 │                  │                                 │
    │ GeneratorAgent  │ ◄──Redis──►      │ ShepherdAgent(s)                │
    │ (OpenAI/Argonne)│                  │   └─► SimAgents (MACE, etc.)    │
    └─────────────────┘                  └─────────────────────────────────┘

Usage:
    # Just start agents on Spark (keep running)
    python scripts/launch_spark.py --endpoint $GC_ENDPOINT --start-agents

    # Start agents and run discovery
    python scripts/launch_spark.py --endpoint $GC_ENDPOINT --start-agents --run-discovery

    # Connect to already-running agents
    python scripts/launch_spark.py --endpoint $GC_ENDPOINT --run-discovery
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


# ============================================================================
# GC Functions (must be self-contained for serialization)
# ============================================================================

def start_spark_agents_gc(config: dict) -> dict:
    """Start Academy agents on Spark. Runs as a GC task.

    This function is SELF-CONTAINED for GC serialization.
    It starts vLLM + all simulation agents + shepherd agents.

    Args:
        config: Dict with:
            - llm_model: Model name for vLLM
            - llm_port: Port for vLLM server
            - redis_host: Redis hostname
            - redis_port: Redis port
            - num_shepherds: Number of shepherd agents
            - agents: List of simulation agent names
            - device: "cpu" or "cuda"
            - run_duration: How long to keep running (0 = indefinite)

    Returns:
        Status dict with agent info
    """
    import asyncio
    import logging
    import signal
    import subprocess
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("spark_agents_gc")

    llm_model = config.get("llm_model", "meta-llama/Llama-3.1-8B-Instruct")
    llm_port = config.get("llm_port", 8000)
    redis_host = config.get("redis_host", "localhost")
    redis_port = config.get("redis_port", 6379)
    num_shepherds = config.get("num_shepherds", 4)
    agent_names = config.get("agents", ["mace", "chgnet", "cantera", "surrogate", "stability"])
    device = config.get("device", "cuda")
    run_duration = config.get("run_duration", 0)  # 0 = run indefinitely

    result = {
        "ok": False,
        "llm_url": f"http://localhost:{llm_port}/v1",
        "redis_host": redis_host,
        "redis_port": redis_port,
        "agents_started": [],
        "shepherds": 0,
    }

    # Step 1: Check/start vLLM server
    logger.info(f"Checking vLLM server on port {llm_port}...")

    import urllib.request
    try:
        req = urllib.request.urlopen(f"http://localhost:{llm_port}/v1/models", timeout=5)
        logger.info("vLLM server already running")
    except Exception:
        logger.info(f"Starting vLLM server with model {llm_model}...")
        # Start vLLM in background via docker
        cmd = [
            "docker", "run", "--gpus", "all",
            "-p", f"{llm_port}:{llm_port}",
            "--rm", "-d", "--name", "vllm-server",
            "nvcr.io/nvidia/vllm:25.11-py3",
            "vllm", "serve", llm_model,
            "--host", "0.0.0.0", "--port", str(llm_port),
            "--enforce-eager", "--trust-remote-code",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("vLLM container started, waiting for model load...")
            # Wait for server to be ready
            for i in range(60):  # Wait up to 5 minutes
                time.sleep(5)
                try:
                    req = urllib.request.urlopen(f"http://localhost:{llm_port}/v1/models", timeout=5)
                    logger.info("vLLM server ready")
                    break
                except Exception:
                    logger.info(f"Still waiting for vLLM... ({(i+1)*5}s)")
            else:
                result["error"] = "vLLM server did not start in time"
                return result
        except subprocess.CalledProcessError as e:
            # Maybe container already exists
            if b"already in use" in e.stderr:
                logger.info("vLLM container already exists")
            else:
                result["error"] = f"Failed to start vLLM: {e.stderr.decode()}"
                return result

    result["vllm_ready"] = True

    # Step 2: Start Academy agents
    logger.info("Starting Academy agents...")

    async def run_agents():
        from academy.manager import Manager
        from academy.exchange import RedisExchangeFactory

        # Import agent classes
        from skills.shepherd import ShepherdAgent
        from skills.llm_proxy_agent import LLMProxyAgent

        AGENT_CLASSES = {
            "mace": ("skills.sim_agents.mace_agent", "MACEAgent"),
            "chgnet": ("skills.sim_agents.chgnet_agent", "CHGNetAgent"),
            "m3gnet": ("skills.sim_agents.m3gnet_agent", "M3GNetAgent"),
            "cantera": ("skills.sim_agents.cantera_agent", "CanteraAgent"),
            "stability": ("skills.sim_agents.stability_agent", "StabilityAgent"),
            "surrogate": ("skills.sim_agents.surrogate_agent", "SurrogateAgent"),
            "qe": ("skills.sim_agents.qe_agent", "QEAgent"),
            "gpaw": ("skills.sim_agents.gpaw_agent", "GPAWAgent"),
            "openmm": ("skills.sim_agents.openmm_agent", "OpenMMAgent"),
            "gromacs": ("skills.sim_agents.gromacs_agent", "GROMACSAgent"),
            "catmap": ("skills.sim_agents.catmap_agent", "CatMAPAgent"),
        }

        def import_agent_class(name):
            if name not in AGENT_CLASSES:
                return None
            module_path, class_name = AGENT_CLASSES[name]
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        exchange_factory = RedisExchangeFactory(hostname=redis_host, port=redis_port)

        async with await Manager.from_exchange_factory(exchange_factory) as manager:
            sim_agents = {}

            # Launch simulation agents
            for name in agent_names:
                try:
                    AgentClass = import_agent_class(name)
                    if AgentClass is None:
                        continue

                    if name in ("mace", "chgnet"):
                        kwargs = {"device": device}
                    elif name == "m3gnet":
                        kwargs = {"device": device, "container_url": "http://localhost:8080"}
                    else:
                        kwargs = {}

                    logger.info(f"Launching {name} agent...")
                    agent = await manager.launch(AgentClass, kwargs=kwargs)
                    status = await agent.get_status({})
                    if status.get("ok") or status.get("ready"):
                        sim_agents[name] = agent
                        result["agents_started"].append(name)
                        logger.info(f"  {name} ready")
                except Exception as e:
                    logger.warning(f"Failed to launch {name}: {e}")

            # Launch LLM Proxy
            logger.info("Launching LLM Proxy Agent...")
            llm_proxy = await manager.launch(
                LLMProxyAgent,
                kwargs={
                    "llm_url": f"http://localhost:{llm_port}/v1",
                    "model": llm_model,
                },
            )

            # Launch Shepherd Agents
            shepherd_config = {
                "llm": {"mode": "remote", "model": llm_model},
                "budget": {"default": 100.0, "max": 1000.0},
                "timeouts": {"llm_call": 60, "test_poll_interval": 1.0, "test_max_wait": 3600},
            }

            shepherds = []
            for i in range(num_shepherds):
                logger.info(f"Launching ShepherdAgent {i+1}/{num_shepherds}...")
                shepherd = await manager.launch(
                    ShepherdAgent,
                    kwargs={
                        "config": shepherd_config,
                        "sim_agents": sim_agents,
                        "llm_proxy": llm_proxy,
                        "llm_url": f"http://localhost:{llm_port}/v1",
                        "llm_model": llm_model,
                        "redis_host": redis_host,
                        "redis_port": redis_port,
                        "shepherd_id": f"S{i+1}",
                    },
                )
                shepherds.append(shepherd)

            result["shepherds"] = len(shepherds)
            result["ok"] = True

            logger.info(f"All agents started: {result['agents_started']}, {result['shepherds']} shepherds")

            # Keep running
            if run_duration > 0:
                logger.info(f"Running for {run_duration} seconds...")
                await asyncio.sleep(run_duration)
            else:
                logger.info("Running indefinitely (Ctrl+C to stop)...")
                shutdown = asyncio.Event()

                def handle_signal():
                    shutdown.set()

                import signal as sig
                asyncio.get_event_loop().add_signal_handler(sig.SIGTERM, handle_signal)
                asyncio.get_event_loop().add_signal_handler(sig.SIGINT, handle_signal)

                await shutdown.wait()

            logger.info("Shutting down agents...")

        return result

    try:
        return asyncio.run(run_agents())
    except Exception as e:
        result["error"] = str(e)
        return result


def check_agents_gc(config: dict) -> dict:
    """Check if agents are running on Spark. Runs as a GC task.

    Args:
        config: Dict with redis_host, redis_port

    Returns:
        Status dict with discovered agents
    """
    import asyncio

    redis_host = config.get("redis_host", "localhost")
    redis_port = config.get("redis_port", 6379)

    result = {
        "ok": False,
        "redis_host": redis_host,
        "redis_port": redis_port,
    }

    # First check Redis directly for agent keys
    try:
        import redis
        r = redis.Redis(host=redis_host, port=redis_port)
        r.ping()
        result["redis_connected"] = True

        # Look for agent-related keys
        keys = r.keys("*")
        result["redis_keys"] = [k.decode() if isinstance(k, bytes) else k for k in keys[:20]]
        result["total_keys"] = len(keys)
    except Exception as e:
        result["redis_connected"] = False
        result["redis_error"] = str(e)
        return result

    # Try Academy discovery
    async def check():
        from academy.exchange import RedisExchangeFactory

        exchange_factory = RedisExchangeFactory(hostname=redis_host, port=redis_port)
        exchange = await exchange_factory()

        # Get all registered agents from the exchange
        # The exchange tracks registrations
        try:
            # Try to list agents via Redis keys
            agent_keys = [k for k in result["redis_keys"] if "agent" in k.lower() or "Agent" in k]
            result["agent_keys"] = agent_keys
            result["ok"] = True
        except Exception as e:
            result["discovery_error"] = str(e)

        await exchange.close()
        return result

    try:
        return asyncio.run(check())
    except Exception as e:
        result["error"] = str(e)
        # Still return what we have from Redis check
        result["ok"] = result.get("redis_connected", False)
        return result


# ============================================================================
# Main script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Launch Spark agents from Mac via Globus Compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start agents on Spark (keeps running)
    python scripts/launch_spark.py --endpoint $GC_ENDPOINT --start-agents

    # Start agents with specific config
    python scripts/launch_spark.py --endpoint $GC_ENDPOINT --start-agents \\
        --llm-model meta-llama/Llama-3.1-8B-Instruct --num-shepherds 8

    # Check if agents are running
    python scripts/launch_spark.py --endpoint $GC_ENDPOINT --check-agents

    # Start agents and run discovery
    python scripts/launch_spark.py --endpoint $GC_ENDPOINT --start-agents --run-discovery
        """,
    )

    parser.add_argument(
        "--endpoint",
        required=True,
        help="Globus Compute endpoint ID on Spark",
    )

    # Actions
    parser.add_argument(
        "--start-agents",
        action="store_true",
        help="Start vLLM + Academy agents on Spark via GC",
    )
    parser.add_argument(
        "--check-agents",
        action="store_true",
        help="Check if agents are running on Spark",
    )
    parser.add_argument(
        "--run-discovery",
        action="store_true",
        help="Run GeneratorAgent locally after starting agents",
    )

    # Agent config
    parser.add_argument(
        "--llm-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="LLM model for Spark vLLM server",
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        default=8000,
        help="Port for vLLM server (default: 8000)",
    )
    parser.add_argument(
        "--num-shepherds",
        type=int,
        default=4,
        help="Number of ShepherdAgents to start (default: 4)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["mace", "chgnet", "cantera", "surrogate", "stability"],
        help="Simulation agents to start",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device for ML agents (default: cuda)",
    )
    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis hostname on Spark (default: localhost)",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )
    parser.add_argument(
        "--run-duration",
        type=int,
        default=3600,
        help="How long to keep agents running in seconds (default: 3600, 0=indefinite)",
    )

    # Generator config (if --run-discovery)
    parser.add_argument(
        "--generator-llm",
        choices=["openai", "argonne"],
        default="openai",
        help="LLM for Generator (default: openai)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max discovery iterations (default: 3)",
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
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.start_agents and not args.check_agents and not args.run_discovery:
        parser.error("Must specify at least one of: --start-agents, --check-agents, --run-discovery")

    from globus_compute_sdk import Client, Executor

    client = Client()

    # Check agents
    if args.check_agents:
        logging.info("Checking agents on Spark...")
        func_id = client.register_function(check_agents_gc)

        with Executor(endpoint_id=args.endpoint) as ex:
            future = ex.submit_to_registered_function(
                func_id,
                args=({"redis_host": args.redis_host, "redis_port": args.redis_port},),
            )
            result = future.result(timeout=60)

        if result.get("ok"):
            logging.info(f"Redis connected: {result.get('redis_connected')}")
            logging.info(f"Total Redis keys: {result.get('total_keys', 0)}")
            if result.get("redis_keys"):
                logging.info("Redis keys:")
                for key in result.get("redis_keys", []):
                    logging.info(f"  - {key}")
            if result.get("agent_keys"):
                logging.info(f"Agent-related keys: {result.get('agent_keys')}")
        else:
            logging.error(f"Check failed: {result.get('error', result.get('redis_error', 'unknown'))}")

        if not args.start_agents and not args.run_discovery:
            return 0

    # Start agents
    if args.start_agents:
        logging.info("=" * 60)
        logging.info("Starting agents on Spark via Globus Compute")
        logging.info("=" * 60)
        logging.info(f"Endpoint: {args.endpoint}")
        logging.info(f"LLM Model: {args.llm_model}")
        logging.info(f"Agents: {args.agents}")
        logging.info(f"Shepherds: {args.num_shepherds}")
        logging.info(f"Device: {args.device}")
        logging.info(f"Run duration: {args.run_duration}s")
        logging.info("=" * 60)

        config = {
            "llm_model": args.llm_model,
            "llm_port": args.llm_port,
            "redis_host": args.redis_host,
            "redis_port": args.redis_port,
            "num_shepherds": args.num_shepherds,
            "agents": args.agents,
            "device": args.device,
            "run_duration": args.run_duration,
        }

        func_id = client.register_function(start_spark_agents_gc)
        logging.info(f"Registered GC function: {func_id}")

        with Executor(endpoint_id=args.endpoint) as ex:
            logging.info("Submitting task to start agents...")
            future = ex.submit_to_registered_function(func_id, args=(config,))

            if args.run_discovery:
                # Don't wait for completion, just wait for agents to be ready
                logging.info("Waiting for agents to start...")
                time.sleep(30)  # Give agents time to initialize

                # Check if they're ready
                check_func_id = client.register_function(check_agents_gc)
                for attempt in range(10):
                    check_future = ex.submit_to_registered_function(
                        check_func_id,
                        args=({"redis_host": args.redis_host, "redis_port": args.redis_port},),
                    )
                    try:
                        check_result = check_future.result(timeout=30)
                        if check_result.get("shepherds_found", 0) > 0:
                            logging.info(f"Agents ready: {check_result['shepherds_found']} shepherds")
                            break
                    except Exception as e:
                        logging.warning(f"Check attempt {attempt+1} failed: {e}")
                    time.sleep(10)
                else:
                    logging.error("Agents did not start in time")
                    return 1
            else:
                # Wait for completion
                logging.info("Waiting for GC task to complete (agents running)...")
                try:
                    result = future.result(timeout=args.run_duration + 600)
                    if result.get("ok"):
                        logging.info("Agents completed successfully")
                        logging.info(f"  Agents: {result.get('agents_started')}")
                        logging.info(f"  Shepherds: {result.get('shepherds')}")
                    else:
                        logging.error(f"Agent startup failed: {result.get('error')}")
                        return 1
                except TimeoutError:
                    logging.info("GC task timed out (agents may still be running)")

    # Run discovery
    if args.run_discovery:
        logging.info("=" * 60)
        logging.info("Running Generator locally")
        logging.info("=" * 60)

        # This requires Redis to be accessible from Mac
        # For now, use GC-based discovery instead
        logging.info("Note: For Redis-based discovery, use run_generator.py")
        logging.info("Using GC-based evaluation instead...")

        # Import and run
        from scripts.run_discovery import run_discovery, load_config

        config_path = Path("config.yaml")
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            config = {}

        # Setup generator LLM
        gen_config = config.setdefault("generator", {})
        gen_llm = gen_config.setdefault("llm", {})

        if args.generator_llm == "openai":
            gen_llm["base_url"] = "https://api.openai.com/v1"
            gen_llm["model"] = "gpt-4o"
            gen_llm["api_key_env"] = "OPENAI_API_KEY"
        else:
            gen_llm["base_url"] = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
            gen_llm["model"] = "meta-llama/Meta-Llama-3.1-70B-Instruct"
            gen_llm["api_key_env"] = "ARGONNE_ACCESS_TOKEN"

        gen_config.setdefault("generation", {})["max_iterations"] = args.max_iterations

        shepherd_config = config.setdefault("shepherd", {})
        shepherd_config["llm_url"] = f"http://localhost:{args.llm_port}/v1"
        shepherd_config["llm_model"] = args.llm_model

        results = asyncio.run(
            run_discovery(
                config=config,
                gc_endpoint=args.endpoint,
                shepherd_llm_url=f"http://localhost:{args.llm_port}/v1",
            )
        )

        print("\n" + "=" * 60)
        print("Discovery Summary")
        print("=" * 60)
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Best score: {results.get('best_score', 0):.1f}")
            print(f"Total evaluated: {results.get('total_evaluated', 0)}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
