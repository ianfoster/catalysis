#!/usr/bin/env python3
"""Run Academy agents on Polaris.

This script starts the agent infrastructure on Polaris for DFT calculations:
1. QEAgent - Quantum ESPRESSO DFT (GPU-accelerated)
2. GPAWAgent - GPAW DFT (MPI-parallel)
3. ShepherdAgent(s) - evaluate candidates using LLM + simulation agents

Architecture:
    Mac (local)                          Polaris (ALCF)
    ┌─────────────┐                      ┌────────────────────────────────┐
    │ Generator   │                      │  PBS Job                       │
    │ Agent       │◄────── Redis ───────►│  ┌────────────────────────────┤
    └─────────────┘       Exchange       │  │ Academy Manager             │
                                         │  │                             │
                                         │  │  ┌─────────┐ ┌─────────┐   │
                                         │  │  │Shepherd │ │Shepherd │   │
                                         │  │  │Agent    │ │Agent    │   │
                                         │  │  └────┬────┘ └────┬────┘   │
                                         │  │       │           │        │
                                         │  │  ┌────▼────┐ ┌────▼────┐   │
                                         │  │  │QEAgent  │ │GPAWAgent│   │
                                         │  │  │(A100 GPU│ │(64 CPUs)│   │
                                         │  │  └─────────┘ └─────────┘   │
                                         │  │                             │
                                         │  │  LLM via Argonne Inference  │
                                         │  └────────────────────────────┤
                                         └────────────────────────────────┘

Usage:
    # On Polaris (within PBS job):
    python scripts/run_polaris_agents.py --redis-host <redis-server>

    # With Argonne inference API:
    python scripts/run_polaris_agents.py \\
        --llm-url https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1 \\
        --redis-host <redis-server>

Prerequisites:
    - PBS job running with GPU allocation
    - Redis server accessible for exchange
    - ARGONNE_ACCESS_TOKEN environment variable set (for LLM)
    - QE and/or GPAW modules loaded
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml


# Polaris-focused agents (DFT-heavy)
POLARIS_AGENT_CLASSES = {
    # DFT agents (primary on Polaris)
    "qe": ("skills.sim_agents.qe_agent", "QEAgent"),
    "gpaw": ("skills.sim_agents.gpaw_agent", "GPAWAgent"),
    # ML potentials (can run on GPU)
    "mace": ("skills.sim_agents.mace_agent", "MACEAgent"),
    "chgnet": ("skills.sim_agents.chgnet_agent", "CHGNetAgent"),
    # Supporting agents
    "cantera": ("skills.sim_agents.cantera_agent", "CanteraAgent"),
    "stability": ("skills.sim_agents.stability_agent", "StabilityAgent"),
    "surrogate": ("skills.sim_agents.surrogate_agent", "SurrogateAgent"),
    # LLM proxy
    "llm_proxy": ("skills.llm_proxy_agent", "LLMProxyAgent"),
}


def check_polaris_environment():
    """Verify we're running on Polaris with proper setup."""
    checks = {}

    # Check hostname
    hostname = os.uname().nodename
    checks["hostname"] = hostname
    checks["is_polaris"] = hostname.startswith(("polaris", "x3"))

    # Check PBS job
    checks["pbs_jobid"] = os.environ.get("PBS_JOBID")
    checks["pbs_nodefile"] = os.environ.get("PBS_NODEFILE")

    # Check GPU
    checks["cuda_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    # Check QE
    try:
        import subprocess
        result = subprocess.run(["which", "pw.x"], capture_output=True, text=True)
        checks["qe_available"] = result.returncode == 0
        checks["qe_path"] = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        checks["qe_available"] = False

    # Check GPAW
    try:
        import gpaw
        checks["gpaw_available"] = True
        checks["gpaw_version"] = gpaw.__version__
    except ImportError:
        checks["gpaw_available"] = False

    # Check Argonne token
    checks["argonne_token_set"] = bool(os.environ.get("ARGONNE_ACCESS_TOKEN"))

    return checks


def import_agent_class(name: str):
    """Dynamically import an agent class."""
    if name not in POLARIS_AGENT_CLASSES:
        raise ValueError(f"Unknown agent: {name}. Available: {list(POLARIS_AGENT_CLASSES.keys())}")

    module_path, class_name = POLARIS_AGENT_CLASSES[name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


async def launch_simulation_agents(manager, agent_names: list[str], device: str = "cuda"):
    """Launch simulation agents.

    Args:
        manager: Academy Manager
        agent_names: List of agent names to launch
        device: Compute device for ML agents (cuda for Polaris A100s)

    Returns:
        Dict mapping agent names to agent instances
    """
    sim_agents = {}

    for name in agent_names:
        if name == "llm_proxy":
            continue  # Launched separately

        try:
            AgentClass = import_agent_class(name)

            # Configure agent based on type
            if name in ("mace", "chgnet"):
                kwargs = {"device": device}
            elif name == "qe":
                # QE agent with Polaris pseudopotential path
                pseudo_dir = os.environ.get(
                    "PSEUDO_DIR",
                    "/eagle/projects/catalyst/pseudopotentials"
                )
                kwargs = {"pseudo_dir": pseudo_dir}
            elif name == "gpaw":
                kwargs = {}  # GPAW uses GPAW_SETUP_PATH env var
            else:
                kwargs = {}

            logging.info(f"Launching {name} agent...")
            agent = await manager.launch(AgentClass, kwargs=kwargs)

            # Verify agent is ready
            status = await agent.get_status({})
            if status.get("ok") or status.get("ready"):
                sim_agents[name] = agent
                logging.info(f"  {name} agent ready: {status}")
            else:
                logging.warning(f"  {name} agent not ready: {status}")

        except ImportError as e:
            logging.warning(f"Cannot load {name} agent: {e}")
        except Exception as e:
            logging.error(f"Failed to launch {name} agent: {e}")

    return sim_agents


async def run_agents(
    llm_url: str,
    llm_model: str,
    agent_names: list[str],
    num_shepherds: int,
    budget: float,
    device: str = "cuda",
    redis_host: str = None,
    redis_port: int = 6379,
):
    """Run Academy agents on Polaris.

    Args:
        llm_url: URL to LLM server (Argonne inference API)
        llm_model: Model name for LLM
        agent_names: List of simulation agent names to launch
        num_shepherds: Number of shepherd agents to spawn
        budget: Default budget per candidate
        device: Compute device (cuda for Polaris A100s)
        redis_host: Redis hostname for distributed exchange
        redis_port: Redis port
    """
    from academy.manager import Manager

    from skills.shepherd import ShepherdAgent

    # Environment check
    env_checks = check_polaris_environment()
    logging.info("Polaris environment check:")
    for key, val in env_checks.items():
        logging.info(f"  {key}: {val}")

    if not env_checks.get("pbs_jobid"):
        logging.warning("Not running in PBS job - some features may not work")

    logging.info("=" * 60)
    logging.info("Starting Academy agents on Polaris")
    logging.info(f"  LLM URL: {llm_url}")
    logging.info(f"  LLM Model: {llm_model}")
    logging.info(f"  Simulation agents: {agent_names}")
    logging.info(f"  Shepherds: {num_shepherds}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Redis: {redis_host}:{redis_port}" if redis_host else "  Exchange: local")
    logging.info("=" * 60)

    # Create exchange factory
    if redis_host:
        from academy.exchange import RedisExchangeFactory
        exchange_factory = RedisExchangeFactory(hostname=redis_host, port=redis_port)
        logging.info(f"Using Redis exchange: {redis_host}:{redis_port}")
    else:
        from academy.exchange import LocalExchangeFactory
        exchange_factory = LocalExchangeFactory()
        logging.warning("Using local exchange - GeneratorAgent must run in same process!")

    # Shutdown event
    shutdown_event = asyncio.Event()

    def handle_signal(sig):
        logging.info(f"Received signal {sig}, shutting down...")
        shutdown_event.set()

    # Register signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

    async with await Manager.from_exchange_factory(exchange_factory) as manager:
        # Launch simulation agents
        logging.info("Launching simulation agents...")
        sim_agents = await launch_simulation_agents(manager, agent_names, device)

        if not sim_agents:
            logging.error("No simulation agents could be launched!")
            return

        logging.info(f"Launched {len(sim_agents)} simulation agents: {list(sim_agents.keys())}")

        # Launch LLM Proxy Agent
        logging.info("Launching LLM Proxy Agent...")
        from skills.llm_proxy_agent import LLMProxyAgent

        # Check for Argonne auth token
        api_key = os.environ.get("ARGONNE_ACCESS_TOKEN")
        if not api_key:
            logging.warning("ARGONNE_ACCESS_TOKEN not set - LLM calls may fail")

        llm_proxy = await manager.launch(
            LLMProxyAgent,
            kwargs={
                "llm_url": llm_url,
                "model": llm_model,
                "api_key": api_key,
            },
        )
        llm_status = await llm_proxy.get_status({})
        logging.info(f"  LLM Proxy ready: {llm_status.get('model')}")

        # Shepherd config
        shepherd_config = {
            "llm": {
                "mode": "remote",
                "model": llm_model,
            },
            "budget": {
                "default": budget,
                "max": budget * 10,
            },
            "timeouts": {
                "llm_call": 120,  # Longer timeout for Argonne API
                "test_poll_interval": 2.0,
                "test_max_wait": 7200,  # 2 hours for DFT
            },
        }

        # Launch ShepherdAgents
        shepherds = []
        for i in range(num_shepherds):
            shepherd_id = f"P{i+1}"  # P for Polaris
            logging.info(f"Launching ShepherdAgent {shepherd_id} ({i+1}/{num_shepherds})...")
            shepherd = await manager.launch(
                ShepherdAgent,
                kwargs={
                    "config": shepherd_config,
                    "sim_agents": sim_agents,
                    "llm_proxy": llm_proxy,
                    "llm_url": llm_url,
                    "llm_model": llm_model,
                    "redis_host": redis_host,
                    "redis_port": redis_port,
                    "shepherd_id": shepherd_id,
                },
            )
            shepherds.append(shepherd)

        logging.info("=" * 60)
        logging.info("All Polaris agents launched!")
        logging.info(f"  DFT agents: {[a for a in sim_agents if a in ('qe', 'gpaw')]}")
        logging.info(f"  ML agents: {[a for a in sim_agents if a in ('mace', 'chgnet')]}")
        logging.info(f"  Other agents: {[a for a in sim_agents if a not in ('qe', 'gpaw', 'mace', 'chgnet')]}")
        logging.info(f"  Shepherd agents: {num_shepherds}")
        logging.info("=" * 60)
        logging.info("Agents ready to receive evaluation requests")
        logging.info("Press Ctrl+C to shutdown")
        logging.info("=" * 60)

        # Export agent info
        agent_info = {
            "host": "polaris",
            "pbs_jobid": os.environ.get("PBS_JOBID"),
            "simulation_agents": list(sim_agents.keys()),
            "shepherd_count": num_shepherds,
            "llm_url": llm_url,
            "llm_model": llm_model,
            "device": device,
            "redis_host": redis_host,
            "redis_port": redis_port,
            "qe_available": env_checks.get("qe_available"),
            "gpaw_available": env_checks.get("gpaw_available"),
        }

        info_file = Path("polaris_agents_info.json")
        with open(info_file, "w") as f:
            json.dump(agent_info, f, indent=2)
        logging.info(f"Agent info saved to {info_file}")

        # Wait for shutdown
        await shutdown_event.wait()
        logging.info("Shutting down agents...")


def main():
    parser = argparse.ArgumentParser(
        description="Run Academy agents on Polaris for DFT calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # With Redis exchange (recommended for distributed)
    python scripts/run_polaris_agents.py --redis-host login1.polaris.alcf.anl.gov

    # With Argonne inference API
    python scripts/run_polaris_agents.py \\
        --llm-url https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1 \\
        --redis-host <redis-server>

    # DFT agents only (QE + GPAW)
    python scripts/run_polaris_agents.py --agents qe gpaw --redis-host <redis-server>

    # Test mode (check agent availability)
    python scripts/run_polaris_agents.py --check-env
        """,
    )

    parser.add_argument(
        "--llm-url",
        default="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        help="URL to LLM server (default: Argonne inference API)",
    )
    parser.add_argument(
        "--llm-model",
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
        help="LLM model name",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["qe", "gpaw", "mace", "chgnet", "cantera", "stability", "surrogate"],
        help="Simulation agents to launch (default: DFT + ML + support agents)",
    )
    parser.add_argument(
        "--num-shepherds",
        type=int,
        default=2,
        help="Number of ShepherdAgents (default: 2 for Polaris)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=200.0,
        help="Default budget per candidate (default: 200 for DFT)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Compute device for ML agents (default: cuda for A100s)",
    )
    parser.add_argument(
        "--redis-host",
        required=False,
        help="Redis hostname for distributed exchange (REQUIRED for multi-node)",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check Polaris environment and exit",
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

    # Environment check mode
    if args.check_env:
        checks = check_polaris_environment()
        print("\nPolaris Environment Check")
        print("=" * 40)
        for key, val in checks.items():
            status = "OK" if val else "MISSING"
            if isinstance(val, bool):
                print(f"  {key}: {status}")
            else:
                print(f"  {key}: {val}")
        print("=" * 40)

        # Summary
        ready = checks.get("pbs_jobid") and (checks.get("qe_available") or checks.get("gpaw_available"))
        if ready:
            print("\nEnvironment ready for DFT agents!")
        else:
            print("\nEnvironment NOT ready. Check:")
            if not checks.get("pbs_jobid"):
                print("  - Submit a PBS job first")
            if not checks.get("qe_available"):
                print("  - Load QE module: module load quantum-espresso")
            if not checks.get("gpaw_available"):
                print("  - Activate conda env with GPAW")
        return

    # Load config for defaults
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Override defaults from config
        shepherds_config = config.get("academy", {}).get("shepherds", {})
        if args.num_shepherds == 2 and "num_concurrent" in shepherds_config:
            args.num_shepherds = min(shepherds_config["num_concurrent"], 4)  # Cap at 4 for Polaris
        if args.budget == 200.0 and "budget_per_candidate" in shepherds_config:
            args.budget = shepherds_config["budget_per_candidate"]

    # Warn if no Redis
    if not args.redis_host:
        logging.warning("No --redis-host specified. Using local exchange.")
        logging.warning("For distributed operation, specify a Redis server.")

    # Run agents
    asyncio.run(run_agents(
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        agent_names=args.agents,
        num_shepherds=args.num_shepherds,
        budget=args.budget,
        device=args.device,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    ))


if __name__ == "__main__":
    main()
