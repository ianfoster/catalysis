#!/usr/bin/env python3
"""Run Academy agents on Spark.

This script starts the agent infrastructure on Spark:
1. Individual simulation agents (MACE, CHGNet, Cantera, etc.)
2. ShepherdAgent(s) - evaluate candidates using LLM + simulation agents

Architecture:
    Spark
    ┌────────────────────────────────────────────────────────────────┐
    │  Academy Manager                                               │
    │                                                                │
    │  ┌─────────────┐      ┌─────────────┐  ┌─────────────┐        │
    │  │ShepherdAgent│─────▶│ MACEAgent   │  │ CHGNetAgent │        │
    │  │             │      └─────────────┘  └─────────────┘        │
    │  │             │      ┌─────────────┐  ┌─────────────┐        │
    │  │             │─────▶│CanteraAgent │  │SurrogateAgent│       │
    │  │             │      └─────────────┘  └─────────────┘        │
    │  │             │      ┌─────────────┐  ┌─────────────┐        │
    │  │             │─────▶│StabilityAgent│ │  QEAgent    │        │
    │  └─────────────┘      └─────────────┘  └─────────────┘        │
    │         │                                                      │
    │         │ LLM calls via HTTP                                   │
    │         ▼                                                      │
    │  ┌─────────────┐                                              │
    │  │ vLLM Server │ (port 8000)                                  │
    │  └─────────────┘                                              │
    └────────────────────────────────────────────────────────────────┘

Usage:
    # On Spark:
    python scripts/run_spark_agents.py --llm-url http://localhost:8000/v1

    # With specific simulation agents:
    python scripts/run_spark_agents.py --llm-url http://localhost:8000/v1 \\
        --agents mace chgnet cantera surrogate stability
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml


# Map of agent names to their classes (11 agents total)
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


def import_agent_class(name: str):
    """Dynamically import an agent class."""
    if name not in AGENT_CLASSES:
        raise ValueError(f"Unknown agent: {name}. Available: {list(AGENT_CLASSES.keys())}")

    module_path, class_name = AGENT_CLASSES[name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


async def launch_simulation_agents(manager, agent_names: list[str], device: str = "cpu"):
    """Launch individual simulation agents.

    Args:
        manager: Academy Manager
        agent_names: List of agent names to launch
        device: Compute device for ML agents

    Returns:
        Dict mapping agent names to agent instances
    """
    sim_agents = {}

    for name in agent_names:
        try:
            AgentClass = import_agent_class(name)

            # Configure agent based on type
            if name in ("mace", "chgnet"):
                kwargs = {"device": device}
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
    device: str = "cpu",
    exchange_url: str | None = None,
    redis_host: str | None = None,
    redis_port: int = 6379,
):
    """Run Academy agents on Spark.

    Args:
        llm_url: URL to vLLM server
        llm_model: Model name for LLM
        agent_names: List of simulation agent names to launch
        num_shepherds: Number of shepherd agents to spawn
        budget: Default budget per candidate
        device: Compute device for ML agents
        exchange_url: Optional Academy exchange URL (for cloud mode)
    """
    from academy.manager import Manager
    from academy.exchange import LocalExchangeFactory

    from skills.shepherd import ShepherdAgent

    logging.info("Starting Academy agents on Spark")
    logging.info(f"  LLM URL: {llm_url}")
    logging.info(f"  LLM Model: {llm_model}")
    logging.info(f"  Simulation agents: {agent_names}")
    logging.info(f"  Shepherds: {num_shepherds}")
    logging.info(f"  Device: {device}")

    # Create exchange factory (local, Redis, or HTTP)
    if redis_host:
        from academy.exchange import RedisExchangeFactory
        exchange_factory = RedisExchangeFactory(hostname=redis_host, port=redis_port)
        logging.info(f"  Exchange: Redis ({redis_host}:{redis_port})")
    elif exchange_url:
        from academy.exchange import HttpExchangeFactory
        exchange_factory = HttpExchangeFactory(url=exchange_url)
        logging.info(f"  Exchange: HTTP ({exchange_url})")
    else:
        exchange_factory = LocalExchangeFactory()
        logging.info("  Exchange: local (in-memory)")

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
        # Launch individual simulation agents
        logging.info("Launching simulation agents...")
        sim_agents = await launch_simulation_agents(manager, agent_names, device)

        if not sim_agents:
            logging.error("No simulation agents could be launched!")
            return

        logging.info(f"Launched {len(sim_agents)} simulation agents: {list(sim_agents.keys())}")

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
                "llm_call": 60,
                "test_poll_interval": 1.0,
                "test_max_wait": 3600,
            },
        }

        # Launch ShepherdAgents
        shepherds = []
        for i in range(num_shepherds):
            logging.info(f"Launching ShepherdAgent {i+1}/{num_shepherds}...")
            shepherd = await manager.launch(
                ShepherdAgent,
                kwargs={
                    "config": shepherd_config,
                    "sim_agents": sim_agents,  # Pass individual agents
                    "llm_url": llm_url,
                    "llm_model": llm_model,
                    "redis_host": redis_host,
                    "redis_port": redis_port,
                },
            )
            shepherds.append(shepherd)

        logging.info("=" * 60)
        logging.info("All agents launched!")
        logging.info(f"  Simulation agents: {list(sim_agents.keys())}")
        logging.info(f"  Shepherd agents: {num_shepherds}")
        logging.info("=" * 60)
        logging.info("Agents are ready to receive evaluation requests")
        logging.info("Press Ctrl+C to shutdown")
        logging.info("=" * 60)

        # Export agent info for external connections
        agent_info = {
            "simulation_agents": list(sim_agents.keys()),
            "shepherd_count": num_shepherds,
            "llm_url": llm_url,
            "llm_model": llm_model,
            "device": device,
        }

        with open("spark_agents_info.json", "w") as f:
            json.dump(agent_info, f, indent=2)
        logging.info("Agent info saved to spark_agents_info.json")

        # Wait for shutdown
        await shutdown_event.wait()

        logging.info("Shutting down agents...")


async def test_agents(agent_names: list[str], device: str = "cpu"):
    """Quick test of simulation agents without full Academy setup."""
    logging.info("Testing simulation agents...")

    test_candidate = {
        "support": "ZrO2",
        "metals": [
            {"element": "Cu", "wt_pct": 55},
            {"element": "Zn", "wt_pct": 30},
            {"element": "Al", "wt_pct": 15},
        ],
    }

    for name in agent_names:
        try:
            AgentClass = import_agent_class(name)

            if name in ("mace", "chgnet"):
                agent = AgentClass(device=device)
            else:
                agent = AgentClass()

            await agent.agent_on_startup()

            status = await agent.get_status({})
            logging.info(f"{name} status: {status}")

            # Run a test action
            if hasattr(agent, "screening"):
                result = await agent.screening({"candidate": test_candidate})
                logging.info(f"{name} screening result: {json.dumps(result, indent=2)}")
            elif hasattr(agent, "analyze"):
                result = await agent.analyze({"candidate": test_candidate})
                logging.info(f"{name} analyze result: {json.dumps(result, indent=2)}")
            elif hasattr(agent, "reactor"):
                result = await agent.reactor({"candidate": test_candidate})
                logging.info(f"{name} reactor result: {json.dumps(result, indent=2)}")

        except ImportError as e:
            logging.warning(f"Cannot test {name}: {e}")
        except Exception as e:
            logging.error(f"Error testing {name}: {e}")

    logging.info("Agent tests complete")


def main():
    parser = argparse.ArgumentParser(
        description="Run Academy agents on Spark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic - use local vLLM
    python scripts/run_spark_agents.py --llm-url http://localhost:8000/v1

    # With specific agents
    python scripts/run_spark_agents.py --llm-url http://localhost:8000/v1 \\
        --agents mace cantera surrogate stability

    # Use GPU for ML agents
    python scripts/run_spark_agents.py --llm-url http://localhost:8000/v1 \\
        --device cuda

    # Test agents only (no LLM needed)
    python scripts/run_spark_agents.py --test
        """,
    )

    parser.add_argument(
        "--llm-url",
        default="http://localhost:8000/v1",
        help="URL to vLLM server (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--llm-model",
        default="neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
        help="LLM model name",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["mace", "chgnet", "m3gnet", "cantera", "surrogate", "stability", "qe", "gpaw", "openmm", "gromacs", "catmap"],
        help="Simulation agents to launch (default: all 11 agents)",
    )
    parser.add_argument(
        "--num-shepherds",
        type=int,
        default=4,
        help="Number of ShepherdAgents to spawn (default: 4)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100.0,
        help="Default budget per candidate (default: 100.0)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device for ML agents (default: cpu)",
    )
    parser.add_argument(
        "--exchange-url",
        help="Academy HTTP exchange URL (for cloud mode)",
    )
    parser.add_argument(
        "--redis-host",
        default=None,
        help="Redis hostname for distributed exchange (e.g., localhost)",
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
        "--test",
        action="store_true",
        help="Test simulation agents only (no Academy setup)",
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

    # Load config for defaults if available
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

        academy_config = config.get("academy", {})
        if not args.exchange_url and academy_config.get("exchange", {}).get("url"):
            args.exchange_url = academy_config["exchange"]["url"]

        shepherds_config = academy_config.get("shepherds", {})
        if args.num_shepherds == 4 and "num_concurrent" in shepherds_config:
            args.num_shepherds = shepherds_config["num_concurrent"]
        if args.budget == 100.0 and "budget_per_candidate" in shepherds_config:
            args.budget = shepherds_config["budget_per_candidate"]

    if args.test:
        # Quick test mode
        asyncio.run(test_agents(args.agents, args.device))
    else:
        # Full agent mode
        asyncio.run(run_agents(
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            agent_names=args.agents,
            num_shepherds=args.num_shepherds,
            budget=args.budget,
            device=args.device,
            exchange_url=args.exchange_url,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
        ))


if __name__ == "__main__":
    main()
