#!/usr/bin/env python3
"""Query status and history from all Academy agents.

Usage:
    python scripts/agent_status.py --redis-host spark
    python scripts/agent_status.py --redis-host spark --history 10
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def query_agents(redis_host: str, redis_port: int, show_history: int = 0):
    """Query all agents for status and optionally history."""
    from academy.manager import Manager
    from academy.exchange import RedisExchangeFactory
    from academy.handle import Handle

    exchange_factory = RedisExchangeFactory(hostname=redis_host, port=redis_port)

    async with await Manager.from_exchange_factory(exchange_factory) as manager:
        exchange = manager.exchange_client

        # Discover all agent types
        from skills.shepherd import ShepherdAgent
        from skills.sim_agents.mace_agent import MACEAgent
        from skills.sim_agents.chgnet_agent import CHGNetAgent
        from skills.sim_agents.surrogate_agent import SurrogateAgent
        from skills.sim_agents.cantera_agent import CanteraAgent
        from skills.sim_agents.stability_agent import StabilityAgent
        from skills.sim_agents.qe_agent import QEAgent

        agent_types = [
            ("ShepherdAgent", ShepherdAgent),
            ("MACEAgent", MACEAgent),
            ("CHGNetAgent", CHGNetAgent),
            ("SurrogateAgent", SurrogateAgent),
            ("CanteraAgent", CanteraAgent),
            ("StabilityAgent", StabilityAgent),
            ("QEAgent", QEAgent),
        ]

        print("=" * 70)
        print("ACADEMY AGENT STATUS")
        print("=" * 70)

        for type_name, agent_class in agent_types:
            try:
                agent_ids = await exchange.discover(agent_class)
            except Exception as e:
                print(f"\n{type_name}: Error discovering - {e}")
                continue

            if not agent_ids:
                print(f"\n{type_name}: No agents found")
                continue

            print(f"\n{type_name}: {len(agent_ids)} agent(s)")
            print("-" * 50)

            for agent_id in agent_ids:
                try:
                    handle = Handle(agent_id)

                    # Get status
                    status = await asyncio.wait_for(
                        handle.get_status({}),
                        timeout=5.0
                    )

                    print(f"  Agent: {agent_id}")
                    print(f"    Ready: {status.get('ready', 'N/A')}")

                    # Print relevant status fields
                    for key in ['model', 'device', 'total_actions', 'total_time_s', 'action_counts']:
                        if key in status:
                            print(f"    {key}: {status[key]}")

                    # Get history if requested
                    if show_history > 0:
                        try:
                            history_resp = await asyncio.wait_for(
                                handle.get_history({"last_n": show_history}),
                                timeout=5.0
                            )
                            history = history_resp.get("history", [])
                            if history:
                                print(f"    Recent actions ({len(history)}):")
                                for h in history:
                                    success = "✓" if h.get("success") else "✗"
                                    print(f"      {success} {h['action']}: {h['duration_s']}s - {h.get('input', '')[:30]}")
                        except Exception as e:
                            print(f"    History: Not available ({e})")

                except asyncio.TimeoutError:
                    print(f"  Agent: {agent_id} - TIMEOUT")
                except Exception as e:
                    print(f"  Agent: {agent_id} - ERROR: {e}")

        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Query Academy agent status")
    parser.add_argument("--redis-host", default="spark", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--history", type=int, default=0, help="Show last N history entries")
    args = parser.parse_args()

    asyncio.run(query_agents(args.redis_host, args.redis_port, args.history))


if __name__ == "__main__":
    main()
