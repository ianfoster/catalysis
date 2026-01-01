#!/usr/bin/env python3
"""Streamlit dashboard for monitoring Academy agents.

Usage:
    streamlit run scripts/dashboard.py -- --redis-host spark

Requires: pip install streamlit
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

# Page config
st.set_page_config(
    page_title="Catalyst Discovery Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_manager(redis_host: str, redis_port: int):
    """Create a cached manager connection."""
    from academy.exchange import RedisExchangeFactory
    return RedisExchangeFactory(hostname=redis_host, port=redis_port)


async def discover_agents(exchange_factory, agent_class):
    """Discover agents of a given type."""
    from academy.manager import Manager

    async with await Manager.from_exchange_factory(exchange_factory) as manager:
        exchange = manager.exchange_client
        try:
            return await exchange.discover(agent_class)
        except Exception:
            return []


async def query_agent(exchange_factory, agent_id, action: str, request: dict):
    """Query a single agent."""
    from academy.manager import Manager
    from academy.handle import Handle

    async with await Manager.from_exchange_factory(exchange_factory) as manager:
        handle = Handle(agent_id)
        return await asyncio.wait_for(
            getattr(handle, action)(request),
            timeout=10.0
        )


def run_async(coro):
    """Run async function in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def main():
    # Sidebar configuration
    st.sidebar.title("🧪 Catalyst Discovery")
    st.sidebar.markdown("---")

    redis_host = st.sidebar.text_input("Redis Host", value="spark")
    redis_port = st.sidebar.number_input("Redis Port", value=6379, min_value=1, max_value=65535)
    auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=False)

    if auto_refresh:
        st.sidebar.info("Auto-refreshing every 10 seconds")
        import time
        time.sleep(0.1)  # Small delay
        st.rerun()

    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Links")
    st.sidebar.markdown("- [Narrative Log](#narrative)")
    st.sidebar.markdown("- [Agent Status](#agents)")
    st.sidebar.markdown("- [Results](#results)")

    # Main content
    st.title("Catalyst Discovery Dashboard")
    st.markdown(f"Connected to Redis: `{redis_host}:{redis_port}`")

    exchange_factory = get_manager(redis_host, redis_port)

    # Import agent classes
    try:
        from skills.shepherd import ShepherdAgent
        from skills.llm_proxy_agent import LLMProxyAgent
        from skills.sim_agents.mace_agent import MACEAgent
        from skills.sim_agents.chgnet_agent import CHGNetAgent
        from skills.sim_agents.surrogate_agent import SurrogateAgent
        from skills.sim_agents.cantera_agent import CanteraAgent
        from skills.sim_agents.stability_agent import StabilityAgent
        from skills.sim_agents.qe_agent import QEAgent
        from skills.sim_agents.openmm_agent import OpenMMAgent
        from skills.sim_agents.m3gnet_agent import M3GNetAgent
        from skills.sim_agents.gpaw_agent import GPAWAgent
        from skills.sim_agents.gromacs_agent import GROMACSAgent
        from skills.sim_agents.catmap_agent import CatMAPAgent

        agent_types = {
            "ShepherdAgent": ShepherdAgent,
            "LLMProxyAgent": LLMProxyAgent,
            "MACEAgent": MACEAgent,
            "CHGNetAgent": CHGNetAgent,
            "SurrogateAgent": SurrogateAgent,
            "CanteraAgent": CanteraAgent,
            "StabilityAgent": StabilityAgent,
            "QEAgent": QEAgent,
            "OpenMMAgent": OpenMMAgent,
            "M3GNetAgent": M3GNetAgent,
            "GPAWAgent": GPAWAgent,
            "GROMACSAgent": GROMACSAgent,
            "CatMAPAgent": CatMAPAgent,
        }
    except ImportError as e:
        st.error(f"Failed to import agent classes: {e}")
        return

    # Agent Status Section
    st.header("🤖 Agent Status", anchor="agents")

    cols = st.columns(3)

    for idx, (type_name, agent_class) in enumerate(agent_types.items()):
        col = cols[idx % 3]

        with col:
            try:
                agent_ids = run_async(discover_agents(exchange_factory, agent_class))
            except Exception as e:
                st.error(f"{type_name}: Connection error")
                continue

            if not agent_ids:
                st.info(f"**{type_name}**: No agents")
                continue

            with st.expander(f"**{type_name}** ({len(agent_ids)})", expanded=True):
                for agent_id in agent_ids:
                    try:
                        status = run_async(query_agent(
                            exchange_factory, agent_id, "get_status", {}
                        ))

                        ready = status.get("ready", False)
                        status_icon = "🟢" if ready else "🔴"

                        st.markdown(f"{status_icon} `{str(agent_id)[:20]}...`")

                        # Show key metrics
                        if "total_actions" in status:
                            st.metric("Actions", status["total_actions"])
                        if "total_time_s" in status:
                            st.metric("Total Time", f"{status['total_time_s']:.1f}s")
                        if "model" in status:
                            st.caption(f"Model: {status['model']}")
                        if "device" in status:
                            st.caption(f"Device: {status['device']}")

                    except Exception as e:
                        st.warning(f"Agent timeout: {str(agent_id)[:20]}")

    # History Section
    st.header("📜 Recent Activity")

    selected_type = st.selectbox(
        "Select agent type",
        list(agent_types.keys()),
        key="history_type"
    )

    history_count = st.slider("Show last N actions", 5, 50, 10)

    if selected_type:
        try:
            agent_ids = run_async(discover_agents(
                exchange_factory, agent_types[selected_type]
            ))

            if agent_ids:
                for agent_id in agent_ids[:1]:  # Just first agent
                    try:
                        history = run_async(query_agent(
                            exchange_factory, agent_id, "get_history",
                            {"last_n": history_count}
                        ))

                        records = history.get("history", [])
                        if records:
                            st.subheader(f"History for {str(agent_id)[:30]}...")

                            # Convert to table format
                            table_data = []
                            for r in records:
                                table_data.append({
                                    "Time": r.get("timestamp", "")[-8:],
                                    "Action": r.get("action", ""),
                                    "Duration": f"{r.get('duration_s', 0):.2f}s",
                                    "Input": r.get("input", "")[:30],
                                    "Status": "✓" if r.get("success") else "✗",
                                })

                            st.table(table_data)

                            # Statistics
                            stats = history.get("statistics", {})
                            if stats:
                                st.subheader("Statistics")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Total Actions", stats.get("total_actions", 0))
                                col2.metric("Total Time", f"{stats.get('total_time_s', 0):.1f}s")

                                avg_times = stats.get("action_avg_times_s", {})
                                if avg_times:
                                    col3.metric(
                                        "Avg Time",
                                        f"{sum(avg_times.values()) / len(avg_times):.2f}s"
                                    )
                        else:
                            st.info("No history records yet")

                    except Exception as e:
                        st.warning(f"Could not get history: {e}")
            else:
                st.info(f"No {selected_type} agents found")

        except Exception as e:
            st.error(f"Error: {e}")

    # Results Section
    st.header("📊 Discovery Results", anchor="results")

    results_file = project_root / "data" / "generator_results.jsonl"
    if results_file.exists():
        import json

        results = []
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        if results:
            st.success(f"Found {len(results)} evaluation records")

            # Show best candidates
            scored = []
            for r in results:
                result = r.get("result", {})
                assessment = result.get("final_assessment", {})
                score = assessment.get("viability_score", 0)
                candidate = result.get("candidate", {})

                metals = candidate.get("metals", [])
                metal_str = "/".join(
                    f"{m.get('element', '?')}{m.get('wt_pct', 0)}"
                    for m in metals
                )
                support = candidate.get("support", "?")

                scored.append({
                    "Score": score,
                    "Candidate": f"{metal_str} on {support}",
                    "Recommendation": assessment.get("recommendation", "?"),
                    "Iteration": r.get("iteration", "?"),
                })

            # Sort by score
            scored.sort(key=lambda x: x["Score"], reverse=True)

            st.subheader("Top Candidates")
            st.table(scored[:10])
        else:
            st.info("No results yet")
    else:
        st.info("No results file found. Run the generator first.")

    # Narrative Section
    st.header("📖 Narrative Log", anchor="narrative")

    narrative_file = project_root / "data" / "narrative.log"
    if narrative_file.exists():
        with open(narrative_file) as f:
            lines = f.readlines()

        if lines:
            # Show last N lines
            n_lines = st.slider("Show last N lines", 10, 100, 30, key="narrative_lines")
            st.code("".join(lines[-n_lines:]), language=None)
        else:
            st.info("Narrative log is empty")
    else:
        st.info("No narrative log found")

    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
