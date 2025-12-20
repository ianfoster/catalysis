from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

from skills.catalyst import CatalystSkill
from skills.performance import PerformanceSkill
from skills.economics import EconomicsSkill  # optional if you want later

from orchestration.tools import make_catalyst_tools, make_performance_tools, make_economics_tools
from orchestration.loop import build_loop_graph

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--out", default="data/runs.jsonl")
    return p.parse_args()

async def main() -> int:
    init_logging(logging.INFO)
    args = parse_args()

    Path("data").mkdir(exist_ok=True)

    async with await Manager.from_exchange_factory(LocalExchangeFactory()) as manager:
        cat = await manager.launch(CatalystSkill)
        perf = await manager.launch(PerformanceSkill)
        econ = await manager.launch(EconomicsSkill)  # not used in v1 score, but ready

        tools = []
        tools.extend(make_catalyst_tools(cat))
        tools.extend(make_performance_tools(perf))
        tools.extend(make_economics_tools(econ))

        # Build a callable context from tools by name
        tool_by_name = {t.name: t for t in tools}
        logger.info("Tools: %s", sorted(tool_by_name.keys()))

        async def call_tool(name: str, **kwargs):
            return await tool_by_name[name].ainvoke(kwargs)

        ctx = {
            "encode_catalyst": lambda **kw: call_tool("encode_catalyst", **kw),
            "predict_performance": lambda **kw: call_tool("predict_performance", **kw),
        }

        graph = build_loop_graph(ctx)

        state = {
            "goal": "CO2 + H2 -> methanol catalyst optimization",
            "candidates": [],
            "evaluations": [],
            "best": None,
            "iteration": 0,
            "max_iterations": args.max_iterations,
        }

        # Run the loop
        result = await graph.ainvoke(state)

        # Persist the final result record
        record = {
            "goal": state["goal"],
            "max_iterations": args.max_iterations,
            "final_best": result.get("best"),
            "final_iteration": result.get("iteration"),
        }

        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(json.dumps(record, indent=2))

    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
