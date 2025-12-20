from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional
import uuid
import time
import random
import numpy as np

from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

from skills.catalyst import CatalystSkill
from skills.performance import PerformanceSkill
from skills.economics import EconomicsSkill  # optional if you want later

from orchestration.tools import make_catalyst_tools, make_performance_tools, make_economics_tools
from orchestration.loop import build_loop_graph

logger = logging.getLogger(__name__)

run_id = str(uuid.uuid4())
logger.info("Run ID: %s", run_id)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--out", default="data/runs.jsonl")
    p.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )
    p.add_argument(
        "-l", "--log-level",
        dest="log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: %(default)s).",
    )
    return p.parse_args()

async def main() -> int:
    init_logging(logging.INFO)
    args = parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    init_logging(level)

    # Ensure non-academy loggers also honor the level
    logging.getLogger().setLevel(level)

    if args.seed is not None:
        #random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info("Random seed set to %d", args.seed)

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
            "estimate_cost": lambda **kw: call_tool("estimate_cost", **kw),
            "estimate_catalyst_cost": lambda **kw: call_tool("estimate_catalyst_cost", **kw),
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
        final_state = None

        async for event in graph.astream(state):
            # Each event is {node_name: node_output}
            for node_name, payload in event.items():

                if node_name == "select":
                    iteration = payload["iteration"]
                    best = payload["best"]
        
                    record = {
                        "run_id": run_id,
                        "iteration": iteration,
                        "candidate_count": len(state["candidates"]),
                        "best_score": best["score"] if best else None,
                        "best_candidate": best["candidate"] if best else None,
                        "performance": best["performance"] if best else None,
                        "ts": time.time(),
                    }

                    # Write one JSONL record per iteration
                    with open(args.out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

                    logger.info(
                        "Iteration %d | best score %.3f | support=%s",
                        iteration,
                        record["best_score"],
                        record["best_candidate"]["support"] if best else "n/a",
                    )

                final_state = payload

        # Persist the final result record
        summary = {
            "run_id": run_id,
            "goal": state["goal"],
            "final_iteration": final_state.get("iteration") if final_state else None,
            "final_best": final_state.get("best") if final_state else None,
        }

        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        print(json.dumps(summary, indent=2))

    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
