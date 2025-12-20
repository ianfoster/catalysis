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

from skills.hpc_characterizer import HPCCharacterizerSkill
from skills.microkinetic import MicrokineticSkill
from orchestration.tools import make_microkinetic_tools
from orchestration.tools import make_hpc_tools

from orchestration.tools import make_catalyst_tools, make_performance_tools, make_economics_tools
from orchestration.loop import build_loop_graph
from orchestration.cache import JsonlCache, make_cache_key, detect_version

logger = logging.getLogger(__name__)

run_id = str(uuid.uuid4())
logger.info("Run ID: %s", run_id)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-path", default="data/char_cache.jsonl", help="JSONL cache path (default: %(default)s).")
    p.add_argument("--no-cache", action="store_true", help="Disable cache reads/writes.")
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--out", default="data/runs.jsonl")
    p.add_argument("--gc-endpoint", default=None, help="Globus Compute endpoint UUID")
    p.add_argument("--gc-func-fast", default=None, help="Function ID for fast characterizer")
    p.add_argument("--gc-timeout", type=float, default=300.0,
                   help="Max seconds to wait for GC batch completion (default: %(default)s).")
    p.add_argument("--poll-interval", type=float, default=0.25,
                   help="Polling interval for GC task status (default: %(default)s).")
    p.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Max concurrent characterizer executions (default: %(default)s).",
    )
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
    p.add_argument(
        "--escalate-k",
        type=int,
        default=5,
        help="Max number of candidates per iteration eligible for microkinetic escalation (default: %(default)s).",
    )
    return p.parse_args()

async def main() -> int:
    init_logging(logging.INFO)
    args = parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    init_logging(level)

    # Ensure non-academy loggers also honor the level
    logging.getLogger().setLevel(level)
    logger.info("Concurrency set to %d", args.concurrency)

    if args.seed is not None:
        #random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info("Random seed set to %d", args.seed)

    version = detect_version()
    logger.info("Code version: %s", version)
    
    cache = None
    if not args.no_cache:
        cache = JsonlCache.load(args.cache_path)
        logger.info("Cache enabled: %s (entries=%d)", args.cache_path, len(cache.index))
    else:
        logger.info("Cache disabled")

    Path("data").mkdir(exist_ok=True)

    async with await Manager.from_exchange_factory(LocalExchangeFactory()) as manager:
        cat = await manager.launch(CatalystSkill)
        perf = await manager.launch(PerformanceSkill)
        econ = await manager.launch(EconomicsSkill)  # not used in v1 score, but ready
        mk = await manager.launch(MicrokineticSkill)
        use_gc = args.gc_endpoint is not None and args.gc_func_fast is not None
        if use_gc:
            logger.info("Globus Compute ENABLED (endpoint=%s)", args.gc_endpoint)
        else:
            logger.info("Globus Compute DISABLED (local-only mode)")

        if use_gc:
            hpc = await manager.launch(
                HPCCharacterizerSkill,
                kwargs={
                    "endpoint_id": args.gc_endpoint,
                    "function_map": {
                        "fast_surrogate": args.gc_func_fast,
                    },
                },
            )

        tools = []
        tools.extend(make_catalyst_tools(cat))
        tools.extend(make_performance_tools(perf))
        tools.extend(make_economics_tools(econ))
        if use_gc: tools.extend(make_hpc_tools(hpc))
        tools.extend(make_microkinetic_tools(mk)) 

        # Build a callable context from tools by name
        tool_by_name = {t.name: t for t in tools}
        logger.info("Tools: %s", sorted(tool_by_name.keys()))

        async def call_tool(name: str, **kwargs):
            return await tool_by_name[name].ainvoke(kwargs)

        ctx = {
            "encode_catalyst": lambda **kw: call_tool("encode_catalyst", **kw),
            "predict_performance": lambda **kw: call_tool("predict_performance", **kw),
            #"estimate_cost": lambda **kw: call_tool("estimate_cost", **kw),
            "estimate_catalyst_cost": lambda **kw: call_tool("estimate_catalyst_cost", **kw),
            "microkinetic_lite": lambda **kw: call_tool("microkinetic_lite", **kw),
        }
        ctx["submit_characterization"] = lambda **kw: call_tool("submit_characterization", **kw)
        ctx["get_characterization"] = lambda **kw: call_tool("get_characterization", **kw)
        if cache is not None:
            ctx["cache_get"] = cache.get
            ctx["cache_set"] = cache.set
            ctx["cache_key"] = lambda candidate, characterizer: make_cache_key(candidate, characterizer, version)

        logger.info(
            "Run config | iterations=%d | concurrency=%d | escalate_k=%d | cache=%s",
            args.max_iterations,
            args.concurrency,
            args.escalate_k,
            "disabled" if args.no_cache else "enabled",
        )

        graph = build_loop_graph(ctx)

        state = {
            "run_id": run_id,
            "goal": "CO2 + H2 -> methanol catalyst optimization",
            "candidates": [],
            "evaluations": [],
            "best": None,
            "iteration": 0,
            "max_iterations": args.max_iterations,
            "char_history": {},
            "concurrency": args.concurrency,  
            "escalate_k": args.escalate_k, 
            "gc_timeout": args.gc_timeout,
            "poll_interval": args.poll_interval,
        }

        # Run the loop
        final_state = None

        async for event in graph.astream(state):
            # Each event is {node_name: node_output}
            for node_name, payload in event.items():

                if node_name == "select":
                    iteration = payload["iteration"]
                    best = payload["best"]

                    logger.info(
                        "Iteration %d | best score %.3f | support=%s",
                        iteration,
                        best["score"],
                        best["candidate"]["support"],
                    )

                    # Log escalation budget decision
                    logger.info(
                        "Iteration %d | escalation budget: top %d / %d candidates",
                        iteration,
                        state["escalate_k"],
                        len(state["candidates"]),
                    )
        
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
                if node_name == "evaluate" and "char_events" in payload:
                    for rec in payload["char_events"]:
                        with open(args.out, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
        
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
