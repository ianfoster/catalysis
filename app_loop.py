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
from orchestration.config import load_config
from orchestration.seen import load_seen
from orchestration.seen import append_seen

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
    p.add_argument("--gc-func-mk", default=None)
    p.add_argument("--gc-timeout", type=float, default=300.0,
                   help="Max seconds to wait for GC batch completion (default: %(default)s).")
    p.add_argument("--poll-interval", type=float, default=0.25,
                   help="Polling interval for GC task status (default: %(default)s).")
    p.add_argument("--gc-retries", type=int, default=2,
                   help="Number of retries for transient GC submit/poll errors (default: %(default)s).")
    p.add_argument("--gc-retry-backoff", type=float, default=1.0,
                   help="Base seconds for exponential backoff between GC retries (default: %(default)s).")
    # Config files (base + local overlay)
    p.add_argument("--config", default="config.yaml", help="Base config file (default: %(default)s).")
    p.add_argument("--config-local", default="config.local.yaml",
                   help="Local override config file (default: %(default)s).")
    p.add_argument("--seen-path", default="data/seen_candidates.jsonl",
               help="Path to persistent seen-candidate log (default: %(default)s).")
    p.add_argument("--no-seen", action="store_true",
               help="Disable seen-candidate tracking (debug).")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of candidates to propose/evaluate per iteration (default: from config or seed size).",
    )
 
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
    args = parse_args()

    # Load config early (base + local overlay)
    cfg = load_config(args.config, args.config_local)

    # Apply logging level from CLI if provided, else config, else INFO
    cfg_log_level = (cfg.get("run", {}) or {}).get("log_level")
    log_level_str = (args.log_level or cfg_log_level or "INFO").upper()
    level = getattr(logging, log_level_str, logging.INFO)
    init_logging(level)
    logging.getLogger().setLevel(level)

    # Resolve defaults from config if CLI not provided
    run_cfg = cfg.get("run", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    gc_cfg = cfg.get("globus_compute", {}) or {}
    gc_funcs = (gc_cfg.get("functions", {}) or {})

    max_iterations = args.max_iterations if args.max_iterations is not None else int(run_cfg.get("max_iterations", 3))
    concurrency = args.concurrency if args.concurrency is not None else int(run_cfg.get("concurrency", 32))
    escalate_k = args.escalate_k if args.escalate_k is not None else int(run_cfg.get("escalate_k", 5))
    poll_interval = args.poll_interval if args.poll_interval is not None else float(run_cfg.get("poll_interval", 0.5))
    gc_timeout = args.gc_timeout if args.gc_timeout is not None else float(run_cfg.get("gc_timeout", 300))
    gc_retries = args.gc_retries if args.gc_retries is not None else int(run_cfg.get("gc_retries", 2))
    gc_retry_backoff = args.gc_retry_backoff if args.gc_retry_backoff is not None else float(run_cfg.get("gc_retry_backoff", 1.0))

    batch_size = args.batch_size if args.batch_size is not None else int(run_cfg.get("batch_size", 6))

    out_path = args.out if args.out is not None else str(paths_cfg.get("out_jsonl", "data/runs.jsonl"))
    cache_path = args.cache_path if args.cache_path is not None else str(paths_cfg.get("cache_jsonl", "data/char_cache.jsonl"))

    # Globus Compute enablement: CLI overrides config
    gc_enabled_cfg = bool(gc_cfg.get("enabled", False))
    gc_endpoint = args.gc_endpoint if args.gc_endpoint is not None else gc_cfg.get("endpoint_id")
    gc_func_fast = args.gc_func_fast if args.gc_func_fast is not None else gc_funcs.get("fast_surrogate")
    gc_func_mk = args.gc_func_mk if args.gc_func_mk is not None else gc_funcs.get("microkinetic_lite")

    use_gc = bool(gc_endpoint and gc_func_fast) and (args.gc_endpoint is not None or args.gc_func_fast is not None or gc_enabled_cfg)
    gc_requested = bool(gc_endpoint and gc_func_fast) and \
        ( args.gc_endpoint is not None or args.gc_func_fast is not None or gc_enabled_cfg )
    cache = None
    if not args.no_cache:
        cache = JsonlCache.load(cache_path)

    seen = set()
    if not args.no_seen:
        seen = load_seen(args.seen_path)
        logger.info("Seen-candidates loaded: %d", len(seen))
    else:
        logger.info("Seen-candidates disabled")

    # Seed and run config logging
    logger.info("Concurrency set to %d", concurrency)
    logger.info("Run config | iterations=%d | concurrency=%d | escalate_k=%d | cache=%s",
                max_iterations, concurrency, escalate_k, "disabled" if args.no_cache else "enabled")
    async with await Manager.from_exchange_factory(LocalExchangeFactory()) as manager:
        cat = await manager.launch(CatalystSkill)
        perf = await manager.launch(PerformanceSkill)
        econ = await manager.launch(EconomicsSkill)
        mk = await manager.launch(MicrokineticSkill)
        rdk = await manager.launch(RDKitSkill)
    
        # Build tool list first (local tools always available)
        tools = []
        tools.extend(make_catalyst_tools(cat))
        tools.extend(make_performance_tools(perf))
        tools.extend(make_economics_tools(econ))
        tools.extend(make_microkinetic_tools(mk))
        tools.extend(make_rdkit_tools(rdk))
    
        # Try to enable GC only if requested
        gc_available = False
        hpc = None
        if gc_requested:
            logger.info("Globus Compute requested (endpoint=%s)", gc_endpoint)
            try:
                function_map = {"fast_surrogate": gc_func_fast}
                if gc_func_mk:
                    function_map["microkinetic_lite"] = gc_func_mk
    
                hpc = await manager.launch(
                    HPCCharacterizerSkill,
                    kwargs={"endpoint_id": gc_endpoint, "function_map": function_map},
                )
                tools.extend(make_hpc_tools(hpc))
                gc_available = True
                logger.info("Globus Compute ENABLED (HPCCharacterizerSkill launched)")
            except Exception as e:
                logger.warning("Globus Compute DISABLED (failed to init HPCCharacterizerSkill): %r", e)
        else:
            logger.info("Globus Compute not requested; running local-only")
    
        # Registry of tools
        tool_by_name = {t.name: t for t in tools}
        have_gc_tools = ("submit_characterization" in tool_by_name) and ("get_characterization" in tool_by_name)
    
        if gc_available and not have_gc_tools:
            logger.warning("GC was enabled but submit/get tools are missing; falling back to local-only")
            gc_available = False
            have_gc_tools = False
    
        logger.info("Tools: %s", sorted(tool_by_name.keys()))
    
        async def call_tool(name: str, **kwargs):
            return await tool_by_name[name].ainvoke(kwargs)
    
        # Callable context passed into the loop
        ctx = {
            "encode_catalyst": lambda **kw: call_tool("encode_catalyst", **kw),
            "predict_performance": lambda **kw: call_tool("predict_performance", **kw),
            "estimate_catalyst_cost": lambda **kw: call_tool("estimate_catalyst_cost", **kw),
            "microkinetic_lite": lambda **kw: call_tool("microkinetic_lite", **kw),
        }
        ctx["rdkit_descriptors"] = lambda **kw: call_tool("rdkit_descriptors", **kw)
    
        if have_gc_tools:
            ctx["submit_characterization"] = lambda **kw: call_tool("submit_characterization", **kw)
            ctx["get_characterization"] = lambda **kw: call_tool("get_characterization", **kw)
        else:
            logger.info("GC tools not available; using local executor")

        if cache is not None:
            ctx["cache_get"] = cache.get
            ctx["cache_set"] = cache.set
            ctx["cache_key"] = lambda candidate, characterizer: make_cache_key(candidate, characterizer, version)

        logger.info("CTX keys: %s", sorted(ctx.keys()))
        assert "submit_characterization" in ctx, "ctx missing submit_characterization"
        assert "get_characterization" in ctx, "ctx missing get_characterization"

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
            "gc_retries": args.gc_retries,
            "gc_retry_backoff": args.gc_retry_backoff,
            "batch_size": args.batch_size,
            "seen_candidates": list(seen),
            "seen_path": args.seen_path,
            "no_seen": args.no_seen,
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
                        "Iteration %d | escalation budget: top %d candidates",
                        iteration,
                        state["escalate_k"],
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
                    events = payload["char_events"]

                    # Write all events in one file open
                    with open(args.out, "a", encoding="utf-8") as f:
                        for rec in events:
                            f.write(json.dumps(rec) + "\n")
                
                    if not state.get("no_seen", False):
                        # persist all candidate_ids evaluated this iteration
                        ids = {
                            rec["candidate_id"]
                            for rec in events
                            if rec.get("characterizer") == "fast_surrogate"
                        }
                        append_seen(state["seen_path"], ids)

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
