from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict
import time
import uuid
import random

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

RUN_ID = str(uuid.uuid4())

class LoopState(TypedDict):
    goal: str
    candidates: List[dict]
    evaluations: List[dict]
    best: Optional[dict]
    iteration: int
    max_iterations: int
    iteration_record: Optional[dict]

def propose_candidates_OLD(_: LoopState) -> Dict[str, Any]:
    candidates = [
        {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 55}, {"element": "Zn", "wt_pct": 25}, {"element": "Al", "wt_pct": 20}]},
        {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 60}, {"element": "Zn", "wt_pct": 25}, {"element": "Al", "wt_pct": 15}]},
        {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 65}, {"element": "Zn", "wt_pct": 20}, {"element": "Al", "wt_pct": 15}]},
        {"support": "ZrO2",  "metals": [{"element": "Cu", "wt_pct": 55}, {"element": "Zn", "wt_pct": 30}, {"element": "Al", "wt_pct": 15}]},
        {"support": "ZrO2",  "metals": [{"element": "Cu", "wt_pct": 60}, {"element": "Zn", "wt_pct": 30}, {"element": "Al", "wt_pct": 10}]},
        {"support": "ZrO2",  "metals": [{"element": "Cu", "wt_pct": 70}, {"element": "Zn", "wt_pct": 20}, {"element": "Al", "wt_pct": 10}]},
    ]
    return {"candidates": candidates}

def _normalize_metals(metals: list[dict]) -> list[dict]:
    # Ensure Cu/Zn/Al exist and sum to 100
    d = {m["element"]: float(m["wt_pct"]) for m in metals}
    cu, zn, al = d.get("Cu", 0.0), d.get("Zn", 0.0), d.get("Al", 0.0)
    s = cu + zn + al
    if s <= 0:
        cu, zn, al = 60.0, 30.0, 10.0
        s = 100.0
    cu, zn, al = 100.0 * cu / s, 100.0 * zn / s, 100.0 * al / s
    return [
        {"element": "Cu", "wt_pct": round(cu, 1)},
        {"element": "Zn", "wt_pct": round(zn, 1)},
        {"element": "Al", "wt_pct": round(al, 1)},
    ]

def _neighbors(best_candidate: dict, k: int = 6, step: float = 5.0) -> list[dict]:
    # Generate k variants by shifting wt% among Cu/Zn/Al while keeping sum=100
    base_support = best_candidate["support"]
    base_metals = _normalize_metals(best_candidate["metals"])
    d = {m["element"]: float(m["wt_pct"]) for m in base_metals}

    out = []
    for _ in range(k):
        cu, zn, al = d["Cu"], d["Zn"], d["Al"]

        # pick two elements to transfer between
        elems = ["Cu", "Zn", "Al"]
        src, dst = random.sample(elems, 2)
        delta = random.choice([-step, step])

        vals = {"Cu": cu, "Zn": zn, "Al": al}
        vals[src] = max(0.0, vals[src] - delta)
        vals[dst] = max(0.0, vals[dst] + delta)

        # renormalize to 100 and clamp
        metals = _normalize_metals(
            [{"element": "Cu", "wt_pct": vals["Cu"]},
             {"element": "Zn", "wt_pct": vals["Zn"]},
             {"element": "Al", "wt_pct": vals["Al"]}]
        )

        # occasionally flip support
        support = base_support
        if random.random() < 0.25:
            support = "Al2O3" if base_support.lower() == "zro2" else "ZrO2"

        out.append({"support": support, "metals": metals})

    return out

def propose_candidates(state: LoopState) -> Dict[str, Any]:
    # Iteration 0: seed set; later: explore around best
    seed = [
        {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 55}, {"element": "Zn", "wt_pct": 25}, {"element": "Al", "wt_pct": 20}]},
        {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 60}, {"element": "Zn", "wt_pct": 25}, {"element": "Al", "wt_pct": 15}]},
        {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 65}, {"element": "Zn", "wt_pct": 20}, {"element": "Al", "wt_pct": 15}]},
        {"support": "ZrO2",  "metals": [{"element": "Cu", "wt_pct": 55}, {"element": "Zn", "wt_pct": 30}, {"element": "Al", "wt_pct": 15}]},
        {"support": "ZrO2",  "metals": [{"element": "Cu", "wt_pct": 60}, {"element": "Zn", "wt_pct": 30}, {"element": "Al", "wt_pct": 10}]},
        {"support": "ZrO2",  "metals": [{"element": "Cu", "wt_pct": 70}, {"element": "Zn", "wt_pct": 20}, {"element": "Al", "wt_pct": 10}]},
    ]

    if state["iteration"] == 0 or not state.get("best"):
        return {"candidates": seed}

    best_candidate = state["best"]["candidate"]
    return {"candidates": _neighbors(best_candidate, k=len(seed), step=5.0)}

def make_evaluate_candidates(ctx: dict):
    async def evaluate_candidates(state: LoopState) -> Dict[str, Any]:
        evals: List[dict] = []
        for c in state["candidates"]:
            enc = await ctx["encode_catalyst"](support=c["support"], metals=c["metals"])
            perf = await ctx["predict_performance"](feature_vector=enc["feature_vector"])
            cost = await ctx["estimate_catalyst_cost"](support=c["support"], metals=c["metals"])
            usd_per_kg = float(cost["usd_per_kg"])

            score = (
                1.0 * float(perf["methanol_sty"])
                + 2.0 * float(perf["methanol_selectivity"])
                + 0.5 * float(perf["co2_conversion"])
                - 0.5 * float(perf.get("uncertainty", 0.0))
                - 0.001 * usd_per_kg   # small penalty
            )

            evals.append({"candidate": c, "encoding": enc, "performance": perf, "score": score})

        return {"evaluations": evals}
    return evaluate_candidates

def select_best(state: LoopState) -> Dict[str, Any]:
    best = max(state["evaluations"], key=lambda x: x["score"]) if state["evaluations"] else None
    iter_idx = state["iteration"] + 1

    iteration_record = {
        "run_id": RUN_ID,
        "ts": time.time(),
        "iteration": iter_idx,
        "goal": state["goal"],
        "candidates": state["candidates"],
        "evaluations": state["evaluations"],
        "best": best,
    }

    return {
        "best": best,
        "iteration": iter_idx,
        "iteration_record": iteration_record,
    }

def should_continue(state: LoopState) -> str:
    return "stop" if state["iteration"] >= state["max_iterations"] else "continue"


def build_loop_graph(ctx: dict):
    g = StateGraph(LoopState)
    g.add_node("propose", propose_candidates)
    g.add_node("evaluate", make_evaluate_candidates(ctx))
    g.add_node("select", select_best)

    g.set_entry_point("propose")
    g.add_edge("propose", "evaluate")
    g.add_edge("evaluate", "select")
    g.add_conditional_edges("select", should_continue, {"continue": "propose", "stop": END})

    return g.compile()
