from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


class LoopState(TypedDict):
    goal: str
    candidates: List[dict]
    evaluations: List[dict]
    best: Optional[dict]
    iteration: int
    max_iterations: int

def propose_candidates(_: LoopState) -> Dict[str, Any]:
    candidates = [
        {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 55}, {"element": "Zn", "wt_pct": 25}, {"element": "Al", "wt_pct": 20}]},
        {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 60}, {"element": "Zn", "wt_pct": 25}, {"element": "Al", "wt_pct": 15}]},
        {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 65}, {"element": "Zn", "wt_pct": 20}, {"element": "Al", "wt_pct": 15}]},
        {"support": "ZrO2",  "metals": [{"element": "Cu", "wt_pct": 55}, {"element": "Zn", "wt_pct": 30}, {"element": "Al", "wt_pct": 15}]},
        {"support": "ZrO2",  "metals": [{"element": "Cu", "wt_pct": 60}, {"element": "Zn", "wt_pct": 30}, {"element": "Al", "wt_pct": 10}]},
        {"support": "ZrO2",  "metals": [{"element": "Cu", "wt_pct": 70}, {"element": "Zn", "wt_pct": 20}, {"element": "Al", "wt_pct": 10}]},
    ]
    return {"candidates": candidates}

def make_evaluate_candidates(ctx: dict):
    async def evaluate_candidates(state: LoopState) -> Dict[str, Any]:
        evals: List[dict] = []
        for c in state["candidates"]:
            enc = await ctx["encode_catalyst"](support=c["support"], metals=c["metals"])
            perf = await ctx["predict_performance"](feature_vector=enc["feature_vector"])

            score = (
                1.0 * float(perf["methanol_sty"])
                + 2.0 * float(perf["methanol_selectivity"])
                + 0.5 * float(perf["co2_conversion"])
                - 0.5 * float(perf.get("uncertainty", 0.0))
            )

            evals.append({"candidate": c, "encoding": enc, "performance": perf, "score": score})

        return {"evaluations": evals}
    return evaluate_candidates

def select_best(state: LoopState) -> Dict[str, Any]:
    best = max(state["evaluations"], key=lambda x: x["score"]) if state["evaluations"] else None
    return {"best": best, "iteration": state["iteration"] + 1}

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
