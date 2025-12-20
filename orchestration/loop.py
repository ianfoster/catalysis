from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict
import time
import uuid
import random
import asyncio

import hashlib
import json

from langgraph.graph import StateGraph, END

from orchestration.characterizers import CHARACTERIZERS
from orchestration.ids import candidate_id
from orchestration.scoring import score_candidate


def choose_characterizers(candidate: dict, history_for_candidate: Dict[str, Any]) -> List[str]:
    """
    Decide which characterizers to run for this candidate, given what's already known.

    v1 policy:
      - always run fast_surrogate first
      - if promising and still uncertain, escalate to microkinetic_lite
      - if very promising, escalate to dft_adsorption
    """
    if "fast_surrogate" not in history_for_candidate:
        return ["fast_surrogate"]

    fs = history_for_candidate["fast_surrogate"]
    sty = float(fs["performance"]["methanol_sty"])

    # Escalation thresholds (tune later)
    if sty > 5.5 and "microkinetic_lite" not in history_for_candidate:
        return ["microkinetic_lite"]

    if sty > 6.5 and "dft_adsorption" not in history_for_candidate:
        return ["dft_adsorption"]

    return []

def score_from_history(history_for_candidate: Dict[str, Any]) -> float:
    """
    Compute a score using whatever information we currently have.

    v1: use fast_surrogate metrics + cost penalty, plus small bonuses for extra characterization.
    """
    if "fast_surrogate" not in history_for_candidate:
        return float("-inf")

    fs = history_for_candidate["fast_surrogate"]
    perf = fs["performance"]
    cost = fs.get("catalyst_cost", {})

    usd_per_kg = float(cost.get("usd_per_kg", 0.0))

    score = (
        1.0 * float(perf["methanol_sty"])
        + 2.0 * float(perf["methanol_selectivity"])
        + 0.5 * float(perf["co2_conversion"])
        - 0.5 * float(perf.get("uncertainty", 0.0))
        - 0.001 * usd_per_kg
    )

    # Bonus for additional evidence (placeholders for now)
    if "microkinetic_lite" in history_for_candidate:
        score += 0.25
    if "dft_adsorption" in history_for_candidate:
        score += 0.50

    return score

def choose_characterizers(candidate: dict, history: dict) -> list[str]:
    """
    Decide which characterizers to run next for a candidate.
    """
    cid = repr(candidate)
    seen = history.get(cid, {})

    # Always start with cheap
    if "fast_surrogate" not in seen:
        return ["fast_surrogate"]

    # Escalate if uncertainty remains and candidate is promising
    if seen["fast_surrogate"]["methanol_sty"] > 5.5:
        if "microkinetic_lite" not in seen:
            return ["microkinetic_lite"]

    # Only escalate to DFT for top-tier
    if seen["fast_surrogate"]["methanol_sty"] > 6.5:
        if "dft_adsorption" not in seen:
            return ["dft_adsorption"]

    return []

logger = logging.getLogger(__name__)

RUN_ID = str(uuid.uuid4())

class LoopState(TypedDict):
    run_id: str
    goal: str
    candidates: List[dict]
    evaluations: List[dict]
    best: Optional[dict]
    iteration: int
    max_iterations: int
    iteration_record: Optional[dict]
    char_history: Dict[str, Dict[str, Any]]  # candidate_key -> {characterizer -> result}

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
        # Ensure history exists
        if "char_history" not in state or state["char_history"] is None:
            state["char_history"] = {}

        char_events: List[dict] = []
        evals: List[dict] = []

        for c in state["candidates"]:
            ck = candidate_id(c)
            state["char_history"].setdefault(ck, {})
            h = state["char_history"][ck]

            # Decide what to run next for this candidate
            to_run = choose_characterizers(c, h)

            # Execute selected characterizers
            for char in to_run:
                ts = time.time()

                if char == "fast_surrogate":
                    cache_key = None
                    cached = None
                    if "cache_key" in ctx and "cache_get" in ctx:
                        cache_key = ctx["cache_key"](c, "fast_surrogate")
                        cached = ctx["cache_get"](cache_key)

                    if cached is not None:
                        # cached value should match what we store (see below)
                        h["fast_surrogate"] = cached
                        char_events.append({
                            "run_id": state["run_id"],
                            "iteration": state["iteration"] + 1,
                            "candidate_key": ck,
                            "candidate": c,
                            "characterizer": "fast_surrogate",
                            "status": "CACHED",
                            "result": cached,
                            "ts": ts,
                        })
                    else:
                        enc = await ctx["encode_catalyst"](support=c["support"], metals=c["metals"])
                        perf = await ctx["predict_performance"](feature_vector=enc["feature_vector"])
                        cost = await ctx["estimate_catalyst_cost"](support=c["support"], metals=c["metals"])

                        computed = {"encoding": enc, "performance": perf, "catalyst_cost": cost}
                        h["fast_surrogate"] = computed

                        if cache_key is not None and "cache_set" in ctx:
                            ctx["cache_set"](cache_key, computed)

                        char_events.append({
                            "run_id": state["run_id"],
                            "iteration": state["iteration"] + 1,
                            "candidate_key": ck,
                            "candidate": c,
                            "characterizer": "fast_surrogate",
                            "status": "SUCCEEDED",
                            "result": computed,
                            "ts": ts,
                        })

                elif char == "microkinetic_lite":
                    # Only run if tool is available; otherwise mark skipped
                    if "microkinetic_lite" in ctx:
                        fs = h.get("fast_surrogate", {})
                        perf = fs.get("performance", {})
                        mk = await ctx["microkinetic_lite"](candidate=c, performance=perf)
                        h["microkinetic_lite"] = mk

                        char_events.append({
                            "run_id": state["run_id"],
                            "iteration": state["iteration"] + 1,
                            "candidate_key": ck,
                            "candidate": c,
                            "characterizer": "microkinetic_lite",
                            "status": "SUCCEEDED",
                            "result": mk,
                            "ts": ts,
                        })
                    else:
                        h["microkinetic_lite"] = {"status": "SKIPPED", "reason": "tool not available"}
                        char_events.append({
                            "run_id": state["run_id"],
                            "iteration": state["iteration"] + 1,
                            "candidate_key": ck,
                            "candidate": c,
                            "characterizer": "microkinetic_lite",
                            "status": "SKIPPED",
                            "result": h["microkinetic_lite"],
                            "ts": ts,
                        })

                elif char == "dft_adsorption":
                    # Keep placeholder until GC/HPC is wired
                    h["dft_adsorption"] = {"status": "SKIPPED", "reason": "HPC not wired"}
                    char_events.append({
                        "run_id": state["run_id"],
                        "iteration": state["iteration"] + 1,
                        "candidate_key": ck,
                        "candidate": c,
                        "characterizer": "dft_adsorption",
                        "status": "SKIPPED",
                        "result": h["dft_adsorption"],
                        "ts": ts,
                    })

            # Build evaluation using accumulated history
            score = score_candidate(h)

            evals.append(
                {
                    "candidate": c,
                    "history": h,  # contains fast_surrogate + any escalations
                    "score": score,
                    # Convenience fields for debugging/selection
                    "performance": h.get("fast_surrogate", {}).get("performance"),
                    "encoding": h.get("fast_surrogate", {}).get("encoding"),
                    "catalyst_cost": h.get("fast_surrogate", {}).get("catalyst_cost"),
                }
            )

        return {"evaluations": evals, "char_history": state["char_history"]}

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
