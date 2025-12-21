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
from orchestration.executors import LocalExecutor, GCExecutor


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

    fs = history_for_candidate.get("fast_surrogate", {})
    perf = fs.get("performance", {}) if isinstance(fs, dict) else {}
    sty = float(perf.get("methanol_sty", 0.0))

    # Escalation thresholds (tune later)
    #if sty > 5.5 and "microkinetic_lite" not in history_for_candidate:
    #    return ["microkinetic_lite"]

    # Always request microkinetic next (we'll gate by top-K elsewhere)
    if "microkinetic_lite" not in history_for_candidate:
        return ["microkinetic_lite"]

    # Disable DFT tier until implemented
    #if sty > 6.5 and "dft_adsorption" not in history_for_candidate:
    #    return ["dft_adsorption"]

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

def choose_characterizers_OLD(candidate: dict, history: dict) -> list[str]:
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
    concurrency: int
    escalate_k: int
    batch_size: int | None
    seen_candidates: List[str]
    seen_path: str
    no_seen: bool

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


from orchestration.ids import candidate_id  # ensure this exists

def _neighbors(best_candidate: dict, k: int = 6, step: float = 5.0) -> list[dict]:
    """
    Generate up to k unique variants by shifting wt% among Cu/Zn/Al while keeping sum=100,
    with optional support flips. Uniqueness is enforced within this batch.
    """
    base_support = best_candidate["support"]
    base_metals = _normalize_metals(best_candidate["metals"])
    d0 = {m["element"]: float(m["wt_pct"]) for m in base_metals}

    out: list[dict] = []
    seen: set[str] = set()

    # Limit attempts to avoid infinite loops when the neighborhood is small
    max_attempts = 50 * k
    attempts = 0

    while len(out) < k and attempts < max_attempts:
        attempts += 1

        cu, zn, al = d0["Cu"], d0["Zn"], d0["Al"]

        elems = ["Cu", "Zn", "Al"]
        src, dst = random.sample(elems, 2)
        delta = random.choice([-step, step])

        vals = {"Cu": cu, "Zn": zn, "Al": al}
        vals[src] = max(0.0, vals[src] - delta)
        vals[dst] = max(0.0, vals[dst] + delta)

        metals = _normalize_metals(
            [
                {"element": "Cu", "wt_pct": vals["Cu"]},
                {"element": "Zn", "wt_pct": vals["Zn"]},
                {"element": "Al", "wt_pct": vals["Al"]},
            ]
        )

        support = base_support
        if random.random() < 0.25:
            support = "Al2O3" if base_support.strip().lower() in ("zro2", "zirconia") else "ZrO2"

        cand = {"support": support, "metals": metals}
        cid = candidate_id(cand)
        if cid in seen:
            continue

        seen.add(cid)
        out.append(cand)

    return out


def propose_candidates(state: LoopState) -> Dict[str, Any]:
    seed = [
        {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 55}, {"element": "Zn", "wt_pct": 25}, {"element": "Al", "wt_pct": 20}]},
        {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 60}, {"element": "Zn", "wt_pct": 25}, {"element": "Al", "wt_pct": 15}]},
        {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 65}, {"element": "Zn", "wt_pct": 20}, {"element": "Al", "wt_pct": 15}]},
        {"support": "ZrO2",  "metals": [{"element": "Cu", "wt_pct": 55}, {"element": "Zn", "wt_pct": 30}, {"element": "Al", "wt_pct": 15}]},
        {"support": "ZrO2",  "metals": [{"element": "Cu", "wt_pct": 60}, {"element": "Zn", "wt_pct": 30}, {"element": "Al", "wt_pct": 10}]},
        {"support": "ZrO2",  "metals": [{"element": "Cu", "wt_pct": 70}, {"element": "Zn", "wt_pct": 20}, {"element": "Al", "wt_pct": 10}]},
    ]

    # Determine requested batch size
    default_n = len(seed)
    n = state.get("batch_size") or default_n
    n = int(n)

    # Generate raw candidates
    if state["iteration"] == 0 or not state.get("best"):
        candidates = list(seed)
    else:
        best_candidate = state["best"]["candidate"]
        # generate more than needed then dedupe down
        candidates = _neighbors(best_candidate, k=max(n, default_n), step=5.0)

    # Deduplicate within iteration
    seen = set()
    unique = []
    dup_count = 0
    for c in candidates:
        cid = candidate_id(c)
        if cid in seen:
            dup_count += 1
            continue
        seen.add(cid)
        unique.append(c)

    if dup_count > 0:
        logger.warning(
            "Duplicate candidates filtered | iteration=%d | generated=%d | kept=%d | dropped=%d",
            state["iteration"],
            len(candidates),
            len(unique),
            dup_count,
        )

    # If too many, truncate (randomize to avoid bias)
    if len(unique) > n:
        random.shuffle(unique)
        unique = unique[:n]

    # If too few, pad from seed (unique only)
    if len(unique) < n:
        for c in seed:
            cid = candidate_id(c)
            if cid in seen:
                continue
            seen.add(cid)
            unique.append(c)
            if len(unique) >= n:
                break

    seen_global = set(state.get("seen_candidates", []))

    # Filter out globally seen candidates
    filtered = []
    dropped_seen = 0
    for c in unique:
        cid = candidate_id(c)
        if cid in seen_global:
            dropped_seen += 1
            continue
        filtered.append(c)

    if dropped_seen > 0:
        logger.warning(
            "Seen candidates filtered | iteration=%d | dropped=%d",
            state["iteration"],
            dropped_seen,
        )

    # Update global seen list with what we will actually evaluate this iteration
    new_ids = {candidate_id(c) for c in filtered}
    seen_global.update(new_ids)

    return {
        "candidates": filtered,
        "seen_candidates": list(seen_global),
    }


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
        if "char_history" not in state or state["char_history"] is None:
            state["char_history"] = {}

        # Switch backends automatically
        use_gc_ctx = ("submit_characterization" in ctx) and ("get_characterization" in ctx)
        logging.getLogger(__name__).debug("make_evaluate_candidates ctx keys: %s", sorted(ctx.keys()))
        executor = GCExecutor(ctx,
                              poll_interval_s=state.get("poll_interval", 0.25),
                              timeout_s=state.get("gc_timeout", 300.0),
                              max_retries=state.get("gc_retries", 2),
                              retry_backoff_s=state.get("gc_retry_backoff", 1.0),
                             ) if use_gc_ctx else LocalExecutor(ctx)
        logger.debug("Executor backend: %s", executor.__class__.__name__)
        print(f"Executor backend: {executor.__class__.__name__}")

        char_events: List[dict] = []
        evals: List[dict] = []

        # Prepare candidate pairs for stable mapping
        cand_pairs = []
        for c in state["candidates"]:
            ck = candidate_id(c)
            state["char_history"].setdefault(ck, {})
            cand_pairs.append((ck, c))

        histories = state["char_history"]

        # 1) Batch run fast_surrogate for all candidates
        logger.debug("loop ctx keys: %s", sorted(ctx.keys()))
        logger.debug("executor: %s", executor.__class__.__name__)
        batch_results = await executor.run_batch(
            "fast_surrogate",
            cand_pairs,
            histories,
            concurrency=state.get("concurrency", 32),
        )

        # Update history + emit events for fast_surrogate
        for ck, c in cand_pairs:
            h = histories.setdefault(ck, {})
            res = batch_results.get(ck)

            if res is None:
                h["fast_surrogate"] = {"status": "FAILED", "error": "missing batch result"}
                continue

            if res.status in ("SUCCEEDED", "CACHED"):
                h["fast_surrogate"] = res.result
            else:
                h["fast_surrogate"] = {"status": res.status, **res.result}

            char_events.append(
                {
                    "run_id": state["run_id"],
                    "iteration": state["iteration"] + 1,
                    "candidate_id": ck,
                    "candidate": c,
                    "characterizer": "fast_surrogate",
                    "status": res.status,
                    "result": res.result,
                    "latency_s": res.latency_s,
                    "ts": time.time(),
                    "provenance": res.provenance,
                }
            )

        # Rank candidates by current score (using only available history)
        ranked = [(ck, c, score_candidate(histories[ck])) for ck, c in cand_pairs]
        ranked.sort(key=lambda x: x[2], reverse=True)
        eligible = {ck for ck, _, _ in ranked[: int(state.get("escalate_k", 0)) ]}

        logger.debug(
            "Iteration %d | escalating %d candidates: %s",
            state["iteration"] + 1,
            len(eligible),
            sorted(eligible),
        )

        # 2) Optional escalation per candidate (microkinetic_lite, dft_adsorption, etc.)
        logger.info("Escalation executor backend: %s", executor.__class__.__name__)
        for ck, c, _ in ranked:
            h = histories.setdefault(ck, {})

            # Decide characterizers, but only escalate if within top-K
            logger.info("Escalation sees keys | candidate_id=%s | keys=%s", ck, list(h.keys()))
            to_run = choose_characterizers(c, h)
            logger.info("Escalation plan pre-gate | candidate_id=%s | plan=%s", ck, to_run)

            # We already ran fast_surrogate in batch; remove it if present
            to_run = [ch for ch in to_run if ch != "fast_surrogate"]

            # Budget gate: only top-K may escalate
            if ck not in eligible:
                to_run = []
                # to_run = [ch for ch in to_run if ch == "fast_surrogate"]
            logger.info("Escalation plan post-gate | candidate_id=%s | to_run=%s", ck, to_run)

            logger.info("Escalation plan | candidate_id=%s | to_run=%s", ck, to_run)
            for ch in to_run:
                ts = time.time()
                res = await executor.run(ch, c, h, candidate_id=ck)

                if res.status in ("SUCCEEDED", "CACHED"):
                    h[ch] = res.result
                else:
                    h[ch] = {"status": res.status, **res.result}

                char_events.append(
                    {
                        "run_id": state["run_id"],
                        "iteration": state["iteration"] + 1,
                        "candidate_id": ck,
                        "candidate": c,
                        "characterizer": ch,
                        "status": res.status,
                        "result": res.result,
                        "latency_s": res.latency_s,
                        "ts": ts,
                        "provenance": res.provenance,
                    }
                )

        # 3) Build evaluations
        for ck, c in cand_pairs:
            h = histories.setdefault(ck, {})
            score = score_candidate(h)

            fs = h.get("fast_surrogate", {})
            perf = fs.get("performance") if isinstance(fs, dict) else None
            enc = fs.get("encoding") if isinstance(fs, dict) else None
            cost = fs.get("catalyst_cost") if isinstance(fs, dict) else None

            evals.append(
                {
                    "candidate": c,
                    "candidate_id": ck,
                    "history": h,
                    "score": score,
                    "performance": perf,
                    "encoding": enc,
                    "catalyst_cost": cost,
                }
            )

        return {"evaluations": evals, "char_history": histories, "char_events": char_events}

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
