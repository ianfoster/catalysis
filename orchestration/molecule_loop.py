from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from rdkit import Chem

import logging
logger = logging.getLogger(__name__)


def mol_id(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:16]


@dataclass
class Constraints:
    max_mw: float = 600.0
    max_hbd: int = 5
    max_hba: int = 10
    max_rings: int = 6
    min_tpsa: float = 10.0


@dataclass
class Weights:
    # for a single "champion" score (Pareto frontier is the main output)
    w_logp: float = 1.0
    w_mw: float = 0.002
    w_tpsa: float = 0.01
    w_novelty: float = 0.25


async def escalate_openmm(ctx, accepted, top_k: int):
    """
    Run OpenMM (via OpenMMEscalator) on top-K molecules.
    Expects ctx["openmm_escalate"] = async fn(smiles)->dict
    """
    if top_k <= 0 or "openmm_escalate" not in ctx:
        return accepted

    ranked = sorted(
        accepted,
        key=lambda x: x["descriptors"]["MolWt"]
    )[: min(top_k, len(accepted))]

    logger.info("OpenMM escalation | selected=%d / accepted=%d", len(ranked), len(accepted))

    for m in ranked:
        m["openmm"] = await ctx["openmm_escalate"](m["smiles"])

    return accepted


async def escalate_gromacs(ctx, accepted, top_k: int):
    if top_k <= 0 or "gromacs_escalate" not in ctx:
        logger.info("GROMACS escalation | disabled or missing ctx key")
        return accepted

    ranked = [m for m in accepted if m.get("openmm", {}).get("ok")]  # only if OpenMM succeeded
    ranked = ranked[:min(top_k, len(ranked))]

    logger.info("GROMACS escalation | selected=%d / accepted=%d", len(ranked), len(accepted))
    
    for m in ranked:
        logger.info("[GROMACS] call start | smiles=%s", m["smiles"])
        res = await ctx["gromacs_escalate"](m["smiles"])
        if isinstance(res, dict) and not res.get("ok"):
            detail = res.get("detail", {})
            logger.info("[GROMACS] stderr_tail=%s", (detail.get("stderr_tail") or "")[:500])
        m["gromacs"] = res
        logger.info(
            "[GROMACS] call done  | smiles=%s | ok=%s | stage=%s | error=%s",
            m["smiles"],
            res.get("ok") if isinstance(res, dict) else None,
            res.get("stage") if isinstance(res, dict) else None,
            res.get("error") if isinstance(res, dict) else None,
        )
    return accepted


def dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """Return True if a dominates b for objectives: maximize logP, minimize MolWt and TPSA."""
    return (
        (a["MolLogP"] >= b["MolLogP"])
        and (a["MolWt"] <= b["MolWt"])
        and (a["TPSA"] <= b["TPSA"])
        and (
            (a["MolLogP"] > b["MolLogP"])
            or (a["MolWt"] < b["MolWt"])
            or (a["TPSA"] < b["TPSA"])
        )
    )


def pareto_frontier(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return non-dominated subset."""
    front = []
    for i, x in enumerate(items):
        xd = x["descriptors"]
        dominated = False
        for j, y in enumerate(items):
            if i == j:
                continue
            yd = y["descriptors"]
            if dominates(yd, xd):
                dominated = True
                break
        if not dominated:
            front.append(x)
    return front


def novelty_score(desc: Dict[str, float], seen_descs: List[Dict[str, float]]) -> float:
    """Simple L2 distance to nearest seen descriptor (higher is more novel)."""
    if not seen_descs:
        return 1.0
    def dist(u, v):
        return (
            (u["MolWt"] - v["MolWt"]) ** 2
            + (u["MolLogP"] - v["MolLogP"]) ** 2
            + (u["TPSA"] - v["TPSA"]) ** 2
            + (u["HBD"] - v["HBD"]) ** 2
            + (u["HBA"] - v["HBA"]) ** 2
            + (u["RingCount"] - v["RingCount"]) ** 2
        ) ** 0.5
    return min(dist(desc, s) for s in seen_descs)


def propose(seed_smiles: List[str], batch_size: int, rng: random.Random) -> List[str]:
    out: List[str] = []
    local_seen: set[str] = set()
    attempts = 0
    max_attempts = 50 * batch_size

    def mutate(s: str) -> str:
        frags = ["C", "N", "O", "Cl", "F", "Br", "CC", "CO", "CN"]
        if rng.random() < 0.5 and len(s) > 3:
            i = rng.randrange(0, len(s) - 1)
            j = min(len(s), i + rng.randrange(1, 3))
            return (s[:i] + s[j:]) or s
        else:
            return s + rng.choice(frags)

    while len(out) < batch_size and attempts < max_attempts:
        attempts += 1
        s = rng.choice(seed_smiles)
        if rng.random() < 0.15:   # reduce mutation rate from 0.35 -> 0.15
            s = mutate(s)

        if s in local_seen:
            continue
        local_seen.add(s)
        out.append(s)

    return out


async def escalate_openmm_DUP(ctx, accepted, top_k: int):
    """
    Run OpenMM (via OpenMMEscalator) on top-K molecules.
    Expects ctx["openmm_escalate"] = async fn(smiles)->dict
    """
    if top_k <= 0 or "openmm_escalate" not in ctx:
        return accepted

    ranked = sorted(
        accepted,
        key=lambda x: x["descriptors"]["MolWt"]
    )[: min(top_k, len(accepted))]

    logger.info("OpenMM escalation | selected=%d / accepted=%d", len(ranked), len(accepted))

    for m in ranked:
        m["openmm"] = await ctx["openmm_escalate"](m["smiles"])

    return accepted


# -----------------------------
# Constraint logic (EXPLICIT)
# -----------------------------

def violates_constraints_OLD(desc: Dict[str, float], constraints: Dict[str, Tuple[float, float]]) -> bool:
    """
    Return True if any descriptor violates constraints.
    """
    for name, (lo, hi) in constraints.items():
        val = desc.get(name)
        if val is None:
            return True
        if val < lo or val > hi:
            return True
    return False

def violates_constraints(desc: dict, constraints) -> bool:
    """
    Return reason if molecule violates any constraint.
    """
    if desc["MolWt"] > constraints.max_mw:
        return f'MolWt {desc["MolWt"]} > {constraints.max_mw}'

    if desc["HBD"] > constraints.max_hbd:
        return f'HBD {desc["HBD"]} > {constraints.max_hbd}'

    if desc["HBA"] > constraints.max_hba:
        return f'HBA {desc["HBA"]} > {constraints.max_hba}'

    if desc["RingCount"] > constraints.max_rings:
        return f'RingCount {desc["RingCount"]} > {constraints.max_rings}'

    if desc["TPSA"] < constraints.min_tpsa:
        return f'TPSA {desc["TPSA"]} < {constraints.min_tpsa}'

    return None


# -----------------------------
# Batch evaluation
# -----------------------------

async def evaluate_batch(
    ctx: Dict,
    smiles_batch: List[str],
    constraints: Dict[str, Tuple[float, float]],
) -> Tuple[List[Dict], List[str]]:
    """
    Evaluate a batch of SMILES:
      - compute RDKit descriptors
      - reject invalid SMILES
      - enforce constraints
      - return accepted frontier molecules
    """

    rdkit = ctx["rdkit_descriptors"]

    t_eval0 = time.time()
    rdkit_call_s = 0.0
    rdkit_calls = 0

    accepted = []
    rejected = []
    invalid  = 0
    constraint_fail = 0

    for smi in smiles_batch:
        t0 = time.time()
        r = await rdkit(smiles=smi)
        rdkit_call_s += (time.time() - t0)
        rdkit_calls += 1
        rec = {
            "smiles": smi,
            "mol_id": mol_id(smi),
            "ts": time.time(),
            "ok": bool(r.get("ok", False)),
            "raw": r,
        }
        if not r.get("ok"):
            invalid += 1
            rec["reason"] = r.get("error", "invalid")
            rejected.append(rec)
            continue
    
        desc = r["descriptors"]
        reason = violates_constraints(desc, constraints)
        if reason != None:
            constraint_fail += 1
            rec["reason"] = reason
            rejected.append(rec)
            continue

        # Cheap "organic scope" filter for OpenFF v1: require at least one carbon
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            has_carbon = any(a.GetSymbol() == "C" for a in mol.GetAtoms())
            if not has_carbon:
                rec["reason"] = "constraint"
                rec["descriptors"] = desc
                rejected.append(rec)
                continue
    
        rec["descriptors"] = desc
        accepted.append(rec)

    logger.info(
        "Batch results | invalid=%d constraint=%d accepted=%d",
        invalid,
        constraint_fail,
        len(accepted),
    )

    wall_s = time.time() - t_eval0
    stats = {
        "rdkit_wall_s": wall_s,
        "rdkit_call_s": rdkit_call_s,
        "rdkit_calls": rdkit_calls,
        "invalid": invalid,
        "constraint": constraint_fail,
        "accepted": len(accepted),
        "avg_call_ms": (1000.0 * rdkit_call_s / rdkit_calls) if rdkit_calls else 0.0,
    }
    logger.info(
        "RDKit timing | wall=%.3fs calls=%d avg=%.2fms invalid=%d constraint=%d accepted=%d",
        stats["rdkit_wall_s"],
        stats["rdkit_calls"],
        stats["avg_call_ms"],
        stats["invalid"],
        stats["constraint"],
        stats["accepted"],
    )

    return accepted, rejected, stats


def champion_score(item: Dict[str, Any], seen_descs: List[Dict[str, float]], w: Weights) -> float:
    d = item["descriptors"]
    nov = novelty_score(d, seen_descs)
    return (w.w_logp * d["MolLogP"]) - (w.w_mw * d["MolWt"]) - (w.w_tpsa * d["TPSA"]) + (w.w_novelty * nov)


def select(accepted: List[Dict[str, Any]], seen_descs: List[Dict[str, float]], w: Weights) -> Dict[str, Any]:
    front = pareto_frontier(accepted)
    # pick a single champion for logging (weighted score)
    best = None
    best_s = float("-inf")
    for item in front:
        s = champion_score(item, seen_descs, w)
        if s > best_s:
            best_s = s
            best = item
    return {"frontier": front, "best": best, "best_score": best_s}
