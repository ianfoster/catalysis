from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List
import re
import subprocess

from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager
from langchain.tools import tool
from langchain_core.tools import BaseTool

from academy.handle import Handle
from skills.rdkit_skill import RDKitSkill
from orchestration.molecule_loop import Constraints, Weights, propose, evaluate_batch, select
from orchestration.openmm_escalation import OpenMMEscalator
from orchestration.gromacs_escalation import GromacsEscalator
from orchestration.cache import OpenMMJsonlCache
from orchestration.molecule_loop import escalate_openmm
from orchestration.molecule_loop import escalate_gromacs

from globus_compute_sdk import Client as GCClient

logger = logging.getLogger(__name__)


"""
RDKit proposal
   ↓
RDKit descriptor filter
   ↓
accept/reject
   ↓
OpenMM (top-K only)
   ↓
(optional later) GROMACS
"""


def make_rdkit_tools(rdkit: Handle[RDKitSkill]) -> List[BaseTool]:
    @tool
    async def rdkit_descriptors(smiles: str) -> dict:
        """Compute RDKit descriptors for a SMILES string."""
        return await rdkit.descriptors({"smiles": smiles})

    return [rdkit_descriptors]


DEFAULT_SEEDS = [
    "c1ccccc1",          # benzene
    "Cc1ccccc1",         # toluene
    "Oc1ccccc1",         # phenol
    "Nc1ccccc1",         # aniline
    "c1ccncc1",          # pyridine
    "CCO",               # ethanol
    "CC(=O)C",           # acetone
    "CC#N",              # acetonitrile
    "CCOC(=O)C",         # ethyl acetate
    "C1CCCCC1",          # cyclohexane
    "O=C=O",             # CO2
    "CCN(CC)CC",         # triethylamine
    "OCCO",              # ethylene glycol
    "CC(=O)O",           # acetic acid
    "CNC",               # methylamine
    "COC",               # dimethyl ether
    "CCOC",              # ethyl methyl ether
    "c1ncccc1",          # pyridine isomer-ish
]


async def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-objective RDKit molecule loop (Pareto demo).")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/molecule_runs.jsonl")
    ap.add_argument("-l", "--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    ap.add_argument("--openmm-k", type=int, default=5, help="Escalate top-K molecules to OpenMM on Spark")
    ap.add_argument("--mac-transfer-ep", default="d2055662-dfcc-11e9-b5de-0ef30f6b83a8")
    ap.add_argument("--spark-transfer-ep", default="a185d484-df50-11f0-a992-0213754b0ca1")
    ap.add_argument("--spark-compute-ep", default="cee14c1d-100c-417b-9049-8c8914f3ff56")
    ap.add_argument("--openmm-function-id", required=True, help="Spark OpenMM file-minimize function UUID")
    ap.add_argument("--openmm-spark-base", default="/home/ian/openmm-runs")
    ap.add_argument("--openmm-platform", default="OpenCL")
    ap.add_argument("--openmm-max-it", type=int, default=2000)
    ap.add_argument("--openmm-cache", default="data/openmm_cache.jsonl")
    ap.add_argument("--no-openmm-cache", action="store_true")
    ap.add_argument("--gmx-k", type=int, default=0)
    ap.add_argument("--gmx-function-id", type=str)
    ap.add_argument("--gmx-spark-dir", default="/home/ian/gmx-mol-runs")
    ap.add_argument("--gmx-ntomp", type=int, default=8)
    ap.add_argument("--gmx-ntmpi", type=int, default=1)
    ap.add_argument("--no-gmx-gpu", action="store_true")
    args = ap.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    init_logging(level)
    logging.getLogger().setLevel(level)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    constraints = Constraints()
    weights = Weights()

    gc = GCClient()

    openmm_cache = None
    if args.openmm_k > 0 and not args.no_openmm_cache:
        openmm_cache = OpenMMJsonlCache.load(args.openmm_cache)

    async with await Manager.from_exchange_factory(LocalExchangeFactory()) as manager:
        rdkit = await manager.launch(RDKitSkill)
        tools = make_rdkit_tools(rdkit)
        tool_by_name = {t.name: t for t in tools}

        async def call_tool(name: str, **kwargs):
            return await tool_by_name[name].ainvoke(kwargs)

        def _sh(cmd: list[str]) -> str:
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr}\nSTDOUT:\n{p.stdout}")
            return p.stdout.strip()
        
        def _parse_task_id(out: str) -> str:
            m = re.search(r"Task ID:\s*([0-9a-f-]+)", out, re.IGNORECASE)
            if not m:
                raise RuntimeError(f"Could not parse Globus task id from:\n{out}")
            return m.group(1)
        
        def _globus_transfer(src_ep: str, src_path: str, dst_ep: str, dst_path: str, *, recursive: bool = True) -> str:
            cmd = ["globus", "transfer", f"{src_ep}:{src_path}", f"{dst_ep}:{dst_path}"]
            if recursive:
                cmd.append("--recursive")
            out = _sh(cmd)
            return _parse_task_id(out)
        
        def _globus_wait(task_id: str) -> None:
            _sh(["globus", "task", "wait", task_id])

        ctx = {"rdkit_descriptors": lambda **kw: call_tool("rdkit_descriptors", **kw)}


        async def globus_transfer_to_spark(*, src_path: str, dst_path: str, recursive: bool = True):
            tid = _globus_transfer(args.mac_transfer_ep, src_path, args.spark_transfer_ep, dst_path, recursive=recursive)
            _globus_wait(tid)
            return tid

        async def globus_transfer_from_spark(*, src_path: str, dst_path: str, recursive: bool = True):
            tid = _globus_transfer(args.spark_transfer_ep, src_path, args.mac_transfer_ep, dst_path, recursive=recursive)
            _globus_wait(tid)
            return tid

        if args.openmm_k > 0:
            if not args.openmm_function_id:
                raise RuntimeError("--openmm-k set but --openmm-function-id not provided")
            openmm = OpenMMEscalator(
                gc_client=gc,
                spark_compute_ep=args.spark_compute_ep,
                openmm_function_id=args.openmm_function_id,
                spark_base_dir=args.openmm_spark_base,
                transfer_to_spark=globus_transfer_to_spark,
                transfer_from_spark=globus_transfer_from_spark,
                platform=args.openmm_platform,
                max_iterations=args.openmm_max_it,
                cache=openmm_cache,
                local_stage_dir="data/openmm_stage",  # optional
            )
            ctx["openmm_escalate"] = openmm.run

        if args.gmx_k > 0:
            if not args.gmx_function_id:
                raise RuntimeError("--gmx-k set but --gmx-function-id not provided")
            gmx = GromacsEscalator(
                gc_client=gc,
                gmx_function_id=args.gmx_function_id,
                spark_compute_ep=args.spark_compute_ep,
                spark_base_dir=args.gmx_spark_dir,
                transfer_to_spark=globus_transfer_to_spark,
                transfer_from_spark=globus_transfer_from_spark,
                ntomp=args.gmx_ntomp,
                ntmpi=args.gmx_ntmpi,
                gpu=not args.no_gmx_gpu,
            )
            ctx["gromacs_escalate"] = gmx.run

        seen_ids: set[str] = set()
        seen_descs: List[Dict[str, float]] = []

        for it in range(args.iterations):
            smiles_batch = propose(DEFAULT_SEEDS, args.batch_size, rng)
            accepted, rejected, rdkit_stats = await evaluate_batch(ctx, smiles_batch, constraints)
            accepted = await escalate_openmm(ctx, accepted, top_k=args.openmm_k)
            accepted = await escalate_gromacs(ctx, accepted, top_k=args.gmx_k)

            logger.info(
                "Iter %d | RDKit wall=%.3fs avg=%.2fms accepted=%d rejected=%d",
                it + 1,
                rdkit_stats["rdkit_wall_s"],
                rdkit_stats["avg_call_ms"],
                len(accepted),
                len(rejected),
            )

            rdkit_cost_path = Path("data/rdkit_cost.jsonl")
            rdkit_cost_path.parent.mkdir(parents=True, exist_ok=True)

            with rdkit_cost_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "iteration": it + 1,
                    **rdkit_stats,
                    "batch_size": len(smiles_batch),
                }) + "\n")

            #print(f'RRRRRRRRRRR\n{rejected}\n====')
            #for r in rejected:
            #    print('--R', r, type(r))

            invalid = sum(1 for r in rejected if r.get("reason") != "constraint")
            constr  = sum(1 for r in rejected if r.get("reason") == "constraint")
            logger.info("Iter %d | invalid=%d constraint=%d", it + 1, invalid, constr)

            accepted = await escalate_openmm(ctx, accepted, top_k=args.openmm_k)
            accepted = await escalate_gromacs(ctx, accepted, top_k=args.gmx_k)

            # Only mark molecules as seen if RDKit parsed them (accepted or constraint-rejected),
            # not if they were invalid strings.
            #print(f'AAAAAAAA')
            #for r in accepted:
            #    print('--A', r, type(r))

            for rec in accepted:
                seen_ids.add(rec["mol_id"])
            for rec in rejected:
                # Pretty sure that the following is wrong -- we do not give this reason? XXXXXX
                if rec.get("reason") == "constraint":
                    seen_ids.add(rec["mol_id"])

            # update novelty memory using accepted descriptors
            for a in accepted:
                seen_descs.append(a["descriptors"])

            sel = select(accepted, seen_descs, weights)
            frontier = sel["frontier"]
            best = sel["best"]

            rec = {
                "ts": time.time(),
                "iteration": it + 1,
                "batch_size": args.batch_size,
                "accepted": len(accepted),
                "rejected": len(rejected),
                "frontier_size": len(frontier),
                "best": best,
                "best_score": sel["best_score"],
                "rdkit_stats": rdkit_stats
            }

            logger.info(
                "Iter %d | accepted=%d rejected=%d frontier=%d best_logP=%.3f best_MW=%.1f best_TPSA=%.1f",
                it + 1,
                len(accepted),
                len(rejected),
                len(frontier),
                float(best["descriptors"]["MolLogP"]) if best else float("nan"),
                float(best["descriptors"]["MolWt"]) if best else float("nan"),
                float(best["descriptors"]["TPSA"]) if best else float("nan"),
            )

            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
