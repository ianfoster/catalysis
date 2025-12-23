from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
from orchestration.cache import OpenMMJsonlCache
from orchestration.escalators.base import BaseEscalator

import logging
logger = logging.getLogger(__name__)

def canonical_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return smiles
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return smiles

def _sh(cmd: list[str], *, cwd: Optional[str] = None) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p.stdout.strip()


def _parse_task_id(globus_transfer_output: str) -> str:
    m = re.search(r"Task ID:\s*([0-9a-f-]+)", globus_transfer_output, re.IGNORECASE)
    if not m:
        raise RuntimeError(f"Could not parse Globus task id from:\n{globus_transfer_output}")
    return m.group(1)


def _globus_transfer(src_ep: str, src_path: str, dst_ep: str, dst_path: str, *, recursive: bool) -> str:
    cmd = ["globus", "transfer", f"{src_ep}:{src_path}", f"{dst_ep}:{dst_path}"]
    if recursive:
        cmd.append("--recursive")
    out = _sh(cmd)
    return _parse_task_id(out)


def _globus_wait(task_id: str) -> None:
    _sh(["globus", "task", "wait", task_id])


from orchestration.escalators.base import BaseEscalator

class OpenMMEscalator(BaseEscalator):
    """
    End-to-end:
      SMILES -> parameterize locally (OpenFF) -> transfer to Spark -> compute minimize -> return result
    """
    def __init__(
        self,
        *,
        gc_client,
        spark_compute_ep: str,
        openmm_function_id: str,
        spark_base_dir: str,
        transfer_to_spark,
        transfer_from_spark,
        platform: str = "OpenCL",
        max_iterations: int = 2000,
        cache=None,
        local_stage_dir: str = "data/openmm_stage",
        cost_path: str = "data/openmm_cost.jsonl",
        timeout_s: float = 600.0,
        poll_s: float = 1.0,
    ):
        super().__init__(
            gc_client=gc_client,
            transfer_to_spark=transfer_to_spark,
            transfer_from_spark=transfer_from_spark,
        )

        self.openmm_function_id = openmm_function_id
        self.spark_base_dir = spark_base_dir
        self.spark_compute_ep = spark_compute_ep
        self.platform = platform
        self.max_iterations = max_iterations
        self.cache = cache

        self.stage_root = Path(local_stage_dir).resolve()
        self.stage_root.mkdir(parents=True, exist_ok=True)

        self.cost_path = Path("data/openmm_cost.jsonl")
        self.cost_path.parent.mkdir(parents=True, exist_ok=True)

    def _parameterize_openff(self, smiles: str, outdir: Path) -> None:
        """
        Calls parameterize_openff.py in a dedicated conda env (openff311).
        Assumes `conda` is available on Mac.
        """
        outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "bash", "-lc",
            # IMPORTANT: openff311 env
            f"conda activate openff311 && "
            f"python parameterize_openff.py --smiles '{smiles}' --outdir '{outdir.as_posix()}'"
        ]
        _sh(cmd)

        # Sanity check expected files exist
        for fn in ("system.xml", "positions.pdb"):
            p = outdir / fn
            if not p.exists():
                raise RuntimeError(f"parameterization did not produce {p}")

    def _submit_openmm(self, spark_run_dir: str) -> str:
        payload = {
            "run_dir": spark_run_dir,
            "system_xml": "system.xml",
            "positions_pdb": "positions.pdb",
            "platform": self.platform,
            "max_iterations": self.max_iterations,
        }
        task_id = self.gc.run(
            payload,
            endpoint_id=self.spark_compute_ep,
            function_id=self.openmm_function_id,
        )
        return task_id

    def _poll_result(self, task_id: str) -> Dict[str, Any]:
        t0 = time.time()
        last_log = 0.0
        last = None
        while True:
            t = self.gc.get_task(task_id)
            last = t
            if not t.get("pending", True):
                break
            elapsed = time.time() - t0
            if elapsed - last_log >= 10.0:  # log every 10s
                logger.info("[OpenMM] poll | task_id=%s | status=%s | pending=%s | elapsed=%.1fs",
                            task_id, t.get("status"), t.get("pending"), elapsed)
                last_log = elapsed
            if elapsed > self.timeout_s:
                raise TimeoutError(f"OpenMM task {task_id} timed out after {self.timeout_s}s (status={t.get('status')})")
            time.sleep(self.poll_s)
        return self.gc.get_result(task_id), (last or {})


    def _poll_result_OLD(self, task_id: str) -> Dict[str, Any]:
        t0 = time.time()
        while True:
            t = self.gc.get_task(task_id)
            if not t.get("pending", True):
                break
            if time.time() - t0 > self.timeout_s:
                raise TimeoutError(f"OpenMM task {task_id} timed out after {self.timeout_s}s (status={t.get('status')})")
            time.sleep(self.poll_s)
        return self.gc.get_result(task_id)

    async def run(self, smiles: str) -> Dict[str, Any]:
        """
        Returns a structured result with timings + task ids.
        """
        can = canonical_smiles(smiles)
        # Cache key includes molecule + settings that affect output
        cache_key = hashlib.sha1(
            json.dumps(
                {
                    "smiles": can,
                    "platform": self.platform,
                    "max_iterations": self.max_iterations,
                    "spark_compute_ep": self.spark_compute_ep,
                    "openmm_function_id": self.openmm_function_id,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]

        if self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                logger.info("[OpenMM] cache hit | smiles=%s | key=%s", smiles, cache_key)
                return {
                    "ok": bool(hit.get("ok", False)),
                    "smiles": smiles,
                    "cache": {"hit": True, "key": cache_key},
                    **hit,
                }
        run_id = uuid.uuid4().hex[:16]
        logger.info("[OpenMM] %s | start | run_id=%s", smiles, run_id)

        local_dir = (self.stage_root / run_id).resolve()
        #src = str(local_dir) + "/"
        spark_dir = f"{self.spark_base_dir}/{run_id}"

        # 1) Parameterize locally
        logger.info("[OpenMM] %s | parameterize start | run_id=%s", smiles, run_id)
        t_param0 = time.time()
        try:
            self._parameterize_openff(smiles, local_dir)
        except Exception as e:
            logger.info("[OpenMM] %s | parameterize error | run_id=%s | %s", smiles, run_id, repr(e))
            fail = {
                "ok": False,
                "smiles": smiles,
                "run_id": run_id,
                "stage": "parameterize",
                "error": repr(e),
                "cache": {"hit": False, "key": cache_key},
            }
            if self.cache is not None:
                self.cache.set(cache_key, fail, meta={"canonical_smiles": can})
            return fail
        t_param = time.time() - t_param0
        logger.info("[OpenMM] %s | parameterize done | run_id=%s | %.2fs", smiles, run_id, t_param)

        # 2) Transfer to Spark (copy contents of local_dir into spark_dir)
        logger.info("[OpenMM] %s | transfer start | run_id=%s | dst=%s", smiles, run_id, spark_dir)
        src = local_dir.as_posix().rstrip("/") + "/"
        dst = spark_dir

        t_in0 = time.time()
        tid_in = _globus_transfer(self.mac_transfer_ep, src, self.spark_transfer_ep, dst, recursive=True)
        logger.info("[OpenMM] %s | transfer task created | run_id=%s | task_id=%s", smiles, run_id, tid_in)
        _globus_wait(tid_in)
        t_in = time.time() - t_in0
        logger.info("[OpenMM] %s | transfer done | run_id=%s | task_id=%s | %.2fs", smiles, run_id, tid_in, t_in)
        logger.info("[OpenMM] %s | transfer done | run_id=%s | task=%s | %.2fs", smiles, run_id, tid_in, t_in)

        # 3) Compute minimize on Spark
        submit_ts = time.time()
        task_id = self._submit_openmm(spark_dir)
        logger.info("[OpenMM] compute submit | run_id=%s | task_id=%s | spark_dir=%s", run_id, task_id, spark_dir)
        result, task = self._poll_result(task_id)
        logger.info("[OpenMM] compute result type=%s keys=%s", type(result), getattr(result, "keys", lambda: [])())
        t_gc = time.time() - submit_ts
    
        details = (task.get("details") or {})
        trans = (details.get("task_transitions") or {})
        exec_start = trans.get("execution-start")
        exec_end = trans.get("execution-end")
        
        queue_s = (exec_start - submit_ts) if (exec_start is not None) else None
        exec_s = (exec_end - exec_start) if (exec_start is not None and exec_end is not None) else None

        final = {
            "ok": bool(result.get("ok", False)),
            "smiles": smiles,
            "run_id": run_id,
            "spark_run_dir": spark_dir,
            "parameterize": {"elapsed_s": t_param},
            "transfer_in": {"task_id": tid_in, "elapsed_s": t_in},
            "compute": {"task_id": task_id, "elapsed_s": t_gc, "result": result},
            "cache": {"hit": False, "key": cache_key},
        }
        final["cost"] = {
            "parameterize_s": t_param,
            "transfer_in_s": t_in,
            "compute_wall_s": t_gc,
            "compute_queue_s": queue_s,
            "compute_exec_s": exec_s,
        }
        final["compute"]["task_transitions"] = trans
        final["compute"]["submit_ts"] = submit_ts

        cost_rec = {
            "ts": time.time(),
            "smiles": smiles,
            "run_id": run_id,
            "spark_run_dir": spark_dir,
            "transfer_task_in": tid_in,
            "compute_task_id": task_id,
            "parameterize_s": t_param,
            "transfer_in_s": t_in,
            "compute_wall_s": t_gc,
            "compute_queue_s": queue_s,
            "compute_exec_s": exec_s,
            "platform": (result.get("platform") if isinstance(result, dict) else None),
            "ok": bool(result.get("ok", False)) if isinstance(result, dict) else False,
        }
        with self.cost_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cost_rec) + "\n")

        if self.cache is not None:
            self.cache.set(
                cache_key,
                final,
                meta={"canonical_smiles": can},
            )

        return final
