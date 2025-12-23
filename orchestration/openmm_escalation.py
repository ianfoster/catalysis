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

import logging
logger = logging.getLogger(__name__)

from globus_compute_sdk import Client


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


@dataclass
class OpenMMEscalatorConfig:
    # Transfer
    mac_transfer_ep: str
    spark_transfer_ep: str

    # Compute
    spark_compute_ep: str
    openmm_function_id: str

    # Paths
    spark_base_dir: str = "/home/ian/openmm-runs"
    local_stage_dir: str = "data/openmm_stage"   # created on Mac

    # OpenMM execution params
    platform: str = "OpenCL"
    max_iterations: int = 2000
    timeout_s: int = 600
    poll_s: float = 1.0


class OpenMMEscalator:
    """
    End-to-end:
      SMILES -> parameterize locally (OpenFF) -> transfer to Spark -> compute minimize -> return result
    """

    def __init__(self, cfg: OpenMMEscalatorConfig):
        self.cfg = cfg
        self.gc = Client()

        self.stage_root = Path(cfg.local_stage_dir)
        self.stage_root.mkdir(parents=True, exist_ok=True)

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
            "platform": self.cfg.platform,
            "max_iterations": self.cfg.max_iterations,
        }
        task_id = self.gc.run(
            payload,
            endpoint_id=self.cfg.spark_compute_ep,
            function_id=self.cfg.openmm_function_id,
        )
        return task_id

    def _poll_result(self, task_id: str) -> Dict[str, Any]:
        t0 = time.time()
        last_log = 0.0
        while True:
            t = self.gc.get_task(task_id)
            if not t.get("pending", True):
                break
    
            elapsed = time.time() - t0
            if elapsed - last_log >= 10.0:  # log every 10s
                logger.info("[OpenMM] poll | task_id=%s | status=%s | pending=%s | elapsed=%.1fs",
                            task_id, t.get("status"), t.get("pending"), elapsed)
                last_log = elapsed
    
            if elapsed > self.cfg.timeout_s:
                raise TimeoutError(f"OpenMM task {task_id} timed out after {self.cfg.timeout_s}s (status={t.get('status')})")
    
            time.sleep(self.cfg.poll_s)
    
        return self.gc.get_result(task_id)


    def _poll_result_OLD(self, task_id: str) -> Dict[str, Any]:
        t0 = time.time()
        while True:
            t = self.gc.get_task(task_id)
            if not t.get("pending", True):
                break
            if time.time() - t0 > self.cfg.timeout_s:
                raise TimeoutError(f"OpenMM task {task_id} timed out after {self.cfg.timeout_s}s (status={t.get('status')})")
            time.sleep(self.cfg.poll_s)
        return self.gc.get_result(task_id)

    async def run(self, smiles: str) -> Dict[str, Any]:
        """
        Returns a structured result with timings + task ids.
        """
        run_id = uuid.uuid4().hex[:16]
        logger.info("[OpenMM] %s | start | run_id=%s", smiles, run_id)

        local_dir = (self.stage_root / run_id).resolve()
        #src = str(local_dir) + "/"
        spark_dir = f"{self.cfg.spark_base_dir}/{run_id}"

        # 1) Parameterize locally
        logger.info("[OpenMM] %s | parameterize start | run_id=%s", smiles, run_id)
        t_param0 = time.time()
        try:
            self._parameterize_openff(smiles, local_dir)
            #self._parameterize_openff(smiles, src)
        except Exception as e:
            logger.info("[OpenMM] %s | parameterize error | run_id=%s | %s", smiles, run_id, repr(e))
            return {
                "ok": False,
                "smiles": smiles,
                "run_id": run_id,
                "stage": "parameterize",
                "error": repr(e),
            }
        t_param = time.time() - t_param0
        logger.info("[OpenMM] %s | parameterize done | run_id=%s | %.2fs", smiles, run_id, t_param)

        # 2) Transfer to Spark (copy contents of local_dir into spark_dir)
        logger.info("[OpenMM] %s | transfer start | run_id=%s | dst=%s", smiles, run_id, spark_dir)
        src = local_dir.as_posix().rstrip("/") + "/"
        dst = spark_dir

        t_in0 = time.time()
        tid_in = _globus_transfer(self.cfg.mac_transfer_ep, src, self.cfg.spark_transfer_ep, dst, recursive=True)
        logger.info("[OpenMM] %s | transfer task created | run_id=%s | task_id=%s", smiles, run_id, tid_in)
        print('XXXX', self.cfg.mac_transfer_ep, src, self.cfg.spark_transfer_ep, dst)
        _globus_wait(tid_in)
        t_in = time.time() - t_in0
        logger.info("[OpenMM] %s | transfer done | run_id=%s | task_id=%s | %.2fs", smiles, run_id, tid_in, t_in)
        logger.info("[OpenMM] %s | transfer done | run_id=%s | task=%s | %.2fs", smiles, run_id, tid_in, t_in)

        # 3) Compute minimize on Spark
        t_gc0 = time.time()
        task_id = self._submit_openmm(spark_dir)
        logger.info("[OpenMM] compute submit | run_id=%s | task_id=%s | spark_dir=%s", run_id, task_id, spark_dir)
        result = self._poll_result(task_id)
        t_gc = time.time() - t_gc0

        return {
            "ok": bool(result.get("ok", False)),
            "smiles": smiles,
            "run_id": run_id,
            "spark_run_dir": spark_dir,
            "transfer_in": {"task_id": tid_in, "elapsed_s": t_in},
            "compute": {"task_id": task_id, "elapsed_s": t_gc, "result": result},
            "parameterize": {"elapsed_s": t_param},
        }
