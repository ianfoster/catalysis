from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

OPENMM_PY = "/home/ian/miniforge3/envs/openmm/bin/python"
WORKER = "/home/ian/catalysis/tools/openmm_worker.py"  # adjust if your repo path differs


def openmm_minimize_dispatch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs OpenMM minimization using conda env python.
    Expects payload with run_dir + system.xml + positions.pdb already on Spark.
    """
    t0 = time.time()

    proc = subprocess.run(
        [OPENMM_PY, WORKER],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        return {
            "ok": False,
            "error": "openmm_worker_failed",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
            "runtime_s": time.time() - t0,
        }

    try:
        out = json.loads(proc.stdout)
    except Exception as e:
        return {
            "ok": False,
            "error": f"failed_to_parse_worker_json: {e!r}",
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
            "runtime_s": time.time() - t0,
        }

    out["runtime_s"] = time.time() - t0
    out["worker_python"] = OPENMM_PY
    out["worker_script"] = WORKER
    return out
