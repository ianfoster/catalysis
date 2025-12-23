from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict


def _run(cmd: list[str], cwd: Path, timeout_s: int | None = None) -> Dict[str, Any]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return {
        "cmd": cmd,
        "returncode": p.returncode,
        "stdout_tail": p.stdout[-8000:],
        "stderr_tail": p.stderr[-8000:],
    }


def run_gromacs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run grompp + mdrun in a run directory on the endpoint.

    payload = {
      "run_id": "...",
      "run_dir": "/home/ian/gmx-runs/<run_id>",

      "grompp": {
        "mdp": "md.mdp",
        "gro": "system.gro",
        "top": "topol.top",
        "tpr": "run.tpr",
        "maxwarn": 1
      },

      "mdrun": {
        "deffnm": "run",
        "ntomp": 8,
        "ntmpi": 1,
        "gpu": true
      }
    }
    """
    print('XXXXX', payload)
    run_id = payload["run_id"]
    run_dir = Path(payload["run_dir"]).resolve()
    in_dir = run_dir / "input"
    out_dir = run_dir / "output"
    work_dir = run_dir / "work"

    for d in (in_dir, out_dir, work_dir):
        d.mkdir(parents=True, exist_ok=True)

    # sanity: gmx must exist
    which = _run(["bash", "-lc", "command -v gmx && gmx --version | head -n 1"], cwd=work_dir)
    if which["returncode"] != 0:
        return {"ok": False, "run_id": run_id, "stage": "precheck", "detail": which}

    g = payload["grompp"]
    m = payload["mdrun"]

    # 1) grompp
    tpr_path = work_dir / g.get("tpr", "run.tpr")
    grompp_cmd = [
        "gmx", "grompp",
        "-f", str(in_dir / g["mdp"]),
        "-c", str(in_dir / g["gro"]),
        "-p", str(in_dir / g["top"]),
        "-o", str(tpr_path),
        "-maxwarn", str(g.get("maxwarn", 0)),
    ]

    t_grompp0 = time.time()
    grompp_res = _run(grompp_cmd, cwd=work_dir, timeout_s=900)
    grompp_s = time.time() - t_grompp0
    if grompp_res["returncode"] != 0:
        return {"ok": False, "run_id": run_id, "stage": "grompp", "detail": grompp_res}

    # 2) mdrun
    deffnm = m.get("deffnm", "run")
    mdrun_cmd = [
        "gmx", "mdrun",
        "-s", str(tpr_path),
        "-deffnm", str(out_dir / deffnm),
        "-ntomp", str(m.get("ntomp", 1)),
        "-ntmpi", str(m.get("ntmpi", 1)),
    ]
    if m.get("gpu", False):
        mdrun_cmd += ["-nb", "gpu"]  # minimal GPU enable; tune later

    t_mdrun0 = time.time()
    mdrun_res = _run(mdrun_cmd, cwd=work_dir, timeout_s=int(m.get("timeout_s", 3600)))
    mdrun_s = time.time() - t_mdrun0

    # Output paths (expected)
    outputs = {
        "tpr": str(tpr_path),
        "log": str(out_dir / f"{deffnm}.log"),
        "xtc": str(out_dir / f"{deffnm}.xtc"),
        "edr": str(out_dir / f"{deffnm}.edr"),
        "gro": str(out_dir / f"{deffnm}.gro"),
        "cpt": str(out_dir / f"{deffnm}.cpt"),
    }

    ok = (mdrun_res["returncode"] == 0)
    return {
        "ok": ok,
        "run_id": run_id,
        "stage": "mdrun" if ok else "mdrun_failed",
        "grompp": grompp_res,
        "mdrun": mdrun_res,
        "outputs": outputs,
        "stage_times_s": {
            "grompp_s": grompp_s,
            "mdrun_s": mdrun_s,
        },

    }
