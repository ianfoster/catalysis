from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path
import re

from globus_compute_sdk import Client

def wait_transfer(task_id: str) -> None:
    sh(["globus", "task", "wait", task_id])

def sh(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr}\nSTDOUT:\n{p.stdout}")
    return p.stdout.strip()

def sh_json(cmd: list[str]) -> dict:
    out = sh(cmd)
    return json.loads(out)

def globus_transfer(src_ep: str, src_path: str, dst_ep: str, dst_path: str, *, recursive: bool = True) -> str:
    cmd = ["globus", "transfer", f"{src_ep}:{src_path}", f"{dst_ep}:{dst_path}"]
    if recursive:
        cmd.append("--recursive")
    out = sh(cmd)
    m = re.search(r"Task ID:\s*([0-9a-f-]+)", out, re.IGNORECASE)
    if not m:
        raise RuntimeError(f"Could not parse transfer task id from:\n{out}")
    return m.group(1)

def wait_transfer(task_id: str) -> None:
    sh(["globus", "task", "wait", task_id])

def transfer_stats(task_id: str) -> dict:
    # Globus CLI supports JSON output for task show
    try:
        return sh_json(["globus", "task", "show", task_id, "--format", "json"])
    except Exception as e:
        # Don’t fail the run just because stats parsing fails
        return {"error": f"task show failed: {e!r}", "task_id": task_id}

def timed_transfer(src_ep: str, src_path: str, dst_ep: str, dst_path: str, *, recursive: bool = True) -> dict:
    t0 = time.time()
    tid = globus_transfer(src_ep, src_path, dst_ep, dst_path, recursive=recursive)
    wait_transfer(tid)
    elapsed = time.time() - t0
    stats = transfer_stats(tid)
    return {"task_id": tid, "elapsed_s": elapsed, "stats": stats}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute-endpoint", required=True, help="Globus Compute endpoint UUID (Spark).")
    ap.add_argument("--gmx-function-id", required=True, help="Registered function UUID for run_gromacs.")
    ap.add_argument("--transfer-ep-local", required=True, help="Globus Transfer endpoint ID for your local machine.")
    ap.add_argument("--transfer-ep-spark", required=True, help="Globus Transfer endpoint ID for Spark.")
    ap.add_argument("--local-input-dir", required=True, help="Local directory containing input files (mdp/gro/top).")
    ap.add_argument("--local-output-dir", required=True, help="Local directory to receive outputs.")
    ap.add_argument("--spark-base-dir", default="/home/ian/gmx-runs", help="Base run dir on Spark.")
    ap.add_argument("--ntomp", type=int, default=8)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--poll", type=float, default=2.0)
    args = ap.parse_args()

    run_id = uuid.uuid4().hex[:16]
    spark_run_dir = f"{args.spark_base_dir}/{run_id}"
    spark_input_dir = f"{spark_run_dir}/input"
    spark_output_dir = f"{spark_run_dir}/output"

    local_input = str(Path(args.local_input_dir).resolve())
    local_output = Path(args.local_output_dir).resolve()
    local_output.mkdir(parents=True, exist_ok=True)

    print(f"RUN_ID: {run_id}")
    print("Transferring inputs to Spark...")
    # Transfer inputs (copy contents of local_input_dir into Spark input/)
    src = str(Path(args.local_input_dir).resolve()).rstrip("/") + "/"
    dst = f"{spark_run_dir}/input"

    transfer_in = timed_transfer(
        args.transfer_ep_local,
        src,
        args.transfer_ep_spark,
        dst,
        recursive=True,
    )
    print("Input transfer:", transfer_in["task_id"], f"{transfer_in['elapsed_s']:.2f}s")

    # Submit compute task
    payload = {
        "run_id": run_id,
        "run_dir": spark_run_dir,
        "grompp": {
            "mdp": "md.mdp",
            "gro": "system.gro",
            "top": "topol.top",
            "tpr": "run.tpr",
            "maxwarn": 1,
        },
        "mdrun": {
            "deffnm": "run",
            "ntomp": args.ntomp,
            "ntmpi": 1,
            "gpu": bool(args.gpu),
            "timeout_s": args.timeout,
        },
    }

    gc = Client()
    task_id = gc.run(payload, endpoint_id=args.compute_endpoint, function_id=args.gmx_function_id)
    print("Submitted compute task:", task_id)

    # Poll until done
    t0 = time.time()
    while True:
        t = gc.get_task(task_id)
        if not t.get("pending", True):
            break
        if time.time() - t0 > args.timeout:
            raise RuntimeError(f"Timed out waiting for task {task_id} after {args.timeout}s (status={t.get('status')})")
        time.sleep(args.poll)

    result = gc.get_result(task_id)

    task = gc.get_task(task_id)
    print("Compute result:", json.dumps(result, indent=2))

    print("Transferring outputs back to local...")
    dst_local = str((Path(args.local_output_dir).resolve() / run_id))
    transfer_out = timed_transfer(args.transfer_ep_spark,
                                  f"{spark_run_dir}/output/",
                                  args.transfer_ep_local, dst_local, recursive=True)
    print("Output transfer:", transfer_out["task_id"], f"{transfer_out['elapsed_s']:.2f}s")
    print("Local outputs at:", local_output / run_id)

    run_cost = {
        "transfer_in": transfer_in,
        "transfer_out": transfer_out,
        "compute_task": task,
    }
    print("Cost summary:", json.dumps(run_cost, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
