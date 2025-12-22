from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path

from globus_compute_sdk import Client


def sh(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout.strip()


def globus_transfer(src_ep: str, src_path: str, dst_ep: str, dst_path: str, recursive: bool = True) -> str:
    cmd = ["globus", "transfer", f"{src_ep}:{src_path}", f"{dst_ep}:{dst_path}"]
    if recursive:
        cmd.append("--recursive")
    out = sh(cmd)
    # output usually includes "Task ID: <uuid>"
    return out


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
    globus_transfer(args.transfer_ep_local, local_input, args.transfer_ep_spark, spark_input_dir, recursive=True)

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
    print("Compute result:", json.dumps(result, indent=2))

    # Transfer outputs back
    print("Transferring outputs back to local...")
    globus_transfer(args.transfer_ep_spark, spark_output_dir, args.transfer_ep_local, str(local_output / run_id), recursive=True)

    print("Done.")
    print("Local outputs at:", local_output / run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
