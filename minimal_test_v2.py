#!/usr/bin/env python3
"""
minimal_test.py (v2)

A robust Globus Compute smoke-test harness compatible with globus-compute-sdk 4.x.

Key improvements vs the original:
- Uses Client.run() -> task_id and polls via get_task()/get_result()
- Registers test functions once and caches function_ids locally (optional)
- Configurable timeouts/poll interval
- Better error reporting and JSONL logging

Usage:
  python minimal_test.py --endpoint-id <UUID>
  python minimal_test.py --endpoint-id <UUID> --no-register   # use cached function IDs only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from globus_compute_sdk import Client


# ----------------------------
# Test functions (run remotely)
# ----------------------------

def test_python_info(_: dict | None = None) -> dict:
    import platform
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "uname": platform.uname()._asdict(),
    }


def test_math(_: dict | None = None) -> dict:
    # tiny deterministic computation
    s = sum(i * i for i in range(10_000))
    return {"sum_sq_0_9999": s}


def test_imports(payload: dict | None = None) -> dict:
    payload = payload or {}
    imports = payload.get("imports", ["numpy"])
    out: Dict[str, Any] = {"imports": {}}
    for name in imports:
        try:
            __import__(name)
            out["imports"][name] = {"ok": True}
        except Exception as e:
            out["imports"][name] = {"ok": False, "error": repr(e)}
    return out


def test_gpu(payload: dict | None = None) -> dict:
    # Avoid heavy deps; try nvidia-smi if present; otherwise just report "unknown"
    import subprocess
    try:
        p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False, timeout=10)
        return {"nvidia_smi_rc": p.returncode, "gpus": p.stdout.strip(), "stderr": p.stderr.strip()}
    except FileNotFoundError:
        return {"nvidia_smi_rc": None, "gpus": None, "stderr": "nvidia-smi not found"}
    except Exception as e:
        return {"nvidia_smi_rc": None, "gpus": None, "stderr": repr(e)}


TESTS: Dict[str, Callable[[Optional[dict]], dict]] = {
    "python_info": test_python_info,
    "math": test_math,
    "imports": test_imports,
    "gpu": test_gpu,
}


# ----------------------------
# Harness
# ----------------------------

@dataclass
class RunResult:
    test: str
    function_id: str
    task_id: str
    status: str
    runtime_s: float
    result: Any = None
    error: Optional[str] = None
    task: Optional[dict] = None


def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache_path: Path, data: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def poll_task(gc: Client, task_id: str, *, timeout_s: float, poll_interval_s: float) -> dict:
    """Poll until task.pending is False or timeout."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        t = gc.get_task(task_id)
        last = t
        if not t.get("pending", True):
            return t
        time.sleep(poll_interval_s)
    # timed out; return last seen state for debugging
    return last or {"pending": True, "status": "unknown"}


def run_one(
    gc: Client,
    endpoint_id: str,
    test_name: str,
    function_id: str,
    payload: Optional[dict],
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> RunResult:
    t0 = time.time()
    try:
        task_id = gc.run(payload, endpoint_id=endpoint_id, function_id=function_id) if payload is not None else gc.run(
            endpoint_id=endpoint_id, function_id=function_id
        )
        # Poll for completion (get_result also blocks, but poll gives more visibility)
        task = poll_task(gc, task_id, timeout_s=timeout_s, poll_interval_s=poll_interval_s)
        pending = task.get("pending", True)
        status = (task.get("status") or "").lower()

        if pending:
            return RunResult(
                test=test_name,
                function_id=function_id,
                task_id=task_id,
                status="TIMEOUT",
                runtime_s=time.time() - t0,
                error=f"timeout after {timeout_s}s (last status={status})",
                task=task,
            )

        if status in ("success", "succeeded"):
            # get_result returns the remote function return value
            result = gc.get_result(task_id)
            return RunResult(
                test=test_name,
                function_id=function_id,
                task_id=task_id,
                status="SUCCEEDED",
                runtime_s=time.time() - t0,
                result=result,
                task=task,
            )

        # Failure: surface exception details if present
        # get_task() may include 'exception' or details; get_result raises TaskExecutionFailed.
        err = task.get("exception") or task.get("details") or str(task)
        return RunResult(
            test=test_name,
            function_id=function_id,
            task_id=task_id,
            status="FAILED",
            runtime_s=time.time() - t0,
            error=str(err),
            task=task,
        )

    except Exception as e:
        return RunResult(
            test=test_name,
            function_id=function_id,
            task_id="",
            status="ERROR",
            runtime_s=time.time() - t0,
            error=repr(e),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Globus Compute endpoint smoke tests (sdk 4.x).")
    ap.add_argument("--endpoint-id", required=True, help="Globus Compute endpoint UUID.")
    ap.add_argument("--timeout", type=float, default=300.0, help="Per-test timeout seconds (default: 300).")
    ap.add_argument("--poll-interval", type=float, default=0.5, help="Polling interval seconds (default: 0.5).")
    ap.add_argument("--cache", default="data/gc_function_ids.json", help="Path to function_id cache JSON.")
    ap.add_argument("--no-register", action="store_true", help="Do not register functions; use cached IDs only.")
    ap.add_argument("--jsonl", default="data/gc_test_runs.jsonl", help="Write one JSONL line per test.")
    ap.add_argument(
        "--imports",
        default="numpy",
        help="Comma-separated import module names for the 'imports' test (default: numpy).",
    )
    ap.add_argument(
        "--tests",
        default="python_info,math,imports,gpu",
        help="Comma-separated list of tests to run (default: python_info,math,imports,gpu).",
    )
    args = ap.parse_args()

    cache_path = Path(args.cache)
    cache = load_cache(cache_path)

    gc = Client()
    endpoint_id = args.endpoint_id

    # Register functions unless disabled; cache IDs by test name
    if not args.no_register:
        for name, fn in TESTS.items():
            if name not in cache:
                print(f"Registering function for test '{name}' ...")
                cache[name] = gc.register_function(fn)
        save_cache(cache_path, cache)
        print(f"Saved function_id cache to {cache_path}")
    else:
        missing = [t for t in args.tests.split(",") if t.strip() and t.strip() not in cache]
        if missing:
            print(f"ERROR: --no-register set but missing cached function IDs for: {missing}")
            print(f"Cache file: {cache_path}")
            return 2

    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    imports_list = [s.strip() for s in args.imports.split(",") if s.strip()]
    jsonl_path = Path(args.jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"USING ENDPOINT: {endpoint_id}")
    print(f"TESTS: {tests}")

    for tname in tests:
        if tname not in TESTS:
            print(f"Skipping unknown test: {tname}")
            continue

        payload = None
        if tname == "imports":
            payload = {"imports": imports_list}

        fid = cache[tname]
        rr = run_one(
            gc,
            endpoint_id=endpoint_id,
            test_name=tname,
            function_id=fid,
            payload=payload,
            timeout_s=args.timeout,
            poll_interval_s=args.poll_interval,
        )

        # Console summary
        print(f"[{rr.status}] {tname} | task_id={rr.task_id} | {rr.runtime_s:.2f}s")
        if rr.status == "SUCCEEDED":
            # show a compact preview
            preview = rr.result
            print(f"  result: {preview}")
        else:
            print(f"  error: {rr.error}")
            if rr.task is not None:
                print(f"  task: {rr.task}")

        # JSONL record
        rec = {
            "ts": time.time(),
            "endpoint_id": endpoint_id,
            "test": rr.test,
            "function_id": rr.function_id,
            "task_id": rr.task_id,
            "status": rr.status,
            "runtime_s": rr.runtime_s,
            "result": rr.result,
            "error": rr.error,
            "task": rr.task,
        }
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote JSONL to {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
