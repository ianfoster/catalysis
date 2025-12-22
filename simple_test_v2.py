#!/usr/bin/env python3
"""
simple_test.py (v2) - Globus Compute "no external files" smoke test

This is a simplified companion to minimal_test_v2.py, focused on a few
common scientific-chemistry checks.

Compatible with globus-compute-sdk 4.x:
- Client.run(...) returns task_id (str)
- Poll with get_task() and fetch with get_result()

Usage:
  python simple_test.py --endpoint-id <UUID>
  python simple_test.py --endpoint-id <UUID> --imports rdkit,numpy
  python simple_test.py --endpoint-id <UUID> --no-register   # reuse cached function IDs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from globus_compute_sdk import Client


# ----------------------------
# Remote test functions
# ----------------------------

def ping(payload: Optional[dict] = None) -> dict:
    import sys
    payload = payload or {}
    return {"ok": True, "echo": payload, "python": sys.version.split()[0]}


def rdkit_smiles(payload: Optional[dict] = None) -> dict:
    """Compute a couple RDKit properties to verify RDKit works."""
    payload = payload or {}
    smiles = payload.get("smiles", "c1ccccc1")  # benzene
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"ok": False, "error": f"RDKit failed to parse SMILES: {smiles}"}
        return {
            "ok": True,
            "smiles": smiles,
            "mw": float(Descriptors.MolWt(mol)),
            "hbd": int(Descriptors.NumHDonors(mol)),
            "hba": int(Descriptors.NumHAcceptors(mol)),
        }
    except ImportError as e:
        return {"ok": False, "error": f"RDKit not installed: {e!r}"}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def openmm_platforms(_: Optional[dict] = None) -> dict:
    """List available OpenMM platforms, check for CUDA."""
    try:
        from openmm import Platform
        plats = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
        return {"ok": True, "platforms": plats, "has_cuda": ("CUDA" in plats)}
    except ImportError as e:
        return {"ok": False, "error": f"OpenMM not installed: {e!r}"}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def nvidia_smi(_: Optional[dict] = None) -> dict:
    """Check whether nvidia-smi is present and list GPUs."""
    import subprocess
    try:
        p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False, timeout=10)
        return {"ok": p.returncode == 0, "rc": p.returncode, "gpus": p.stdout.strip(), "stderr": p.stderr.strip()}
    except FileNotFoundError:
        return {"ok": False, "rc": None, "gpus": None, "stderr": "nvidia-smi not found"}
    except Exception as e:
        return {"ok": False, "rc": None, "gpus": None, "stderr": repr(e)}


TESTS: Dict[str, Callable[[Optional[dict]], dict]] = {
    "ping": ping,
    "rdkit_smiles": rdkit_smiles,
    "openmm_platforms": openmm_platforms,
    "nvidia_smi": nvidia_smi,
}


# ----------------------------
# Client-side harness
# ----------------------------

def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def poll_task(gc: Client, task_id: str, *, timeout_s: float, poll_interval_s: float) -> dict:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        t = gc.get_task(task_id)
        last = t
        if not t.get("pending", True):
            return t
        time.sleep(poll_interval_s)
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
) -> dict:
    t0 = time.time()
    task_id = gc.run(payload, endpoint_id=endpoint_id, function_id=function_id) if payload is not None else gc.run(
        endpoint_id=endpoint_id, function_id=function_id
    )
    task = poll_task(gc, task_id, timeout_s=timeout_s, poll_interval_s=poll_interval_s)
    status_raw = (task.get("status") or "").lower()
    pending = task.get("pending", True)

    if pending:
        return {
            "test": test_name,
            "task_id": task_id,
            "status": "TIMEOUT",
            "runtime_s": time.time() - t0,
            "task": task,
        }

    if status_raw in ("success", "succeeded"):
        result = gc.get_result(task_id)
        return {
            "test": test_name,
            "task_id": task_id,
            "status": "SUCCEEDED",
            "runtime_s": time.time() - t0,
            "result": result,
            "task": task,
        }

    # failure
    return {
        "test": test_name,
        "task_id": task_id,
        "status": "FAILED",
        "runtime_s": time.time() - t0,
        "error": task.get("exception") or task.get("details") or str(task),
        "task": task,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Simple Globus Compute smoke test (sdk 4.x).")
    ap.add_argument("--endpoint-id", required=True, help="Endpoint UUID.")
    ap.add_argument("--timeout", type=float, default=300.0, help="Per-test timeout seconds (default: 300).")
    ap.add_argument("--poll-interval", type=float, default=0.5, help="Polling interval seconds (default: 0.5).")
    ap.add_argument("--cache", default="data/gc_simple_function_ids.json", help="Function ID cache JSON path.")
    ap.add_argument("--no-register", action="store_true", help="Do not register functions; use cached IDs only.")
    ap.add_argument("--jsonl", default="data/gc_simple_runs.jsonl", help="Write one JSONL line per test.")
    ap.add_argument("--smiles", default="c1ccccc1", help="SMILES for rdkit_smiles test (default: benzene).")
    ap.add_argument("--tests", default="ping,rdkit_smiles,openmm_platforms,nvidia_smi",
                    help="Comma-separated tests (default: ping,rdkit_smiles,openmm_platforms,nvidia_smi).")
    args = ap.parse_args()

    gc = Client()
    endpoint_id = args.endpoint_id

    cache_path = Path(args.cache)
    cache = load_cache(cache_path)

    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    for t in tests:
        if t not in TESTS:
            print(f"ERROR: unknown test '{t}'. Valid: {sorted(TESTS.keys())}")
            return 2

    if not args.no_register:
        for name in tests:
            if name not in cache:
                print(f"Registering function for '{name}' ...")
                cache[name] = gc.register_function(TESTS[name])
        save_cache(cache_path, cache)
        print(f"Saved function_id cache to {cache_path}")
    else:
        missing = [t for t in tests if t not in cache]
        if missing:
            print(f"ERROR: --no-register set but missing cached function IDs for: {missing}")
            print(f"Cache file: {cache_path}")
            return 2

    jsonl_path = Path(args.jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"USING ENDPOINT: {endpoint_id}")
    print(f"TESTS: {tests}")

    # Run tests
    passed = 0
    total = 0
    for t in tests:
        payload = None
        if t == "ping":
            payload = {"msg": "hello"}
        elif t == "rdkit_smiles":
            payload = {"smiles": args.smiles}

        total += 1
        rec = run_one(
            gc,
            endpoint_id=endpoint_id,
            test_name=t,
            function_id=cache[t],
            payload=payload,
            timeout_s=args.timeout,
            poll_interval_s=args.poll_interval,
        )

        status = rec["status"]
        ok = (status == "SUCCEEDED") and (not isinstance(rec.get("result"), dict) or rec["result"].get("ok", True))
        passed += 1 if ok else 0

        print(f"[{status}] {t} | task_id={rec.get('task_id','')} | {rec.get('runtime_s',0.0):.2f}s")
        if "result" in rec:
            print(f"  result: {rec['result']}")
        if "error" in rec:
            print(f"  error: {rec['error']}")

        # Persist JSONL
        rec_out = {
            "ts": time.time(),
            "endpoint_id": endpoint_id,
            **rec,
            "function_id": cache[t],
        }
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec_out) + "\n")

    print(f"\nPassed: {passed}/{total}")
    print(f"Wrote JSONL to {jsonl_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise
