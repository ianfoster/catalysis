#!/usr/bin/env python3
"""Check Globus Compute endpoint environment."""

import argparse
import sys


def check_environment(config: dict) -> dict:
    """Check Python environment on GC endpoint."""
    import os
    import sys

    result = {
        "python": sys.executable,
        "version": sys.version,
        "path": sys.path[:10],  # First 10 entries
        "cwd": os.getcwd(),
    }

    # Check if skills is importable
    try:
        import skills
        result["skills_found"] = True
        result["skills_path"] = skills.__file__
    except ImportError as e:
        result["skills_found"] = False
        result["skills_error"] = str(e)

    # Check for key packages
    for pkg in ["academy", "mace", "chgnet", "torch", "ase"]:
        try:
            mod = __import__(pkg)
            result[f"{pkg}_found"] = True
        except ImportError:
            result[f"{pkg}_found"] = False

    return result


def main():
    parser = argparse.ArgumentParser(description="Check GC endpoint environment")
    parser.add_argument("--endpoint", required=True, help="GC endpoint ID")
    args = parser.parse_args()

    from globus_compute_sdk import Client, Executor

    client = Client()
    func_id = client.register_function(check_environment)

    print(f"Checking environment on endpoint {args.endpoint}...")

    with Executor(endpoint_id=args.endpoint) as ex:
        future = ex.submit_to_registered_function(func_id, args=({},))
        result = future.result(timeout=60)

    print("\nGC Endpoint Environment:")
    print(f"  Python: {result.get('python')}")
    print(f"  Version: {result.get('version', '').split()[0]}")
    print(f"  CWD: {result.get('cwd')}")
    print(f"\nPackages:")
    print(f"  skills: {result.get('skills_found')} - {result.get('skills_path', result.get('skills_error', ''))}")
    print(f"  academy: {result.get('academy_found')}")
    print(f"  mace: {result.get('mace_found')}")
    print(f"  chgnet: {result.get('chgnet_found')}")
    print(f"  torch: {result.get('torch_found')}")
    print(f"  ase: {result.get('ase_found')}")
    print(f"\nPython path (first 5):")
    for p in result.get('path', [])[:5]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
