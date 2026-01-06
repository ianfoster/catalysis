#!/usr/bin/env python3
"""Attempt to fix M3GNet/MatGL installation on Spark.

The issue: matgl requires torch<=2.8.0 but Spark has torch 2.9.0.
Solution: Try installing matgl with --no-deps and see if it works anyway.

Usage:
    python scripts/fix_m3gnet.py
"""

import subprocess
import sys


def run_cmd(cmd: str, check: bool = False) -> tuple[int, str]:
    """Run a command and return (returncode, output)."""
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result.returncode, output


def check_current_state():
    """Check what's currently installed."""
    print("=" * 60)
    print("CURRENT STATE")
    print("=" * 60)

    # Check torch version
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        return False

    # Check matgl
    try:
        import matgl
        print(f"MatGL: {matgl.__version__} - ALREADY INSTALLED")
        return True
    except ImportError as e:
        print(f"MatGL: NOT INSTALLED ({e})")

    # Check legacy m3gnet
    try:
        import m3gnet
        print(f"m3gnet (legacy): INSTALLED")
        return True
    except ImportError:
        print("m3gnet (legacy): NOT INSTALLED")

    return False


def try_matgl_no_deps():
    """Try installing matgl without dependency checking."""
    print("\n" + "=" * 60)
    print("ATTEMPTING: pip install matgl --no-deps")
    print("=" * 60)

    # First uninstall if exists
    run_cmd("pip uninstall matgl -y 2>/dev/null || true")

    # Install without deps
    code, _ = run_cmd("pip install matgl --no-deps")
    if code != 0:
        print("Failed to install matgl")
        return False

    # Install missing deps that matgl needs (except torch)
    print("\nInstalling matgl dependencies (except torch)...")
    deps = [
        "dgl",  # Deep Graph Library
        "ase",  # Atomic Simulation Environment
        "pymatgen",  # Materials analysis
    ]
    for dep in deps:
        run_cmd(f"pip install {dep} 2>/dev/null || true")

    return True


def test_matgl():
    """Test if matgl works."""
    print("\n" + "=" * 60)
    print("TESTING MATGL")
    print("=" * 60)

    # Need to reload modules
    import importlib

    try:
        import matgl
        print(f"1. Import matgl: OK (version {matgl.__version__})")
    except ImportError as e:
        print(f"1. Import matgl: FAILED - {e}")
        return False

    try:
        from matgl.ext.ase import PESCalculator
        print("2. Import PESCalculator: OK")
    except ImportError as e:
        print(f"2. Import PESCalculator: FAILED - {e}")
        return False

    try:
        potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        print("3. Load M3GNet model: OK")
    except Exception as e:
        print(f"3. Load M3GNet model: FAILED - {e}")
        return False

    try:
        from ase.build import bulk
        import numpy as np

        atoms = bulk("Cu", "fcc", a=3.6)
        calc = PESCalculator(potential=potential)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"4. Calculate energy: OK (E = {energy:.4f} eV)")
    except Exception as e:
        print(f"4. Calculate energy: FAILED - {e}")
        return False

    print("\n*** M3GNet is working! ***")
    return True


def try_legacy_m3gnet():
    """Try installing legacy TensorFlow-based m3gnet."""
    print("\n" + "=" * 60)
    print("ATTEMPTING: pip install m3gnet (legacy TF version)")
    print("=" * 60)

    code, _ = run_cmd("pip install m3gnet")
    if code != 0:
        print("Failed to install legacy m3gnet")
        return False

    # Test it
    print("\nTesting legacy m3gnet...")
    try:
        from m3gnet.models import M3GNet, Potential, M3GNetCalculator
        print("1. Import m3gnet: OK")

        potential = Potential(M3GNet.load())
        print("2. Load model: OK")

        from ase.build import bulk
        atoms = bulk("Cu", "fcc", a=3.6)
        calc = M3GNetCalculator(potential=potential)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"3. Calculate energy: OK (E = {energy:.4f} eV)")

        print("\n*** Legacy m3gnet is working! ***")
        return True
    except Exception as e:
        print(f"Legacy m3gnet test failed: {e}")
        return False


def main():
    print("M3GNet Fix Script")
    print("=" * 60)

    # Check current state
    if check_current_state():
        print("\nM3GNet already working, testing...")
        if test_matgl():
            return 0

    # Try matgl with --no-deps
    if try_matgl_no_deps():
        if test_matgl():
            print("\n" + "=" * 60)
            print("SUCCESS: matgl installed and working with --no-deps")
            print("=" * 60)
            return 0

    # Fall back to legacy m3gnet
    print("\nmatgl didn't work, trying legacy m3gnet...")
    if try_legacy_m3gnet():
        print("\n" + "=" * 60)
        print("SUCCESS: legacy m3gnet installed and working")
        print("=" * 60)
        return 0

    print("\n" + "=" * 60)
    print("FAILED: Could not get M3GNet working")
    print("=" * 60)
    print("\nOptions:")
    print("1. Create a separate conda env with torch<=2.8.0 for M3GNet")
    print("2. Use MACE and CHGNet only (both work)")
    print("3. Wait for matgl to support torch 2.9+")
    return 1


if __name__ == "__main__":
    sys.exit(main())
