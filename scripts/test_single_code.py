#!/usr/bin/env python3
"""Test individual simulation codes via Globus Compute.

This script registers and tests a single function, useful for debugging.

Usage:
    # Test MACE ML potential
    python scripts/test_single_code.py --endpoint ENDPOINT_ID --code mace

    # Test OpenMM
    python scripts/test_single_code.py --endpoint ENDPOINT_ID --code openmm

    # Test all ML potentials
    python scripts/test_single_code.py --endpoint ENDPOINT_ID --code ml

Available codes:
    mace     - MACE ML potential
    chgnet   - CHGNet ML potential
    m3gnet   - M3GNet ML potential
    ml       - All ML potentials (auto-select best)
    openmm   - OpenMM structure relaxation
    gromacs  - GROMACS molecular dynamics
    gpaw     - GPAW DFT
    qe       - Quantum Espresso DFT
    catmap   - CatMAP microkinetics
    cantera  - Cantera reactor
"""

import argparse
import json
import sys
import time


# Test candidate
TEST_CANDIDATE = {
    "support": "ZrO2",
    "metals": [
        {"element": "Cu", "wt_pct": 55},
        {"element": "Zn", "wt_pct": 30},
        {"element": "Al", "wt_pct": 15},
    ],
}


# =============================================================================
# TEST FUNCTIONS (self-contained, registered on-the-fly)
# =============================================================================

def test_mace(payload: dict) -> dict:
    """Test MACE ML potential."""
    try:
        from ase.build import fcc111
        from mace.calculators import mace_mp

        candidate = payload.get("candidate", {})
        metals = candidate.get("metals", [])
        cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 60)
        zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 25)

        # Build structure
        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
        n_zn = int(4 * zn / (cu + zn + 0.01))
        symbols = list(slab.get_chemical_symbols())
        surface_idx = [i for i, z in enumerate(slab.positions[:, 2])
                       if z > slab.positions[:, 2].max() - 2.0]
        for i in range(min(n_zn, len(surface_idx))):
            symbols[surface_idx[i]] = "Zn"
        slab.set_chemical_symbols(symbols)

        # MACE calculation
        calc = mace_mp(model="small", device="cpu", default_dtype="float32")
        slab.calc = calc
        energy = slab.get_potential_energy()
        forces = slab.get_forces()

        return {
            "ok": True,
            "code": "MACE",
            "total_energy_eV": round(float(energy), 6),
            "energy_per_atom_eV": round(float(energy) / len(slab), 6),
            "max_force_eV_A": round(float(abs(forces).max()), 6),
            "n_atoms": len(slab),
        }
    except Exception as e:
        return {"ok": False, "code": "MACE", "error": str(e)}


def test_chgnet(payload: dict) -> dict:
    """Test CHGNet ML potential."""
    try:
        from ase.build import fcc111
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        candidate = payload.get("candidate", {})
        metals = candidate.get("metals", [])
        cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 60)
        zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 25)

        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
        n_zn = int(4 * zn / (cu + zn + 0.01))
        symbols = list(slab.get_chemical_symbols())
        surface_idx = [i for i, z in enumerate(slab.positions[:, 2])
                       if z > slab.positions[:, 2].max() - 2.0]
        for i in range(min(n_zn, len(surface_idx))):
            symbols[surface_idx[i]] = "Zn"
        slab.set_chemical_symbols(symbols)

        model = CHGNet.load()
        calc = CHGNetCalculator(model)
        slab.calc = calc
        energy = slab.get_potential_energy()
        forces = slab.get_forces()

        return {
            "ok": True,
            "code": "CHGNet",
            "total_energy_eV": round(float(energy), 6),
            "energy_per_atom_eV": round(float(energy) / len(slab), 6),
            "max_force_eV_A": round(float(abs(forces).max()), 6),
            "n_atoms": len(slab),
        }
    except Exception as e:
        return {"ok": False, "code": "CHGNet", "error": str(e)}


def test_m3gnet(payload: dict) -> dict:
    """Test M3GNet ML potential."""
    try:
        from ase.build import fcc111
        from m3gnet.models import M3GNet, Potential, M3GNetCalculator

        candidate = payload.get("candidate", {})
        metals = candidate.get("metals", [])
        cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 60)
        zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 25)

        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
        n_zn = int(4 * zn / (cu + zn + 0.01))
        symbols = list(slab.get_chemical_symbols())
        surface_idx = [i for i, z in enumerate(slab.positions[:, 2])
                       if z > slab.positions[:, 2].max() - 2.0]
        for i in range(min(n_zn, len(surface_idx))):
            symbols[surface_idx[i]] = "Zn"
        slab.set_chemical_symbols(symbols)

        potential = Potential(M3GNet.load())
        calc = M3GNetCalculator(potential=potential)
        slab.calc = calc
        energy = slab.get_potential_energy()
        forces = slab.get_forces()

        return {
            "ok": True,
            "code": "M3GNet",
            "total_energy_eV": round(float(energy), 6),
            "energy_per_atom_eV": round(float(energy) / len(slab), 6),
            "max_force_eV_A": round(float(abs(forces).max()), 6),
            "n_atoms": len(slab),
        }
    except Exception as e:
        return {"ok": False, "code": "M3GNet", "error": str(e)}


def test_ml_auto(payload: dict) -> dict:
    """Test best available ML potential."""
    # Try MACE first
    result = test_mace(payload)
    if result.get("ok"):
        return result

    # Try CHGNet
    result = test_chgnet(payload)
    if result.get("ok"):
        return result

    # Try M3GNet
    return test_m3gnet(payload)


def test_openmm(payload: dict) -> dict:
    """Test OpenMM structure relaxation."""
    try:
        import openmm
        from openmm import unit
        from openmm.app import Simulation, ForceField, PDBFile, Modeller

        return {
            "ok": True,
            "code": "OpenMM",
            "version": openmm.__version__,
            "note": "OpenMM available (full test requires structure files)",
        }
    except ImportError as e:
        return {"ok": False, "code": "OpenMM", "error": str(e)}


def test_gromacs(payload: dict) -> dict:
    """Test GROMACS availability."""
    try:
        import subprocess
        result = subprocess.run(
            ["gmx", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.split("\n")[0] if result.stdout else "unknown"
            return {
                "ok": True,
                "code": "GROMACS",
                "version": version,
                "note": "GROMACS available (full test requires structure files)",
            }
        return {"ok": False, "code": "GROMACS", "error": result.stderr[:500]}
    except Exception as e:
        return {"ok": False, "code": "GROMACS", "error": str(e)}


def test_gpaw(payload: dict) -> dict:
    """Test GPAW DFT."""
    try:
        import gpaw
        from ase.build import molecule

        # Quick test with H2 molecule
        atoms = molecule("H2")
        atoms.center(vacuum=3.0)

        from gpaw import GPAW
        calc = GPAW(mode="fd", xc="LDA", txt=None, h=0.3)
        atoms.calc = calc

        energy = atoms.get_potential_energy()

        return {
            "ok": True,
            "code": "GPAW",
            "version": gpaw.__version__,
            "test_energy_eV": round(float(energy), 4),
            "test_molecule": "H2",
        }
    except Exception as e:
        return {"ok": False, "code": "GPAW", "error": str(e)}


def test_qe(payload: dict) -> dict:
    """Test Quantum Espresso availability."""
    try:
        import subprocess
        result = subprocess.run(
            ["pw.x", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # QE returns version info even with non-zero exit code
        version = result.stdout.strip() or result.stderr.strip()
        return {
            "ok": True,
            "code": "Quantum ESPRESSO",
            "version": version[:100],
            "note": "QE available (full test requires pseudopotentials)",
        }
    except Exception as e:
        return {"ok": False, "code": "Quantum ESPRESSO", "error": str(e)}


def test_catmap(payload: dict) -> dict:
    """Test CatMAP microkinetics."""
    try:
        import catmap
        return {
            "ok": True,
            "code": "CatMAP",
            "version": catmap.__version__,
            "note": "CatMAP available",
        }
    except ImportError as e:
        return {"ok": False, "code": "CatMAP", "error": str(e)}


def test_cantera(payload: dict) -> dict:
    """Test Cantera chemical kinetics."""
    try:
        import cantera as ct

        # Quick test with GRI-Mech
        gas = ct.Solution("gri30.yaml")
        gas.TPX = 300, 101325, "CH4:1, O2:2, N2:7.52"

        return {
            "ok": True,
            "code": "Cantera",
            "version": ct.__version__,
            "n_species": gas.n_species,
            "n_reactions": gas.n_reactions,
        }
    except Exception as e:
        return {"ok": False, "code": "Cantera", "error": str(e)}


# =============================================================================
# TEST DISPATCH
# =============================================================================

TEST_FUNCTIONS = {
    "mace": test_mace,
    "chgnet": test_chgnet,
    "m3gnet": test_m3gnet,
    "ml": test_ml_auto,
    "openmm": test_openmm,
    "gromacs": test_gromacs,
    "gpaw": test_gpaw,
    "qe": test_qe,
    "catmap": test_catmap,
    "cantera": test_cantera,
}


def main():
    parser = argparse.ArgumentParser(
        description="Test individual simulation codes via Globus Compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--endpoint", required=True, help="Globus Compute endpoint ID")
    parser.add_argument("--code", required=True, choices=list(TEST_FUNCTIONS.keys()),
                        help="Code to test")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--local", action="store_true",
                        help="Run locally instead of via Globus Compute")

    args = parser.parse_args()

    test_func = TEST_FUNCTIONS[args.code]
    payload = {"candidate": TEST_CANDIDATE}

    print(f"Testing: {args.code}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Candidate: {json.dumps(TEST_CANDIDATE, indent=2)}")
    print()

    if args.local:
        print("Running locally...")
        t0 = time.time()
        result = test_func(payload)
        elapsed = time.time() - t0
    else:
        print("Running via Globus Compute...")
        from globus_compute_sdk import Client, Executor

        client = Client()

        # Register the function
        print("Registering function...")
        func_id = client.register_function(test_func)
        print(f"Function ID: {func_id}")

        # Execute
        print("Executing...")
        t0 = time.time()

        with Executor(endpoint_id=args.endpoint) as ex:
            future = ex.submit_to_registered_function(func_id, args=(payload,))
            result = future.result(timeout=args.timeout)

        elapsed = time.time() - t0

    print(f"\nResult ({elapsed:.2f}s):")
    print(json.dumps(result, indent=2))

    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
