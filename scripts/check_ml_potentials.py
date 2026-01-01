#!/usr/bin/env python3
"""Diagnostic script to check ML potential availability.

Run on Spark to see which potentials are working and what needs fixing.

Usage:
    python scripts/check_ml_potentials.py
"""

import sys
import traceback


def header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def check_basic_deps() -> dict:
    """Check basic dependencies."""
    header("Basic Dependencies")
    results = {}

    # Python version
    print(f"Python: {sys.version}")

    # PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        results["torch"] = torch.__version__
    except ImportError as e:
        print(f"PyTorch: NOT INSTALLED - {e}")
        results["torch"] = None

    # torchvision
    try:
        import torchvision
        print(f"torchvision: {torchvision.__version__}")
        results["torchvision"] = torchvision.__version__
    except ImportError as e:
        print(f"torchvision: NOT INSTALLED - {e}")
        results["torchvision"] = None

    # ASE
    try:
        import ase
        print(f"ASE: {ase.__version__}")
        results["ase"] = ase.__version__
    except ImportError as e:
        print(f"ASE: NOT INSTALLED - {e}")
        results["ase"] = None

    # numpy
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
        results["numpy"] = np.__version__
    except ImportError as e:
        print(f"NumPy: NOT INSTALLED - {e}")
        results["numpy"] = None

    return results


def check_chgnet() -> bool:
    """Check CHGNet availability."""
    header("CHGNet")

    # Step 1: Import
    try:
        import chgnet
        print(f"1. Import chgnet: OK (version {chgnet.__version__})")
    except ImportError as e:
        print(f"1. Import chgnet: FAILED")
        print(f"   Error: {e}")
        print(f"   Fix: pip install chgnet")
        return False

    # Step 2: Load model
    try:
        from chgnet.model import CHGNet
        print("2. Import CHGNet class: OK")
    except ImportError as e:
        print(f"2. Import CHGNet class: FAILED")
        print(f"   Error: {e}")
        return False

    try:
        model = CHGNet.load()
        print("3. Load pretrained model: OK")
    except Exception as e:
        print(f"3. Load pretrained model: FAILED")
        print(f"   Error: {e}")
        return False

    # Step 3: Calculator
    try:
        from chgnet.model.dynamics import CHGNetCalculator
        print("4. Import CHGNetCalculator: OK")
    except ImportError as e:
        print(f"4. Import CHGNetCalculator: FAILED")
        print(f"   Error: {e}")
        print(f"   This may be an ASE compatibility issue")
        return False

    # Step 4: Test calculation
    try:
        from ase.build import bulk
        atoms = bulk("Cu", "fcc", a=3.6)
        calc = CHGNetCalculator(model=model)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"5. Test calculation: OK (E = {energy:.4f} eV)")
        return True
    except Exception as e:
        print(f"5. Test calculation: FAILED")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False


def check_mace() -> bool:
    """Check MACE availability."""
    header("MACE")

    # Step 1: Import
    try:
        import mace
        print(f"1. Import mace: OK")
    except ImportError as e:
        print(f"1. Import mace: FAILED")
        print(f"   Error: {e}")
        print(f"   Fix: pip install mace-torch")
        return False

    # Step 2: Import calculator
    try:
        from mace.calculators import mace_mp
        print("2. Import mace_mp: OK")
    except ImportError as e:
        print(f"2. Import mace_mp: FAILED")
        print(f"   Error: {e}")
        if "torchvision" in str(e).lower():
            print(f"   Fix: pip install 'torchvision>=0.15'")
        return False

    # Step 3: Load model
    try:
        calc = mace_mp(model="small", device="cpu", default_dtype="float32")
        print("3. Load MACE-MP model: OK")
    except Exception as e:
        print(f"3. Load MACE-MP model: FAILED")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False

    # Step 4: Test calculation
    try:
        from ase.build import bulk
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"4. Test calculation: OK (E = {energy:.4f} eV)")
        return True
    except Exception as e:
        print(f"4. Test calculation: FAILED")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False


def check_m3gnet() -> bool:
    """Check M3GNet/MatGL availability."""
    header("M3GNet (MatGL)")

    # Step 1: Try matgl first (newer)
    try:
        import matgl
        print(f"1. Import matgl: OK (version {matgl.__version__})")
        use_matgl = True
    except ImportError as e:
        print(f"1. Import matgl: FAILED")
        print(f"   Error: {e}")
        if "torchdata" in str(e).lower():
            print(f"   Fix: pip install torchdata")
        use_matgl = False

    if not use_matgl:
        # Try legacy m3gnet
        try:
            import m3gnet
            print(f"1b. Import m3gnet (legacy): OK")
        except ImportError as e:
            print(f"1b. Import m3gnet (legacy): FAILED")
            print(f"   Error: {e}")
            print(f"   Fix: pip install matgl torchdata  (recommended)")
            print(f"   Or:  pip install m3gnet  (legacy)")
            return False

    # Step 2: Load model
    if use_matgl:
        try:
            potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            print("2. Load M3GNet model: OK")
        except Exception as e:
            print(f"2. Load M3GNet model: FAILED")
            print(f"   Error: {e}")
            return False

        # Step 3: Calculator
        try:
            from matgl.ext.ase import PESCalculator
            calc = PESCalculator(potential=potential)
            print("3. Create PESCalculator: OK")
        except ImportError as e:
            print(f"3. Create PESCalculator: FAILED")
            print(f"   Error: {e}")
            return False
    else:
        # Legacy m3gnet
        try:
            from m3gnet.models import M3GNet, Potential, M3GNetCalculator
            potential = Potential(M3GNet.load())
            calc = M3GNetCalculator(potential=potential)
            print("2-3. Load legacy M3GNet: OK")
        except Exception as e:
            print(f"2-3. Load legacy M3GNet: FAILED")
            print(f"   Error: {e}")
            return False

    # Step 4: Test calculation
    try:
        from ase.build import bulk
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"4. Test calculation: OK (E = {energy:.4f} eV)")
        return True
    except Exception as e:
        print(f"4. Test calculation: FAILED")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False


def check_cantera() -> bool:
    """Check Cantera availability."""
    header("Cantera")

    try:
        import cantera as ct
        print(f"1. Import cantera: OK (version {ct.__version__})")
    except ImportError as e:
        print(f"1. Import cantera: FAILED")
        print(f"   Error: {e}")
        print(f"   Fix: conda install -c cantera cantera")
        return False

    # Test basic functionality
    try:
        gas = ct.Solution("gri30.yaml")
        gas.TPX = 300, ct.one_atm, "CH4:1, O2:2, N2:7.52"
        print(f"2. Load GRI-Mech 3.0: OK ({gas.n_species} species)")
        return True
    except Exception as e:
        print(f"2. Load GRI-Mech 3.0: FAILED")
        print(f"   Error: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("  ML Potential Diagnostic Tool")
    print("="*60)

    deps = check_basic_deps()

    results = {
        "chgnet": check_chgnet(),
        "mace": check_mace(),
        "m3gnet": check_m3gnet(),
        "cantera": check_cantera(),
    }

    # Summary
    header("SUMMARY")

    working = [k for k, v in results.items() if v]
    broken = [k for k, v in results.items() if not v]

    if working:
        print(f"Working: {', '.join(working)}")
    else:
        print("Working: (none)")

    if broken:
        print(f"Broken:  {', '.join(broken)}")

    print("\nRecommended fixes:")
    if not results["chgnet"]:
        print("  CHGNet:  pip install --upgrade chgnet")
    if not results["mace"]:
        tv = deps.get("torchvision")
        if tv and tv.startswith("0."):
            print(f"  MACE:    pip install 'torchvision>=0.15' (current: {tv})")
        else:
            print("  MACE:    pip install mace-torch")
    if not results["m3gnet"]:
        print("  M3GNet:  pip install matgl torchdata")
    if not results["cantera"]:
        print("  Cantera: conda install -c cantera cantera")

    print("\n" + "="*60)
    n_working = sum(results.values())
    print(f"  {n_working}/4 ML potentials ready")
    print("="*60 + "\n")

    return 0 if n_working > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
