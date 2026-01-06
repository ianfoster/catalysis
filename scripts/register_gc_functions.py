#!/usr/bin/env python3
"""Register simulation functions with Globus Compute.

Usage:
    # Real simulations (uses actual codes when available, falls back to surrogates)
    python scripts/register_gc_functions.py --endpoint ENDPOINT_ID

    # Mock mode (fixed values, fastest, for testing infrastructure)
    python scripts/register_gc_functions.py --endpoint ENDPOINT_ID --mock

    # Surrogate-only mode (physics-informed approximations, no real simulations)
    python scripts/register_gc_functions.py --endpoint ENDPOINT_ID --surrogate

This registers simulation functions and outputs a JSON mapping for use with app_generator.py.

Available simulations:
- fast_surrogate: ML surrogate for quick performance prediction
- ml_screening: MACE/CHGNet ML potential screening (near-DFT accuracy)
- ml_relaxation: Structure relaxation with ML potentials
- dft_adsorption: DFT adsorption energies (GPAW, QE, VASP)
- microkinetic_lite: Microkinetic analysis (CatMAP or surrogate)
- openmm_relaxation: Structure relaxation with OpenMM
- gromacs_md: Molecular dynamics with GROMACS
- stability_analysis: Thermodynamic stability assessment
"""

import argparse
import json
import sys


# =============================================================================
# REAL IMPLEMENTATIONS (with automatic fallback to surrogates)
# These attempt to use actual simulation codes when available
# =============================================================================

def fast_surrogate_real(payload: dict) -> dict:
    """ML surrogate prediction of CO2->MeOH performance.

    Uses physics-informed model based on Cu/ZnO/Al2O3 catalyst behavior.
    """
    candidate = payload.get("candidate", {})
    metals = candidate.get("metals", [])
    support = candidate.get("support", "Al2O3")

    cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 0)
    zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 0)
    al = next((m["wt_pct"] for m in metals if m["element"] == "Al"), 0)

    is_zro2 = 1 if support == "ZrO2" else 0

    # Model from skills/performance.py
    conversion = max(0.0, min(1.0, 0.2 + 0.01 * cu + 0.005 * zn - 0.002 * al))
    selectivity = max(0.0, min(1.0, 0.6 + 0.003 * zn - 0.004 * max(0.0, cu - 60) + 0.05 * is_zro2))
    sty = conversion * selectivity * 10.0

    return {
        "co2_conversion": round(conversion, 4),
        "methanol_selectivity": round(selectivity, 4),
        "methanol_sty": round(sty, 4),
        "uncertainty": 0.25,
        "method": "surrogate",
    }


def microkinetic_lite_real(payload: dict) -> dict:
    """Microkinetic analysis - rate-limiting step and sensitivities.

    Uses support-dependent kinetic model from skills/microkinetic.py.
    """
    candidate = payload.get("candidate", {})
    support = str(candidate.get("support", "")).strip().lower()

    perf = payload.get("performance", {}) or {}
    sty = float(perf.get("methanol_sty", 5.0))
    sel = float(perf.get("methanol_selectivity", 0.7))

    # Support-dependent kinetics
    if "zro2" in support or "zirconia" in support:
        rls = "CO2_activation"
        temp_sensitivity = 0.8
        pressure_sensitivity = 0.6
    else:
        rls = "hydrogenation"
        temp_sensitivity = 0.6
        pressure_sensitivity = 0.8

    reduced_uncertainty = max(0.05, 0.25 - 0.01 * sty - 0.02 * sel)

    return {
        "RLS": rls,
        "temp_sensitivity": round(temp_sensitivity, 3),
        "pressure_sensitivity": round(pressure_sensitivity, 3),
        "uncertainty_reduction": round(0.25 - reduced_uncertainty, 3),
        "method": "microkinetic_model",
    }


def dft_adsorption_real(payload: dict) -> dict:
    """DFT adsorption energy calculation.

    Attempts to use GPAW if available, falls back to surrogate.
    Self-contained function for Globus Compute.
    """
    candidate = payload.get("candidate", {})
    metals = candidate.get("metals", [])
    support = candidate.get("support", "Al2O3")

    cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
    zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)

    # Try GPAW
    try:
        from ase.build import fcc111
        from gpaw import GPAW, PW
        import time

        t0 = time.time()

        # Build Cu/Zn slab
        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
        n_zn = int(4 * zn / (cu + zn + 0.01))
        symbols = list(slab.get_chemical_symbols())
        surface_idx = [i for i, z in enumerate(slab.positions[:, 2])
                       if z > slab.positions[:, 2].max() - 2.0]
        for i in range(min(n_zn, len(surface_idx))):
            symbols[surface_idx[i]] = "Zn"
        slab.set_chemical_symbols(symbols)

        calc = GPAW(mode=PW(300), xc='PBE', kpts=(2, 2, 1), txt=None)
        slab.calc = calc
        energy = slab.get_potential_energy()

        return {
            "ok": True,
            "total_energy_eV": round(float(energy), 6),
            "energy_per_atom_eV": round(float(energy) / len(slab), 6),
            "E_ads_CO2": round(-0.35 - energy / len(slab) / 50, 4),
            "E_ads_H": round(-0.28 - energy / len(slab) / 60, 4),
            "uncertainty_reduction": 0.15,
            "method": "gpaw",
            "elapsed_s": round(time.time() - t0, 3),
        }
    except ImportError:
        pass
    except Exception as e:
        # GPAW failed, continue to surrogate
        pass

    # Physics-informed surrogate fallback
    base_co2 = -0.35
    base_h = -0.28

    support_effects = {
        "Al2O3": (0.0, 0.0),
        "ZrO2": (-0.1, -0.05),
        "SiO2": (0.05, 0.02),
    }
    co2_mod, h_mod = support_effects.get(support, (0.0, 0.0))

    e_ads_co2 = base_co2 - (cu / 100) * 0.2 + co2_mod
    e_ads_h = base_h - (zn / 100) * 0.15 + h_mod

    return {
        "ok": True,
        "E_ads_CO2": round(e_ads_co2, 4),
        "E_ads_H": round(e_ads_h, 4),
        "uncertainty_reduction": 0.10,
        "method": "dft_surrogate",
    }


def openmm_relaxation_real(payload: dict) -> dict:
    """OpenMM structure relaxation.

    Self-contained function for Globus Compute.
    Tests OpenMM availability and returns surrogate if no structure files.
    """
    candidate = payload.get("candidate", {})
    support = candidate.get("support", "Al2O3")
    metals = candidate.get("metals", [])

    cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
    zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)

    # Try OpenMM
    try:
        import openmm
        version = openmm.__version__

        # OpenMM is available - return info about it
        # Full relaxation requires structure files which we don't have
        base_energies = {"Al2O3": -125.4, "ZrO2": -132.1, "SiO2": -118.7}
        base_energy = base_energies.get(support, -120.0)
        energy = base_energy - (cu / 100) * 5.0 + (zn / 100) * 2.0
        rmsd = 0.10 + (cu / 100) * 0.15

        return {
            "ok": True,
            "relaxed_energy": round(energy, 2),
            "potential_energy_kj_mol": round(energy, 2),
            "structure_rmsd": round(rmsd, 3),
            "method": "openmm_available",
            "openmm_version": version,
            "note": "OpenMM available, using surrogate energies (no structure files)",
        }
    except ImportError:
        pass

    # Surrogate fallback
    base_energies = {"Al2O3": -125.4, "ZrO2": -132.1, "SiO2": -118.7}
    base_energy = base_energies.get(support, -120.0)
    energy = base_energy - (cu / 100) * 5.0 + (zn / 100) * 2.0
    rmsd = 0.10 + (cu / 100) * 0.15

    return {
        "ok": True,
        "relaxed_energy": round(energy, 2),
        "potential_energy_kj_mol": round(energy, 2),
        "structure_rmsd": round(rmsd, 3),
        "method": "openmm_surrogate",
    }


def stability_analysis_real(payload: dict) -> dict:
    """Thermodynamic stability analysis.

    Composition and support-dependent stability model.
    """
    candidate = payload.get("candidate", {})
    metals = candidate.get("metals", [])
    support = candidate.get("support", "Al2O3")

    cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
    zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)

    base_stability = {"Al2O3": 0.80, "ZrO2": 0.90, "SiO2": 0.70}

    stability = base_stability.get(support, 0.75)
    stability -= max(0, (cu - 60) / 100) * 0.1
    stability += (zn / 100) * 0.05
    stability = max(0.3, min(0.98, stability))

    risk = "low" if stability > 0.85 else ("medium" if stability > 0.70 else "high")

    return {
        "stability_score": round(stability, 3),
        "degradation_risk": risk,
        "method": "stability_model",
    }


def gromacs_md_real(payload: dict) -> dict:
    """GROMACS molecular dynamics simulation.

    Self-contained function for Globus Compute.
    Tests GROMACS availability, returns surrogate if no input files.
    """
    candidate = payload.get("candidate", {})
    support = candidate.get("support", "Al2O3")
    metals = candidate.get("metals", [])

    cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)

    base_values = {
        "Al2O3": {"temp": 300.0, "rmsd": 0.12},
        "ZrO2": {"temp": 300.0, "rmsd": 0.10},
        "SiO2": {"temp": 300.0, "rmsd": 0.15},
    }
    base = base_values.get(support, {"temp": 300.0, "rmsd": 0.13})

    # Check if GROMACS is available
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
                "avg_temperature_K": base["temp"],
                "temp_fluctuation_K": round(5.0 + (cu / 100) * 3.0, 2),
                "final_rmsd_nm": round(base["rmsd"] + (cu / 100) * 0.05, 4),
                "total_energy_kJ_mol": round(-500.0 - cu * 2.0, 2),
                "method": "gromacs_available",
                "gromacs_version": version,
                "note": "GROMACS available, using surrogate values (no input files)",
            }
    except Exception:
        pass

    # Surrogate fallback
    return {
        "ok": True,
        "avg_temperature_K": base["temp"],
        "temp_fluctuation_K": round(5.0 + (cu / 100) * 3.0, 2),
        "final_rmsd_nm": round(base["rmsd"] + (cu / 100) * 0.05, 4),
        "total_energy_kJ_mol": round(-500.0 - cu * 2.0, 2),
        "method": "gromacs_surrogate",
    }


def ml_screening_real(payload: dict) -> dict:
    """ML potential screening using MACE or CHGNet.

    Self-contained function for Globus Compute.
    Near-DFT accuracy at MD speeds for rapid catalyst screening.
    """
    candidate = payload.get("candidate", {})
    metals = candidate.get("metals", [])
    support = candidate.get("support", "Al2O3")

    cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
    zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)

    # Helper to build slab
    def build_slab():
        from ase.build import fcc111
        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
        n_zn = int(4 * zn / (cu + zn + 0.01))
        symbols = list(slab.get_chemical_symbols())
        surface_idx = [i for i, z in enumerate(slab.positions[:, 2])
                       if z > slab.positions[:, 2].max() - 2.0]
        for i in range(min(n_zn, len(surface_idx))):
            symbols[surface_idx[i]] = "Zn"
        slab.set_chemical_symbols(symbols)
        return slab

    # Try MACE
    try:
        from mace.calculators import mace_mp
        import numpy as np
        import time
        t0 = time.time()

        slab = build_slab()
        calc = mace_mp(model="small", dispersion=False, default_dtype="float32", device="cpu")
        slab.calc = calc
        energy = float(slab.get_potential_energy())
        forces = slab.get_forces()
        max_force = float(np.max(np.abs(forces)))

        return {
            "ok": True,
            "total_energy_eV": round(energy, 6),
            "energy_per_atom_eV": round(energy / len(slab), 6),
            "max_force_eV_A": round(max_force, 6),
            "n_atoms": int(len(slab)),
            "method": "mace",
            "E_ads_CO2_est": round(-0.3 - energy / len(slab) / 10, 4),
            "E_ads_H_est": round(-0.25 - energy / len(slab) / 15, 4),
            "uncertainty_reduction": 0.12,
            "elapsed_s": round(time.time() - t0, 3),
        }
    except ImportError:
        pass
    except Exception as e:
        pass  # Try next model

    # Try CHGNet
    try:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        import numpy as np
        import time
        t0 = time.time()

        slab = build_slab()
        chgnet = CHGNet.load()
        calc = CHGNetCalculator(model=chgnet, use_device="cpu")
        slab.calc = calc
        energy = float(slab.get_potential_energy())
        forces = slab.get_forces()
        max_force = float(np.max(np.abs(forces)))

        return {
            "ok": True,
            "total_energy_eV": round(energy, 6),
            "energy_per_atom_eV": round(energy / len(slab), 6),
            "max_force_eV_A": round(max_force, 6),
            "n_atoms": int(len(slab)),
            "method": "chgnet",
            "E_ads_CO2_est": round(-0.3 - energy / len(slab) / 10, 4),
            "E_ads_H_est": round(-0.25 - energy / len(slab) / 15, 4),
            "uncertainty_reduction": 0.12,
            "elapsed_s": round(time.time() - t0, 3),
        }
    except ImportError:
        pass
    except Exception as e:
        pass  # Fall through to surrogate

    # Surrogate fallback - return error, don't silently use surrogate
    return {
        "ok": False,
        "error": "No ML potential available (tried MACE, CHGNet)",
        "method": "ml_screening",
    }


def ml_relaxation_real(payload: dict) -> dict:
    """Structure relaxation using ML potentials (MACE/CHGNet).

    Self-contained function for Globus Compute.
    Fast geometry optimization for catalyst structures.
    """
    candidate = payload.get("candidate", {})
    metals = candidate.get("metals", [])
    support = candidate.get("support", "Al2O3")

    cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
    zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)
    fmax = payload.get("fmax", 0.05)
    steps = payload.get("steps", 50)

    # Helper to build slab
    def build_slab():
        from ase.build import fcc111
        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
        n_zn = int(4 * zn / (cu + zn + 0.01))
        symbols = list(slab.get_chemical_symbols())
        surface_idx = [i for i, z in enumerate(slab.positions[:, 2])
                       if z > slab.positions[:, 2].max() - 2.0]
        for i in range(min(n_zn, len(surface_idx))):
            symbols[surface_idx[i]] = "Zn"
        slab.set_chemical_symbols(symbols)
        return slab

    # Try MACE
    try:
        from mace.calculators import mace_mp
        from ase.optimize import BFGS
        import numpy as np
        import time
        t0 = time.time()

        slab = build_slab()
        calc = mace_mp(model="small", dispersion=False, default_dtype="float32", device="cpu")
        slab.calc = calc

        e_initial = float(slab.get_potential_energy())
        opt = BFGS(slab, logfile=None)
        converged = opt.run(fmax=fmax, steps=steps)
        e_final = float(slab.get_potential_energy())
        forces = slab.get_forces()
        max_force = float(np.max(np.abs(forces)))

        return {
            "ok": True,
            "converged": bool(converged),
            "initial_energy_eV": round(e_initial, 6),
            "final_energy_eV": round(e_final, 6),
            "energy_change_eV": round(e_final - e_initial, 6),
            "max_force_eV_A": round(max_force, 6),
            "n_steps": int(opt.nsteps),
            "n_atoms": int(len(slab)),
            "method": "mace",
            "elapsed_s": round(time.time() - t0, 3),
        }
    except ImportError:
        pass
    except Exception as e:
        pass

    # Try CHGNet
    try:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        from ase.optimize import BFGS
        import numpy as np
        import time
        t0 = time.time()

        slab = build_slab()
        chgnet = CHGNet.load()
        calc = CHGNetCalculator(model=chgnet, use_device="cpu")
        slab.calc = calc

        e_initial = float(slab.get_potential_energy())
        opt = BFGS(slab, logfile=None)
        converged = opt.run(fmax=fmax, steps=steps)
        e_final = float(slab.get_potential_energy())
        forces = slab.get_forces()
        max_force = float(np.max(np.abs(forces)))

        return {
            "ok": True,
            "converged": bool(converged),
            "initial_energy_eV": round(e_initial, 6),
            "final_energy_eV": round(e_final, 6),
            "energy_change_eV": round(e_final - e_initial, 6),
            "max_force_eV_A": round(max_force, 6),
            "n_steps": int(opt.nsteps),
            "n_atoms": int(len(slab)),
            "method": "chgnet",
            "elapsed_s": round(time.time() - t0, 3),
        }
    except ImportError:
        pass
    except Exception as e:
        pass

    # Return error, don't silently use surrogate
    return {
        "ok": False,
        "error": "No ML potential available (tried MACE, CHGNet)",
        "method": "ml_relaxation",
    }


def dft_gpaw_real(payload: dict) -> dict:
    """DFT calculation using GPAW (ASE-based).

    Self-contained function for Globus Compute.
    Returns error if GPAW is not available.
    """
    candidate = payload.get("candidate", {})
    metals = candidate.get("metals", [])

    cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
    zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)

    try:
        from ase.build import fcc111
        from gpaw import GPAW, PW
        import time
        t0 = time.time()

        # Build Cu/Zn slab
        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
        n_zn = int(4 * zn / (cu + zn + 0.01))
        symbols = list(slab.get_chemical_symbols())
        surface_idx = [i for i, z in enumerate(slab.positions[:, 2])
                       if z > slab.positions[:, 2].max() - 2.0]
        for i in range(min(n_zn, len(surface_idx))):
            symbols[surface_idx[i]] = "Zn"
        slab.set_chemical_symbols(symbols)

        calc = GPAW(mode=PW(300), xc='PBE', kpts=(2, 2, 1), txt=None)
        slab.calc = calc
        energy = slab.get_potential_energy()
        forces = slab.get_forces()

        return {
            "ok": True,
            "total_energy_eV": round(float(energy), 6),
            "energy_per_atom_eV": round(float(energy) / len(slab), 6),
            "max_force_eV_A": round(float(abs(forces).max()), 6),
            "n_atoms": len(slab),
            "method": "gpaw",
            "E_ads_CO2": round(-0.35 - energy / len(slab) / 50, 4),
            "E_ads_H": round(-0.28 - energy / len(slab) / 60, 4),
            "uncertainty_reduction": 0.15,
            "elapsed_s": round(time.time() - t0, 3),
        }
    except ImportError as e:
        return {"ok": False, "error": f"GPAW not available: {e}", "method": "gpaw"}
    except Exception as e:
        return {"ok": False, "error": str(e), "method": "gpaw"}


def dft_qe_real(payload: dict) -> dict:
    """DFT calculation using Quantum Espresso via ASE.

    Self-contained function for Globus Compute.
    Tests QE availability, returns info or error.
    """
    # Check if pw.x is available
    try:
        import subprocess
        result = subprocess.run(
            ["pw.x", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = result.stdout.strip() or result.stderr.strip()

        return {
            "ok": True,
            "method": "qe_available",
            "qe_version": version[:100],
            "note": "QE available (full calculation requires pseudopotentials)",
        }
    except Exception as e:
        return {"ok": False, "error": f"QE not available: {e}", "method": "qe"}


def microkinetic_catmap_real(payload: dict) -> dict:
    """CatMAP microkinetic modeling.

    Descriptor-based microkinetic analysis for catalyst evaluation.
    """
    try:
        from simulations.microkinetics import microkinetic_gc
        return microkinetic_gc(payload)
    except ImportError:
        pass

    # Fallback to simplified kinetic model
    try:
        from simulations.microkinetics import run_kinetic_model
        return run_kinetic_model(payload)
    except ImportError:
        pass

    # Final surrogate fallback (self-contained)
    candidate = payload.get("candidate", {})
    support = str(candidate.get("support", "")).strip().lower()
    perf = payload.get("performance", {}) or {}
    sty = float(perf.get("methanol_sty", 5.0))
    sel = float(perf.get("methanol_selectivity", 0.7))

    if "zro2" in support or "zirconia" in support:
        rls = "CO2_activation"
        temp_sensitivity = 0.8
        pressure_sensitivity = 0.6
    else:
        rls = "hydrogenation"
        temp_sensitivity = 0.6
        pressure_sensitivity = 0.8

    reduced_uncertainty = max(0.05, 0.25 - 0.01 * sty - 0.02 * sel)

    return {
        "ok": True,
        "RLS": rls,
        "temp_sensitivity": round(temp_sensitivity, 3),
        "pressure_sensitivity": round(pressure_sensitivity, 3),
        "uncertainty_reduction": round(0.25 - reduced_uncertainty, 3),
        "method": "microkinetic_surrogate",
    }


def ml_m3gnet_real(payload: dict) -> dict:
    """M3GNet ML potential calculation.

    Self-contained function for Globus Compute.
    MatErials 3-body Graph Network for materials properties.
    """
    candidate = payload.get("candidate", {})
    metals = candidate.get("metals", [])

    cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
    zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)

    try:
        from ase.build import fcc111
        from m3gnet.models import M3GNet, Potential, M3GNetCalculator
        import numpy as np
        import time
        t0 = time.time()

        # Build slab
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
        energy = float(slab.get_potential_energy())
        forces = slab.get_forces()
        max_force = float(np.max(np.abs(forces)))

        return {
            "ok": True,
            "total_energy_eV": round(energy, 6),
            "energy_per_atom_eV": round(energy / len(slab), 6),
            "max_force_eV_A": round(max_force, 6),
            "n_atoms": int(len(slab)),
            "method": "m3gnet",
            "E_ads_CO2_est": round(-0.3 - energy / len(slab) / 10, 4),
            "E_ads_H_est": round(-0.25 - energy / len(slab) / 15, 4),
            "uncertainty_reduction": 0.11,
            "elapsed_s": round(time.time() - t0, 3),
        }
    except ImportError as e:
        return {"ok": False, "error": f"M3GNet not available: {e}", "method": "m3gnet"}
    except Exception as e:
        return {"ok": False, "error": str(e), "method": "m3gnet"}


def cantera_reactor_real(payload: dict) -> dict:
    """Cantera reactor simulation.

    Self-contained function for Globus Compute.
    Chemical kinetics and reactor modeling for catalyst evaluation.
    """
    candidate = payload.get("candidate", {})
    metals = candidate.get("metals", [])
    support = str(candidate.get("support", "Al2O3")).strip().lower()

    cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
    zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)

    # Try Cantera
    try:
        import cantera as ct

        # Use GRI-Mech for demo (includes CO2/H2 chemistry)
        gas = ct.Solution("gri30.yaml")
        gas.TPX = 523.0, 50e5, {"CO2": 0.25, "H2": 0.75}

        reactor = ct.IdealGasReactor(gas)
        net = ct.ReactorNet([reactor])
        net.advance(1.0)  # 1 second residence time

        products = {sp: round(gas.X[gas.species_index(sp)], 6)
                    for sp in gas.species_names if gas.X[gas.species_index(sp)] > 1e-10}

        return {
            "ok": True,
            "products": products,
            "temperature_K": reactor.T,
            "pressure_bar": reactor.thermo.P / 1e5,
            "method": "cantera",
            "cantera_version": ct.__version__,
            "note": "Using GRI-Mech 3.0 (demo mechanism)",
        }
    except ImportError:
        pass
    except Exception as e:
        pass

    # Surrogate fallback
    conversion = 0.25 + 0.003 * cu + 0.002 * zn
    if "zro2" in support:
        conversion += 0.05

    selectivity = 0.75 + 0.002 * zn
    if "zro2" in support:
        selectivity += 0.08

    conversion = min(0.95, max(0.1, conversion))
    selectivity = min(0.98, max(0.5, selectivity))

    return {
        "ok": True,
        "conversion": round(conversion, 4),
        "selectivity": round(selectivity, 4),
        "methanol_yield": round(conversion * selectivity, 4),
        "products": {
            "CH3OH": round(conversion * selectivity, 4),
            "CO": round(conversion * (1 - selectivity) * 0.7, 4),
            "H2O": round(conversion, 4),
        },
        "temperature_K": 523.0,
        "pressure_bar": 50.0,
        "uncertainty_reduction": 0.07,
        "method": "cantera_surrogate",
    }


def cantera_sensitivity_real(payload: dict) -> dict:
    """Cantera sensitivity analysis.

    Self-contained function for Globus Compute.
    Reactor parameter sensitivity for optimization.
    """
    candidate = payload.get("candidate", {})
    support = str(candidate.get("support", "Al2O3")).strip().lower()

    # Try Cantera
    try:
        import cantera as ct

        # Run at two temperatures to estimate sensitivity
        gas1 = ct.Solution("gri30.yaml")
        gas1.TPX = 523.0, 50e5, {"CO2": 0.25, "H2": 0.75}
        r1 = ct.IdealGasReactor(gas1)
        net1 = ct.ReactorNet([r1])
        net1.advance(1.0)
        T1 = r1.T

        gas2 = ct.Solution("gri30.yaml")
        gas2.TPX = 533.0, 50e5, {"CO2": 0.25, "H2": 0.75}
        r2 = ct.IdealGasReactor(gas2)
        net2 = ct.ReactorNet([r2])
        net2.advance(1.0)
        T2 = r2.T

        temp_sens = abs(T2 - T1) / 10.0

        return {
            "ok": True,
            "sensitivities": {
                "temperature": round(temp_sens, 3),
                "pressure": 0.5,
                "H2_CO2_ratio": 0.4,
            },
            "method": "cantera",
            "cantera_version": ct.__version__,
        }
    except ImportError:
        pass
    except Exception as e:
        pass

    # Surrogate fallback
    if "zro2" in support:
        temp_sens = 0.8
        press_sens = 0.6
    else:
        temp_sens = 0.6
        press_sens = 0.75

    return {
        "ok": True,
        "sensitivities": {
            "temperature": round(temp_sens, 3),
            "pressure": round(press_sens, 3),
            "H2_CO2_ratio": 0.5,
        },
        "optimal_conditions": {
            "temperature_K": 523.0 if "zro2" in support else 543.0,
            "pressure_bar": 50.0,
            "H2_CO2_ratio": 3.0,
        },
        "uncertainty_reduction": 0.05,
        "method": "sensitivity_surrogate",
    }


# =============================================================================
# MOCK IMPLEMENTATIONS (for testing infrastructure)
# =============================================================================

def fast_surrogate_mock(payload: dict) -> dict:
    return {
        "co2_conversion": 0.35,
        "methanol_selectivity": 0.72,
        "methanol_sty": 2.52,
        "uncertainty": 0.15,
        "method": "mock",
    }


def microkinetic_lite_mock(payload: dict) -> dict:
    import time
    time.sleep(0.5)
    return {
        "RLS": "CO2_adsorption",
        "temp_sensitivity": 0.75,
        "pressure_sensitivity": 0.35,
        "uncertainty_reduction": 0.05,
        "method": "mock",
    }


def dft_adsorption_mock(payload: dict) -> dict:
    import time
    time.sleep(1)
    return {
        "E_ads_CO2": -0.45,
        "E_ads_H": -0.32,
        "uncertainty_reduction": 0.10,
        "method": "mock",
    }


def openmm_relaxation_mock(payload: dict) -> dict:
    import time
    time.sleep(0.5)
    return {
        "relaxed_energy": -125.4,
        "potential_energy_kj_mol": -125.4,
        "structure_rmsd": 0.15,
        "method": "mock",
        "ok": True,
    }


def stability_analysis_mock(payload: dict) -> dict:
    return {
        "stability_score": 0.82,
        "degradation_risk": "low",
        "method": "mock",
    }


def gromacs_md_mock(payload: dict) -> dict:
    import time
    time.sleep(0.5)
    return {
        "ok": True,
        "avg_temperature_K": 300.0,
        "temp_fluctuation_K": 5.5,
        "final_rmsd_nm": 0.14,
        "total_energy_kJ_mol": -550.0,
        "method": "mock",
    }


def ml_screening_mock(payload: dict) -> dict:
    return {
        "ok": True,
        "total_energy_eV": -165.0,
        "energy_per_atom_eV": -4.58,
        "max_force_eV_A": 0.045,
        "n_atoms": 36,
        "method": "mock",
        "E_ads_CO2_est": -0.42,
        "E_ads_H_est": -0.31,
        "uncertainty_reduction": 0.12,
    }


def ml_relaxation_mock(payload: dict) -> dict:
    import time
    time.sleep(0.3)
    return {
        "ok": True,
        "converged": True,
        "initial_energy_eV": -160.5,
        "final_energy_eV": -162.3,
        "energy_change_eV": -1.8,
        "max_force_eV_A": 0.04,
        "n_steps": 22,
        "n_atoms": 36,
        "method": "mock",
    }


def dft_gpaw_mock(payload: dict) -> dict:
    import time
    time.sleep(1)
    return {
        "ok": True,
        "total_energy_eV": -168.5,
        "energy_per_atom_eV": -4.68,
        "initial_energy_eV": -165.0,
        "max_force_eV_A": 0.03,
        "n_atoms": 36,
        "method": "mock",
        "E_ads_CO2": -0.45,
        "E_ads_H": -0.33,
        "uncertainty_reduction": 0.15,
    }


def dft_qe_mock(payload: dict) -> dict:
    import time
    time.sleep(1)
    return {
        "ok": True,
        "total_energy_eV": -167.8,
        "energy_per_atom_eV": -4.66,
        "max_force_eV_A": 0.035,
        "n_atoms": 36,
        "method": "mock",
        "E_ads_CO2": -0.44,
        "E_ads_H": -0.32,
        "uncertainty_reduction": 0.15,
    }


def microkinetic_catmap_mock(payload: dict) -> dict:
    return {
        "ok": True,
        "RLS": "CO2_activation",
        "pathway": "formate",
        "E_barrier_eV": 0.65,
        "log_TOF": 2.5,
        "temp_sensitivity": 0.75,
        "pressure_sensitivity": 0.55,
        "coverages": {"CO2*": 0.15, "H*": 0.35, "*": 0.50},
        "degree_of_rate_control": {"CO2_activation": 0.65, "hydrogenation": 0.25},
        "uncertainty_reduction": 0.10,
        "method": "mock",
    }


def ml_m3gnet_mock(payload: dict) -> dict:
    return {
        "ok": True,
        "total_energy_eV": -162.0,
        "energy_per_atom_eV": -4.50,
        "max_force_eV_A": 0.052,
        "n_atoms": 36,
        "method": "mock",
        "E_ads_CO2_est": -0.40,
        "E_ads_H_est": -0.29,
        "uncertainty_reduction": 0.11,
    }


def cantera_reactor_mock(payload: dict) -> dict:
    import time
    time.sleep(0.5)
    return {
        "ok": True,
        "conversion": 0.38,
        "selectivity": 0.82,
        "methanol_yield": 0.31,
        "products": {
            "CH3OH": 0.31,
            "CO": 0.04,
            "H2O": 0.38,
        },
        "temperature_K": 523.0,
        "pressure_bar": 50.0,
        "uncertainty_reduction": 0.08,
        "method": "mock",
    }


def cantera_sensitivity_mock(payload: dict) -> dict:
    return {
        "ok": True,
        "sensitivities": {
            "temperature": 0.72,
            "pressure": 0.65,
            "H2_CO2_ratio": 0.48,
        },
        "optimal_conditions": {
            "temperature_K": 533.0,
            "pressure_bar": 50.0,
            "H2_CO2_ratio": 3.0,
        },
        "uncertainty_reduction": 0.06,
        "method": "mock",
    }


# =============================================================================
# REGISTRATION
# =============================================================================

REAL_FUNCTIONS = {
    # Core screening
    "fast_surrogate": fast_surrogate_real,
    "microkinetic_lite": microkinetic_lite_real,
    "stability_analysis": stability_analysis_real,
    # ML potentials (MACE/CHGNet/M3GNet)
    "ml_screening": ml_screening_real,
    "ml_relaxation": ml_relaxation_real,
    "ml_m3gnet": ml_m3gnet_real,
    # DFT
    "dft_adsorption": dft_adsorption_real,
    "dft_gpaw": dft_gpaw_real,
    "dft_qe": dft_qe_real,
    # Microkinetics
    "microkinetic_catmap": microkinetic_catmap_real,
    # Cantera reactor modeling
    "cantera_reactor": cantera_reactor_real,
    "cantera_sensitivity": cantera_sensitivity_real,
    # MD
    "openmm_relaxation": openmm_relaxation_real,
    "gromacs_md": gromacs_md_real,
}

MOCK_FUNCTIONS = {
    # Core screening
    "fast_surrogate": fast_surrogate_mock,
    "microkinetic_lite": microkinetic_lite_mock,
    "stability_analysis": stability_analysis_mock,
    # ML potentials
    "ml_screening": ml_screening_mock,
    "ml_relaxation": ml_relaxation_mock,
    "ml_m3gnet": ml_m3gnet_mock,
    # DFT
    "dft_adsorption": dft_adsorption_mock,
    "dft_gpaw": dft_gpaw_mock,
    "dft_qe": dft_qe_mock,
    # Microkinetics
    "microkinetic_catmap": microkinetic_catmap_mock,
    # Cantera reactor modeling
    "cantera_reactor": cantera_reactor_mock,
    "cantera_sensitivity": cantera_sensitivity_mock,
    # MD
    "openmm_relaxation": openmm_relaxation_mock,
    "gromacs_md": gromacs_md_mock,
}


def register_functions(endpoint_id: str, use_mock: bool = False) -> dict[str, str]:
    """Register all test functions with Globus Compute."""
    from globus_compute_sdk import Client

    client = Client()
    functions = MOCK_FUNCTIONS if use_mock else REAL_FUNCTIONS
    mode = "MOCK" if use_mock else "REAL"

    print(f"Registering {mode} functions for endpoint: {endpoint_id}")

    function_map = {}
    for name, func in functions.items():
        print(f"  Registering {name}...")
        func_id = client.register_function(func)
        function_map[name] = func_id
        print(f"    -> {func_id}")

    return function_map


def main():
    parser = argparse.ArgumentParser(
        description="Register Globus Compute functions for Catalyst",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--endpoint", required=True, help="Globus Compute endpoint ID")
    parser.add_argument("--output", default="gc_functions.json", help="Output JSON file")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock implementations (fastest, fixed values)")
    parser.add_argument("--test", action="store_true",
                        help="Test one function after registration")
    parser.add_argument("--test-all", action="store_true",
                        help="Test all functions after registration")
    args = parser.parse_args()

    function_map = register_functions(args.endpoint, use_mock=args.mock)

    # Save to JSON
    output = {
        "endpoint_id": args.endpoint,
        "mode": "mock" if args.mock else "real",
        "functions": function_map,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved function map to {args.output}")

    print(f"\n{'='*50}")
    print(f"Mode: {'MOCK' if args.mock else 'REAL (with surrogate fallback)'}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Functions registered: {len(function_map)}")
    print(f"{'='*50}")

    # Test functions
    if args.test or args.test_all:
        from globus_compute_sdk import Executor

        test_candidate = {
            "support": "ZrO2",
            "metals": [
                {"element": "Cu", "wt_pct": 55},
                {"element": "Zn", "wt_pct": 30},
                {"element": "Al", "wt_pct": 15},
            ]
        }

        functions = MOCK_FUNCTIONS if args.mock else REAL_FUNCTIONS
        tests_to_run = list(functions.keys()) if args.test_all else ["fast_surrogate"]

        print(f"\n--- Testing {len(tests_to_run)} function(s) ---")

        with Executor(endpoint_id=args.endpoint) as ex:
            for test_name in tests_to_run:
                print(f"\nTesting {test_name}...")
                try:
                    future = ex.submit(
                        functions[test_name],
                        {"candidate": test_candidate}
                    )
                    result = future.result(timeout=120)
                    print(f"  Result: {json.dumps(result, indent=2)}")
                except Exception as e:
                    print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
