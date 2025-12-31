"""MACEAgent - MACE ML potential calculations."""

from __future__ import annotations

import logging
from typing import Any

from academy import Agent, action

logger = logging.getLogger(__name__)


class MACEAgent(Agent):
    """Academy agent for MACE ML potential calculations.

    MACE provides near-DFT accuracy at MD speeds for rapid catalyst screening.
    Supports single-point energy calculations and structure relaxation.
    """

    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        default_dtype: str = "float32",
    ):
        """Initialize MACEAgent.

        Args:
            model: MACE model size ("small", "medium", "large")
            device: Compute device ("cpu" or "cuda")
            default_dtype: Default dtype for calculations
        """
        super().__init__()
        self._model_name = model
        self._device = device
        self._default_dtype = default_dtype
        self._calc = None
        self._ready = False

    async def agent_on_startup(self) -> None:
        """Load MACE model."""
        logger.info(f"MACEAgent starting: model={self._model_name}, device={self._device}")

        try:
            from mace.calculators import mace_mp

            self._calc = mace_mp(
                model=self._model_name,
                dispersion=False,
                default_dtype=self._default_dtype,
                device=self._device,
            )
            self._ready = True
            logger.info("MACE model loaded successfully")
        except ImportError as e:
            logger.error(f"MACE not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load MACE: {e}")
            raise

    @action
    async def screening(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run MACE single-point energy calculation.

        Args:
            request: Dict with:
                - candidate: Catalyst candidate specification

        Returns:
            Dict with energies and forces
        """
        if not self._ready:
            return {"ok": False, "error": "MACE not ready"}

        candidate = request.get("candidate", {})
        slab = self._build_slab(candidate)
        slab.calc = self._calc

        import numpy as np

        energy = float(slab.get_potential_energy())
        forces = slab.get_forces()

        return {
            "ok": True,
            "method": "mace",
            "total_energy_eV": round(energy, 6),
            "energy_per_atom_eV": round(energy / len(slab), 6),
            "max_force_eV_A": round(float(np.max(np.abs(forces))), 6),
            "n_atoms": len(slab),
            "E_ads_CO2_est": round(-0.3 - energy / len(slab) / 10, 4),
            "E_ads_H_est": round(-0.25 - energy / len(slab) / 15, 4),
        }

    @action
    async def relaxation(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run MACE structure relaxation.

        Args:
            request: Dict with:
                - candidate: Catalyst candidate specification
                - fmax: Force convergence criterion (default: 0.05)
                - max_steps: Maximum optimization steps (default: 100)

        Returns:
            Dict with relaxation results
        """
        if not self._ready:
            return {"ok": False, "error": "MACE not ready"}

        from ase.optimize import BFGS
        import numpy as np

        candidate = request.get("candidate", {})
        fmax = request.get("fmax", 0.05)
        max_steps = request.get("max_steps", 100)

        slab = self._build_slab(candidate)
        slab.calc = self._calc

        e_initial = float(slab.get_potential_energy())

        opt = BFGS(slab, logfile=None)
        converged = opt.run(fmax=fmax, steps=max_steps)

        e_final = float(slab.get_potential_energy())
        forces = slab.get_forces()

        return {
            "ok": True,
            "method": "mace",
            "converged": bool(converged),
            "initial_energy_eV": round(e_initial, 6),
            "final_energy_eV": round(e_final, 6),
            "energy_change_eV": round(e_final - e_initial, 6),
            "max_force_eV_A": round(float(np.max(np.abs(forces))), 6),
            "n_steps": opt.nsteps,
            "n_atoms": len(slab),
        }

    @action
    async def get_status(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get agent status."""
        return {
            "ok": True,
            "ready": self._ready,
            "model": self._model_name,
            "device": self._device,
        }

    def _build_slab(self, candidate: dict[str, Any]):
        """Build ASE slab from candidate specification."""
        from ase.build import fcc111

        metals = candidate.get("metals", [])
        cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 60)
        zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 25)

        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)

        # Substitute some Cu with Zn
        n_zn = int(4 * zn / (cu + zn + 0.01))
        symbols = list(slab.get_chemical_symbols())
        surface_idx = [
            i for i, z in enumerate(slab.positions[:, 2])
            if z > slab.positions[:, 2].max() - 2.0
        ]
        for i in range(min(n_zn, len(surface_idx))):
            symbols[surface_idx[i]] = "Zn"
        slab.set_chemical_symbols(symbols)

        return slab
