"""MACEAgent - MACE ML potential calculations."""

from __future__ import annotations

import logging
from typing import Any

from academy.agent import action
from skills.base_agent import TrackedAgent

logger = logging.getLogger(__name__)


class MACEAgent(TrackedAgent):
    """Academy agent for MACE ML potential calculations.

    MACE provides near-DFT accuracy at MD speeds for rapid catalyst screening.
    Supports single-point energy calculations and structure relaxation.

    Inherits from TrackedAgent for automatic history tracking.
    Query history via get_history() and get_statistics() actions.
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
        super().__init__(max_history=100)
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
            logger.warning(f"MACE not available: {e}")
            self._ready = False
        except Exception as e:
            logger.warning(f"Failed to load MACE: {e}")
            self._ready = False

    @action
    async def screening(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run MACE single-point energy calculation.

        Args:
            request: Dict with:
                - candidate: Catalyst candidate specification

        Returns:
            Dict with energies and forces
        """
        with self.track_action("screening", request) as tracker:
            if not self._ready:
                result = {"ok": False, "error": "MACE not ready"}
                tracker.set_result(result)
                return result

            candidate = request.get("candidate", {})
            slab = self._build_slab(candidate)
            slab.calc = self._calc

            import numpy as np

            energy = float(slab.get_potential_energy())
            forces = slab.get_forces()

            result = {
                "ok": True,
                "method": "mace",
                "total_energy_eV": round(energy, 6),
                "energy_per_atom_eV": round(energy / len(slab), 6),
                "max_force_eV_A": round(float(np.max(np.abs(forces))), 6),
                "n_atoms": len(slab),
                "E_ads_CO2_est": round(-0.3 - energy / len(slab) / 10, 4),
                "E_ads_H_est": round(-0.25 - energy / len(slab) / 15, 4),
            }
            tracker.set_result(result)
            return result

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
        with self.track_action("relaxation", request) as tracker:
            if not self._ready:
                result = {"ok": False, "error": "MACE not ready"}
                tracker.set_result(result)
                return result

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

            result = {
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
            tracker.set_result(result)
            return result

    @action
    async def get_status(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get agent status including history statistics."""
        stats = self._get_statistics()
        return {
            "ok": True,
            "ready": self._ready,
            "model": self._model_name,
            "device": self._device,
            "total_actions": stats["total_actions"],
            "total_time_s": stats["total_time_s"],
            "action_counts": stats["action_counts"],
        }

    def _build_slab(self, candidate: dict[str, Any]):
        """Build ASE slab from candidate specification.

        Creates a Cu-based FCC(111) slab with Zn and Al substitutions
        based on the candidate's metal composition.

        Note: Support material (Al2O3, ZrO2, SiO2) affects the calculation
        via lattice constant adjustment but isn't explicitly modeled as
        a separate phase in this simplified approach.
        """
        from ase.build import fcc111
        import numpy as np

        metals = candidate.get("metals", [])
        support = candidate.get("support", "Al2O3")

        # Extract weight percentages
        cu_wt = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 60)
        zn_wt = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 25)
        al_wt = next((m["wt_pct"] for m in metals if m["element"] == "Al"), 15)

        total_wt = cu_wt + zn_wt + al_wt + 0.01  # Avoid division by zero

        # Support affects lattice constant (metal-support interaction)
        lattice_scale = {
            "Al2O3": 1.00,  # Reference
            "ZrO2": 1.02,   # Slight expansion
            "SiO2": 0.99,   # Slight contraction
        }.get(support, 1.00)

        # Build Cu FCC(111) slab - 3x3x4 for better statistics
        a_cu = 3.615 * lattice_scale  # Cu lattice constant in Angstrom
        slab = fcc111("Cu", size=(3, 3, 4), a=a_cu, vacuum=10.0, periodic=True)

        n_atoms = len(slab)
        n_surface = 9  # Top layer of 3x3

        # Calculate number of atoms to substitute
        n_zn = int(n_atoms * zn_wt / total_wt)
        n_al = int(n_atoms * al_wt / total_wt)

        symbols = list(slab.get_chemical_symbols())
        positions_z = slab.positions[:, 2]

        # Identify layers by z-coordinate
        z_sorted = np.sort(np.unique(np.round(positions_z, 1)))

        # Substitute Zn preferentially in surface/subsurface layers
        surface_mask = positions_z > z_sorted[-2] - 0.5
        surface_idx = np.where(surface_mask)[0]
        bulk_idx = np.where(~surface_mask)[0]

        # Shuffle for random distribution
        np.random.seed(hash(str(candidate)) % 2**31)  # Reproducible per candidate
        np.random.shuffle(surface_idx)
        np.random.shuffle(bulk_idx)

        # Place Zn in surface region first
        zn_placed = 0
        for i in surface_idx:
            if zn_placed >= n_zn:
                break
            symbols[i] = "Zn"
            zn_placed += 1

        # Place remaining Zn in bulk
        for i in bulk_idx:
            if zn_placed >= n_zn:
                break
            if symbols[i] == "Cu":
                symbols[i] = "Zn"
                zn_placed += 1

        # Place Al throughout (Al tends to disperse)
        all_idx = list(range(n_atoms))
        np.random.shuffle(all_idx)
        al_placed = 0
        for i in all_idx:
            if al_placed >= n_al:
                break
            if symbols[i] == "Cu":
                symbols[i] = "Al"
                al_placed += 1

        slab.set_chemical_symbols(symbols)

        logger.info(
            "Built slab: %d atoms (Cu:%d, Zn:%d, Al:%d) on %s",
            n_atoms,
            symbols.count("Cu"),
            symbols.count("Zn"),
            symbols.count("Al"),
            support,
        )

        return slab
