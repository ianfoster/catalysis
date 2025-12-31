"""OpenMMAgent - OpenMM molecular dynamics."""

from __future__ import annotations

import logging
from typing import Any

from academy import Agent, action

logger = logging.getLogger(__name__)


class OpenMMAgent(Agent):
    """Academy agent for OpenMM molecular dynamics.

    Provides structure relaxation and MD simulations.
    """

    def __init__(self):
        """Initialize OpenMMAgent."""
        super().__init__()
        self._openmm = None
        self._ready = False

    async def agent_on_startup(self) -> None:
        """Check OpenMM availability."""
        logger.info("OpenMMAgent starting")

        try:
            import openmm
            self._openmm = openmm
            self._ready = True
            logger.info(f"OpenMM {openmm.__version__} available")
        except ImportError as e:
            logger.warning(f"OpenMM not available: {e}")
            self._ready = True  # Still ready, will use surrogate

    @action
    async def relaxation(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run structure relaxation.

        Note: Full relaxation requires structure files.
        Returns surrogate values based on composition.

        Args:
            request: Dict with candidate specification

        Returns:
            Dict with relaxation results
        """
        candidate = request.get("candidate", {})
        support = candidate.get("support", "Al2O3")
        metals = candidate.get("metals", [])

        cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
        zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)

        # Surrogate energies based on composition
        base_energies = {"Al2O3": -125.4, "ZrO2": -132.1, "SiO2": -118.7}
        base_energy = base_energies.get(support, -120.0)
        energy = base_energy - (cu / 100) * 5.0 + (zn / 100) * 2.0
        rmsd = 0.10 + (cu / 100) * 0.15

        return {
            "ok": True,
            "method": "openmm_surrogate",
            "relaxed_energy_kJ_mol": round(energy, 2),
            "structure_rmsd": round(rmsd, 3),
            "openmm_available": self._openmm is not None,
            "openmm_version": self._openmm.__version__ if self._openmm else None,
            "note": "Using surrogate (no structure files provided)",
        }

    @action
    async def get_status(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get agent status."""
        return {
            "ok": True,
            "ready": self._ready,
            "openmm_available": self._openmm is not None,
            "openmm_version": self._openmm.__version__ if self._openmm else None,
        }
