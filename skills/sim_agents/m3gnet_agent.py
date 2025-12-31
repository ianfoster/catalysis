"""M3GNetAgent - M3GNet ML potential calculations."""

from __future__ import annotations

import logging
import time
from typing import Any

from academy.agent import Agent, action

logger = logging.getLogger(__name__)


class M3GNetAgent(Agent):
    """Academy agent for M3GNet ML potential calculations.

    MatErials 3-body Graph Network for materials properties.
    Provides fast ML-based energy and force predictions.
    """

    def __init__(self, device: str = "cpu"):
        """Initialize M3GNetAgent.

        Args:
            device: Compute device ('cpu' or 'cuda')
        """
        super().__init__()
        self._device = device
        self._m3gnet_available = False
        self._ase_available = False
        self._potential = None

    async def agent_on_startup(self) -> None:
        """Check M3GNet/ASE availability and load model."""
        logger.info(f"M3GNetAgent starting: device={self._device}")

        try:
            import ase
            self._ase_available = True
            logger.info(f"ASE {ase.__version__} available")
        except ImportError as e:
            logger.warning(f"ASE not available: {e}")
            return

        try:
            from m3gnet.models import M3GNet, Potential
            self._potential = Potential(M3GNet.load())
            self._m3gnet_available = True
            logger.info("M3GNet model loaded")
        except ImportError as e:
            logger.error(f"M3GNet not available: {e}")
        except Exception as e:
            logger.error(f"Failed to load M3GNet model: {e}")

    @action
    async def screening(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run M3GNet energy calculation.

        Args:
            request: Dict with candidate specification

        Returns:
            Dict with energy and estimated adsorption energies
        """
        candidate = request.get("candidate", {})
        metals = candidate.get("metals", [])

        cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
        zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)

        if not self._m3gnet_available or not self._ase_available:
            return {
                "ok": False,
                "error": f"M3GNet not available (ase={self._ase_available}, m3gnet={self._m3gnet_available})",
                "method": "m3gnet",
            }

        try:
            from ase.build import fcc111
            from m3gnet.models import M3GNetCalculator
            import numpy as np
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

            calc = M3GNetCalculator(potential=self._potential)
            slab.calc = calc
            energy = float(slab.get_potential_energy())
            forces = slab.get_forces()
            max_force = float(np.max(np.abs(forces)))

            return {
                "ok": True,
                "method": "m3gnet",
                "total_energy_eV": round(energy, 6),
                "energy_per_atom_eV": round(energy / len(slab), 6),
                "max_force_eV_A": round(max_force, 6),
                "n_atoms": int(len(slab)),
                "E_ads_CO2_est": round(-0.3 - energy / len(slab) / 10, 4),
                "E_ads_H_est": round(-0.25 - energy / len(slab) / 15, 4),
                "uncertainty_reduction": 0.11,
                "elapsed_s": round(time.time() - t0, 3),
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "method": "m3gnet"}

    @action
    async def get_status(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get agent status."""
        return {
            "ok": True,
            "ready": self._m3gnet_available and self._ase_available,
            "m3gnet_available": self._m3gnet_available,
            "ase_available": self._ase_available,
            "device": self._device,
        }
