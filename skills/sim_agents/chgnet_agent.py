"""CHGNetAgent - CHGNet ML potential calculations."""

from __future__ import annotations

import logging
from typing import Any

from academy.agent import action
from skills.base_agent import TrackedAgent

logger = logging.getLogger(__name__)


class CHGNetAgent(TrackedAgent):
    """Academy agent for CHGNet ML potential calculations.

    CHGNet is a universal ML potential trained on Materials Project data.
    Good for oxide and metal oxide systems.

    Inherits from TrackedAgent for automatic history tracking.
    """

    def __init__(self, device: str = "cpu"):
        """Initialize CHGNetAgent."""
        super().__init__(max_history=100)
        self._device = device
        self._model = None
        self._ready = False

    async def agent_on_startup(self) -> None:
        """Load CHGNet model."""
        logger.info(f"CHGNetAgent starting: device={self._device}")

        try:
            from chgnet.model import CHGNet

            self._model = CHGNet.load()
            self._ready = True
            logger.info("CHGNet model loaded successfully")
        except ImportError as e:
            logger.error(f"CHGNet not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load CHGNet: {e}")
            raise

    @action
    async def screening(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run CHGNet single-point energy calculation."""
        with self.track_action("screening", request) as tracker:
            if not self._ready:
                result = {"ok": False, "error": "CHGNet not ready"}
                tracker.set_result(result)
                return result

            from chgnet.model.dynamics import CHGNetCalculator
            import numpy as np

            candidate = request.get("candidate", {})
            slab = self._build_slab(candidate)

            calc = CHGNetCalculator(model=self._model, use_device=self._device)
            slab.calc = calc

            energy = float(slab.get_potential_energy())
            forces = slab.get_forces()

            result = {
                "ok": True,
                "method": "chgnet",
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
    async def get_status(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get agent status including history statistics."""
        stats = self._get_statistics()
        return {
            "ok": True,
            "ready": self._ready,
            "device": self._device,
            "total_actions": stats["total_actions"],
            "total_time_s": stats["total_time_s"],
            "action_counts": stats["action_counts"],
        }

    def _build_slab(self, candidate: dict[str, Any]):
        """Build ASE slab from candidate specification."""
        from ase.build import fcc111

        metals = candidate.get("metals", [])
        cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 60)
        zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 25)

        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)

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
