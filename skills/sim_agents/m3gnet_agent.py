"""M3GNetAgent - M3GNet/MatGL ML potential calculations."""

from __future__ import annotations

import logging
from typing import Any

from academy.agent import action
from skills.base_agent import TrackedAgent

logger = logging.getLogger(__name__)


class M3GNetAgent(TrackedAgent):
    """Academy agent for M3GNet ML potential calculations.

    MatErials 3-body Graph Network for materials properties.
    Provides fast ML-based energy and force predictions.
    Uses matgl (the maintained successor to m3gnet).

    Inherits from TrackedAgent for automatic history tracking.
    """

    def __init__(self, device: str = "cpu"):
        """Initialize M3GNetAgent."""
        super().__init__(max_history=100)
        self._device = device
        self._m3gnet_available = False
        self._ase_available = False
        self._potential = None
        self._calculator_class = None
        self._use_matgl = False

    async def agent_on_startup(self) -> None:
        """Check M3GNet/MatGL/ASE availability and load model."""
        logger.info(f"M3GNetAgent starting: device={self._device}")

        try:
            import ase
            self._ase_available = True
            logger.info(f"ASE {ase.__version__} available")
        except ImportError as e:
            logger.warning(f"ASE not available: {e}")
            return

        # Try matgl first (newer, maintained)
        try:
            import matgl
            logger.info(f"MatGL {matgl.__version__} found")

            # matgl 2.x uses PESCalculator
            from matgl.ext.ase import PESCalculator
            logger.info("PESCalculator imported")

            self._potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            logger.info("M3GNet model loaded")

            self._calculator_class = PESCalculator
            self._m3gnet_available = True
            self._use_matgl = True
            logger.info("MatGL M3GNet ready")
        except ImportError as e:
            logger.warning(f"MatGL import failed: {e}")
            # Fall back to old m3gnet
            try:
                from m3gnet.models import M3GNet, Potential, M3GNetCalculator
                self._potential = Potential(M3GNet.load())
                self._calculator_class = M3GNetCalculator
                self._m3gnet_available = True
                self._use_matgl = False
                logger.info("Legacy m3gnet model loaded")
            except ImportError as e2:
                logger.error(f"Neither matgl nor m3gnet available: matgl={e}, m3gnet={e2}")
            except Exception as e2:
                logger.error(f"Failed to load m3gnet model: {e2}")
        except Exception as e:
            logger.error(f"Failed to load MatGL model: {e}")

    @action
    async def screening(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run M3GNet energy calculation."""
        with self.track_action("screening", request) as tracker:
            candidate = request.get("candidate", {})
            metals = candidate.get("metals", [])

            cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
            zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)

            if not self._m3gnet_available or not self._ase_available:
                result = {
                    "ok": False,
                    "error": f"M3GNet not available (ase={self._ase_available}, m3gnet={self._m3gnet_available})",
                    "method": "m3gnet",
                }
                tracker.set_result(result)
                return result

            try:
                from ase.build import fcc111
                import numpy as np

                # Build Cu/Zn slab
                slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
                n_zn = int(4 * zn / (cu + zn + 0.01))
                symbols = list(slab.get_chemical_symbols())
                surface_idx = [i for i, z in enumerate(slab.positions[:, 2])
                               if z > slab.positions[:, 2].max() - 2.0]
                for i in range(min(n_zn, len(surface_idx))):
                    symbols[surface_idx[i]] = "Zn"
                slab.set_chemical_symbols(symbols)

                # Create calculator - API differs between matgl and legacy m3gnet
                if getattr(self, '_use_matgl', False):
                    # matgl 2.x API
                    calc = self._calculator_class(potential=self._potential)
                else:
                    # Legacy m3gnet API
                    calc = self._calculator_class(potential=self._potential)
                slab.calc = calc
                energy = float(slab.get_potential_energy())
                forces = slab.get_forces()
                max_force = float(np.max(np.abs(forces)))

                result = {
                    "ok": True,
                    "method": "m3gnet",
                    "total_energy_eV": round(energy, 6),
                    "energy_per_atom_eV": round(energy / len(slab), 6),
                    "max_force_eV_A": round(max_force, 6),
                    "n_atoms": int(len(slab)),
                    "E_ads_CO2_est": round(-0.3 - energy / len(slab) / 10, 4),
                    "E_ads_H_est": round(-0.25 - energy / len(slab) / 15, 4),
                    "uncertainty_reduction": 0.11,
                }
                tracker.set_result(result)
                return result
            except Exception as e:
                result = {"ok": False, "error": str(e), "method": "m3gnet"}
                tracker.set_result(result)
                return result

    @action
    async def get_status(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get agent status including history statistics."""
        stats = self._get_statistics()
        return {
            "ok": True,
            "ready": self._m3gnet_available and self._ase_available,
            "m3gnet_available": self._m3gnet_available,
            "ase_available": self._ase_available,
            "device": self._device,
            "total_actions": stats["total_actions"],
            "total_time_s": stats["total_time_s"],
            "action_counts": stats["action_counts"],
        }
