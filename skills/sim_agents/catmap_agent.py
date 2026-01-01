"""CatMAPAgent - CatMAP microkinetic modeling."""

from __future__ import annotations

import logging
from typing import Any

from academy.agent import action
from skills.base_agent import TrackedAgent

logger = logging.getLogger(__name__)


class CatMAPAgent(TrackedAgent):
    """Academy agent for CatMAP microkinetic modeling.

    Provides descriptor-based microkinetic analysis for catalyst evaluation.
    Falls back to surrogate if CatMAP not available.

    Inherits from TrackedAgent for automatic history tracking.
    """

    def __init__(self):
        """Initialize CatMAPAgent."""
        super().__init__(max_history=100)
        self._catmap_available = False
        self._custom_kinetics_available = False

    async def agent_on_startup(self) -> None:
        """Check CatMAP availability."""
        logger.info("CatMAPAgent starting")

        try:
            import catmap
            self._catmap_available = True
            logger.info("CatMAP available")
        except ImportError:
            logger.warning("CatMAP not available")

        # Check for custom kinetics module
        try:
            from simulations.microkinetics import microkinetic_gc
            self._custom_kinetics_available = True
            logger.info("Custom microkinetics module available")
        except ImportError:
            pass

    @action
    async def microkinetic(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run microkinetic analysis."""
        with self.track_action("microkinetic", request) as tracker:
            candidate = request.get("candidate", {})
            performance = request.get("performance", {})

            support = str(candidate.get("support", "Al2O3")).lower()
            sty = float(performance.get("methanol_sty", 5.0))
            sel = float(performance.get("methanol_selectivity", 0.7))

            # Try custom kinetics first
            if self._custom_kinetics_available:
                try:
                    from simulations.microkinetics import microkinetic_gc
                    result = microkinetic_gc(request)
                    result["method"] = "microkinetic_custom"
                    tracker.set_result(result)
                    return result
                except Exception as e:
                    logger.warning(f"Custom kinetics failed: {e}")

            # Try CatMAP
            if self._catmap_available:
                try:
                    # CatMAP requires setup files - use surrogate for now
                    logger.info("CatMAP available but using surrogate (no setup files)")
                except Exception as e:
                    logger.warning(f"CatMAP failed: {e}")

            # Surrogate fallback
            if "zro2" in support or "zirconia" in support:
                rls = "CO2_activation"
                temp_sensitivity = 0.8
                pressure_sensitivity = 0.6
            else:
                rls = "hydrogenation"
                temp_sensitivity = 0.6
                pressure_sensitivity = 0.8

            reduced_uncertainty = max(0.05, 0.25 - 0.01 * sty - 0.02 * sel)

            result = {
                "ok": True,
                "method": "microkinetic_surrogate",
                "RLS": rls,
                "temp_sensitivity": round(temp_sensitivity, 3),
                "pressure_sensitivity": round(pressure_sensitivity, 3),
                "uncertainty_reduction": round(0.25 - reduced_uncertainty, 3),
                "catmap_available": self._catmap_available,
            }
            tracker.set_result(result)
            return result

    @action
    async def get_status(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get agent status including history statistics."""
        stats = self._get_statistics()
        return {
            "ok": True,
            "ready": True,
            "catmap_available": self._catmap_available,
            "custom_kinetics_available": self._custom_kinetics_available,
            "total_actions": stats["total_actions"],
            "total_time_s": stats["total_time_s"],
            "action_counts": stats["action_counts"],
        }
