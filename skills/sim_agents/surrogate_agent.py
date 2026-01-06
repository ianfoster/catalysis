"""SurrogateAgent - Fast physics-informed surrogate predictions."""

from __future__ import annotations

import logging
from typing import Any

from academy.agent import action
from skills.base_agent import TrackedAgent

logger = logging.getLogger(__name__)


class SurrogateAgent(TrackedAgent):
    """Academy agent for fast surrogate predictions.

    Provides quick performance estimates based on physics-informed models.
    Useful for initial screening before more expensive simulations.

    Inherits from TrackedAgent for automatic history tracking.
    """

    def __init__(self):
        """Initialize SurrogateAgent."""
        super().__init__(max_history=100)

    async def agent_on_startup(self) -> None:
        """Initialize surrogate models."""
        logger.info("SurrogateAgent starting")

    @action
    async def screening(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run fast surrogate prediction."""
        with self.track_action("screening", request) as tracker:
            candidate = request.get("candidate", {})
            support = str(candidate.get("support", "Al2O3")).lower()
            metals = candidate.get("metals", [])

            cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
            zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)
            al = next((m["wt_pct"] for m in metals if m["element"] == "Al"), 20)

            # Support effects
            if "zro2" in support:
                support_bonus = 1.15
            elif "sio2" in support:
                support_bonus = 0.95
            else:
                support_bonus = 1.0

            # Optimal composition is around Cu55/Zn30/Al15
            cu_opt, zn_opt = 55, 30
            cu_factor = 1 - 0.005 * abs(cu - cu_opt)
            zn_factor = 1 - 0.003 * abs(zn - zn_opt)

            # Calculate metrics
            base_conversion = 0.25
            base_selectivity = 0.80

            conversion = base_conversion * support_bonus * cu_factor * zn_factor
            conversion = max(0.05, min(0.45, conversion))

            selectivity = base_selectivity * (1 + 0.002 * zn) * support_bonus
            selectivity = max(0.6, min(0.95, selectivity))

            sty = conversion * selectivity * 100  # g MeOH / kg cat / h

            result = {
                "ok": True,
                "method": "surrogate",
                "co2_conversion": round(conversion, 4),
                "methanol_selectivity": round(selectivity, 4),
                "methanol_sty": round(sty, 2),
                "uncertainty": 0.15,
            }
            tracker.set_result(result)
            return result

    @action
    async def microkinetic(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run microkinetic surrogate analysis."""
        with self.track_action("microkinetic", request) as tracker:
            candidate = request.get("candidate", {})
            performance = request.get("performance", {})

            support = str(candidate.get("support", "Al2O3")).lower()
            sty = float(performance.get("methanol_sty", 5.0))
            sel = float(performance.get("methanol_selectivity", 0.7))

            # Support-dependent kinetics
            if "zro2" in support:
                rls = "CO2_activation"
                temp_sens = 0.8
                press_sens = 0.6
            else:
                rls = "hydrogenation"
                temp_sens = 0.6
                press_sens = 0.8

            # Uncertainty reduction based on performance
            uncertainty_reduction = max(0.05, 0.25 - 0.01 * sty - 0.02 * sel)

            result = {
                "ok": True,
                "method": "microkinetic_surrogate",
                "RLS": rls,
                "temp_sensitivity": round(temp_sens, 3),
                "pressure_sensitivity": round(press_sens, 3),
                "uncertainty_reduction": round(0.25 - uncertainty_reduction, 3),
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
            "total_actions": stats["total_actions"],
            "total_time_s": stats["total_time_s"],
            "action_counts": stats["action_counts"],
        }
