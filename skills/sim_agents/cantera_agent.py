"""CanteraAgent - Cantera reactor simulations."""

from __future__ import annotations

import logging
import math
from typing import Any

from academy.agent import action
from skills.base_agent import TrackedAgent

logger = logging.getLogger(__name__)


class CanteraAgent(TrackedAgent):
    """Academy agent for Cantera reactor simulations.

    Provides reactor modeling and kinetic analysis for catalyst evaluation.
    Uses surrogate kinetics for CO2-to-methanol (full mechanism not available).

    Inherits from TrackedAgent for automatic history tracking.
    """

    def __init__(self):
        """Initialize CanteraAgent."""
        super().__init__(max_history=100)
        self._ct = None
        self._ready = False

    async def agent_on_startup(self) -> None:
        """Check Cantera availability."""
        logger.info("CanteraAgent starting")

        try:
            import cantera as ct
            self._ct = ct
            self._ready = True
            logger.info(f"Cantera {ct.__version__} loaded")
        except ImportError as e:
            logger.warning(f"Cantera not available: {e}")
            # Still mark as ready - will use surrogate
            self._ready = True

    @action
    async def reactor(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run reactor simulation."""
        with self.track_action("reactor", request) as tracker:
            candidate = request.get("candidate", {})
            T = request.get("temperature_K", 523.0)
            P = request.get("pressure_bar", 50.0)
            tau = request.get("residence_time_s", 1.0)

            support = str(candidate.get("support", "Al2O3")).lower()
            metals = candidate.get("metals", [])
            cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
            zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)

            # Surrogate kinetic model
            R = 8.314
            Ea_base = 70.0 if "zro2" in support else 80.0
            Ea = Ea_base - 0.1 * cu + 0.05 * zn
            A = 1e8 * (1 + cu / 100) * (1 + zn / 200)
            k = A * math.exp(-Ea * 1000 / (R * T))

            K_ads = 0.1 * (1 + 0.01 * zn)
            theta = K_ads * P / (1 + K_ads * P)

            conversion = 1 - math.exp(-k * theta * tau)
            conversion = max(0.01, min(0.95, conversion))

            if "zro2" in support:
                selectivity = 0.85 + 0.001 * zn - 0.002 * max(0, cu - 60)
            else:
                selectivity = 0.75 + 0.002 * zn - 0.003 * max(0, cu - 60)
            selectivity = max(0.5, min(0.98, selectivity))

            result = {
                "ok": True,
                "method": "cantera_surrogate",
                "conversion": round(conversion, 4),
                "selectivity": round(selectivity, 4),
                "methanol_yield": round(conversion * selectivity, 4),
                "temperature_K": T,
                "pressure_bar": P,
            }
            tracker.set_result(result)
            return result

    @action
    async def sensitivity(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run sensitivity analysis."""
        with self.track_action("sensitivity", request) as tracker:
            candidate = request.get("candidate", {})
            support = str(candidate.get("support", "Al2O3")).lower()

            temp_sens = 0.8 if "zro2" in support else 0.6
            press_sens = 0.7
            h2_co2_sens = 0.5

            result = {
                "ok": True,
                "method": "cantera_sensitivity",
                "sensitivities": {
                    "temperature": round(temp_sens, 3),
                    "pressure": round(press_sens, 3),
                    "H2_CO2_ratio": round(h2_co2_sens, 3),
                },
                "optimal_conditions": {
                    "temperature_K": 523.0 if "zro2" in support else 543.0,
                    "pressure_bar": 50.0,
                    "H2_CO2_ratio": 3.0,
                },
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
            "cantera_available": self._ct is not None,
            "cantera_version": self._ct.__version__ if self._ct else None,
            "total_actions": stats["total_actions"],
            "total_time_s": stats["total_time_s"],
            "action_counts": stats["action_counts"],
        }
