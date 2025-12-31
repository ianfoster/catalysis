"""StabilityAgent - Thermodynamic stability analysis."""

from __future__ import annotations

import logging
from typing import Any

from academy import Agent, action

logger = logging.getLogger(__name__)


class StabilityAgent(Agent):
    """Academy agent for thermodynamic stability analysis.

    Evaluates catalyst stability based on composition and support effects.
    """

    def __init__(self):
        """Initialize StabilityAgent."""
        super().__init__()

    async def agent_on_startup(self) -> None:
        """Initialize stability models."""
        logger.info("StabilityAgent starting")

    @action
    async def analyze(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run stability analysis.

        Args:
            request: Dict with:
                - candidate: Catalyst candidate specification

        Returns:
            Dict with stability score and risk assessment
        """
        candidate = request.get("candidate", {})
        support = str(candidate.get("support", "Al2O3")).lower()
        metals = candidate.get("metals", [])

        cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
        zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)

        # Base stability by support
        if "zro2" in support:
            base_stability = 0.85
        elif "al2o3" in support:
            base_stability = 0.80
        else:
            base_stability = 0.70

        # High Cu can lead to sintering
        cu_penalty = max(0, (cu - 60) * 0.005)

        # Zn helps with stability (promoter effect)
        zn_bonus = min(0.1, zn * 0.002)

        stability_score = base_stability - cu_penalty + zn_bonus
        stability_score = max(0.3, min(0.95, stability_score))

        # Risk assessment
        if stability_score > 0.80:
            risk = "low"
            degradation_mechanism = "none_expected"
        elif stability_score > 0.65:
            risk = "medium"
            degradation_mechanism = "slow_sintering" if cu > 60 else "support_interaction"
        else:
            risk = "high"
            degradation_mechanism = "rapid_sintering" if cu > 70 else "phase_separation"

        return {
            "ok": True,
            "method": "stability_model",
            "stability_score": round(stability_score, 3),
            "degradation_risk": risk,
            "degradation_mechanism": degradation_mechanism,
            "factors": {
                "support_contribution": round(base_stability, 3),
                "cu_penalty": round(cu_penalty, 3),
                "zn_bonus": round(zn_bonus, 3),
            },
        }

    @action
    async def get_status(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get agent status."""
        return {
            "ok": True,
            "ready": True,
        }
