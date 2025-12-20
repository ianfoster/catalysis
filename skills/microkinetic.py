from __future__ import annotations

from academy.agent import Agent, action


class MicrokineticSkill(Agent):
    """
    Local 'microkinetic_lite' characterizer (stub).
    Returns: rate-limiting step (RLS), sensitivities, and a reduced uncertainty proxy.
    """

    @action
    async def microkinetic_lite(self, req: dict) -> dict:
        # req can include candidate/spec/performance later; for now be simple & deterministic
        candidate = req.get("candidate", {})
        perf = req.get("performance", {}) or {}
        sty = float(perf.get("methanol_sty", 0.0))
        sel = float(perf.get("methanol_selectivity", 0.0))

        support = str(candidate.get("support", "")).strip().lower()

        # Toy "mechanism-ish" outputs
        # Pretend zirconia helps CO2 activation; alumina shifts hydrogenation balance.
        if "zro2" in support or "zirconia" in support:
            rls = "CO2_activation"
            temp_sensitivity = 0.8
            pressure_sensitivity = 0.6
        else:
            rls = "hydrogenation"
            temp_sensitivity = 0.6
            pressure_sensitivity = 0.8

        # Proxy: higher sty + sel => lower effective uncertainty
        # (in a real model you'd reduce posterior uncertainty)
        reduced_uncertainty = max(0.05, 0.25 - 0.01 * sty - 0.02 * sel)

        return {
            "rls": rls,
            "temp_sensitivity": temp_sensitivity,
            "pressure_sensitivity": pressure_sensitivity,
            "reduced_uncertainty": reduced_uncertainty,
        }
