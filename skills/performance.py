from __future__ import annotations
from academy.agent import Agent, action

class PerformanceSkill(Agent):
    """Surrogate prediction of CO2->MeOH performance (stub)."""

    @action
    async def predict(self, req: dict) -> dict:
        x = req["feature_vector"]
        # toy proxy: Cu helps, too much Cu hurts selectivity; support matters slightly
        cu, zn, al, is_al2o3, is_zro2, is_sio2 = x

        conversion = max(0.0, min(1.0, 0.2 + 0.01 * cu + 0.005 * zn - 0.002 * al))
        selectivity = max(0.0, min(1.0, 0.6 + 0.003 * zn - 0.004 * max(0.0, cu - 60) + 0.05 * is_zro2))
        sty = conversion * selectivity * 10.0  # arbitrary units

        return {
            "co2_conversion": conversion,
            "methanol_selectivity": selectivity,
            "methanol_sty": sty,
            "uncertainty": 0.25,  # placeholder
        }
