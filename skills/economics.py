from __future__ import annotations
from dataclasses import dataclass, asdict
from academy.agent import Agent, action

@dataclass
class CostRequest:
    smiles: str
    currency: str = "USD"
    basis: str = "per_g"

@dataclass
class CostResult:
    cost: float
    currency: str
    basis: str
    confidence: str

class EconomicsSkill(Agent):
    """Economics-related skills (catalog/heuristic cost)."""

    @action
    async def estimate_cost(self, req: dict) -> dict:
        r = CostRequest(**req)
        out = CostResult(cost=50.0, currency=r.currency, basis=r.basis, confidence="low")
        return asdict(out)

    @action
    async def estimate_catalyst_cost(self, req: dict) -> dict:
        # crude proxy: base + per-wt% metal costs
        metals = req["metals"]
        support = req["support"].lower()
    
        metal_cost = 0.0
        for m in metals:
            el = m["element"]
            wt = float(m["wt_pct"])
            # toy weights: Cu cheap, Zn cheap, Al very cheap
            per = {"Cu": 8.0, "Zn": 6.0, "Al": 2.0}.get(el, 20.0)
            metal_cost += per * (wt / 100.0)

        support_cost = 3.0 if "zro2" in support else 2.0
        cost = 50.0 * (support_cost + metal_cost)  # arbitrary scaling
    
        return {"usd_per_kg": cost, "confidence": "low"}
