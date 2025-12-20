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
