from __future__ import annotations
from dataclasses import dataclass, asdict
from academy.agent import Agent, action

@dataclass
class CatalystSpec:
    support: str
    metals: list[dict]          # [{"element":"Cu","wt_pct":50.0}, ...]
    promoters: list[dict] = None
    prep: str = "coprecipitation"

class CatalystSkill(Agent):
    """Catalyst representation and validation."""

    @action
    async def encode(self, req: dict) -> dict:
        spec = CatalystSpec(**req)
        # Minimal, transparent encoding (replace later)
        # Example features: wt% Cu, wt% Zn, wt% Al, plus one-hot support
        metals = {m["element"]: float(m["wt_pct"]) for m in spec.metals}
        cu = metals.get("Cu", 0.0)
        zn = metals.get("Zn", 0.0)
        al = metals.get("Al", 0.0)
        support = spec.support.lower()
        support_feats = [
            1.0 if support == "al2o3" else 0.0,
            1.0 if support == "zro2" else 0.0,
            1.0 if support == "sio2" else 0.0,
        ]
        feature_vector = [cu, zn, al] + support_feats
        return {"feature_vector": feature_vector, "spec": asdict(spec)}
