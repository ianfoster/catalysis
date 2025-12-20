from __future__ import annotations
from academy.agent import Agent, action

class CandidateSkill(Agent):
    """Propose candidate molecules (placeholder)."""

    @action
    async def propose(self, req: dict) -> dict:
        # req could include product/feedstocks later
        return {"candidates": ["c1ccccc1", "CCO", "CC(C)O"]}
