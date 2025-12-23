from __future__ import annotations

from typing import Any, Dict

from academy.agent import Agent, action
from tools.rdkit_tool import rdkit_descriptors


class RDKitSkill(Agent):
    """Cheap chemistry descriptors from SMILES."""

    @action
    async def descriptors(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return rdkit_descriptors(payload)
