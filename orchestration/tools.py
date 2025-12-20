from __future__ import annotations
from typing import List

from langchain.tools import tool
from langchain_core.tools import BaseTool

from academy.handle import Handle
from skills.chemistry import ChemistrySkill
from skills.economics import EconomicsSkill

def make_chemistry_tools(chem: Handle[ChemistrySkill]) -> List[BaseTool]:
    @tool
    async def compute_ionization_energy(smiles: str) -> dict:
        """Compute ionization energy (eV) from a SMILES string. Returns JSON."""
        return await chem.compute_ionization_energy({"smiles": smiles})

    @tool
    async def compute_color(smiles: str) -> dict:
        """Compute approximate color from a SMILES string. Returns JSON {color_name, hex}."""
        return await chem.compute_color({"smiles": smiles})

    return [compute_ionization_energy, compute_color]

def make_economics_tools(econ: Handle[EconomicsSkill]) -> List[BaseTool]:
    @tool
    async def estimate_cost(smiles: str, currency: str = "USD", basis: str = "per_g") -> dict:
        """Estimate cost for a molecule from SMILES. Returns JSON {cost, currency, basis, confidence}."""
        return await econ.estimate_cost({"smiles": smiles, "currency": currency, "basis": basis})

    return [estimate_cost]
