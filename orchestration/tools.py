from __future__ import annotations
from typing import List

from langchain.tools import tool
from langchain_core.tools import BaseTool

from academy.handle import Handle

from skills.chemistry import ChemistrySkill
from skills.economics import EconomicsSkill
from skills.catalyst import CatalystSkill
from skills.performance import PerformanceSkill
from skills.hpc_characterizer import HPCCharacterizerSkill
from skills.microkinetic import MicrokineticSkill


def make_microkinetic_tools(mk: Handle[MicrokineticSkill]) -> list[BaseTool]:
    @tool
    async def microkinetic_lite(candidate: dict, performance: dict) -> dict:
        """Run microkinetic-lite characterization using candidate + predicted performance."""
        return await mk.microkinetic_lite({"candidate": candidate, "performance": performance})

    return [microkinetic_lite]


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
        """Estimate molecular cost from SMILES."""
        return await econ.estimate_cost(
            {"smiles": smiles, "currency": currency, "basis": basis}
        )

    @tool
    async def estimate_catalyst_cost(support: str, metals: list) -> dict:
        """Estimate catalyst material cost (USD/kg)."""
        return await econ.estimate_catalyst_cost(
            {"support": support, "metals": metals}
        )

    return [
        estimate_cost,
        estimate_catalyst_cost,
    ]


def make_catalyst_tools(cat: Handle[CatalystSkill]) -> list[BaseTool]:
    @tool
    async def encode_catalyst(
        support: str,
        metals: list,
        promoters: list | None = None,
        prep: str = "coprecipitation",
    ) -> dict:
        """Encode a catalyst spec into a feature vector for CO2->MeOH models."""
        return await cat.encode(
            {"support": support, "metals": metals, "promoters": promoters, "prep": prep}
        )

    return [encode_catalyst]


def make_performance_tools(perf: Handle[PerformanceSkill]) -> list[BaseTool]:
    @tool
    async def predict_performance(feature_vector: list) -> dict:
        """Predict CO2->MeOH performance metrics from catalyst feature vector."""
        return await perf.predict({"feature_vector": feature_vector})

    return [predict_performance]


def make_hpc_tools(hpc: Handle[HPCCharacterizerSkill]) -> list[BaseTool]:
    @tool
    async def submit_characterization(characterizer: str, payload: dict) -> dict:
        """Submit an HPC characterization job via Globus Compute. Returns {task_id,...}."""
        return await hpc.submit({"characterizer": characterizer, "payload": payload})

    @tool
    async def get_characterization(task_id: str) -> dict:
        """Poll an HPC characterization job. Returns {status, result|error}."""
        return await hpc.get({"task_id": task_id})

    return [submit_characterization, get_characterization]
