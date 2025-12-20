from __future__ import annotations
from dataclasses import dataclass, asdict
from academy.agent import Agent, action

@dataclass
class IonizationEnergyRequest:
    smiles: str

@dataclass
class IonizationEnergyResult:
    ionization_energy_ev: float

@dataclass
class ColorRequest:
    smiles: str

@dataclass
class ColorResult:
    color_name: str
    hex: str

class ChemistrySkill(Agent):
    """Chemistry-related property skills."""

    @action
    async def compute_ionization_energy(self, req: dict) -> dict:
        # In production: parse & validate, call QC surrogate/HPC, etc.
        r = IonizationEnergyRequest(**req)
        out = IonizationEnergyResult(ionization_energy_ev=0.5)
        return asdict(out)

    @action
    async def compute_color(self, req: dict) -> dict:
        r = ColorRequest(**req)
        out = ColorResult(color_name="colorless", hex="#fefff0")
        return asdict(out)
