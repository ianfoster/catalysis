"""Simulation agents - one Academy agent per simulation code.

Each agent wraps a specific simulation code and exposes it as an action.
ShepherdAgent dispatches to the appropriate agent based on test type.

Available agents (11 total):
- MACEAgent: MACE ML potential (screening, relaxation)
- CHGNetAgent: CHGNet ML potential
- M3GNetAgent: M3GNet ML potential
- CanteraAgent: Cantera reactor simulations
- StabilityAgent: Thermodynamic stability analysis
- SurrogateAgent: Fast physics-informed surrogates
- QEAgent: Quantum ESPRESSO DFT
- GPAWAgent: GPAW DFT calculations
- OpenMMAgent: OpenMM molecular dynamics
- GROMACSAgent: GROMACS molecular dynamics
- CatMAPAgent: CatMAP microkinetic modeling
"""

from .mace_agent import MACEAgent
from .chgnet_agent import CHGNetAgent
from .m3gnet_agent import M3GNetAgent
from .cantera_agent import CanteraAgent
from .stability_agent import StabilityAgent
from .surrogate_agent import SurrogateAgent
from .qe_agent import QEAgent
from .gpaw_agent import GPAWAgent
from .openmm_agent import OpenMMAgent
from .gromacs_agent import GROMACSAgent
from .catmap_agent import CatMAPAgent

__all__ = [
    "MACEAgent",
    "CHGNetAgent",
    "M3GNetAgent",
    "CanteraAgent",
    "StabilityAgent",
    "SurrogateAgent",
    "QEAgent",
    "GPAWAgent",
    "OpenMMAgent",
    "GROMACSAgent",
    "CatMAPAgent",
]
