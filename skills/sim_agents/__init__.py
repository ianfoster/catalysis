"""Simulation agents - one Academy agent per simulation code.

Each agent wraps a specific simulation code and exposes it as an action.
ShepherdAgent dispatches to the appropriate agent based on test type.

Available agents:
- MACEAgent: MACE ML potential (screening, relaxation)
- CHGNetAgent: CHGNet ML potential
- CanteraAgent: Cantera reactor simulations
- StabilityAgent: Thermodynamic stability analysis
- SurrogateAgent: Fast physics-informed surrogates
- QEAgent: Quantum ESPRESSO DFT (if available)
- OpenMMAgent: OpenMM molecular dynamics
"""

from .mace_agent import MACEAgent
from .chgnet_agent import CHGNetAgent
from .cantera_agent import CanteraAgent
from .stability_agent import StabilityAgent
from .surrogate_agent import SurrogateAgent

__all__ = [
    "MACEAgent",
    "CHGNetAgent",
    "CanteraAgent",
    "StabilityAgent",
    "SurrogateAgent",
]

# Optional agents (may not be available on all systems)
try:
    from .qe_agent import QEAgent
    __all__.append("QEAgent")
except ImportError:
    pass

try:
    from .openmm_agent import OpenMMAgent
    __all__.append("OpenMMAgent")
except ImportError:
    pass
