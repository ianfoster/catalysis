"""Polaris workflow configuration for Catalyst.

This package provides:
- Globus Compute endpoint setup for Polaris
- PBS job scripts for launching GC endpoint
- DFT functions (QE, GPAW) optimized for Polaris A100 GPUs

Quick Start:
    1. SSH to Polaris
    2. Run setup_endpoint.py to configure GC
    3. Submit start_gc_endpoint.pbs to get nodes
    4. Register functions with dft_functions.py
    5. Update config.yaml with endpoint ID
"""

from .dft_functions import (
    POLARIS_DFT_FUNCTIONS,
    qe_scf_polaris,
    qe_relax_polaris,
    gpaw_scf_polaris,
    gpaw_relax_polaris,
    register_polaris_functions,
)

__all__ = [
    "POLARIS_DFT_FUNCTIONS",
    "qe_scf_polaris",
    "qe_relax_polaris",
    "gpaw_scf_polaris",
    "gpaw_relax_polaris",
    "register_polaris_functions",
]
