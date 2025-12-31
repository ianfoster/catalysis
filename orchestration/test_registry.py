"""Test registry for ShepherdAgent.

Defines available characterization tests with metadata for LLM reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TestSpec:
    """Specification for a characterization test.

    Attributes:
        name: Unique identifier for the test
        description: Human-readable description for LLM reasoning
        cost: Abstract cost units (for budget tracking)
        endpoint: Execution endpoint type ("cheap" or "gpu")
        prerequisites: List of test names that must complete first
        timeout: Maximum wait time in seconds
        outputs: List of output field names
        reduces_uncertainty: Whether this test reduces prediction uncertainty
        expected_runtime_s: Expected runtime for informational purposes
        gc_function: Globus Compute function name for this test
        simulation_method: Underlying simulation code (e.g., 'mace', 'quantum_espresso')
        method_status: Status of the underlying method ('available', 'surrogate', 'unavailable')
    """

    name: str
    description: str
    cost: float
    endpoint: str  # "cheap" or "gpu"
    prerequisites: tuple[str, ...] = field(default_factory=tuple)
    timeout: float = 300.0
    outputs: tuple[str, ...] = field(default_factory=tuple)
    reduces_uncertainty: bool = False
    expected_runtime_s: int = 60
    gc_function: str = ""  # Globus Compute function name
    simulation_method: str = ""  # Underlying simulation code
    method_status: str = "available"  # 'available', 'surrogate', 'unavailable'


# Default test registry
# Cost units are abstract and can be tuned based on actual compute costs
#
# gc_function: Name of the registered Globus Compute function
# simulation_method: Underlying simulation code used
# method_status: 'available' (real code works), 'surrogate' (using approximation),
#                'unavailable' (broken, do not use)
#
AVAILABLE_TESTS: dict[str, TestSpec] = {
    "fast_surrogate": TestSpec(
        name="fast_surrogate",
        description=(
            "Quick ML surrogate prediction (~1s). "
            "Returns CO2 conversion, methanol selectivity, and space-time yield (STY). "
            "Low cost, good for initial screening. Does not reduce uncertainty."
        ),
        cost=0.01,
        endpoint="cheap",
        prerequisites=(),
        timeout=30,
        outputs=("co2_conversion", "methanol_selectivity", "methanol_sty", "uncertainty"),
        reduces_uncertainty=False,
        expected_runtime_s=1,
        gc_function="fast_surrogate",
        simulation_method="surrogate",
        method_status="available",
    ),
    "ml_screening": TestSpec(
        name="ml_screening",
        description=(
            "ML potential screening with MACE (~10s). "
            "Returns total energy, energy per atom, and maximum force. "
            "Fast and accurate for initial structure evaluation."
        ),
        cost=0.5,
        endpoint="cheap",
        prerequisites=(),
        timeout=120,
        outputs=("total_energy_eV", "energy_per_atom_eV", "max_force_eV_A"),
        reduces_uncertainty=True,
        expected_runtime_s=10,
        gc_function="ml_screening",
        simulation_method="mace",
        method_status="available",
    ),
    "ml_relaxation": TestSpec(
        name="ml_relaxation",
        description=(
            "ML potential structure relaxation with MACE (~60s). "
            "Relaxes atomic positions to minimize energy. "
            "Returns converged energy and structure."
        ),
        cost=2.0,
        endpoint="cheap",
        prerequisites=("ml_screening",),
        timeout=300,
        outputs=("final_energy_eV", "converged", "n_steps"),
        reduces_uncertainty=True,
        expected_runtime_s=60,
        gc_function="ml_relaxation",
        simulation_method="mace",
        method_status="available",
    ),
    "microkinetic_lite": TestSpec(
        name="microkinetic_lite",
        description=(
            "Lite microkinetic analysis (~60s). "
            "Returns rate-limiting step (RLS), temperature sensitivity, pressure sensitivity. "
            "Medium cost. Reduces prediction uncertainty."
        ),
        cost=1.0,
        endpoint="cheap",
        prerequisites=("fast_surrogate",),
        timeout=120,
        outputs=("RLS", "temp_sensitivity", "pressure_sensitivity", "uncertainty_reduction"),
        reduces_uncertainty=True,
        expected_runtime_s=60,
        gc_function="microkinetic_lite",
        simulation_method="catmap",
        method_status="surrogate",
    ),
    "dft_adsorption": TestSpec(
        name="dft_adsorption",
        description=(
            "DFT adsorption energy calculation with Quantum ESPRESSO (~1h). "
            "Returns CO2 and H adsorption energies with high accuracy. "
            "High cost. Significantly reduces uncertainty."
        ),
        cost=100.0,
        endpoint="gpu",
        prerequisites=("ml_screening",),
        timeout=7200,
        outputs=("E_ads_CO2", "E_ads_H", "uncertainty_reduction"),
        reduces_uncertainty=True,
        expected_runtime_s=3600,
        gc_function="dft_qe",
        simulation_method="quantum_espresso",
        method_status="available",
    ),
    "openmm_relaxation": TestSpec(
        name="openmm_relaxation",
        description=(
            "OpenMM structure relaxation (~5min). "
            "Optimizes catalyst surface structure using molecular mechanics. "
            "Medium cost, benefits from GPU."
        ),
        cost=5.0,
        endpoint="gpu",
        prerequisites=("fast_surrogate",),
        timeout=600,
        outputs=("relaxed_energy", "structure_rmsd"),
        reduces_uncertainty=True,
        expected_runtime_s=300,
        gc_function="openmm_relaxation",
        simulation_method="openmm",
        method_status="available",
    ),
    "stability_analysis": TestSpec(
        name="stability_analysis",
        description=(
            "Thermodynamic stability analysis (~30s). "
            "Estimates catalyst stability under reaction conditions. "
            "Low cost. Useful for identifying unstable candidates early."
        ),
        cost=0.1,
        endpoint="cheap",
        prerequisites=("fast_surrogate",),
        timeout=60,
        outputs=("stability_score", "degradation_risk"),
        reduces_uncertainty=False,
        expected_runtime_s=30,
        gc_function="stability_analysis",
        simulation_method="surrogate",
        method_status="available",
    ),
    "cantera_reactor": TestSpec(
        name="cantera_reactor",
        description=(
            "Cantera reactor simulation (~30s). "
            "Simulates reactor performance under operating conditions. "
            "Returns conversion, selectivity, and product distribution."
        ),
        cost=1.0,
        endpoint="cheap",
        prerequisites=("fast_surrogate",),
        timeout=120,
        outputs=("conversion", "selectivity", "products"),
        reduces_uncertainty=True,
        expected_runtime_s=30,
        gc_function="cantera_reactor",
        simulation_method="cantera",
        method_status="surrogate",
    ),
}


def get_test(name: str) -> TestSpec:
    """Get test specification by name.

    Args:
        name: Test name

    Returns:
        TestSpec for the test

    Raises:
        KeyError: If test not found
    """
    if name not in AVAILABLE_TESTS:
        raise KeyError(f"Unknown test: {name}. Available: {list(AVAILABLE_TESTS.keys())}")
    return AVAILABLE_TESTS[name]


def check_prerequisites(test_name: str, completed_tests: set[str]) -> tuple[bool, list[str]]:
    """Check if prerequisites for a test are satisfied.

    Args:
        test_name: Name of test to check
        completed_tests: Set of test names that have completed

    Returns:
        Tuple of (satisfied: bool, missing: list of missing prerequisite names)
    """
    spec = get_test(test_name)
    missing = [p for p in spec.prerequisites if p not in completed_tests]
    return (len(missing) == 0, missing)


def format_tests_for_prompt(
    completed_tests: set[str] | None = None,
    budget_remaining: float | None = None,
) -> str:
    """Format test registry as table for LLM prompt.

    Args:
        completed_tests: Set of already-completed test names (to show status)
        budget_remaining: Remaining budget (to show affordability)

    Returns:
        Formatted markdown table string
    """
    completed_tests = completed_tests or set()

    lines = [
        "| Test | Cost | Method | Status | Prerequisites | Description |",
        "|------|------|--------|--------|---------------|-------------|",
    ]

    for name, spec in AVAILABLE_TESTS.items():
        # Status indicator
        if name in completed_tests:
            avail = "[DONE]"
        elif spec.method_status == "unavailable":
            avail = "[BROKEN]"
        elif budget_remaining is not None and spec.cost > budget_remaining:
            avail = "[OVER BUDGET]"
        elif spec.method_status == "surrogate":
            avail = "[surrogate]"
        else:
            avail = "[available]"

        # Prerequisites
        prereqs = ", ".join(spec.prerequisites) if spec.prerequisites else "none"

        # Truncate description for table
        desc = spec.description[:50] + "..." if len(spec.description) > 50 else spec.description

        lines.append(
            f"| {name} | {spec.cost} | {spec.simulation_method} | {avail} | {prereqs} | {desc} |"
        )

    return "\n".join(lines)


def get_affordable_tests(
    budget_remaining: float,
    completed_tests: set[str] | None = None,
) -> list[TestSpec]:
    """Get tests that are affordable and have prerequisites met.

    Args:
        budget_remaining: Remaining budget
        completed_tests: Set of completed test names

    Returns:
        List of TestSpecs that can be run
    """
    completed_tests = completed_tests or set()
    affordable = []

    for name, spec in AVAILABLE_TESTS.items():
        if name in completed_tests:
            continue
        if spec.cost > budget_remaining:
            continue
        satisfied, _ = check_prerequisites(name, completed_tests)
        if satisfied:
            affordable.append(spec)

    return affordable


def estimate_total_cost(test_names: list[str]) -> float:
    """Estimate total cost for a sequence of tests.

    Args:
        test_names: List of test names

    Returns:
        Sum of test costs
    """
    return sum(get_test(name).cost for name in test_names)
