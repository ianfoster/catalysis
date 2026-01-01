"""Test registry for ShepherdAgent.

Defines available characterization tests with metadata for LLM reasoning.
Includes RuntimeTracker for adaptive test classification based on observed runtimes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


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
        prerequisites=(),  # Removed temporarily
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
        prerequisites=(),  # Removed temporarily
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
        cost=10.0,  # Reduced from 100 for testing
        endpoint="gpu",
        prerequisites=(),  # Removed temporarily
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
        prerequisites=(),  # Removed temporarily
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
        prerequisites=(),  # Removed temporarily
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
        prerequisites=(),  # Removed temporarily
        timeout=120,
        outputs=("conversion", "selectivity", "products"),
        reduces_uncertainty=True,
        expected_runtime_s=30,
        gc_function="cantera_reactor",
        simulation_method="cantera",
        method_status="surrogate",
    ),
}

# Cost threshold for cheap vs expensive tests
CHEAP_TEST_THRESHOLD = 2.0


def get_cheap_tests() -> list[TestSpec]:
    """Get all tests below the cheap threshold."""
    return [t for t in AVAILABLE_TESTS.values() if t.cost < CHEAP_TEST_THRESHOLD]


def get_expensive_tests() -> list[TestSpec]:
    """Get all tests at or above the cheap threshold."""
    return [t for t in AVAILABLE_TESTS.values() if t.cost >= CHEAP_TEST_THRESHOLD]


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


# ============================================================================
# Runtime Tracking for Adaptive Classification
# ============================================================================

# Default threshold for "fast" tests (seconds)
FAST_TEST_THRESHOLD_S = 30.0

# Maximum number of observations to keep per test
MAX_RUNTIME_OBSERVATIONS = 100


class RuntimeTracker:
    """Tracks observed test runtimes for adaptive classification.

    Stores runtime observations in Redis for distributed visibility.
    Falls back to in-memory storage if Redis unavailable.

    Usage:
        tracker = RuntimeTracker(redis_host="spark")
        tracker.record("ml_screening", 8.5)
        avg = tracker.get_average("ml_screening")
        fast_tests = tracker.get_fast_tests(threshold_s=30.0)
    """

    REDIS_KEY_PREFIX = "test_runtime:"

    def __init__(
        self,
        redis_host: str | None = None,
        redis_port: int = 6379,
    ):
        """Initialize RuntimeTracker.

        Args:
            redis_host: Redis hostname. If None, uses in-memory storage.
            redis_port: Redis port (default 6379).
        """
        self._redis = None
        self._local_cache: dict[str, list[float]] = {}  # Fallback storage

        if redis_host:
            try:
                import redis
                self._redis = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True,
                )
                self._redis.ping()
                logger.info("RuntimeTracker connected to Redis at %s:%d", redis_host, redis_port)
            except Exception as e:
                logger.warning("RuntimeTracker: Redis unavailable (%s), using local cache", e)
                self._redis = None

    def record(self, test_name: str, runtime_s: float) -> None:
        """Record an observed runtime for a test.

        Args:
            test_name: Name of the test
            runtime_s: Observed runtime in seconds
        """
        if self._redis:
            try:
                key = f"{self.REDIS_KEY_PREFIX}{test_name}"
                self._redis.lpush(key, runtime_s)
                self._redis.ltrim(key, 0, MAX_RUNTIME_OBSERVATIONS - 1)
            except Exception as e:
                logger.warning("Failed to record runtime to Redis: %s", e)
                self._record_local(test_name, runtime_s)
        else:
            self._record_local(test_name, runtime_s)

    def _record_local(self, test_name: str, runtime_s: float) -> None:
        """Record runtime to local cache."""
        if test_name not in self._local_cache:
            self._local_cache[test_name] = []
        self._local_cache[test_name].insert(0, runtime_s)
        self._local_cache[test_name] = self._local_cache[test_name][:MAX_RUNTIME_OBSERVATIONS]

    def get_average(self, test_name: str) -> float | None:
        """Get average observed runtime for a test.

        Args:
            test_name: Name of the test

        Returns:
            Average runtime in seconds, or None if no observations
        """
        observations = self._get_observations(test_name)
        if not observations:
            return None
        return sum(observations) / len(observations)

    def get_observations(self, test_name: str) -> list[float]:
        """Get all observed runtimes for a test.

        Args:
            test_name: Name of the test

        Returns:
            List of observed runtimes (most recent first)
        """
        return self._get_observations(test_name)

    def _get_observations(self, test_name: str) -> list[float]:
        """Internal method to fetch observations."""
        if self._redis:
            try:
                key = f"{self.REDIS_KEY_PREFIX}{test_name}"
                values = self._redis.lrange(key, 0, -1)
                return [float(v) for v in values]
            except Exception as e:
                logger.warning("Failed to get observations from Redis: %s", e)
                return self._local_cache.get(test_name, [])
        else:
            return self._local_cache.get(test_name, [])

    def get_estimated_runtime(self, test_name: str) -> float:
        """Get estimated runtime for a test.

        Uses observed average if available, falls back to expected_runtime_s from TestSpec.

        Args:
            test_name: Name of the test

        Returns:
            Estimated runtime in seconds
        """
        observed = self.get_average(test_name)
        if observed is not None:
            return observed

        # Fall back to expected runtime from spec
        try:
            spec = get_test(test_name)
            return float(spec.expected_runtime_s)
        except KeyError:
            return FAST_TEST_THRESHOLD_S  # Default to threshold if unknown

    def get_fast_tests(self, threshold_s: float = FAST_TEST_THRESHOLD_S) -> list[TestSpec]:
        """Get tests classified as "fast" based on observed or expected runtime.

        Args:
            threshold_s: Runtime threshold in seconds (default 30s)

        Returns:
            List of TestSpecs for fast tests
        """
        fast = []
        for name, spec in AVAILABLE_TESTS.items():
            runtime = self.get_estimated_runtime(name)
            if runtime < threshold_s:
                fast.append(spec)
        return fast

    def get_slow_tests(self, threshold_s: float = FAST_TEST_THRESHOLD_S) -> list[TestSpec]:
        """Get tests classified as "slow" based on observed or expected runtime.

        Args:
            threshold_s: Runtime threshold in seconds (default 30s)

        Returns:
            List of TestSpecs for slow tests
        """
        slow = []
        for name, spec in AVAILABLE_TESTS.items():
            runtime = self.get_estimated_runtime(name)
            if runtime >= threshold_s:
                slow.append(spec)
        return slow

    def get_classification_summary(self, threshold_s: float = FAST_TEST_THRESHOLD_S) -> dict[str, Any]:
        """Get summary of test classifications with runtime data.

        Args:
            threshold_s: Runtime threshold in seconds

        Returns:
            Dict with fast/slow tests and their runtime info
        """
        summary = {
            "threshold_s": threshold_s,
            "fast": [],
            "slow": [],
        }

        for name, spec in AVAILABLE_TESTS.items():
            observed = self.get_average(name)
            estimated = self.get_estimated_runtime(name)
            n_obs = len(self._get_observations(name))

            entry = {
                "name": name,
                "observed_avg_s": round(observed, 2) if observed else None,
                "expected_s": spec.expected_runtime_s,
                "estimated_s": round(estimated, 2),
                "n_observations": n_obs,
                "source": "observed" if observed else "expected",
            }

            if estimated < threshold_s:
                summary["fast"].append(entry)
            else:
                summary["slow"].append(entry)

        return summary


# Global runtime tracker instance
_runtime_tracker: RuntimeTracker | None = None


def get_runtime_tracker(
    redis_host: str | None = None,
    redis_port: int = 6379,
) -> RuntimeTracker:
    """Get or create the global runtime tracker.

    Args:
        redis_host: Redis hostname (only used on first call)
        redis_port: Redis port (only used on first call)

    Returns:
        RuntimeTracker instance
    """
    global _runtime_tracker
    if _runtime_tracker is None:
        _runtime_tracker = RuntimeTracker(redis_host=redis_host, redis_port=redis_port)
    return _runtime_tracker


def reset_runtime_tracker(
    redis_host: str | None = None,
    redis_port: int = 6379,
) -> RuntimeTracker:
    """Reset and return a fresh runtime tracker.

    Args:
        redis_host: Redis hostname
        redis_port: Redis port

    Returns:
        New RuntimeTracker instance
    """
    global _runtime_tracker
    _runtime_tracker = RuntimeTracker(redis_host=redis_host, redis_port=redis_port)
    return _runtime_tracker
