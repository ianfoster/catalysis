#!/usr/bin/env python3
"""Test Globus Compute functions for all 11 simulation codes.

Usage:
    # Test all functions (requires gc_functions.json from registration)
    python scripts/test_gc_functions.py --endpoint ENDPOINT_ID

    # Test specific function
    python scripts/test_gc_functions.py --endpoint ENDPOINT_ID --function ml_screening

    # Test with mock mode
    python scripts/test_gc_functions.py --endpoint ENDPOINT_ID --mock

    # Quick test (fast functions only)
    python scripts/test_gc_functions.py --endpoint ENDPOINT_ID --quick

Supported codes (11 total):
    1. OpenMM - Structure relaxation
    2. GROMACS - Molecular dynamics
    3. Quantum ESPRESSO - DFT calculations
    4. ASE - Structure manipulation (via other codes)
    5. pymatgen - Materials analysis (via other codes)
    6. MACE - ML potential screening
    7. CHGNet - ML potential calculations
    8. M3GNet - ML potential calculations
    9. GPAW - Python-based DFT
   10. CatMAP - Microkinetic modeling
   11. Cantera - Chemical kinetics
"""

import argparse
import json
import sys
import time
from pathlib import Path


# Test candidate for all simulations
TEST_CANDIDATE = {
    "support": "ZrO2",
    "metals": [
        {"element": "Cu", "wt_pct": 55},
        {"element": "Zn", "wt_pct": 30},
        {"element": "Al", "wt_pct": 15},
    ],
}


# Function descriptions and expected outputs
FUNCTION_TESTS = {
    # Core screening
    "fast_surrogate": {
        "description": "Fast ML surrogate for performance prediction",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["co2_conversion", "methanol_selectivity", "methanol_sty"],
        "quick": True,
    },
    "microkinetic_lite": {
        "description": "Simplified microkinetic analysis",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["RLS", "temp_sensitivity", "pressure_sensitivity"],
        "quick": True,
    },
    "stability_analysis": {
        "description": "Thermodynamic stability assessment",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["stability_score", "degradation_risk"],
        "quick": True,
    },

    # ML Potentials (MACE/CHGNet/M3GNet)
    "ml_screening": {
        "description": "MACE/CHGNet ML potential screening (Code 6/7)",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["total_energy_eV", "energy_per_atom_eV", "max_force_eV_A"],
        "quick": True,
        "codes": ["MACE", "CHGNet"],
    },
    "ml_relaxation": {
        "description": "ML potential structure relaxation (Code 6/7)",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["final_energy_eV", "converged", "n_steps"],
        "quick": False,
        "codes": ["MACE", "CHGNet"],
    },
    "ml_m3gnet": {
        "description": "M3GNet ML potential calculation (Code 8)",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["total_energy_eV", "energy_per_atom_eV"],
        "quick": True,
        "codes": ["M3GNet"],
    },

    # DFT
    "dft_adsorption": {
        "description": "DFT adsorption energy calculation (Code 3)",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["E_ads_CO2", "E_ads_H"],
        "quick": False,
        "codes": ["Quantum ESPRESSO", "VASP"],
    },
    "dft_gpaw": {
        "description": "GPAW DFT calculation (Code 9)",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["total_energy_eV", "energy_per_atom_eV"],
        "quick": False,
        "codes": ["GPAW"],
    },
    "dft_qe": {
        "description": "Quantum Espresso DFT via ASE (Code 3/4)",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["total_energy_eV"],
        "quick": False,
        "codes": ["Quantum ESPRESSO", "ASE"],
    },

    # Microkinetics
    "microkinetic_catmap": {
        "description": "CatMAP microkinetic modeling (Code 10)",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["RLS", "temp_sensitivity", "pressure_sensitivity"],
        "quick": True,
        "codes": ["CatMAP"],
    },

    # Cantera
    "cantera_reactor": {
        "description": "Cantera reactor simulation (Code 11)",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["conversion", "selectivity", "products"],
        "quick": False,
        "codes": ["Cantera"],
    },
    "cantera_sensitivity": {
        "description": "Cantera sensitivity analysis (Code 11)",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["sensitivities", "optimal_conditions"],
        "quick": True,
        "codes": ["Cantera"],
    },

    # MD
    "openmm_relaxation": {
        "description": "OpenMM structure relaxation (Code 1)",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["relaxed_energy", "structure_rmsd"],
        "quick": False,
        "codes": ["OpenMM"],
    },
    "gromacs_md": {
        "description": "GROMACS MD simulation (Code 2)",
        "payload": {"candidate": TEST_CANDIDATE},
        "expected_keys": ["avg_temperature_K", "final_rmsd_nm"],
        "quick": False,
        "codes": ["GROMACS"],
    },
}


def test_function(
    executor,
    func_name: str,
    func_id: str,
    payload: dict,
    timeout: int = 300,
) -> dict:
    """Test a single Globus Compute function."""
    from globus_compute_sdk import Executor

    print(f"\n{'='*60}")
    print(f"Testing: {func_name}")
    print(f"Function ID: {func_id}")
    print(f"{'='*60}")

    test_info = FUNCTION_TESTS.get(func_name, {})
    description = test_info.get("description", "No description")
    codes = test_info.get("codes", [])

    print(f"Description: {description}")
    if codes:
        print(f"Codes tested: {', '.join(codes)}")

    t0 = time.time()

    try:
        # Submit the function
        future = executor.submit_to_registered_function(func_id, args=(payload,))

        # Wait for result
        result = future.result(timeout=timeout)
        elapsed = time.time() - t0

        print(f"\nResult ({elapsed:.2f}s):")
        print(json.dumps(result, indent=2))

        # Check expected keys
        expected_keys = test_info.get("expected_keys", [])
        missing_keys = [k for k in expected_keys if k not in result]
        if missing_keys:
            print(f"\nWARNING: Missing expected keys: {missing_keys}")

        # Check for errors
        if result.get("ok") == False:
            print(f"\nERROR: Function returned error: {result.get('error', 'Unknown')}")
            return {"success": False, "error": result.get("error"), "elapsed": elapsed}

        return {"success": True, "result": result, "elapsed": elapsed}

    except TimeoutError:
        elapsed = time.time() - t0
        print(f"\nTIMEOUT after {elapsed:.2f}s")
        return {"success": False, "error": "Timeout", "elapsed": elapsed}

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\nEXCEPTION: {e}")
        return {"success": False, "error": str(e), "elapsed": elapsed}


def run_tests(
    endpoint_id: str,
    function_map: dict,
    functions_to_test: list | None = None,
    quick: bool = False,
    timeout: int = 300,
):
    """Run tests for specified functions."""
    from globus_compute_sdk import Executor

    if functions_to_test is None:
        functions_to_test = list(function_map.keys())

    if quick:
        # Filter to quick functions only
        functions_to_test = [
            f for f in functions_to_test
            if FUNCTION_TESTS.get(f, {}).get("quick", False)
        ]
        print(f"Quick mode: Testing {len(functions_to_test)} fast functions")

    results = {}

    with Executor(endpoint_id=endpoint_id) as executor:
        for func_name in functions_to_test:
            if func_name not in function_map:
                print(f"\nSkipping {func_name}: Not in function map")
                continue

            test_info = FUNCTION_TESTS.get(func_name, {})
            payload = test_info.get("payload", {"candidate": TEST_CANDIDATE})

            result = test_function(
                executor,
                func_name,
                function_map[func_name],
                payload,
                timeout=timeout,
            )
            results[func_name] = result

    return results


def print_summary(results: dict):
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    successful = [k for k, v in results.items() if v.get("success")]
    failed = [k for k, v in results.items() if not v.get("success")]

    print(f"\nTotal: {len(results)}")
    print(f"Passed: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\nSuccessful tests:")
        for name in successful:
            elapsed = results[name].get("elapsed", 0)
            method = results[name].get("result", {}).get("method", "unknown")
            print(f"  ✓ {name} ({elapsed:.2f}s, method={method})")

    if failed:
        print(f"\nFailed tests:")
        for name in failed:
            error = results[name].get("error", "Unknown")
            print(f"  ✗ {name}: {error}")

    # Code coverage
    print("\n" + "-"*40)
    print("CODE COVERAGE")
    print("-"*40)

    codes_tested = set()
    for name in successful:
        test_info = FUNCTION_TESTS.get(name, {})
        codes_tested.update(test_info.get("codes", []))

    all_codes = [
        "OpenMM", "GROMACS", "Quantum ESPRESSO", "ASE", "pymatgen",
        "MACE", "CHGNet", "M3GNet", "GPAW", "CatMAP", "Cantera"
    ]

    for i, code in enumerate(all_codes, 1):
        status = "✓" if code in codes_tested else "✗"
        print(f"  {i:2}. {status} {code}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Globus Compute functions for all 11 simulation codes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--endpoint", required=True, help="Globus Compute endpoint ID")
    parser.add_argument("--functions-file", default="gc_functions.json",
                        help="JSON file with function IDs")
    parser.add_argument("--function", "-f", action="append",
                        help="Specific function(s) to test (can specify multiple)")
    parser.add_argument("--mock", action="store_true",
                        help="Expect mock results")
    parser.add_argument("--quick", action="store_true",
                        help="Only run quick tests")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per function in seconds")
    parser.add_argument("--list", action="store_true",
                        help="List available functions and exit")

    args = parser.parse_args()

    if args.list:
        print("Available functions:\n")
        for name, info in FUNCTION_TESTS.items():
            codes = info.get("codes", [])
            quick = "✓" if info.get("quick") else " "
            print(f"  [{quick}] {name:25} - {info['description']}")
            if codes:
                print(f"       Codes: {', '.join(codes)}")
        return 0

    # Load function map
    functions_file = Path(args.functions_file)
    if not functions_file.exists():
        print(f"Error: Functions file not found: {functions_file}")
        print("Run register_gc_functions.py first to generate it.")
        return 1

    with open(functions_file) as f:
        data = json.load(f)

    function_map = data.get("functions", {})
    mode = data.get("mode", "unknown")

    print(f"Loaded {len(function_map)} functions from {functions_file}")
    print(f"Mode: {mode}")
    print(f"Endpoint: {args.endpoint}")

    # Run tests
    functions_to_test = args.function if args.function else None

    results = run_tests(
        endpoint_id=args.endpoint,
        function_map=function_map,
        functions_to_test=functions_to_test,
        quick=args.quick,
        timeout=args.timeout,
    )

    # Print summary
    print_summary(results)

    # Return exit code
    failed = [k for k, v in results.items() if not v.get("success")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
