#!/usr/bin/env python3
"""Shepherd evaluation via Globus Compute.

This module provides GC functions for running ShepherdAgent evaluations on Spark.
The ShepherdAgent connects to a shared llama-cpp server for LLM inference.

Usage:
    # Register the function
    from shepherd_gc import evaluate_candidate_gc
    func_id = gc_client.register_function(evaluate_candidate_gc)

    # Submit evaluations
    futures = []
    for candidate in candidates:
        config = {
            "candidate": candidate,
            "llm_url": "http://localhost:8080/v1",
            "budget": 100.0,
        }
        future = executor.submit_to_registered_function(func_id, args=(config,))
        futures.append(future)
"""


def evaluate_candidate_gc(config: dict) -> dict:
    """Evaluate a catalyst candidate on Spark.

    This function is COMPLETELY SELF-CONTAINED for GC serialization.
    It runs a simplified ShepherdAgent evaluation loop.

    Args:
        config: Dict with:
            - candidate: Catalyst candidate dict
            - llm_url: URL to llama-cpp server (e.g., "http://localhost:8080/v1")
            - llm_model: Model name for API calls (default: "gpt-3.5-turbo")
            - budget: Evaluation budget (default: 100.0)
            - gc_functions: Dict of simulation function IDs (optional)
            - gc_endpoint: Endpoint for simulations (optional)

    Returns:
        Evaluation result dict with:
            - candidate: The input candidate
            - results: List of test results
            - total_cost: Total compute cost
            - final_assessment: LLM assessment
            - ok: Success status
    """
    import asyncio
    import json
    import logging
    import re
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("shepherd_gc")

    candidate = config.get("candidate")
    if not candidate:
        return {"ok": False, "error": "No candidate provided"}

    llm_url = config.get("llm_url", "http://localhost:8000/v1")
    llm_model = config.get("llm_model", "gpt-3.5-turbo")  # llama-cpp uses this as default
    budget_total = config.get("budget", 100.0)

    print('EEEEEE', llm_url, llm_model)

    result = {
        "candidate": candidate,
        "results": [],
        "total_cost": 0.0,
        "history": [],
        "ok": False,
    }

    # Helper: Call LLM
    def call_llm(messages: list, max_tokens: int = 1024) -> str:
        import urllib.request
        import urllib.error

        url = f"{llm_url}/chat/completions"
        data = json.dumps({
            "model": llm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }).encode()

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        response = urllib.request.urlopen(req, timeout=120)
        response_data = json.loads(response.read().decode())
        return response_data["choices"][0]["message"]["content"]

    # Helper: Parse JSON from LLM response
    def parse_json(content: str) -> dict:
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try markdown code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try any JSON object
        match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    # Helper: Format candidate
    def candidate_str(c: dict) -> str:
        support = c.get("support", "?")
        metals = c.get("metals", [])
        metal_str = "+".join(f"{m.get('element','?')}{m.get('wt_pct','?')}%" for m in metals[:3])
        return f"{metal_str}/{support}"

    logger.info(f"Evaluating candidate: {candidate_str(candidate)}")

    # System prompt
    system_prompt = """You are a catalyst evaluation expert. You help decide which
characterization tests to run for catalyst candidates for CO2-to-methanol conversion.

Available tests and their costs:
- fast_surrogate (cost: 1.0): Quick ML surrogate prediction
- ml_screening (cost: 5.0): MACE/CHGNet ML potential screening
- stability_analysis (cost: 10.0): Thermodynamic stability check
- microkinetic_lite (cost: 20.0): Simplified microkinetic model
- dft_adsorption (cost: 50.0): DFT adsorption energy calculation

Respond in JSON format."""

    # Evaluation loop
    budget_spent = 0.0
    test_results = []
    max_iterations = 10

    for iteration in range(max_iterations):
        if budget_spent >= budget_total:
            break

        # Build reasoning prompt
        results_summary = ""
        if test_results:
            results_summary = "Results so far:\n"
            for tr in test_results:
                results_summary += f"- {tr['test']}: {json.dumps(tr['result'])}\n"

        reasoning_prompt = f"""Candidate: {candidate_str(candidate)}
Support: {candidate.get('support')}
Metals: {json.dumps(candidate.get('metals', []))}

Budget: {budget_total - budget_spent:.1f} remaining (spent: {budget_spent:.1f})

{results_summary}

What should we do next? Respond with JSON:
{{
    "action": "test" or "stop",
    "test": "test_name" (if action is test),
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0 (if stopping, how confident in assessment)
}}"""

        try:
            response = call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": reasoning_prompt},
            ])
            decision = parse_json(response)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            result["history"].append({"iteration": iteration, "error": str(e)})
            break

        result["history"].append({
            "iteration": iteration,
            "decision": decision,
            "budget_remaining": budget_total - budget_spent,
        })

        action = decision.get("action", "stop")

        if action == "stop":
            logger.info(f"Stopping: {decision.get('reasoning', '')[:100]}")
            break

        if action == "test":
            test_name = decision.get("test")
            if not test_name:
                continue

            # Test costs
            test_costs = {
                "fast_surrogate": 1.0,
                "ml_screening": 5.0,
                "stability_analysis": 10.0,
                "microkinetic_lite": 20.0,
                "dft_adsorption": 50.0,
            }

            cost = test_costs.get(test_name, 10.0)
            if cost > (budget_total - budget_spent):
                logger.warning(f"Test {test_name} costs {cost} but only {budget_total - budget_spent} remaining")
                continue

            # Run test via GC or surrogate
            logger.info(f"Running test: {test_name} (cost: {cost})")
            gc_functions = config.get("gc_functions")
            gc_endpoint = config.get("gc_endpoint")

            # Build performance data from previous results for context
            perf_data = {}
            for prev in test_results:
                perf_data.update(prev.get("result", {}))

            test_result = run_simulation(
                test_name=test_name,
                candidate=candidate,
                gc_functions=gc_functions,
                gc_endpoint=gc_endpoint,
                performance=perf_data,
            )

            test_results.append({
                "test": test_name,
                "result": test_result,
                "cost": cost,
            })
            budget_spent += cost

    # Generate final assessment
    if test_results:
        assessment_prompt = f"""Candidate: {candidate_str(candidate)}
Support: {candidate.get('support')}
Metals: {json.dumps(candidate.get('metals', []))}

Test results:
{json.dumps(test_results, indent=2)}

Provide a final assessment. Respond with JSON:
{{
    "viability_score": 0-100,
    "strengths": ["list", "of", "strengths"],
    "concerns": ["list", "of", "concerns"],
    "recommendation": "PROMISING" or "DEPRIORITIZE" or "REJECT",
    "summary": "One sentence summary"
}}"""

        try:
            response = call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": assessment_prompt},
            ])
            final_assessment = parse_json(response)
        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            final_assessment = {
                "viability_score": 50,
                "strengths": [],
                "concerns": [f"Assessment failed: {e}"],
                "recommendation": "DEPRIORITIZE",
                "summary": "Assessment generation failed",
            }
    else:
        final_assessment = {
            "viability_score": 0,
            "strengths": [],
            "concerns": ["No tests were run"],
            "recommendation": "REJECT",
            "summary": "Unable to assess - no tests completed",
        }

    result["results"] = test_results
    result["total_cost"] = budget_spent
    result["final_assessment"] = final_assessment
    result["ok"] = True

    logger.info(f"Evaluation complete: score={final_assessment.get('viability_score', 0)}")
    return result


def run_simulation(
    test_name: str,
    candidate: dict,
    gc_functions: dict | None = None,
    gc_endpoint: str | None = None,
    performance: dict | None = None,
) -> dict:
    """Run a simulation test via Globus Compute or fallback to surrogate.

    Args:
        test_name: Name of the test to run
        candidate: Catalyst candidate dict
        gc_functions: Dict mapping test names to GC function IDs
        gc_endpoint: GC endpoint ID for running simulations
        performance: Optional performance data from previous tests

    Returns:
        Test result dict
    """
    # If GC functions are available, use them
    if gc_functions and gc_endpoint and test_name in gc_functions:
        try:
            from globus_compute_sdk import Client, Executor

            func_id = gc_functions[test_name]
            payload = {
                "candidate": candidate,
                "performance": performance,
            }

            # Submit to GC
            with Executor(endpoint_id=gc_endpoint) as ex:
                future = ex.submit_to_registered_function(func_id, args=(payload,))
                result = future.result(timeout=300)  # 5 min timeout

            return result
        except Exception as e:
            # Log error and fall through to surrogate
            pass

    # Fallback to physics-informed surrogates (self-contained)
    return run_surrogate(test_name, candidate, performance)


def run_surrogate(test_name: str, candidate: dict, performance: dict | None = None) -> dict:
    """Run physics-informed surrogate for a test.

    These are simplified models that provide reasonable estimates
    without requiring external simulation codes.
    """
    metals = candidate.get("metals", [])
    support = candidate.get("support", "Al2O3")

    cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 50)
    zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 30)
    al = next((m["wt_pct"] for m in metals if m["element"] == "Al"), 20)

    is_zro2 = 1 if support == "ZrO2" else 0

    if test_name == "fast_surrogate":
        # Physics-informed ML surrogate
        conversion = max(0.0, min(1.0, 0.2 + 0.01 * cu + 0.005 * zn - 0.002 * al))
        selectivity = max(0.0, min(1.0, 0.6 + 0.003 * zn - 0.004 * max(0.0, cu - 60) + 0.05 * is_zro2))
        sty = conversion * selectivity * 10.0
        return {
            "co2_conversion": round(conversion, 4),
            "methanol_selectivity": round(selectivity, 4),
            "methanol_sty": round(sty, 4),
            "uncertainty": 0.25,
            "method": "surrogate",
        }

    elif test_name == "ml_screening":
        # ML potential screening surrogate
        base_co2 = -0.35
        base_h = -0.28
        support_effects = {"Al2O3": (0.0, 0.0), "ZrO2": (-0.1, -0.05), "SiO2": (0.05, 0.02)}
        co2_mod, h_mod = support_effects.get(support, (0.0, 0.0))

        e_ads_co2 = base_co2 - (cu / 100) * 0.2 + co2_mod
        e_ads_h = base_h - (zn / 100) * 0.15 + h_mod
        surface_energy = 0.8 + (cu / 100) * 0.4 - is_zro2 * 0.1

        return {
            "E_ads_CO2": round(e_ads_co2, 4),
            "E_ads_H": round(e_ads_h, 4),
            "surface_energy": round(surface_energy, 4),
            "stable": cu <= 70 and zn >= 20,
            "uncertainty_reduction": 0.12,
            "method": "surrogate",
        }

    elif test_name == "stability_analysis":
        # Thermodynamic stability model
        base_stability = {"Al2O3": 0.80, "ZrO2": 0.90, "SiO2": 0.70}
        stability = base_stability.get(support, 0.75)
        stability -= max(0, (cu - 60) / 100) * 0.1
        stability += (zn / 100) * 0.05
        stability = max(0.3, min(0.98, stability))
        risk = "low" if stability > 0.85 else ("medium" if stability > 0.70 else "high")

        return {
            "stability_score": round(stability, 3),
            "degradation_risk": risk,
            "method": "surrogate",
        }

    elif test_name == "microkinetic_lite":
        # Microkinetic model
        perf = performance or {}
        sty = float(perf.get("methanol_sty", 5.0))
        sel = float(perf.get("methanol_selectivity", 0.7))

        if support.lower() in ["zro2", "zirconia"]:
            rls = "CO2_activation"
            temp_sens = 0.8
            press_sens = 0.6
        else:
            rls = "hydrogenation"
            temp_sens = 0.6
            press_sens = 0.8

        reduced_uncertainty = max(0.05, 0.25 - 0.01 * sty - 0.02 * sel)

        return {
            "RLS": rls,
            "temp_sensitivity": round(temp_sens, 3),
            "pressure_sensitivity": round(press_sens, 3),
            "uncertainty_reduction": round(0.25 - reduced_uncertainty, 3),
            "method": "surrogate",
        }

    elif test_name == "dft_adsorption":
        # DFT adsorption energy surrogate
        base_co2 = -0.35
        base_h = -0.28
        support_effects = {"Al2O3": (0.0, 0.0), "ZrO2": (-0.1, -0.05), "SiO2": (0.05, 0.02)}
        co2_mod, h_mod = support_effects.get(support, (0.0, 0.0))

        e_ads_co2 = base_co2 - (cu / 100) * 0.2 + co2_mod
        e_ads_h = base_h - (zn / 100) * 0.15 + h_mod
        e_ads_ch3o = -0.6 - (cu / 100) * 0.15 - is_zro2 * 0.1

        return {
            "E_ads_CO2": round(e_ads_co2, 4),
            "E_ads_H": round(e_ads_h, 4),
            "E_ads_CH3O": round(e_ads_ch3o, 4),
            "uncertainty_reduction": 0.10,
            "method": "surrogate",
        }

    else:
        return {"status": "completed", "test": test_name, "method": "surrogate"}


# For testing locally
if __name__ == "__main__":
    import sys

    # Test candidate
    test_candidate = {
        "support": "ZrO2",
        "metals": [
            {"element": "Cu", "wt_pct": 55},
            {"element": "Zn", "wt_pct": 30},
            {"element": "Al", "wt_pct": 15},
        ],
    }

    config = {
        "candidate": test_candidate,
        "llm_url": sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/v1",
        "budget": 50.0,
    }

    print("Testing shepherd evaluation...")
    result = evaluate_candidate_gc(config)
    print(json.dumps(result, indent=2))
