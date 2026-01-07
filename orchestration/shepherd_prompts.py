"""Prompt templates for ShepherdAgent LLM reasoning."""

from __future__ import annotations

import json
from typing import Any

from orchestration.test_registry import format_tests_for_prompt
from orchestration.capabilities import get_capabilities_prompt, load_capabilities


def get_system_prompt(include_capabilities: bool = True) -> str:
    """Get the system prompt, optionally including capabilities context.

    Args:
        include_capabilities: Whether to include simulation capabilities info.

    Returns:
        System prompt string.
    """
    base = """\
You are a materials scientist AI assistant evaluating catalyst candidates for CO2-to-methanol conversion.

Your task is to decide which characterization tests to run on a catalyst candidate to determine its viability. You must balance:
- Information value: Which test would most reduce uncertainty?
- Cost efficiency: More expensive tests should only run if cheaper ones suggest promise
- Prerequisites: Some tests require others to complete first

Always respond with valid JSON only. Do not include any text outside the JSON object."""

    if include_capabilities:
        capabilities_info = get_capabilities_prompt()
        return f"{base}\n\n{capabilities_info}"

    return base


# Legacy constant for backwards compatibility
SYSTEM_PROMPT = get_system_prompt(include_capabilities=False)


REASONING_PROMPT = """\
## Candidate
```json
{candidate_json}
```

## VALID TEST NAMES (you MUST use one of these exact names)
{valid_test_names}

## Available Tests
{tests_table}

## Tests Already Completed
{completed_list}

## Results So Far
{results_section}

## Budget
- Total budget: {budget_total} units
- Spent so far: {budget_spent} units
- Remaining: {budget_remaining} units

## Task
Based on the candidate properties and test results so far, decide what to do next.

Options:
1. Run another test to gain more information (if budget allows and prerequisites met)
2. Stop and provide a final assessment (if confident enough or budget exhausted)

IMPORTANT RULES:
- You MUST use an exact test name from the VALID TEST NAMES list above
- DO NOT re-run tests listed in "Tests Already Completed"
- Each test can only be run ONCE per candidate
- If all affordable tests are [DONE], you MUST choose action "stop"
- NEVER invent or modify test names - use ONLY the exact names listed

Consider:
- Have you run the minimum tests needed to assess this candidate?
- Would additional tests meaningfully change your assessment?
- Are there any red flags that suggest stopping early?

Respond with JSON in this exact format:
```json
{{
  "action": "test",
  "test": "test_name_here",
  "reasoning": "Brief explanation of why this test is most valuable next"
}}
```

OR if you have enough information or budget is exhausted:
```json
{{
  "action": "stop",
  "reasoning": "Brief explanation of why stopping now is appropriate",
  "confidence": 0.85
}}
```

The confidence value (0.0-1.0) indicates how confident you are in your assessment of this candidate."""


FINAL_ASSESSMENT_PROMPT = """\
## Candidate
```json
{candidate_json}
```

## Test Results
{results_json}

## Total Cost
{total_cost} units spent

## Task
Based on all test results for this catalyst candidate, provide a final assessment.

Evaluate:
1. Overall viability for CO2-to-methanol conversion
2. Key strengths (what makes this candidate promising)
3. Key concerns (potential issues or limitations)
4. Recommendation for next steps

Respond with JSON in this exact format:
```json
{{
  "viability_score": 75,
  "strengths": [
    "High selectivity predicted",
    "Stable under reaction conditions"
  ],
  "concerns": [
    "Moderate conversion rate",
    "High uncertainty in DFT results"
  ],
  "recommendation": "PURSUE",
  "summary": "One-sentence summary of the assessment"
}}
```

The viability_score is 0-100 (higher is better).
The recommendation must be one of: PURSUE, DEPRIORITIZE, or REJECT."""


def format_candidate(candidate: dict[str, Any]) -> str:
    """Format candidate dict as JSON string for prompt."""
    return json.dumps(candidate, indent=2)


def format_results(results: list[dict[str, Any]]) -> str:
    """Format test results for prompt.

    Args:
        results: List of test result dicts with 'test', 'result', 'cost' keys

    Returns:
        Formatted string for prompt
    """
    if not results:
        return "No tests have been run yet."

    lines = []
    for r in results:
        test_name = r.get("test", "unknown")
        cost = r.get("cost", 0)
        result_data = r.get("result", {})

        lines.append(f"### {test_name} (cost: {cost})")
        lines.append("```json")
        lines.append(json.dumps(result_data, indent=2))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def build_reasoning_prompt(
    candidate: dict[str, Any],
    results: list[dict[str, Any]],
    budget_total: float,
    budget_spent: float,
    only_tests: set[str] | None = None,
) -> str:
    """Build the reasoning prompt for next action decision.

    Args:
        candidate: Candidate specification dict
        results: List of completed test results
        budget_total: Total budget allocated
        budget_spent: Budget spent so far
        only_tests: If provided, only show these tests (for phase 2 slow-only)

    Returns:
        Formatted prompt string
    """
    from orchestration.test_registry import AVAILABLE_TESTS

    completed_tests = {r["test"] for r in results if r.get("result", {}).get("ok", True)}
    budget_remaining = budget_total - budget_spent

    # Build explicit list of completed tests
    if completed_tests:
        completed_list = ", ".join(sorted(completed_tests)) + " (DO NOT re-run these)"
    else:
        completed_list = "None yet"

    # Build explicit list of valid test names (filtered if only_tests specified)
    if only_tests:
        valid_test_names = ", ".join(sorted(only_tests))
    else:
        valid_test_names = ", ".join(sorted(AVAILABLE_TESTS.keys()))

    return REASONING_PROMPT.format(
        candidate_json=format_candidate(candidate),
        valid_test_names=valid_test_names,
        tests_table=format_tests_for_prompt(completed_tests, budget_remaining, only_tests),
        completed_list=completed_list,
        results_section=format_results(results),
        budget_total=budget_total,
        budget_spent=budget_spent,
        budget_remaining=budget_remaining,
    )


def build_final_assessment_prompt(
    candidate: dict[str, Any],
    results: list[dict[str, Any]],
    total_cost: float,
) -> str:
    """Build the final assessment prompt.

    Args:
        candidate: Candidate specification dict
        results: List of all test results
        total_cost: Total cost spent on this candidate

    Returns:
        Formatted prompt string
    """
    return FINAL_ASSESSMENT_PROMPT.format(
        candidate_json=format_candidate(candidate),
        results_json=format_results(results),
        total_cost=total_cost,
    )


# Response validation schemas (for documentation; actual validation in shepherd.py)
REASONING_RESPONSE_SCHEMA = {
    "type": "object",
    "oneOf": [
        {
            "properties": {
                "action": {"const": "test"},
                "test": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["action", "test", "reasoning"],
        },
        {
            "properties": {
                "action": {"const": "stop"},
                "reasoning": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["action", "reasoning", "confidence"],
        },
    ],
}

ASSESSMENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "viability_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "concerns": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"enum": ["PURSUE", "DEPRIORITIZE", "REJECT"]},
        "summary": {"type": "string"},
    },
    "required": ["viability_score", "strengths", "concerns", "recommendation", "summary"],
}
