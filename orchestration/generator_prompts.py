"""Prompt templates for GeneratorAgent LLM reasoning."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """\
You are a materials scientist AI assistant designing catalyst candidates for CO2-to-methanol conversion.

Your task is to propose promising catalyst compositions based on evaluation history. You must balance:
- Exploitation: variations on top performers to refine promising regions
- Exploration: new compositions to discover unexplored regions
- Constraints: chemistry validity (weight sum = 100%, valid supports)

Always respond with valid JSON only. Do not include any text outside the JSON object."""


PROPOSAL_PROMPT = """\
## Goal
Propose {n_candidates} new catalyst candidates for CO2-to-methanol conversion.

## Top Performers So Far
{top_performers_table}

## Evaluation History Summary
- Iterations completed: {iteration}
- Total candidates evaluated: {total_evaluated}
- Best viability score: {best_score}/100
- Best candidate: {best_candidate}

## Patterns Observed
{patterns_summary}

## Constraints
- Metals: Cu, Zn, Al, Pd, Pt, Ni, Fe, Co (weight percentages must sum to 100%)
- Supports: Al2O3, ZrO2, SiO2, TiO2, CeO2, MgO, ZnO
- Each metal wt_pct must be between 0 and 100
- Avoid duplicates of previously evaluated candidates

## Previously Evaluated (avoid these)
{seen_candidates_sample}

## Task
Propose {n_candidates} diverse, promising candidates. Consider:
1. Exploiting: variations on top performers (small wt% adjustments)
2. Exploring: new regions not yet tested (different ratios, supports)
3. Balance exploration vs exploitation based on iteration number

Respond with JSON in this exact format:
```json
{{
  "candidates": [
    {{
      "support": "Al2O3",
      "metals": [
        {{"element": "Cu", "wt_pct": 60}},
        {{"element": "Zn", "wt_pct": 25}},
        {{"element": "Al", "wt_pct": 15}}
      ]
    }}
  ],
  "reasoning": "Brief explanation of proposal strategy"
}}
```
"""


CONVERGENCE_PROMPT = """\
## Evaluation History
{history_summary}

## Score Progression
{score_progression}

## Recent Progress
- Last {n_recent} iterations best scores: {recent_scores}
- Best score overall: {best_score}
- Score improvement (last vs previous): {improvement}

## Task
Based on the evaluation history, should we:
1. CONTINUE - there are still promising regions to explore
2. STOP - we have converged or are seeing diminishing returns

Consider:
- Is the score still improving?
- Have we explored diverse regions of the search space?
- Are there obvious unexplored combinations?

Respond with JSON in this exact format:
```json
{{
  "decision": "CONTINUE",
  "reasoning": "Brief explanation of your recommendation",
  "confidence": 0.85
}}
```

The confidence value (0.0-1.0) indicates how confident you are in the decision.
"""


def format_candidate(candidate: dict[str, Any]) -> str:
    """Format a single candidate as concise string."""
    support = candidate.get("support", "?")
    metals = candidate.get("metals", [])
    metal_str = "/".join(
        f"{m.get('element', '?')}{m.get('wt_pct', 0):.0f}"
        for m in metals
    )
    return f"{metal_str} on {support}"


def format_top_performers_table(performers: list[dict[str, Any]]) -> str:
    """Format top performers as markdown table."""
    if not performers:
        return "No candidates evaluated yet."

    lines = [
        "| Rank | Candidate | Score | Recommendation |",
        "|------|-----------|-------|----------------|",
    ]

    for i, p in enumerate(performers[:10], 1):
        candidate = p.get("candidate", {})
        assessment = p.get("final_assessment", {})
        score = assessment.get("viability_score", 0)
        rec = assessment.get("recommendation", "N/A")
        lines.append(f"| {i} | {format_candidate(candidate)} | {score} | {rec} |")

    return "\n".join(lines)


def format_patterns_summary(performers: list[dict[str, Any]]) -> str:
    """Analyze patterns in top performers."""
    if len(performers) < 3:
        return "Not enough data to identify patterns yet."

    # Analyze support distribution
    supports = {}
    cu_range = []
    zn_range = []

    for p in performers[:10]:
        candidate = p.get("candidate", {})
        support = candidate.get("support", "unknown")
        supports[support] = supports.get(support, 0) + 1

        for m in candidate.get("metals", []):
            if m.get("element") == "Cu":
                cu_range.append(m.get("wt_pct", 0))
            elif m.get("element") == "Zn":
                zn_range.append(m.get("wt_pct", 0))

    lines = []

    # Support preference
    if supports:
        best_support = max(supports, key=supports.get)
        lines.append(f"- Best support: {best_support} ({supports[best_support]}/10 top performers)")

    # Cu range
    if cu_range:
        lines.append(f"- Cu wt% in top performers: {min(cu_range):.0f}-{max(cu_range):.0f}%")

    # Zn range
    if zn_range:
        lines.append(f"- Zn wt% in top performers: {min(zn_range):.0f}-{max(zn_range):.0f}%")

    return "\n".join(lines) if lines else "Patterns unclear - more exploration needed."


def format_seen_candidates_sample(seen: set[str], max_show: int = 10) -> str:
    """Show sample of seen candidates."""
    if not seen:
        return "None evaluated yet."

    sample = list(seen)[:max_show]
    if len(seen) > max_show:
        return "\n".join(sample) + f"\n... and {len(seen) - max_show} more"
    return "\n".join(sample)


def format_score_progression(score_history: list[float]) -> str:
    """Format score history as simple chart."""
    if not score_history:
        return "No scores recorded yet."

    lines = ["```"]
    max_score = max(score_history) if score_history else 100
    for i, score in enumerate(score_history):
        bar_len = int((score / max_score) * 30) if max_score > 0 else 0
        bar = "#" * bar_len
        lines.append(f"Iter {i+1}: {bar} {score:.1f}")
    lines.append("```")

    return "\n".join(lines)


def format_history_summary(
    iteration: int,
    total_evaluated: int,
    best_score: float,
    score_history: list[float],
) -> str:
    """Format overall history summary."""
    lines = [
        f"- Iterations completed: {iteration}",
        f"- Total candidates evaluated: {total_evaluated}",
        f"- Best score achieved: {best_score:.1f}/100",
    ]

    if len(score_history) >= 2:
        improvement = score_history[-1] - score_history[-2]
        lines.append(f"- Last iteration improvement: {improvement:+.2f}")

    return "\n".join(lines)


def build_proposal_prompt(
    n_candidates: int,
    iteration: int,
    total_evaluated: int,
    best_score: float,
    best_candidate: dict[str, Any] | None,
    top_performers: list[dict[str, Any]],
    seen_candidates: set[str],
) -> str:
    """Build the proposal prompt for candidate generation.

    Args:
        n_candidates: Number of candidates to propose
        iteration: Current iteration number
        total_evaluated: Total candidates evaluated so far
        best_score: Best viability score achieved
        best_candidate: Best candidate found
        top_performers: List of top performing candidates with assessments
        seen_candidates: Set of candidate hashes/descriptions already evaluated

    Returns:
        Formatted prompt string
    """
    return PROPOSAL_PROMPT.format(
        n_candidates=n_candidates,
        iteration=iteration,
        total_evaluated=total_evaluated,
        best_score=best_score,
        best_candidate=format_candidate(best_candidate) if best_candidate else "None yet",
        top_performers_table=format_top_performers_table(top_performers),
        patterns_summary=format_patterns_summary(top_performers),
        seen_candidates_sample=format_seen_candidates_sample(seen_candidates),
    )


def build_convergence_prompt(
    iteration: int,
    total_evaluated: int,
    best_score: float,
    score_history: list[float],
    n_recent: int = 3,
) -> str:
    """Build the convergence judgment prompt.

    Args:
        iteration: Current iteration number
        total_evaluated: Total candidates evaluated
        best_score: Best score achieved
        score_history: List of best scores per iteration
        n_recent: Number of recent iterations to highlight

    Returns:
        Formatted prompt string
    """
    recent_scores = score_history[-n_recent:] if score_history else []
    recent_str = ", ".join(f"{s:.1f}" for s in recent_scores) if recent_scores else "N/A"

    improvement = 0.0
    if len(score_history) >= 2:
        improvement = score_history[-1] - score_history[-2]

    return CONVERGENCE_PROMPT.format(
        history_summary=format_history_summary(iteration, total_evaluated, best_score, score_history),
        score_progression=format_score_progression(score_history),
        n_recent=n_recent,
        recent_scores=recent_str,
        best_score=best_score,
        improvement=f"{improvement:+.2f}",
    )


# Response schemas for documentation
PROPOSAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "support": {"type": "string", "enum": ["Al2O3", "ZrO2", "SiO2"]},
                    "metals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "element": {"type": "string", "enum": ["Cu", "Zn", "Al"]},
                                "wt_pct": {"type": "number", "minimum": 0, "maximum": 100},
                            },
                            "required": ["element", "wt_pct"],
                        },
                    },
                },
                "required": ["support", "metals"],
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["candidates", "reasoning"],
}

CONVERGENCE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["CONTINUE", "STOP"]},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["decision", "reasoning", "confidence"],
}
