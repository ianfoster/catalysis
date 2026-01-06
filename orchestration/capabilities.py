"""Simulation capabilities registry for LLM agents.

This module loads the capabilities YAML and provides formatted summaries
that can be included in LLM prompts to guide simulation method selection.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

# Default path to capabilities file
DEFAULT_CAPABILITIES_PATH = Path(__file__).parent.parent / "config" / "simulation_capabilities.yaml"


def load_capabilities(path: Path | str | None = None) -> dict[str, Any]:
    """Load the simulation capabilities registry.

    Args:
        path: Path to capabilities YAML. Uses default if not specified.

    Returns:
        Parsed capabilities dictionary.
    """
    path = Path(path) if path else DEFAULT_CAPABILITIES_PATH

    if not path.exists():
        raise FileNotFoundError(f"Capabilities file not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)


def get_available_methods(capabilities: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Get list of available methods by category.

    Returns:
        Dict mapping category to list of available method names.
    """
    if capabilities is None:
        capabilities = load_capabilities()

    available = {}

    for category in ["ml_potentials", "dft", "md", "kinetics"]:
        cat_data = capabilities.get(category, {})
        available[category] = [
            name for name, info in cat_data.items()
            if info.get("status") in ("available", "surrogate_only")
        ]

    return available


def get_capabilities_prompt(capabilities: dict[str, Any] | None = None) -> str:
    """Generate a prompt-friendly summary of available simulation capabilities.

    This string can be included in LLM prompts to help agents select
    appropriate simulation methods.

    Returns:
        Formatted string for inclusion in prompts.
    """
    if capabilities is None:
        capabilities = load_capabilities()

    lines = [
        "## Available Simulation Methods",
        "",
        "### ML Potentials (fast, seconds per structure)",
    ]

    ml = capabilities.get("ml_potentials", {})
    for name, info in ml.items():
        status = info.get("status", "unknown")
        if status == "available":
            lines.append(f"- **{name.upper()}**: {info.get('use_case', '')} [RECOMMENDED]")
        elif status == "unavailable":
            lines.append(f"- ~~{name.upper()}~~: UNAVAILABLE - {info.get('reason', '')}")

    lines.extend([
        "",
        "### DFT Codes (slow, hours per structure)",
    ])

    dft = capabilities.get("dft", {})
    for name, info in dft.items():
        status = info.get("status", "unknown")
        if status == "available":
            lines.append(f"- **{name.upper()}**: {info.get('use_case', '')}")
        elif status == "unavailable":
            lines.append(f"- ~~{name.upper()}~~: UNAVAILABLE - {info.get('reason', '')}")

    lines.extend([
        "",
        "### Kinetics (surrogate models available)",
    ])

    kinetics = capabilities.get("kinetics", {})
    for name, info in kinetics.items():
        status = info.get("status", "unknown")
        if status in ("available", "surrogate_only"):
            note = " (surrogate)" if status == "surrogate_only" else ""
            lines.append(f"- **{name.upper()}**{note}: {info.get('use_case', '')}")

    # Add selection rules
    rules = capabilities.get("selection_rules", [])
    if rules:
        lines.extend([
            "",
            "### Selection Rules",
        ])
        for rule in rules:
            lines.append(f"- {rule.get('rule', rule)}")

    return "\n".join(lines)


def get_gc_function_for_method(method: str, capabilities: dict[str, Any] | None = None) -> str | None:
    """Get the Globus Compute function name for a simulation method.

    Args:
        method: Method name (e.g., 'mace', 'quantum_espresso')
        capabilities: Capabilities dict, loaded if not provided.

    Returns:
        GC function name or None if not found.
    """
    if capabilities is None:
        capabilities = load_capabilities()

    # Search all categories
    for category in ["ml_potentials", "dft", "md", "kinetics"]:
        cat_data = capabilities.get(category, {})
        if method.lower() in cat_data:
            return cat_data[method.lower()].get("gc_function")

    return None


def get_method_status(method: str, capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get status information for a simulation method.

    Args:
        method: Method name (e.g., 'mace', 'gpaw')

    Returns:
        Dict with status, reason (if unavailable), and gc_function.
    """
    if capabilities is None:
        capabilities = load_capabilities()

    for category in ["ml_potentials", "dft", "md", "kinetics"]:
        cat_data = capabilities.get(category, {})
        if method.lower() in cat_data:
            info = cat_data[method.lower()]
            return {
                "method": method,
                "category": category,
                "status": info.get("status", "unknown"),
                "reason": info.get("reason"),
                "gc_function": info.get("gc_function"),
                "workaround": info.get("workaround"),
            }

    return {"method": method, "status": "not_found"}


def validate_simulation_request(
    method: str,
    capabilities: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Validate that a simulation method can be used.

    Args:
        method: Requested method name.

    Returns:
        Tuple of (is_valid, message).
    """
    status = get_method_status(method, capabilities)

    if status["status"] == "not_found":
        return False, f"Unknown method: {method}"

    if status["status"] == "unavailable":
        workaround = status.get("workaround", "No workaround available")
        return False, f"{method} is unavailable: {status.get('reason')}. {workaround}"

    if status["status"] == "surrogate_only":
        return True, f"{method} will use surrogate model (real code not configured)"

    return True, f"{method} is available"


# Convenience function for agents
def get_recommended_method(task: str, capabilities: dict[str, Any] | None = None) -> str:
    """Get recommended method for a task type.

    Args:
        task: One of 'screening', 'relaxation', 'dft', 'kinetics'

    Returns:
        Recommended method name.
    """
    recommendations = {
        "screening": "mace",
        "ml_screening": "mace",
        "relaxation": "mace",
        "structure_relaxation": "mace",
        "dft": "quantum_espresso",
        "adsorption": "quantum_espresso",
        "kinetics": "cantera",
        "reactor": "cantera",
        "microkinetics": "catmap",
    }

    return recommendations.get(task.lower(), "mace")


if __name__ == "__main__":
    # Print capabilities summary when run directly
    print(get_capabilities_prompt())
    print("\n" + "="*60 + "\n")
    print("Available methods:")
    for cat, methods in get_available_methods().items():
        print(f"  {cat}: {methods}")
