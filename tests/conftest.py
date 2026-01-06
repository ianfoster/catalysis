"""Shared test fixtures and utilities for Catalyst tests."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# --- Sample Data Fixtures ---


@pytest.fixture
def sample_candidate() -> dict[str, Any]:
    """A valid catalyst candidate."""
    return {
        "support": "Al2O3",
        "metals": [
            {"element": "Cu", "wt_pct": 55},
            {"element": "Zn", "wt_pct": 30},
            {"element": "Al", "wt_pct": 15},
        ],
    }


@pytest.fixture
def sample_candidates() -> list[dict[str, Any]]:
    """Multiple valid catalyst candidates."""
    return [
        {
            "support": "Al2O3",
            "metals": [
                {"element": "Cu", "wt_pct": 55},
                {"element": "Zn", "wt_pct": 30},
                {"element": "Al", "wt_pct": 15},
            ],
        },
        {
            "support": "ZrO2",
            "metals": [
                {"element": "Cu", "wt_pct": 60},
                {"element": "Zn", "wt_pct": 25},
                {"element": "Al", "wt_pct": 15},
            ],
        },
        {
            "support": "SiO2",
            "metals": [
                {"element": "Cu", "wt_pct": 50},
                {"element": "Zn", "wt_pct": 35},
                {"element": "Al", "wt_pct": 15},
            ],
        },
    ]


@pytest.fixture
def sample_shepherd_result(sample_candidate) -> dict[str, Any]:
    """A sample ShepherdAgent evaluation result."""
    return {
        "candidate": sample_candidate,
        "results": [
            {
                "test": "fast_surrogate",
                "result": {
                    "co2_conversion": 0.35,
                    "methanol_selectivity": 0.72,
                    "methanol_sty": 0.25,
                    "uncertainty": 0.15,
                },
                "cost": 0.01,
            },
            {
                "test": "microkinetic_lite",
                "result": {
                    "RLS": "CO2_adsorption",
                    "temp_sensitivity": 0.8,
                    "pressure_sensitivity": 0.3,
                },
                "cost": 1.0,
            },
        ],
        "total_cost": 1.01,
        "final_assessment": {
            "viability_score": 75,
            "strengths": ["High selectivity", "Good conversion"],
            "concerns": ["Moderate uncertainty"],
            "recommendation": "PURSUE",
            "summary": "Promising candidate for further evaluation",
        },
        "confidence": 0.75,
        "history": [],
        "iterations": 3,
    }


@pytest.fixture
def sample_test_results() -> list[dict[str, Any]]:
    """Sample test results for a candidate."""
    return [
        {
            "test": "fast_surrogate",
            "result": {
                "co2_conversion": 0.35,
                "methanol_selectivity": 0.72,
                "methanol_sty": 0.25,
                "uncertainty": 0.15,
            },
            "cost": 0.01,
        },
    ]


# --- Configuration Fixtures ---


@pytest.fixture
def shepherd_config() -> dict[str, Any]:
    """Configuration for ShepherdAgent."""
    return {
        "llm": {
            "mode": "shared",
            "model": "test-model",
            "shared_url": "http://localhost:8000/v1",
            "local_url": "http://localhost:8000/v1",
        },
        "budget": {
            "default": 100.0,
            "max": 1000.0,
        },
        "cache": {
            "enabled": False,
        },
        "endpoints": {
            "cheap": None,
            "gpu": None,
        },
        "timeouts": {
            "llm_call": 5,
            "test_poll_interval": 0.1,
            "test_max_wait": 60,
        },
    }


@pytest.fixture
def generator_config() -> dict[str, Any]:
    """Configuration for GeneratorAgent."""
    return {
        "llm": {
            "mode": "shared",
            "model": "test-model",
            "shared_url": "http://localhost:8000/v1",
        },
        "generation": {
            "candidates_per_iteration": 3,
            "max_iterations": 5,
        },
        "convergence": {
            "patience": 2,
            "min_improvement": 0.01,
            "llm_judgment": False,
        },
        "shepherd": {
            "budget_per_candidate": 10.0,
            "max_concurrent_shepherds": 2,
            "timeout": 60,
        },
        "state": {
            "checkpoint_path": "",  # Will be set per test
            "results_path": "",
        },
    }


# --- Mock Fixtures ---


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock = AsyncMock()
    mock.model = "test-model"
    mock.base_url = "http://localhost:8000/v1"
    mock.reason = AsyncMock(return_value="Test response")
    mock.reason_json = AsyncMock(return_value={"action": "stop", "confidence": 0.9})
    return mock


@pytest.fixture
def mock_llm_client_for_shepherd(mock_llm_client):
    """Mock LLM client configured for shepherd reasoning."""
    mock_llm_client.reason_json = AsyncMock(side_effect=[
        # First call: run fast_surrogate
        {"action": "test", "test": "fast_surrogate", "reasoning": "Initial screening"},
        # Second call: stop
        {"action": "stop", "reasoning": "Sufficient data", "confidence": 0.85},
        # Third call: final assessment
        {
            "viability_score": 75,
            "strengths": ["Good metrics"],
            "concerns": [],
            "recommendation": "PURSUE",
            "summary": "Promising",
        },
    ])
    return mock_llm_client


@pytest.fixture
def mock_llm_client_for_generator(mock_llm_client):
    """Mock LLM client configured for generator proposals."""
    mock_llm_client.reason_json = AsyncMock(return_value={
        "candidates": [
            {
                "support": "Al2O3",
                "metals": [
                    {"element": "Cu", "wt_pct": 60},
                    {"element": "Zn", "wt_pct": 25},
                    {"element": "Al", "wt_pct": 15},
                ],
            },
        ],
        "reasoning": "Exploring high-Cu region",
    })
    return mock_llm_client


@pytest.fixture
def mock_gc_adapter():
    """Mock Globus Compute adapter."""
    mock = MagicMock()
    mock.submit = MagicMock(return_value=MagicMock(
        task_id="test-task-123",
        endpoint_id="test-endpoint",
        function_id="test-function",
    ))
    mock.try_result = MagicMock(return_value={
        "task_id": "test-task-123",
        "status": "SUCCEEDED",
        "result": {"test": "result"},
    })
    return mock


# --- Temporary Directory Fixtures ---


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_cache_path(temp_dir):
    """Temporary path for cache file."""
    return temp_dir / "test_cache.jsonl"


@pytest.fixture
def temp_checkpoint_path(temp_dir):
    """Temporary path for checkpoint file."""
    return temp_dir / "test_checkpoint.json"


@pytest.fixture
def temp_results_path(temp_dir):
    """Temporary path for results file."""
    return temp_dir / "test_results.jsonl"


# --- Helper Functions ---


def make_shepherd_result(
    candidate: dict[str, Any],
    score: int = 75,
    recommendation: str = "PURSUE",
    cost: float = 1.0,
) -> dict[str, Any]:
    """Create a shepherd result with given parameters."""
    return {
        "candidate": candidate,
        "results": [{"test": "fast_surrogate", "result": {}, "cost": cost}],
        "total_cost": cost,
        "final_assessment": {
            "viability_score": score,
            "strengths": [],
            "concerns": [],
            "recommendation": recommendation,
            "summary": f"Score {score}",
        },
        "confidence": score / 100,
        "history": [],
        "iterations": 1,
    }


def make_candidate(
    support: str = "Al2O3",
    cu: float = 60,
    zn: float = 25,
    al: float = 15,
) -> dict[str, Any]:
    """Create a candidate with given composition."""
    return {
        "support": support,
        "metals": [
            {"element": "Cu", "wt_pct": cu},
            {"element": "Zn", "wt_pct": zn},
            {"element": "Al", "wt_pct": al},
        ],
    }


# --- Async Test Utilities ---


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
