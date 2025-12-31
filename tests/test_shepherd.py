"""Tests for ShepherdAgent.

Run with: pytest tests/test_shepherd.py -v
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from academy.manager import Manager
from academy.exchange import LocalExchangeFactory

from skills.shepherd import ShepherdAgent
from orchestration.llm_client import LLMClient, _extract_json
from orchestration.test_registry import (
    get_test,
    check_prerequisites,
    format_tests_for_prompt,
    get_affordable_tests,
    AVAILABLE_TESTS,
)


# --- LLM Client Tests ---


class TestExtractJson:
    """Tests for JSON extraction from LLM responses."""

    def test_pure_json(self):
        text = '{"action": "test", "test": "fast_surrogate"}'
        result = _extract_json(text)
        assert result["action"] == "test"
        assert result["test"] == "fast_surrogate"

    def test_json_in_markdown_block(self):
        text = '''Here is my response:
```json
{"action": "stop", "confidence": 0.85}
```
'''
        result = _extract_json(text)
        assert result["action"] == "stop"
        assert result["confidence"] == 0.85

    def test_json_with_surrounding_text(self):
        text = '''I think we should run a test.
{"action": "test", "test": "microkinetic_lite", "reasoning": "need more data"}
That's my recommendation.'''
        result = _extract_json(text)
        assert result["action"] == "test"
        assert result["test"] == "microkinetic_lite"

    def test_invalid_json_raises(self):
        text = "This is not JSON at all"
        with pytest.raises(Exception):
            _extract_json(text)

    def test_nested_json(self):
        text = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = _extract_json(text)
        assert result["outer"]["inner"] == "value"
        assert result["list"] == [1, 2, 3]


# --- Test Registry Tests ---


class TestTestRegistry:
    """Tests for the test registry."""

    def test_get_test_exists(self):
        spec = get_test("fast_surrogate")
        assert spec.name == "fast_surrogate"
        assert spec.cost == 0.01
        assert spec.endpoint == "cheap"

    def test_get_test_not_found(self):
        with pytest.raises(KeyError):
            get_test("nonexistent_test")

    def test_check_prerequisites_none(self):
        satisfied, missing = check_prerequisites("fast_surrogate", set())
        assert satisfied is True
        assert missing == []

    def test_check_prerequisites_satisfied(self):
        completed = {"fast_surrogate"}
        satisfied, missing = check_prerequisites("microkinetic_lite", completed)
        assert satisfied is True
        assert missing == []

    def test_check_prerequisites_missing(self):
        satisfied, missing = check_prerequisites("microkinetic_lite", set())
        assert satisfied is False
        assert "fast_surrogate" in missing

    def test_format_tests_for_prompt(self):
        output = format_tests_for_prompt()
        assert "fast_surrogate" in output
        assert "microkinetic_lite" in output
        assert "|" in output  # Table format

    def test_format_tests_shows_completed(self):
        output = format_tests_for_prompt(completed_tests={"fast_surrogate"})
        assert "[DONE]" in output

    def test_get_affordable_tests(self):
        tests = get_affordable_tests(budget_remaining=1.0, completed_tests=set())
        # Should include fast_surrogate (0.01) and stability_analysis (0.1)
        names = [t.name for t in tests]
        assert "fast_surrogate" in names

    def test_get_affordable_tests_respects_prerequisites(self):
        tests = get_affordable_tests(budget_remaining=10.0, completed_tests=set())
        names = [t.name for t in tests]
        # microkinetic_lite requires fast_surrogate, so should not be included
        assert "microkinetic_lite" not in names

    def test_get_affordable_tests_with_prereqs_met(self):
        tests = get_affordable_tests(
            budget_remaining=10.0,
            completed_tests={"fast_surrogate"},
        )
        names = [t.name for t in tests]
        assert "microkinetic_lite" in names


# --- ShepherdAgent Tests ---


class TestShepherdAgent:
    """Tests for ShepherdAgent."""

    @pytest.fixture
    def mock_config(self):
        return {
            "llm": {
                "mode": "shared",
                "model": "test-model",
                "shared_url": "http://localhost:8000/v1",
            },
            "budget": {
                "default": 10.0,
                "max": 100.0,
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
            },
        }

    @pytest.fixture
    def sample_candidate(self):
        return {
            "support": "Al2O3",
            "metals": [
                {"element": "Cu", "wt_pct": 55},
                {"element": "Zn", "wt_pct": 30},
                {"element": "Al", "wt_pct": 15},
            ],
        }

    @pytest.mark.asyncio
    async def test_shepherd_initialization(self, mock_config):
        """Test that ShepherdAgent initializes correctly."""
        agent = ShepherdAgent(config=mock_config)
        assert agent._config == mock_config
        assert agent._llm is None  # Not initialized until startup

    @pytest.mark.asyncio
    async def test_shepherd_evaluate_stops_immediately(self, mock_config, sample_candidate):
        """Test shepherd stops when LLM says stop."""
        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            # Mock LLM to return stop immediately
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(side_effect=[
                # First call: reasoning - stop
                {"action": "stop", "reasoning": "Candidate looks unpromising", "confidence": 0.3},
                # Second call: final assessment
                {
                    "viability_score": 30,
                    "strengths": [],
                    "concerns": ["Low potential"],
                    "recommendation": "REJECT",
                    "summary": "Not viable",
                },
            ])
            mock_llm_factory.return_value = mock_llm

            agent = ShepherdAgent(config=mock_config)
            await agent.agent_on_startup()

            result = await agent.evaluate({"candidate": sample_candidate})

            assert result["candidate"] == sample_candidate
            assert result["total_cost"] == 0.0
            assert result["iterations"] == 1
            assert len(result["results"]) == 0

    @pytest.mark.asyncio
    async def test_shepherd_runs_tests(self, mock_config, sample_candidate):
        """Test shepherd runs tests when LLM requests them."""
        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(side_effect=[
                # First call: run fast_surrogate
                {"action": "test", "test": "fast_surrogate", "reasoning": "Initial screening"},
                # Second call: stop
                {"action": "stop", "reasoning": "Enough info", "confidence": 0.8},
                # Third call: final assessment
                {
                    "viability_score": 75,
                    "strengths": ["Good conversion"],
                    "concerns": [],
                    "recommendation": "PURSUE",
                    "summary": "Promising candidate",
                },
            ])
            mock_llm_factory.return_value = mock_llm

            agent = ShepherdAgent(config=mock_config)
            await agent.agent_on_startup()

            result = await agent.evaluate({"candidate": sample_candidate})

            assert result["total_cost"] == 0.01  # fast_surrogate cost
            assert len(result["results"]) == 1
            assert result["results"][0]["test"] == "fast_surrogate"

    @pytest.mark.asyncio
    async def test_shepherd_respects_budget(self, mock_config, sample_candidate):
        """Test shepherd stops when budget exhausted."""
        mock_config["budget"]["default"] = 0.005  # Less than fast_surrogate cost

        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(side_effect=[
                # Try to run test (should fail due to budget)
                {"action": "test", "test": "fast_surrogate", "reasoning": "Need data"},
                # Stop
                {"action": "stop", "reasoning": "Budget exhausted", "confidence": 0.5},
                # Final assessment
                {
                    "viability_score": 50,
                    "strengths": [],
                    "concerns": ["Incomplete evaluation"],
                    "recommendation": "DEPRIORITIZE",
                    "summary": "Incomplete",
                },
            ])
            mock_llm_factory.return_value = mock_llm

            agent = ShepherdAgent(config=mock_config)
            await agent.agent_on_startup()

            result = await agent.evaluate({"candidate": sample_candidate})

            # Should have 0 results due to budget constraint
            assert result["total_cost"] == 0.0
            assert len(result["results"]) == 0

    @pytest.mark.asyncio
    async def test_shepherd_validates_prerequisites(self, mock_config, sample_candidate):
        """Test shepherd validates test prerequisites."""
        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(side_effect=[
                # Try to run microkinetic without fast_surrogate
                {"action": "test", "test": "microkinetic_lite", "reasoning": "Skip ahead"},
                # Run fast_surrogate
                {"action": "test", "test": "fast_surrogate", "reasoning": "Start from beginning"},
                # Stop
                {"action": "stop", "reasoning": "Done", "confidence": 0.7},
                # Final assessment
                {
                    "viability_score": 70,
                    "strengths": ["Screened"],
                    "concerns": [],
                    "recommendation": "PURSUE",
                    "summary": "OK",
                },
            ])
            mock_llm_factory.return_value = mock_llm

            agent = ShepherdAgent(config=mock_config)
            await agent.agent_on_startup()

            result = await agent.evaluate({"candidate": sample_candidate})

            # Should have 1 result (fast_surrogate), microkinetic skipped
            assert len(result["results"]) == 1
            assert result["results"][0]["test"] == "fast_surrogate"

    @pytest.mark.asyncio
    async def test_shepherd_handles_unknown_test(self, mock_config, sample_candidate):
        """Test shepherd handles LLM requesting unknown test."""
        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(side_effect=[
                # Request unknown test
                {"action": "test", "test": "nonexistent_test", "reasoning": "Try this"},
                # Then stop
                {"action": "stop", "reasoning": "No valid tests", "confidence": 0.5},
                # Final assessment
                {
                    "viability_score": 40,
                    "strengths": [],
                    "concerns": ["Could not run tests"],
                    "recommendation": "REJECT",
                    "summary": "Failed",
                },
            ])
            mock_llm_factory.return_value = mock_llm

            agent = ShepherdAgent(config=mock_config)
            await agent.agent_on_startup()

            result = await agent.evaluate({"candidate": sample_candidate})

            # Should complete without crashing
            assert result["total_cost"] == 0.0
            assert "final_assessment" in result

    @pytest.mark.asyncio
    async def test_shepherd_runs_multiple_tests(self, mock_config, sample_candidate):
        """Test shepherd runs sequence of tests."""
        mock_config["budget"]["default"] = 100.0  # Enough for multiple tests

        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(side_effect=[
                # Run fast_surrogate first
                {"action": "test", "test": "fast_surrogate", "reasoning": "Initial screening"},
                # Then microkinetic_lite
                {"action": "test", "test": "microkinetic_lite", "reasoning": "Deeper analysis"},
                # Then stop
                {"action": "stop", "reasoning": "Complete evaluation", "confidence": 0.9},
                # Final assessment
                {
                    "viability_score": 85,
                    "strengths": ["High performance", "Good kinetics"],
                    "concerns": [],
                    "recommendation": "PURSUE",
                    "summary": "Excellent candidate",
                },
            ])
            mock_llm_factory.return_value = mock_llm

            agent = ShepherdAgent(config=mock_config)
            await agent.agent_on_startup()

            result = await agent.evaluate({"candidate": sample_candidate})

            assert len(result["results"]) == 2
            assert result["results"][0]["test"] == "fast_surrogate"
            assert result["results"][1]["test"] == "microkinetic_lite"
            assert result["total_cost"] == 0.01 + 1.0  # fast_surrogate + microkinetic_lite

    @pytest.mark.asyncio
    async def test_shepherd_max_iterations_safety(self, mock_config, sample_candidate):
        """Test shepherd stops after max iterations to prevent infinite loops."""
        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            # LLM keeps requesting tests (would loop forever without safety limit)
            mock_llm.reason_json = AsyncMock(return_value={
                "action": "test", "test": "fast_surrogate", "reasoning": "Keep testing"
            })
            mock_llm_factory.return_value = mock_llm

            agent = ShepherdAgent(config=mock_config)
            await agent.agent_on_startup()

            result = await agent.evaluate({"candidate": sample_candidate, "budget": 1000})

            # Should stop due to max_iterations (20) safety limit
            assert result["iterations"] <= 20

    @pytest.mark.asyncio
    async def test_shepherd_handles_llm_error(self, mock_config, sample_candidate):
        """Test shepherd handles LLM errors gracefully."""
        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(side_effect=Exception("LLM connection failed"))
            mock_llm_factory.return_value = mock_llm

            agent = ShepherdAgent(config=mock_config)
            await agent.agent_on_startup()

            result = await agent.evaluate({"candidate": sample_candidate})

            # Should complete with error recorded in history
            assert "history" in result
            assert len(result["history"]) > 0

    @pytest.mark.asyncio
    async def test_shepherd_returns_correct_structure(self, mock_config, sample_candidate):
        """Test shepherd returns all expected fields."""
        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(side_effect=[
                {"action": "stop", "reasoning": "Quick stop", "confidence": 0.5},
                {
                    "viability_score": 50,
                    "strengths": ["Test strength"],
                    "concerns": ["Test concern"],
                    "recommendation": "DEPRIORITIZE",
                    "summary": "Test summary",
                },
            ])
            mock_llm_factory.return_value = mock_llm

            agent = ShepherdAgent(config=mock_config)
            await agent.agent_on_startup()

            result = await agent.evaluate({"candidate": sample_candidate})

            # Check all required fields present
            assert "candidate" in result
            assert "results" in result
            assert "total_cost" in result
            assert "final_assessment" in result
            assert "confidence" in result
            assert "history" in result
            assert "iterations" in result

            # Check final_assessment structure
            assessment = result["final_assessment"]
            assert "viability_score" in assessment
            assert "strengths" in assessment
            assert "concerns" in assessment
            assert "recommendation" in assessment
            assert "summary" in assessment

    @pytest.mark.asyncio
    async def test_shepherd_custom_budget(self, mock_config, sample_candidate):
        """Test shepherd respects custom budget in request."""
        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(side_effect=[
                # Try expensive test
                {"action": "test", "test": "dft_adsorption", "reasoning": "High accuracy"},
                # Should fail due to budget, then stop
                {"action": "stop", "reasoning": "Budget exceeded", "confidence": 0.3},
                {"viability_score": 30, "strengths": [], "concerns": [], "recommendation": "REJECT", "summary": ""},
            ])
            mock_llm_factory.return_value = mock_llm

            agent = ShepherdAgent(config=mock_config)
            await agent.agent_on_startup()

            # Request with small budget (DFT costs 100)
            result = await agent.evaluate({"candidate": sample_candidate, "budget": 50})

            # DFT should not have run due to budget
            assert all(r["test"] != "dft_adsorption" for r in result["results"])


# --- Test Registry Additional Tests ---


class TestTestRegistryAdvanced:
    """Advanced tests for test registry."""

    def test_all_tests_have_required_fields(self):
        """Verify all tests have required fields."""
        for name, spec in AVAILABLE_TESTS.items():
            assert spec.name == name
            assert spec.description
            assert spec.cost >= 0
            assert spec.endpoint in ["cheap", "gpu"]
            assert isinstance(spec.prerequisites, tuple)
            assert spec.timeout > 0

    def test_dft_requires_fast_surrogate(self):
        """DFT should require fast_surrogate."""
        spec = get_test("dft_adsorption")
        assert "fast_surrogate" in spec.prerequisites

    def test_fast_surrogate_is_cheapest(self):
        """Fast surrogate should be the cheapest test."""
        fast_cost = get_test("fast_surrogate").cost
        for name, spec in AVAILABLE_TESTS.items():
            if name != "fast_surrogate":
                assert spec.cost >= fast_cost

    def test_gpu_tests_more_expensive(self):
        """GPU tests should generally be more expensive."""
        gpu_tests = [s for s in AVAILABLE_TESTS.values() if s.endpoint == "gpu"]
        cheap_tests = [s for s in AVAILABLE_TESTS.values() if s.endpoint == "cheap"]

        avg_gpu_cost = sum(t.cost for t in gpu_tests) / len(gpu_tests) if gpu_tests else 0
        avg_cheap_cost = sum(t.cost for t in cheap_tests) / len(cheap_tests) if cheap_tests else 0

        assert avg_gpu_cost > avg_cheap_cost

    def test_format_tests_over_budget(self):
        """Format shows over budget indicator."""
        output = format_tests_for_prompt(budget_remaining=0.001)
        assert "[OVER BUDGET]" in output

    def test_estimate_total_cost(self):
        """Test cost estimation for multiple tests."""
        from orchestration.test_registry import estimate_total_cost

        tests = ["fast_surrogate", "microkinetic_lite"]
        cost = estimate_total_cost(tests)
        assert cost == 0.01 + 1.0


# --- Shepherd Prompts Tests ---


class TestShepherdPrompts:
    """Tests for shepherd prompt generation."""

    def test_build_reasoning_prompt_contains_candidate(self, sample_candidate):
        """Reasoning prompt should include candidate info."""
        from orchestration.shepherd_prompts import build_reasoning_prompt

        prompt = build_reasoning_prompt(
            candidate=sample_candidate,
            results=[],
            budget_total=100,
            budget_spent=0,
        )

        assert "Cu" in prompt
        assert "Al2O3" in prompt
        assert "100" in prompt  # budget

    def test_build_reasoning_prompt_with_results(self, sample_candidate):
        """Reasoning prompt includes previous results."""
        from orchestration.shepherd_prompts import build_reasoning_prompt

        results = [
            {
                "test": "fast_surrogate",
                "result": {"co2_conversion": 0.35},
                "cost": 0.01,
            }
        ]

        prompt = build_reasoning_prompt(
            candidate=sample_candidate,
            results=results,
            budget_total=100,
            budget_spent=0.01,
        )

        assert "fast_surrogate" in prompt
        assert "0.35" in prompt or "co2_conversion" in prompt

    def test_final_assessment_prompt_structure(self, sample_candidate):
        """Final assessment prompt has correct structure."""
        from orchestration.shepherd_prompts import build_final_assessment_prompt

        prompt = build_final_assessment_prompt(
            candidate=sample_candidate,
            results=[],
            total_cost=1.0,
        )

        assert "viability_score" in prompt
        assert "strengths" in prompt
        assert "recommendation" in prompt
        assert "PURSUE" in prompt or "REJECT" in prompt


# --- Integration Test ---


@pytest.mark.asyncio
@pytest.mark.integration
async def test_shepherd_full_integration():
    """Full integration test with Academy Manager.

    Requires a running LLM server. Skip if not available.
    """
    pytest.skip("Integration test requires LLM server")

    config = {
        "llm": {
            "mode": "shared",
            "model": "meta-llama/Llama-3-8B-Instruct",
            "shared_url": "http://localhost:8000/v1",
        },
        "budget": {"default": 10.0},
        "cache": {"enabled": False},
        "endpoints": {"cheap": None, "gpu": None},
        "timeouts": {"llm_call": 30, "test_poll_interval": 1.0},
    }

    candidate = {
        "support": "Al2O3",
        "metals": [
            {"element": "Cu", "wt_pct": 55},
            {"element": "Zn", "wt_pct": 30},
        ],
    }

    async with await Manager.from_exchange_factory(LocalExchangeFactory()) as manager:
        shepherd = await manager.launch(ShepherdAgent, kwargs={"config": config})
        result = await shepherd.evaluate({"candidate": candidate})

        assert "candidate" in result
        assert "results" in result
        assert "final_assessment" in result
        assert result["final_assessment"]["recommendation"] in ["PURSUE", "DEPRIORITIZE", "REJECT"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
