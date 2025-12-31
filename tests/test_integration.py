"""End-to-end integration tests for Catalyst.

These tests verify the complete workflow from GeneratorAgent
through ShepherdAgent evaluation.

Run with: pytest tests/test_integration.py -v -m integration
Skip with: pytest tests/ -v -m "not integration"
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestration.generator_state import GenerationState
from orchestration.llm_client import LLMClient


# --- Mocked End-to-End Tests ---


class TestMockedEndToEnd:
    """End-to-end tests with mocked LLM."""

    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def generator_config(self, temp_dir):
        """Generator config for testing."""
        return {
            "llm": {
                "mode": "shared",
                "model": "test-model",
                "shared_url": "http://localhost:8000/v1",
            },
            "generation": {
                "candidates_per_iteration": 2,
                "max_iterations": 3,
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
                "checkpoint_path": str(temp_dir / "checkpoint.json"),
                "results_path": str(temp_dir / "results.jsonl"),
            },
        }

    @pytest.fixture
    def shepherd_config(self):
        """Shepherd config for testing."""
        return {
            "llm": {
                "mode": "shared",
                "model": "test-model",
                "shared_url": "http://localhost:8000/v1",
            },
            "budget": {"default": 10.0},
            "cache": {"enabled": False},
            "endpoints": {"cheap": None, "gpu": None},
            "timeouts": {"llm_call": 5, "test_poll_interval": 0.1},
        }

    @pytest.mark.asyncio
    async def test_single_iteration_flow(self, generator_config, shepherd_config, temp_dir):
        """Test a single iteration of the generation loop."""
        from skills.generator import GeneratorAgent
        from skills.shepherd import ShepherdAgent

        # Mock LLM responses
        proposal_response = {
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
            "reasoning": "Initial exploration",
        }

        shepherd_reasoning = [
            {"action": "test", "test": "fast_surrogate", "reasoning": "Start"},
            {"action": "stop", "reasoning": "Done", "confidence": 0.8},
            {
                "viability_score": 75,
                "strengths": ["Good conversion"],
                "concerns": [],
                "recommendation": "PURSUE",
                "summary": "Promising",
            },
        ]

        with patch("skills.generator.create_llm_client_from_config") as gen_llm_mock, \
             patch("skills.shepherd.create_llm_client_from_config") as shep_llm_mock:

            # Setup generator LLM
            gen_llm = AsyncMock()
            gen_llm.reason_json = AsyncMock(return_value=proposal_response)
            gen_llm_mock.return_value = gen_llm

            # Setup shepherd LLM
            shep_llm = AsyncMock()
            shep_llm.reason_json = AsyncMock(side_effect=shepherd_reasoning)
            shep_llm_mock.return_value = shep_llm

            # Create and initialize generator
            generator = GeneratorAgent(
                config=generator_config,
                shepherd_config=shepherd_config,
            )

            # Manually initialize components
            generator._llm = gen_llm
            generator._state = GenerationState()

            # Test proposal
            candidates = await generator._propose_candidates(2)
            assert len(candidates) == 1
            assert candidates[0]["support"] == "Al2O3"

    @pytest.mark.asyncio
    async def test_shepherd_evaluation_chain(self, shepherd_config):
        """Test shepherd runs through evaluation chain correctly."""
        from skills.shepherd import ShepherdAgent

        candidate = {
            "support": "Al2O3",
            "metals": [
                {"element": "Cu", "wt_pct": 60},
                {"element": "Zn", "wt_pct": 25},
                {"element": "Al", "wt_pct": 15},
            ],
        }

        # Shepherd should run fast_surrogate, then microkinetic_lite, then stop
        reasoning_sequence = [
            {"action": "test", "test": "fast_surrogate", "reasoning": "Initial"},
            {"action": "test", "test": "microkinetic_lite", "reasoning": "Deeper"},
            {"action": "stop", "reasoning": "Complete", "confidence": 0.9},
            {
                "viability_score": 85,
                "strengths": ["High selectivity", "Good kinetics"],
                "concerns": [],
                "recommendation": "PURSUE",
                "summary": "Excellent candidate",
            },
        ]

        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(side_effect=reasoning_sequence)
            mock_llm_factory.return_value = mock_llm

            shepherd = ShepherdAgent(config=shepherd_config)
            await shepherd.agent_on_startup()

            result = await shepherd.evaluate({"candidate": candidate, "budget": 100})

            # Verify execution
            assert len(result["results"]) == 2
            assert result["results"][0]["test"] == "fast_surrogate"
            assert result["results"][1]["test"] == "microkinetic_lite"
            assert result["final_assessment"]["viability_score"] == 85
            assert result["final_assessment"]["recommendation"] == "PURSUE"

    @pytest.mark.asyncio
    async def test_state_persistence_across_iterations(self, generator_config, temp_dir):
        """Test that state persists correctly across iterations."""
        from orchestration.generator_state import GenerationState

        state = GenerationState()

        # Simulate multiple iterations
        for i in range(3):
            results = [
                {
                    "candidate": {
                        "support": "Al2O3",
                        "metals": [{"element": "Cu", "wt_pct": 60 + i}],
                    },
                    "results": [{"test": "fast_surrogate", "result": {}, "cost": 0.01}],
                    "final_assessment": {
                        "viability_score": 70 + i * 5,
                        "recommendation": "PURSUE",
                    },
                    "total_cost": 1.0,
                }
            ]
            state.update_with_results(results)

        # Save checkpoint
        checkpoint_path = temp_dir / "checkpoint.json"
        state.save_checkpoint(checkpoint_path)

        # Load and verify
        loaded = GenerationState.load_checkpoint(checkpoint_path)

        assert loaded.iteration == 3
        assert loaded.best_score == 80  # 70 + 2*5
        assert len(loaded.candidates_evaluated) == 3
        assert len(loaded.score_history) == 3

    @pytest.mark.asyncio
    async def test_convergence_detection(self, generator_config):
        """Test that convergence is detected correctly."""
        state = GenerationState()

        # Add iterations with improving scores
        for score in [50, 60, 70]:
            state.update_with_results([{
                "candidate": {"support": "Al2O3", "metals": []},
                "results": [],
                "final_assessment": {"viability_score": score, "recommendation": "PURSUE"},
                "total_cost": 1.0,
            }])

        # Should not converge yet
        assert state.check_convergence(patience=2, min_improvement=1.0) is False

        # Add stagnant iterations
        for _ in range(3):
            state.update_with_results([{
                "candidate": {"support": "Al2O3", "metals": []},
                "results": [],
                "final_assessment": {"viability_score": 70, "recommendation": "PURSUE"},
                "total_cost": 1.0,
            }])

        # Now should converge
        assert state.check_convergence(patience=2, min_improvement=1.0) is True

    @pytest.mark.asyncio
    async def test_candidate_deduplication(self):
        """Test that duplicate candidates are filtered."""
        from orchestration.generator_state import GenerationState, hash_candidate

        state = GenerationState()

        # Add a candidate
        candidate = {
            "support": "Al2O3",
            "metals": [{"element": "Cu", "wt_pct": 60}],
        }

        state.update_with_results([{
            "candidate": candidate,
            "results": [],
            "final_assessment": {"viability_score": 75, "recommendation": "PURSUE"},
            "total_cost": 1.0,
        }])

        # Same candidate should be detected as seen
        assert state.is_candidate_seen(candidate) is True

        # Different candidate should not be seen
        different = {
            "support": "ZrO2",
            "metals": [{"element": "Cu", "wt_pct": 60}],
        }
        assert state.is_candidate_seen(different) is False

    @pytest.mark.asyncio
    async def test_budget_tracking(self, shepherd_config):
        """Test that shepherd tracks budget correctly."""
        from skills.shepherd import ShepherdAgent

        candidate = {
            "support": "Al2O3",
            "metals": [{"element": "Cu", "wt_pct": 100}],
        }

        # Try to run expensive test with limited budget
        reasoning_sequence = [
            {"action": "test", "test": "dft_adsorption", "reasoning": "High accuracy"},
            # DFT costs 100, budget is 50, so this should fail
            {"action": "stop", "reasoning": "Budget exceeded", "confidence": 0.5},
            {
                "viability_score": 50,
                "strengths": [],
                "concerns": ["Incomplete"],
                "recommendation": "DEPRIORITIZE",
                "summary": "Limited evaluation",
            },
        ]

        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(side_effect=reasoning_sequence)
            mock_llm_factory.return_value = mock_llm

            shepherd = ShepherdAgent(config=shepherd_config)
            await shepherd.agent_on_startup()

            result = await shepherd.evaluate({"candidate": candidate, "budget": 50})

            # DFT should not have run (costs 100, budget is 50)
            dft_ran = any(r["test"] == "dft_adsorption" for r in result["results"])
            assert dft_ran is False


# --- Cache Tests ---


class TestCaching:
    """Tests for caching functionality."""

    @pytest.fixture
    def temp_cache(self, tmp_path):
        """Temporary cache file."""
        return tmp_path / "cache.jsonl"

    def test_jsonl_cache_roundtrip(self, temp_cache):
        """Test cache save and load."""
        from orchestration.cache import JsonlCache

        cache = JsonlCache.load(str(temp_cache))

        # Set values
        cache.set("key1", {"data": "value1"})
        cache.set("key2", {"data": "value2"})

        # Reload and verify
        cache2 = JsonlCache.load(str(temp_cache))

        assert cache2.get("key1") == {"data": "value1"}
        assert cache2.get("key2") == {"data": "value2"}
        assert cache2.get("key3") is None

    def test_cache_key_generation(self):
        """Test cache key generation is deterministic."""
        from orchestration.cache import make_cache_key

        candidate = {
            "support": "Al2O3",
            "metals": [{"element": "Cu", "wt_pct": 60}],
        }

        key1 = make_cache_key(candidate, "fast_surrogate", "v1")
        key2 = make_cache_key(candidate, "fast_surrogate", "v1")

        assert key1 == key2

        # Different version = different key
        key3 = make_cache_key(candidate, "fast_surrogate", "v2")
        assert key1 != key3


# --- Error Handling Tests ---


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_shepherd_handles_llm_timeout(self, shepherd_config):
        """Test shepherd handles LLM timeout gracefully."""
        from skills.shepherd import ShepherdAgent

        with patch("skills.shepherd.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(
                side_effect=asyncio.TimeoutError("LLM timeout")
            )
            mock_llm_factory.return_value = mock_llm

            shepherd = ShepherdAgent(config=shepherd_config)
            await shepherd.agent_on_startup()

            candidate = {"support": "Al2O3", "metals": []}
            result = await shepherd.evaluate({"candidate": candidate})

            # Should complete with error in history
            assert "history" in result
            assert result["iterations"] >= 1

    @pytest.mark.asyncio
    async def test_generator_handles_no_valid_candidates(
        self, generator_config, shepherd_config
    ):
        """Test generator handles LLM returning no valid candidates."""
        from skills.generator import GeneratorAgent
        from orchestration.generator_state import GenerationState

        with patch("skills.generator.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            # LLM returns invalid candidates
            mock_llm.reason_json = AsyncMock(return_value={
                "candidates": [
                    {"support": "InvalidSupport", "metals": []},  # Invalid
                ],
                "reasoning": "Bad proposal",
            })
            mock_llm_factory.return_value = mock_llm

            generator = GeneratorAgent(
                config=generator_config,
                shepherd_config=shepherd_config,
            )
            generator._llm = mock_llm
            generator._state = GenerationState()

            candidates = await generator._propose_candidates(3)

            # Should return empty list for invalid proposals
            assert candidates == []


# --- Performance Tests ---


class TestPerformance:
    """Basic performance/scaling tests."""

    def test_state_handles_many_candidates(self):
        """Test state management with many candidates."""
        from orchestration.generator_state import GenerationState

        state = GenerationState()

        # Add 1000 candidates
        for i in range(100):
            results = [
                {
                    "candidate": {
                        "support": "Al2O3",
                        "metals": [{"element": "Cu", "wt_pct": i % 100}],
                    },
                    "results": [],
                    "final_assessment": {
                        "viability_score": 50 + (i % 50),
                        "recommendation": "PURSUE",
                    },
                    "total_cost": 1.0,
                }
                for _ in range(10)
            ]
            state.update_with_results(results)

        # Should handle many candidates without issue
        assert len(state.candidates_evaluated) == 1000
        assert len(state.seen_hashes) <= 1000  # Some might be duplicates

        # Top performers should be sorted
        top = state.get_top_performers(10)
        scores = [p["final_assessment"]["viability_score"] for p in top]
        assert scores == sorted(scores, reverse=True)

    def test_hash_candidate_performance(self):
        """Test candidate hashing performance."""
        from orchestration.generator_state import hash_candidate
        import time

        candidate = {
            "support": "Al2O3",
            "metals": [
                {"element": "Cu", "wt_pct": 60},
                {"element": "Zn", "wt_pct": 25},
                {"element": "Al", "wt_pct": 15},
            ],
        }

        start = time.time()
        for _ in range(10000):
            hash_candidate(candidate)
        elapsed = time.time() - start

        # Should be fast (< 1 second for 10k hashes)
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
