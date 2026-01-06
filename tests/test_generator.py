"""Tests for GeneratorAgent.

Run with: pytest tests/test_generator.py -v
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestration.generator_state import (
    GenerationState,
    EvaluatedCandidate,
    hash_candidate,
    candidate_to_str,
    validate_candidate,
    normalize_candidate,
)
from orchestration.generator_prompts import (
    format_candidate,
    format_top_performers_table,
    format_patterns_summary,
    build_proposal_prompt,
    build_convergence_prompt,
)


# --- Utility Function Tests ---


class TestHashCandidate:
    """Tests for candidate hashing."""

    def test_same_candidate_same_hash(self):
        c1 = {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 60}]}
        c2 = {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 60}]}
        assert hash_candidate(c1) == hash_candidate(c2)

    def test_different_support_different_hash(self):
        c1 = {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 60}]}
        c2 = {"support": "ZrO2", "metals": [{"element": "Cu", "wt_pct": 60}]}
        assert hash_candidate(c1) != hash_candidate(c2)

    def test_different_wt_pct_different_hash(self):
        c1 = {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 60}]}
        c2 = {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 65}]}
        assert hash_candidate(c1) != hash_candidate(c2)

    def test_order_independent(self):
        c1 = {
            "support": "Al2O3",
            "metals": [
                {"element": "Cu", "wt_pct": 60},
                {"element": "Zn", "wt_pct": 40},
            ],
        }
        c2 = {
            "support": "Al2O3",
            "metals": [
                {"element": "Zn", "wt_pct": 40},
                {"element": "Cu", "wt_pct": 60},
            ],
        }
        assert hash_candidate(c1) == hash_candidate(c2)


class TestValidateCandidate:
    """Tests for candidate validation."""

    def test_valid_candidate(self):
        candidate = {
            "support": "Al2O3",
            "metals": [
                {"element": "Cu", "wt_pct": 60},
                {"element": "Zn", "wt_pct": 25},
                {"element": "Al", "wt_pct": 15},
            ],
        }
        valid, error = validate_candidate(candidate)
        assert valid is True
        assert error is None

    def test_invalid_support(self):
        candidate = {
            "support": "InvalidSupport",
            "metals": [{"element": "Cu", "wt_pct": 100}],
        }
        valid, error = validate_candidate(candidate)
        assert valid is False
        assert "support" in error.lower()

    def test_invalid_element(self):
        candidate = {
            "support": "Al2O3",
            "metals": [{"element": "Fe", "wt_pct": 100}],
        }
        valid, error = validate_candidate(candidate)
        assert valid is False
        assert "element" in error.lower()

    def test_weight_sum_not_100(self):
        candidate = {
            "support": "Al2O3",
            "metals": [
                {"element": "Cu", "wt_pct": 50},
                {"element": "Zn", "wt_pct": 30},
            ],
        }
        valid, error = validate_candidate(candidate)
        assert valid is False
        assert "100" in error

    def test_duplicate_element(self):
        candidate = {
            "support": "Al2O3",
            "metals": [
                {"element": "Cu", "wt_pct": 50},
                {"element": "Cu", "wt_pct": 50},
            ],
        }
        valid, error = validate_candidate(candidate)
        assert valid is False
        assert "duplicate" in error.lower()

    def test_already_seen(self):
        candidate = {
            "support": "Al2O3",
            "metals": [{"element": "Cu", "wt_pct": 100}],
        }
        seen = {hash_candidate(candidate)}
        valid, error = validate_candidate(candidate, seen)
        assert valid is False
        assert "duplicate" in error.lower()


class TestNormalizeCandidate:
    """Tests for candidate normalization."""

    def test_rounds_wt_pct(self):
        candidate = {
            "support": "Al2O3",
            "metals": [{"element": "Cu", "wt_pct": 60.123}],
        }
        normalized = normalize_candidate(candidate)
        assert normalized["metals"][0]["wt_pct"] == 60.1

    def test_sorts_metals(self):
        candidate = {
            "support": "Al2O3",
            "metals": [
                {"element": "Zn", "wt_pct": 40},
                {"element": "Cu", "wt_pct": 60},
            ],
        }
        normalized = normalize_candidate(candidate)
        assert normalized["metals"][0]["element"] == "Cu"
        assert normalized["metals"][1]["element"] == "Zn"


# --- State Management Tests ---


class TestGenerationState:
    """Tests for GenerationState."""

    def test_initial_state(self):
        state = GenerationState()
        assert state.iteration == 0
        assert state.best_score == 0.0
        assert len(state.candidates_evaluated) == 0

    def test_update_with_results(self):
        state = GenerationState()
        results = [
            {
                "candidate": {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 100}]},
                "results": [],
                "final_assessment": {"viability_score": 75, "recommendation": "PURSUE"},
                "total_cost": 10.0,
            }
        ]
        state.update_with_results(results)

        assert state.iteration == 1
        assert state.best_score == 75
        assert len(state.candidates_evaluated) == 1
        assert len(state.seen_hashes) == 1

    def test_update_tracks_best(self):
        state = GenerationState()

        # First batch
        state.update_with_results([
            {
                "candidate": {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 100}]},
                "results": [],
                "final_assessment": {"viability_score": 50, "recommendation": "DEPRIORITIZE"},
                "total_cost": 10.0,
            }
        ])
        assert state.best_score == 50

        # Second batch with better candidate
        state.update_with_results([
            {
                "candidate": {"support": "ZrO2", "metals": [{"element": "Cu", "wt_pct": 100}]},
                "results": [],
                "final_assessment": {"viability_score": 80, "recommendation": "PURSUE"},
                "total_cost": 10.0,
            }
        ])
        assert state.best_score == 80
        assert state.best_candidate["support"] == "ZrO2"

    def test_convergence_detection(self):
        state = GenerationState()

        # Add iterations with no improvement
        for _ in range(5):
            state.update_with_results([
                {
                    "candidate": {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 100}]},
                    "results": [],
                    "final_assessment": {"viability_score": 50, "recommendation": "DEPRIORITIZE"},
                    "total_cost": 10.0,
                }
            ])

        # Should detect convergence after patience=3 with no improvement
        converged = state.check_convergence(patience=3, min_improvement=0.01)
        assert converged is True
        assert state.converged is True

    def test_no_convergence_if_improving(self):
        state = GenerationState()

        # Add iterations with improvement each time
        for i in range(5):
            state.update_with_results([
                {
                    "candidate": {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 100}]},
                    "results": [],
                    "final_assessment": {"viability_score": 50 + i * 5, "recommendation": "PURSUE"},
                    "total_cost": 10.0,
                }
            ])

        converged = state.check_convergence(patience=3, min_improvement=0.01)
        assert converged is False

    def test_get_top_performers(self):
        state = GenerationState()

        # Add candidates with varying scores
        for score in [30, 80, 50, 90, 60]:
            state.update_with_results([
                {
                    "candidate": {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": score}]},
                    "results": [],
                    "final_assessment": {"viability_score": score, "recommendation": "PURSUE"},
                    "total_cost": 10.0,
                }
            ])

        top = state.get_top_performers(3)
        scores = [p["final_assessment"]["viability_score"] for p in top]
        assert scores == [90, 80, 60]

    def test_checkpoint_save_load(self):
        state = GenerationState()
        state.update_with_results([
            {
                "candidate": {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 100}]},
                "results": [],
                "final_assessment": {"viability_score": 75, "recommendation": "PURSUE"},
                "total_cost": 10.0,
            }
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            state.save_checkpoint(checkpoint_path)

            loaded = GenerationState.load_checkpoint(checkpoint_path)

            assert loaded.iteration == state.iteration
            assert loaded.best_score == state.best_score
            assert len(loaded.candidates_evaluated) == len(state.candidates_evaluated)
            assert loaded.seen_hashes == state.seen_hashes


# --- Prompt Tests ---


class TestPromptFormatting:
    """Tests for prompt formatting functions."""

    def test_format_candidate(self):
        candidate = {
            "support": "Al2O3",
            "metals": [
                {"element": "Cu", "wt_pct": 60},
                {"element": "Zn", "wt_pct": 40},
            ],
        }
        result = format_candidate(candidate)
        assert "Cu60" in result
        assert "Zn40" in result
        assert "Al2O3" in result

    def test_format_top_performers_empty(self):
        result = format_top_performers_table([])
        assert "No candidates" in result

    def test_format_top_performers_with_data(self):
        performers = [
            {
                "candidate": {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 100}]},
                "final_assessment": {"viability_score": 80, "recommendation": "PURSUE"},
            }
        ]
        result = format_top_performers_table(performers)
        assert "80" in result
        assert "PURSUE" in result
        assert "|" in result  # Table format

    def test_build_proposal_prompt(self):
        prompt = build_proposal_prompt(
            n_candidates=6,
            iteration=3,
            total_evaluated=18,
            best_score=75.0,
            best_candidate={"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 100}]},
            top_performers=[],
            seen_candidates=set(),
        )
        assert "6" in prompt  # n_candidates
        assert "3" in prompt  # iteration
        assert "75" in prompt  # best_score
        assert "JSON" in prompt

    def test_build_convergence_prompt(self):
        prompt = build_convergence_prompt(
            iteration=5,
            total_evaluated=30,
            best_score=80.0,
            score_history=[60, 70, 75, 78, 80],
        )
        assert "CONTINUE" in prompt
        assert "STOP" in prompt
        assert "80" in prompt


# --- GeneratorAgent Tests ---


class TestGeneratorAgent:
    """Tests for GeneratorAgent."""

    @pytest.fixture
    def mock_config(self):
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
                "checkpoint_path": "/tmp/test_checkpoint.json",
                "results_path": "/tmp/test_results.jsonl",
            },
        }

    @pytest.fixture
    def mock_shepherd_config(self):
        return {
            "llm": {
                "mode": "shared",
                "model": "test-model",
                "shared_url": "http://localhost:8000/v1",
            },
            "budget": {"default": 10.0},
            "cache": {"enabled": False},
            "endpoints": {"cheap": None, "gpu": None},
        }

    @pytest.mark.asyncio
    async def test_generator_proposes_candidates(self, mock_config, mock_shepherd_config):
        """Test that generator proposes candidates via LLM."""
        from skills.generator import GeneratorAgent

        with patch("skills.generator.create_llm_client_from_config") as mock_llm_factory, \
             patch("skills.generator.Manager") as mock_manager_class:

            # Mock LLM
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(return_value={
                "candidates": [
                    {
                        "support": "Al2O3",
                        "metals": [
                            {"element": "Cu", "wt_pct": 60},
                            {"element": "Zn", "wt_pct": 25},
                            {"element": "Al", "wt_pct": 15},
                        ],
                    }
                ],
                "reasoning": "Test proposal",
            })
            mock_llm_factory.return_value = mock_llm

            # Mock Manager
            mock_manager = AsyncMock()
            mock_manager_class.from_exchange_factory = AsyncMock(return_value=mock_manager)

            agent = GeneratorAgent(
                config=mock_config,
                shepherd_config=mock_shepherd_config,
            )
            agent._llm = mock_llm
            agent._state = GenerationState()

            # Test proposal
            candidates = await agent._propose_candidates(3)

            assert len(candidates) == 1
            assert candidates[0]["support"] == "Al2O3"
            mock_llm.reason_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_generator_validates_proposals(self, mock_config, mock_shepherd_config):
        """Test that generator validates LLM proposals."""
        from skills.generator import GeneratorAgent

        with patch("skills.generator.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.reason_json = AsyncMock(return_value={
                "candidates": [
                    # Invalid: weights don't sum to 100
                    {
                        "support": "Al2O3",
                        "metals": [{"element": "Cu", "wt_pct": 50}],
                    },
                    # Valid
                    {
                        "support": "Al2O3",
                        "metals": [
                            {"element": "Cu", "wt_pct": 60},
                            {"element": "Zn", "wt_pct": 40},
                        ],
                    },
                ],
                "reasoning": "Test",
            })
            mock_llm_factory.return_value = mock_llm

            agent = GeneratorAgent(
                config=mock_config,
                shepherd_config=mock_shepherd_config,
            )
            agent._llm = mock_llm
            agent._state = GenerationState()

            candidates = await agent._propose_candidates(3)

            # Only the valid candidate should be returned
            assert len(candidates) == 1
            assert candidates[0]["metals"][0]["wt_pct"] == 60

    @pytest.mark.asyncio
    async def test_generator_deduplicates_candidates(self, mock_config, mock_shepherd_config):
        """Test generator removes duplicate candidates from LLM proposals."""
        from skills.generator import GeneratorAgent

        with patch("skills.generator.create_llm_client_from_config") as mock_llm_factory:
            mock_llm = AsyncMock()
            # LLM proposes duplicate candidates
            mock_llm.reason_json = AsyncMock(return_value={
                "candidates": [
                    {
                        "support": "Al2O3",
                        "metals": [
                            {"element": "Cu", "wt_pct": 60},
                            {"element": "Zn", "wt_pct": 25},
                            {"element": "Al", "wt_pct": 15},
                        ],
                    },
                    {
                        "support": "Al2O3",
                        "metals": [
                            {"element": "Cu", "wt_pct": 60},
                            {"element": "Zn", "wt_pct": 25},
                            {"element": "Al", "wt_pct": 15},
                        ],
                    },
                ],
                "reasoning": "Test duplicates",
            })
            mock_llm_factory.return_value = mock_llm

            agent = GeneratorAgent(
                config=mock_config,
                shepherd_config=mock_shepherd_config,
            )
            agent._llm = mock_llm
            agent._state = GenerationState()

            # Mark first candidate as already seen
            from orchestration.generator_state import hash_candidate
            first_candidate = {
                "support": "Al2O3",
                "metals": [
                    {"element": "Cu", "wt_pct": 60},
                    {"element": "Zn", "wt_pct": 25},
                    {"element": "Al", "wt_pct": 15},
                ],
            }
            agent._state.seen_hashes.add(hash_candidate(first_candidate))

            candidates = await agent._propose_candidates(3)

            # Should filter out duplicates and already-seen
            assert len(candidates) == 0


class TestGeneratorStateAdvanced:
    """Advanced tests for GenerationState."""

    def test_score_history_tracking(self):
        """Test that score history is tracked correctly per iteration."""
        state = GenerationState()

        # Iteration 1: best score 60
        state.update_with_results([
            make_result(60),
            make_result(50),
        ])
        assert state.score_history == [60]

        # Iteration 2: best score 70
        state.update_with_results([
            make_result(70),
            make_result(55),
        ])
        assert state.score_history == [60, 70]

        # Iteration 3: best overall still 70, but this iteration only 65
        state.update_with_results([
            make_result(65),
        ])
        assert state.score_history == [60, 70, 70]  # Maintains global best

    def test_convergence_requires_sufficient_history(self):
        """Convergence check requires enough iterations."""
        state = GenerationState()

        # Only 2 iterations
        state.update_with_results([make_result(50)])
        state.update_with_results([make_result(50)])

        # Should not converge with patience=3 and only 2 iterations
        assert state.check_convergence(patience=3) is False

    def test_convergence_with_improvement(self):
        """No convergence when scores are improving."""
        state = GenerationState()

        for score in [50, 55, 60, 65, 70]:
            state.update_with_results([make_result(score)])

        assert state.check_convergence(patience=3, min_improvement=1.0) is False

    def test_convergence_with_stagnation(self):
        """Convergence detected when scores stagnate."""
        state = GenerationState()

        # First some improvement
        for score in [50, 60, 70]:
            state.update_with_results([make_result(score)])

        # Then stagnation
        for _ in range(4):
            state.update_with_results([make_result(70)])

        assert state.check_convergence(patience=3, min_improvement=1.0) is True

    def test_is_candidate_seen(self):
        """Test seen candidate detection."""
        state = GenerationState()

        candidate = {
            "support": "Al2O3",
            "metals": [{"element": "Cu", "wt_pct": 100}],
        }

        assert state.is_candidate_seen(candidate) is False

        state.update_with_results([{
            "candidate": candidate,
            "results": [],
            "final_assessment": {"viability_score": 50, "recommendation": "PURSUE"},
            "total_cost": 1.0,
        }])

        assert state.is_candidate_seen(candidate) is True

    def test_get_summary(self):
        """Test summary generation."""
        state = GenerationState()
        state.update_with_results([make_result(75)])

        summary = state.get_summary()

        assert "iteration" in summary
        assert "total_candidates" in summary
        assert "best_score" in summary
        assert "elapsed_seconds" in summary
        assert summary["best_score"] == 75

    def test_checkpoint_roundtrip_complex(self, temp_checkpoint_path):
        """Test checkpoint with complex state."""
        state = GenerationState()

        # Add multiple iterations with various scores
        for i in range(5):
            state.update_with_results([
                make_result(50 + i * 5, support="Al2O3" if i % 2 == 0 else "ZrO2"),
                make_result(45 + i * 5),
            ])

        state.converged = True
        state.stop_reason = "Test stop"

        # Save and reload
        state.save_checkpoint(temp_checkpoint_path)
        loaded = GenerationState.load_checkpoint(temp_checkpoint_path)

        assert loaded.iteration == state.iteration
        assert loaded.best_score == state.best_score
        assert loaded.converged == state.converged
        assert loaded.stop_reason == state.stop_reason
        assert len(loaded.candidates_evaluated) == len(state.candidates_evaluated)
        assert loaded.score_history == state.score_history


class TestGeneratorPromptsAdvanced:
    """Advanced tests for generator prompts."""

    def test_proposal_prompt_empty_history(self):
        """Proposal prompt works with no history."""
        prompt = build_proposal_prompt(
            n_candidates=6,
            iteration=0,
            total_evaluated=0,
            best_score=0,
            best_candidate=None,
            top_performers=[],
            seen_candidates=set(),
        )

        assert "6" in prompt
        assert "None" in prompt or "yet" in prompt.lower()

    def test_proposal_prompt_with_history(self):
        """Proposal prompt includes evaluation history."""
        top_performers = [
            {
                "candidate": {"support": "Al2O3", "metals": [{"element": "Cu", "wt_pct": 60}]},
                "final_assessment": {"viability_score": 85, "recommendation": "PURSUE"},
            },
            {
                "candidate": {"support": "ZrO2", "metals": [{"element": "Cu", "wt_pct": 55}]},
                "final_assessment": {"viability_score": 78, "recommendation": "PURSUE"},
            },
        ]

        prompt = build_proposal_prompt(
            n_candidates=6,
            iteration=5,
            total_evaluated=30,
            best_score=85,
            best_candidate=top_performers[0]["candidate"],
            top_performers=top_performers,
            seen_candidates={"Cu60@Al2O3", "Cu55@ZrO2"},
        )

        assert "85" in prompt
        assert "5" in prompt  # iteration
        assert "30" in prompt  # total evaluated

    def test_convergence_prompt_content(self):
        """Convergence prompt includes progress indicators."""
        prompt = build_convergence_prompt(
            iteration=10,
            total_evaluated=60,
            best_score=82,
            score_history=[50, 60, 70, 75, 78, 80, 81, 81, 82, 82],
        )

        assert "CONTINUE" in prompt
        assert "STOP" in prompt
        assert "82" in prompt

    def test_pattern_summary_with_data(self):
        """Pattern summary identifies trends."""
        performers = [
            {
                "candidate": {"support": "Al2O3", "metals": [
                    {"element": "Cu", "wt_pct": 60},
                    {"element": "Zn", "wt_pct": 25},
                ]},
                "final_assessment": {"viability_score": 85},
            },
            {
                "candidate": {"support": "Al2O3", "metals": [
                    {"element": "Cu", "wt_pct": 55},
                    {"element": "Zn", "wt_pct": 30},
                ]},
                "final_assessment": {"viability_score": 80},
            },
            {
                "candidate": {"support": "Al2O3", "metals": [
                    {"element": "Cu", "wt_pct": 58},
                    {"element": "Zn", "wt_pct": 27},
                ]},
                "final_assessment": {"viability_score": 82},
            },
        ]

        summary = format_patterns_summary(performers)

        assert "Al2O3" in summary
        assert "Cu" in summary


class TestCandidateValidationAdvanced:
    """Advanced tests for candidate validation."""

    def test_validate_with_all_supports(self):
        """All valid supports should pass."""
        for support in ["Al2O3", "ZrO2", "SiO2"]:
            candidate = {
                "support": support,
                "metals": [{"element": "Cu", "wt_pct": 100}],
            }
            valid, error = validate_candidate(candidate)
            assert valid is True, f"Support {support} should be valid"

    def test_validate_weight_tolerance(self):
        """Weight sum has small tolerance."""
        # Should pass: 99.9 is close enough to 100
        candidate = {
            "support": "Al2O3",
            "metals": [
                {"element": "Cu", "wt_pct": 59.9},
                {"element": "Zn", "wt_pct": 40},
            ],
        }
        valid, _ = validate_candidate(candidate)
        assert valid is True

        # Should fail: 99 is too far from 100
        candidate["metals"][0]["wt_pct"] = 59
        valid, _ = validate_candidate(candidate)
        assert valid is False

    def test_normalize_handles_missing_fields(self):
        """Normalize handles incomplete candidates."""
        candidate = {"metals": []}
        normalized = normalize_candidate(candidate)
        assert normalized["support"] == ""
        assert normalized["metals"] == []


# --- Helper function for tests ---


def make_result(score: int, support: str = "Al2O3") -> dict:
    """Create a simple shepherd result for testing."""
    return {
        "candidate": {
            "support": support,
            "metals": [{"element": "Cu", "wt_pct": 100}],
        },
        "results": [],
        "final_assessment": {
            "viability_score": score,
            "recommendation": "PURSUE" if score >= 60 else "DEPRIORITIZE",
        },
        "total_cost": 1.0,
    }


# --- Integration Test ---


@pytest.mark.asyncio
@pytest.mark.integration
async def test_generator_full_loop():
    """Full integration test with mocked LLM and shepherds.

    This tests the complete generation loop with mocked dependencies.
    """
    pytest.skip("Integration test - run manually")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
