"""State management for GeneratorAgent."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def hash_candidate(candidate: dict[str, Any]) -> str:
    """Generate unique hash for a candidate."""
    # Normalize the candidate to ensure consistent hashing
    normalized = {
        "support": candidate.get("support", ""),
        "metals": sorted(
            [
                {"element": m.get("element", ""), "wt_pct": round(m.get("wt_pct", 0), 1)}
                for m in candidate.get("metals", [])
            ],
            key=lambda x: x["element"],
        ),
    }
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def candidate_to_str(candidate: dict[str, Any]) -> str:
    """Convert candidate to human-readable string for seen set."""
    support = candidate.get("support", "?")
    metals = candidate.get("metals", [])
    metal_str = "/".join(
        f"{m.get('element', '?')}{m.get('wt_pct', 0):.0f}"
        for m in sorted(metals, key=lambda x: x.get("element", ""))
    )
    return f"{metal_str}@{support}"


@dataclass
class EvaluatedCandidate:
    """Record of an evaluated candidate."""

    candidate: dict[str, Any]
    candidate_hash: str
    results: list[dict[str, Any]]
    final_assessment: dict[str, Any]
    viability_score: float
    recommendation: str
    total_cost: float
    iteration: int
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_shepherd_result(
        cls,
        result: dict[str, Any],
        iteration: int,
    ) -> "EvaluatedCandidate":
        """Create from ShepherdAgent result."""
        candidate = result.get("candidate", {})
        assessment = result.get("final_assessment", {})
        return cls(
            candidate=candidate,
            candidate_hash=hash_candidate(candidate),
            results=result.get("results", []),
            final_assessment=assessment,
            viability_score=assessment.get("viability_score", 0),
            recommendation=assessment.get("recommendation", "UNKNOWN"),
            total_cost=result.get("total_cost", 0),
            iteration=iteration,
        )


@dataclass
class GenerationState:
    """State for the generation loop.

    Tracks all evaluated candidates, convergence metrics, and iteration history.
    """

    iteration: int = 0
    candidates_evaluated: list[EvaluatedCandidate] = field(default_factory=list)
    seen_hashes: set[str] = field(default_factory=set)
    seen_strings: set[str] = field(default_factory=set)  # Human-readable for prompts
    best_score: float = 0.0
    best_candidate: dict[str, Any] | None = None
    best_assessment: dict[str, Any] | None = None
    score_history: list[float] = field(default_factory=list)  # Best score per iteration
    converged: bool = False
    stop_reason: str | None = None
    total_cost: float = 0.0
    start_time: float = field(default_factory=time.time)

    def update_with_results(self, results: list[dict[str, Any]]) -> None:
        """Update state with new shepherd results.

        Args:
            results: List of ShepherdAgent result dicts
        """
        iteration_best_score = self.best_score

        for result in results:
            if "error" in result:
                logger.warning("Skipping failed result: %s", result.get("error"))
                continue

            evaluated = EvaluatedCandidate.from_shepherd_result(result, self.iteration)

            # Add to evaluated list
            self.candidates_evaluated.append(evaluated)

            # Track seen candidates
            self.seen_hashes.add(evaluated.candidate_hash)
            self.seen_strings.add(candidate_to_str(evaluated.candidate))

            # Track costs
            self.total_cost += evaluated.total_cost

            # Update best if improved
            if evaluated.viability_score > self.best_score:
                self.best_score = evaluated.viability_score
                self.best_candidate = evaluated.candidate
                self.best_assessment = evaluated.final_assessment
                logger.info(
                    "New best candidate: %s (score=%.1f)",
                    candidate_to_str(evaluated.candidate),
                    evaluated.viability_score,
                )

            # Track iteration best
            if evaluated.viability_score > iteration_best_score:
                iteration_best_score = evaluated.viability_score

        # Record best score this iteration
        self.score_history.append(iteration_best_score)
        self.iteration += 1

    def check_convergence(
        self,
        patience: int = 3,
        min_improvement: float = 0.01,
    ) -> bool:
        """Check if optimization has converged.

        Convergence is detected when the best score hasn't improved
        by at least min_improvement for `patience` consecutive iterations.

        Args:
            patience: Number of iterations without improvement to trigger convergence
            min_improvement: Minimum score improvement to count as progress

        Returns:
            True if converged, False otherwise
        """
        if len(self.score_history) < patience + 1:
            return False

        recent = self.score_history[-patience:]
        baseline = self.score_history[-(patience + 1)]

        # Check if any recent iteration improved over baseline
        improved = any(score > baseline + min_improvement for score in recent)

        if not improved:
            self.converged = True
            self.stop_reason = f"No improvement > {min_improvement} for {patience} iterations"
            return True

        return False

    def get_top_performers(self, n: int = 10) -> list[dict[str, Any]]:
        """Get top N candidates by viability score.

        Args:
            n: Number of top performers to return

        Returns:
            List of dicts with 'candidate' and 'final_assessment' keys
        """
        sorted_candidates = sorted(
            self.candidates_evaluated,
            key=lambda x: x.viability_score,
            reverse=True,
        )

        return [
            {
                "candidate": c.candidate,
                "final_assessment": c.final_assessment,
                "total_cost": c.total_cost,
                "iteration": c.iteration,
            }
            for c in sorted_candidates[:n]
        ]

    def is_candidate_seen(self, candidate: dict[str, Any]) -> bool:
        """Check if a candidate has already been evaluated."""
        return hash_candidate(candidate) in self.seen_hashes

    def get_summary(self) -> dict[str, Any]:
        """Get summary of current state."""
        elapsed = time.time() - self.start_time
        return {
            "iteration": self.iteration,
            "total_candidates": len(self.candidates_evaluated),
            "best_score": self.best_score,
            "best_candidate": candidate_to_str(self.best_candidate) if self.best_candidate else None,
            "total_cost": self.total_cost,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "elapsed_seconds": elapsed,
            "score_history": self.score_history,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save state to checkpoint file.

        Args:
            path: Path to checkpoint file (JSON)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "iteration": self.iteration,
            "candidates_evaluated": [asdict(c) for c in self.candidates_evaluated],
            "seen_hashes": list(self.seen_hashes),
            "seen_strings": list(self.seen_strings),
            "best_score": self.best_score,
            "best_candidate": self.best_candidate,
            "best_assessment": self.best_assessment,
            "score_history": self.score_history,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "total_cost": self.total_cost,
            "start_time": self.start_time,
            "checkpoint_time": time.time(),
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved checkpoint to %s", path)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "GenerationState":
        """Load state from checkpoint file.

        Args:
            path: Path to checkpoint file

        Returns:
            Restored GenerationState
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        state = cls(
            iteration=data["iteration"],
            seen_hashes=set(data["seen_hashes"]),
            seen_strings=set(data["seen_strings"]),
            best_score=data["best_score"],
            best_candidate=data["best_candidate"],
            best_assessment=data["best_assessment"],
            score_history=data["score_history"],
            converged=data["converged"],
            stop_reason=data["stop_reason"],
            total_cost=data["total_cost"],
            start_time=data["start_time"],
        )

        # Reconstruct evaluated candidates
        for c_data in data["candidates_evaluated"]:
            state.candidates_evaluated.append(
                EvaluatedCandidate(
                    candidate=c_data["candidate"],
                    candidate_hash=c_data["candidate_hash"],
                    results=c_data["results"],
                    final_assessment=c_data["final_assessment"],
                    viability_score=c_data["viability_score"],
                    recommendation=c_data["recommendation"],
                    total_cost=c_data["total_cost"],
                    iteration=c_data["iteration"],
                    timestamp=c_data.get("timestamp", 0),
                )
            )

        logger.info("Loaded checkpoint from %s (iteration %d)", path, state.iteration)
        return state


def validate_candidate(
    candidate: dict[str, Any],
    seen_hashes: set[str] | None = None,
) -> tuple[bool, str | None]:
    """Validate a candidate meets chemistry constraints.

    Args:
        candidate: Candidate dict with 'support' and 'metals' keys
        seen_hashes: Optional set of already-evaluated candidate hashes

    Returns:
        Tuple of (valid: bool, error_message: str | None)
    """
    # Check support
    support = candidate.get("support", "")
    valid_supports = ["Al2O3", "ZrO2", "SiO2"]
    if support not in valid_supports:
        return False, f"Invalid support '{support}'. Must be one of {valid_supports}"

    # Check metals
    metals = candidate.get("metals", [])
    if not metals:
        return False, "No metals specified"

    valid_elements = ["Cu", "Zn", "Al"]
    total_wt = 0.0
    seen_elements = set()

    for m in metals:
        element = m.get("element", "")
        wt_pct = m.get("wt_pct", 0)

        if element not in valid_elements:
            return False, f"Invalid element '{element}'. Must be one of {valid_elements}"

        if element in seen_elements:
            return False, f"Duplicate element '{element}'"
        seen_elements.add(element)

        if not isinstance(wt_pct, (int, float)):
            return False, f"wt_pct must be numeric, got {type(wt_pct)}"

        if wt_pct < 0 or wt_pct > 100:
            return False, f"wt_pct must be 0-100, got {wt_pct}"

        total_wt += wt_pct

    # Check weight sum
    if abs(total_wt - 100) > 0.5:
        return False, f"Metal weights must sum to 100%, got {total_wt:.1f}%"

    # Check uniqueness
    if seen_hashes:
        candidate_hash = hash_candidate(candidate)
        if candidate_hash in seen_hashes:
            return False, "Duplicate candidate (already evaluated)"

    return True, None


def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Normalize a candidate to canonical form.

    - Rounds wt_pct to 1 decimal
    - Sorts metals by element
    - Ensures all required fields present

    Args:
        candidate: Raw candidate dict

    Returns:
        Normalized candidate dict
    """
    metals = candidate.get("metals", [])
    normalized_metals = sorted(
        [
            {
                "element": m.get("element", ""),
                "wt_pct": round(m.get("wt_pct", 0), 1),
            }
            for m in metals
        ],
        key=lambda x: x["element"],
    )

    return {
        "support": candidate.get("support", ""),
        "metals": normalized_metals,
    }
