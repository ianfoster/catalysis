"""Narrative logger for catalyst discovery.

Captures a human-readable story of the discovery process.
Supports Redis pub/sub for distributed logging across machines.
"""

import json
import socket
from datetime import datetime
from pathlib import Path
from typing import Any


def get_hostname() -> str:
    """Get short hostname for location tagging."""
    hostname = socket.gethostname()
    # Shorten common patterns
    if "." in hostname:
        hostname = hostname.split(".")[0]
    return hostname


class NarrativeLogger:
    """Logs a human-readable narrative of the discovery process.

    Can optionally publish to Redis for distributed visibility.
    """

    CHANNEL = "catalyst:narrative"

    def __init__(
        self,
        path: str = "data/narrative.log",
        redis_host: str | None = None,
        redis_port: int = 6379,
    ):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Start fresh for local file
        self._path.write_text("")
        self._hostname = get_hostname()

        # Redis pub/sub setup
        self._redis = None
        if redis_host:
            try:
                import redis
                self._redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                self._redis.ping()  # Test connection
            except Exception as e:
                print(f"Warning: Could not connect to Redis at {redis_host}:{redis_port}: {e}")
                self._redis = None

    def _write(self, text: str) -> None:
        """Append text to narrative log and optionally publish to Redis."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}"

        # Write to local file
        with open(self._path, "a") as f:
            f.write(line + "\n")

        # Publish to Redis if connected
        if self._redis:
            try:
                self._redis.publish(self.CHANNEL, line)
            except Exception:
                pass  # Don't fail on Redis errors

    def _loc(self, location: str = None) -> str:
        """Format location tag."""
        loc = location or self._hostname
        return f"[{loc}]"

    def section(self, title: str) -> None:
        """Write a section header."""
        self._write("")
        self._write("=" * 60)
        self._write(f"  {title}")
        self._write("=" * 60)

    def generator_start(self, iteration: int, max_iter: int, candidates_per_iter: int) -> None:
        """Log generator starting."""
        self.section(f"ITERATION {iteration} OF {max_iter}")
        self._write(f"{self._loc()} Generator starting iteration {iteration}")
        self._write(f"{self._loc()} Will propose {candidates_per_iter} new catalyst candidates")

    def generator_proposing(self, top_performers: list[dict]) -> None:
        """Log generator proposing candidates."""
        self._write("")
        self._write(f"{self._loc()} GENERATOR: Asking LLM to propose new candidates...")
        if top_performers:
            self._write(f"   Context: {len(top_performers)} top performers from previous iterations")
            best = top_performers[0] if top_performers else {}
            if best:
                score = best.get("final_assessment", {}).get("viability_score", 0)
                self._write(f"   Best so far: score={score}")

    def generator_proposed(self, candidates: list[dict], reasoning: str) -> None:
        """Log proposed candidates."""
        self._write("")
        self._write(f"{self._loc()} GENERATOR: LLM proposed {len(candidates)} candidates")
        self._write(f"   Reasoning: {reasoning[:200]}..." if len(reasoning) > 200 else f"   Reasoning: {reasoning}")
        for i, c in enumerate(candidates, 1):
            metals = c.get("metals", [])
            metal_str = "/".join(f"{m['element']}{m['wt_pct']}" for m in metals)
            self._write(f"   {i}. {metal_str} on {c.get('support', '?')}")

    def shepherd_start(self, candidate: dict, budget: float) -> None:
        """Log shepherd starting evaluation."""
        metals = candidate.get("metals", [])
        metal_str = "/".join(f"{m['element']}{m['wt_pct']}" for m in metals)
        candidate_str = f"{metal_str} on {candidate.get('support', '?')}"

        self._write("")
        self._write(f"{self._loc()} SHEPHERD: Starting evaluation of {candidate_str}")
        self._write(f"   Budget: {budget} units")

    def shepherd_thinking(self, prompt_summary: str) -> None:
        """Log shepherd asking LLM for next action."""
        self._write("")
        self._write(f"{self._loc()}    Shepherd asking LLM: What test should I run next?")
        self._write(f"      Context: {prompt_summary[:80]}...")

    def shepherd_decision(self, test_name: str, reasoning: str, cost: float) -> None:
        """Log shepherd's decision."""
        self._write("")
        self._write(f"{self._loc()}    LLM Decision: Run '{test_name}' (cost: {cost})")
        self._write(f"      Reasoning: {reasoning[:200]}..." if len(reasoning) > 200 else f"      Reasoning: {reasoning}")

    def shepherd_running_test(self, test_name: str, agent_name: str = None) -> None:
        """Log test execution start."""
        if agent_name:
            self._write(f"{self._loc()}    Dispatching '{test_name}' to {agent_name} agent...")
        else:
            self._write(f"{self._loc()}    Running test '{test_name}'...")

    def shepherd_test_result(self, test_name: str, result: dict, elapsed: float = None) -> None:
        """Log test result."""
        # Extract key metrics (skip internal fields)
        key_metrics = []
        skip_keys = {"ok", "cost", "elapsed_s", "error"}
        for k, v in result.items():
            if k in skip_keys:
                continue
            if isinstance(v, float):
                key_metrics.append(f"{k}={v:.3f}")
            elif isinstance(v, int):
                key_metrics.append(f"{k}={v}")

        # Use elapsed from result if not passed explicitly
        if elapsed is None:
            elapsed = result.get("elapsed_s")

        elapsed_str = f" ({elapsed:.1f}s)" if elapsed else ""
        self._write(f"{self._loc()}    Test '{test_name}' complete{elapsed_str}")
        if key_metrics:
            self._write(f"      Results: {', '.join(key_metrics[:5])}")

    def shepherd_done(self, candidate: dict, final_score: float, recommendation: str, tests_run: int, total_cost: float) -> None:
        """Log shepherd completion."""
        metals = candidate.get("metals", [])
        metal_str = "/".join(f"{m['element']}{m['wt_pct']}" for m in metals)
        candidate_str = f"{metal_str} on {candidate.get('support', '?')}"

        self._write("")
        self._write(f"{self._loc()} SHEPHERD: Evaluation complete for {candidate_str}")
        self._write(f"   Final Score: {final_score}")
        self._write(f"   Recommendation: {recommendation}")
        self._write(f"   Tests run: {tests_run}, Total cost: {total_cost:.1f}")

    def generator_dispatching(self, candidate: dict, shepherd_id: str, target_host: str = "Spark") -> None:
        """Log dispatching candidate to remote shepherd."""
        metals = candidate.get("metals", [])
        metal_str = "/".join(f"{m['element']}{m['wt_pct']}" for m in metals)
        candidate_str = f"{metal_str} on {candidate.get('support', '?')}"

        self._write("")
        self._write(f"{self._loc()} Dispatching {candidate_str} to ShepherdAgent on {target_host}")
        self._write(f"   Shepherd ID: {shepherd_id}")

    def generator_received_result(self, candidate: dict, score: float, from_host: str = "Spark") -> None:
        """Log receiving result from remote shepherd."""
        metals = candidate.get("metals", [])
        metal_str = "/".join(f"{m['element']}{m['wt_pct']}" for m in metals)
        candidate_str = f"{metal_str} on {candidate.get('support', '?')}"

        self._write(f"{self._loc()} Received result for {candidate_str} from {from_host}: score={score}")

    def generator_iteration_done(self, iteration: int, results: list[dict]) -> None:
        """Log iteration completion."""
        scores = [r.get("final_assessment", {}).get("viability_score", 0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        best_score = max(scores) if scores else 0

        self._write("")
        self._write(f"{self._loc()} ITERATION {iteration} COMPLETE")
        self._write(f"   Candidates evaluated: {len(results)}")
        self._write(f"   Average score: {avg_score:.1f}")
        self._write(f"   Best score this iteration: {best_score}")

    def generator_converged(self, reason: str, total_candidates: int, best_score: float) -> None:
        """Log convergence."""
        self.section("DISCOVERY COMPLETE")
        self._write(f"{self._loc()} Reason: {reason}")
        self._write(f"{self._loc()} Total candidates evaluated: {total_candidates}")
        self._write(f"{self._loc()} Best score achieved: {best_score}")


# Global instance
_narrative: NarrativeLogger | None = None


def get_narrative(
    path: str = "data/narrative.log",
    redis_host: str | None = None,
    redis_port: int = 6379,
) -> NarrativeLogger:
    """Get or create the global narrative logger."""
    global _narrative
    if _narrative is None:
        _narrative = NarrativeLogger(path, redis_host=redis_host, redis_port=redis_port)
    return _narrative


def reset_narrative(
    path: str = "data/narrative.log",
    redis_host: str | None = None,
    redis_port: int = 6379,
) -> NarrativeLogger:
    """Reset and return a fresh narrative logger."""
    global _narrative
    _narrative = NarrativeLogger(path, redis_host=redis_host, redis_port=redis_port)
    return _narrative


def subscribe_narrative(redis_host: str, redis_port: int = 6379) -> None:
    """Subscribe to narrative events from Redis and print them.

    This is a blocking function that prints all narrative events.
    Run this in a separate terminal to watch the distributed narrative.
    """
    import redis

    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    pubsub = r.pubsub()
    pubsub.subscribe(NarrativeLogger.CHANNEL)

    print(f"Subscribed to {NarrativeLogger.CHANNEL} on {redis_host}:{redis_port}")
    print("Waiting for narrative events...\n")

    for message in pubsub.listen():
        if message["type"] == "message":
            print(message["data"])
