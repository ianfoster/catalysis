#!/usr/bin/env python3
"""Monitor catalyst discovery progress.

Usage:
    python scripts/monitor_progress.py
    python scripts/monitor_progress.py --watch  # Auto-refresh every 10s
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime


def load_state(path: str = "data/generator_state.json") -> dict | None:
    """Load generator state."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_results(path: str = "data/generator_results.jsonl") -> list[dict]:
    """Load all results."""
    p = Path(path)
    if not p.exists():
        return []
    results = []
    with open(p) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def format_candidate(c: dict) -> str:
    """Format candidate as string."""
    metals = c.get("metals", [])
    metal_str = "/".join(f"{m['element']}{m['wt_pct']}" for m in metals)
    return f"{metal_str} on {c.get('support', '?')}"


def print_summary(state: dict, results: list[dict]) -> None:
    """Print progress summary."""
    print("=" * 70)
    print(f"  CATALYST DISCOVERY PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Basic stats
    iteration = state.get("iteration", 0)
    total = len(state.get("candidates_evaluated", []))
    best_score = state.get("best_score", 0)

    print(f"\n📊 OVERVIEW")
    print(f"   Iteration:          {iteration}")
    print(f"   Total candidates:   {total}")
    print(f"   Best score:         {best_score}")
    print(f"   Converged:          {state.get('converged', False)}")
    if state.get("stop_reason"):
        print(f"   Stop reason:        {state.get('stop_reason')}")

    # Score history
    score_history = state.get("score_history", [])
    if score_history:
        print(f"\n📈 SCORE HISTORY (best per iteration)")
        for i, score in enumerate(score_history, 1):
            bar = "█" * int(score / 10)
            print(f"   Iter {i}: {score:5.1f} {bar}")

    # Top performers
    top = state.get("top_performers", [])[:5]
    if top:
        print(f"\n🏆 TOP 5 CANDIDATES")
        for i, entry in enumerate(top, 1):
            c = entry.get("candidate", {})
            assessment = entry.get("final_assessment", {})
            score = assessment.get("viability_score", 0)
            rec = assessment.get("recommendation", "?")
            print(f"   {i}. Score={score:3.0f} [{rec:12s}] {format_candidate(c)}")

    # Recent evaluations
    evaluated = state.get("candidates_evaluated", [])[-5:]
    if evaluated:
        print(f"\n🔬 RECENT EVALUATIONS")
        for entry in evaluated:
            c = entry.get("candidate", {})
            assessment = entry.get("final_assessment", {})
            score = assessment.get("viability_score", 0)
            tests = [r.get("test", "?") for r in entry.get("results", [])]
            print(f"   • {format_candidate(c)}")
            print(f"     Score: {score}, Tests: {', '.join(tests[:4])}")

    # Test breakdown from results
    if results:
        print(f"\n🧪 TEST STATISTICS")
        test_counts = {}
        for r in results:
            for test_result in r.get("results", []):
                test = test_result.get("test", "unknown")
                test_counts[test] = test_counts.get(test, 0) + 1
        for test, count in sorted(test_counts.items(), key=lambda x: -x[1]):
            print(f"   {test}: {count} runs")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Monitor catalyst discovery progress")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch mode (refresh every 10s)")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds")
    parser.add_argument("--state", default="data/generator_state.json", help="State file path")
    parser.add_argument("--results", default="data/generator_results.jsonl", help="Results file path")
    args = parser.parse_args()

    while True:
        # Clear screen in watch mode
        if args.watch:
            print("\033[2J\033[H", end="")

        state = load_state(args.state)
        results = load_results(args.results)

        if state:
            print_summary(state, results)
        else:
            print("No state file found. Waiting for generator to start...")

        if not args.watch:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
