"""Base agent with history tracking for all simulation agents."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from collections import deque

from academy.agent import Agent, action


@dataclass
class ActionRecord:
    """Record of a single action execution."""

    action_name: str
    timestamp: str
    duration_s: float
    input_summary: str
    output_summary: str
    success: bool
    error: str | None = None


class TrackedAgent(Agent):
    """Base agent class with built-in history tracking.

    All simulation agents should inherit from this class to get
    automatic action tracking and history querying.

    Usage:
        class MyAgent(TrackedAgent):
            @action
            async def my_action(self, request: dict) -> dict:
                with self.track_action("my_action", request):
                    # do work
                    result = {...}
                    return result
    """

    def __init__(self, max_history: int = 100):
        """Initialize TrackedAgent.

        Args:
            max_history: Maximum number of action records to keep
        """
        super().__init__()
        self._history: deque[ActionRecord] = deque(maxlen=max_history)
        self._total_actions: int = 0
        self._total_time_s: float = 0.0
        self._action_counts: dict[str, int] = {}
        self._action_times: dict[str, float] = {}
        self._start_time: str = datetime.now().isoformat()

    def track_action(
        self,
        action_name: str,
        request: dict[str, Any],
    ) -> "ActionTracker":
        """Context manager to track an action's execution.

        Args:
            action_name: Name of the action being executed
            request: The request dict passed to the action

        Returns:
            ActionTracker context manager

        Usage:
            with self.track_action("screening", request) as tracker:
                result = do_calculation()
                tracker.set_result(result)
                return result
        """
        return ActionTracker(self, action_name, request)

    def _record_action(
        self,
        action_name: str,
        duration_s: float,
        input_summary: str,
        output_summary: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record an action execution in history."""
        record = ActionRecord(
            action_name=action_name,
            timestamp=datetime.now().isoformat(),
            duration_s=round(duration_s, 3),
            input_summary=input_summary,
            output_summary=output_summary,
            success=success,
            error=error,
        )
        self._history.append(record)

        # Update statistics
        self._total_actions += 1
        self._total_time_s += duration_s
        self._action_counts[action_name] = self._action_counts.get(action_name, 0) + 1
        self._action_times[action_name] = self._action_times.get(action_name, 0.0) + duration_s

    @action
    async def get_history(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get action history.

        Args:
            request: Optional filters:
                - last_n: Return only last N records (default: all)
                - action_name: Filter by action name

        Returns:
            Dict with history records and statistics
        """
        last_n = request.get("last_n")
        action_filter = request.get("action_name")

        records = list(self._history)

        if action_filter:
            records = [r for r in records if r.action_name == action_filter]

        if last_n:
            records = records[-last_n:]

        return {
            "ok": True,
            "history": [
                {
                    "action": r.action_name,
                    "timestamp": r.timestamp,
                    "duration_s": r.duration_s,
                    "input": r.input_summary,
                    "output": r.output_summary,
                    "success": r.success,
                    "error": r.error,
                }
                for r in records
            ],
            "statistics": self._get_statistics(),
        }

    @action
    async def get_statistics(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dict with action counts, times, and averages
        """
        return {
            "ok": True,
            **self._get_statistics(),
        }

    def _get_statistics(self) -> dict[str, Any]:
        """Build statistics dict."""
        avg_times = {
            name: round(self._action_times[name] / self._action_counts[name], 3)
            for name in self._action_counts
        }

        return {
            "agent_start_time": self._start_time,
            "total_actions": self._total_actions,
            "total_time_s": round(self._total_time_s, 2),
            "action_counts": self._action_counts.copy(),
            "action_total_times_s": {k: round(v, 2) for k, v in self._action_times.items()},
            "action_avg_times_s": avg_times,
        }

    @staticmethod
    def summarize_input(request: dict[str, Any], max_len: int = 100) -> str:
        """Create a brief summary of request input."""
        candidate = request.get("candidate")
        if candidate:
            metals = candidate.get("metals", [])
            support = candidate.get("support", "?")
            metal_str = "/".join(f"{m.get('element', '?')}{m.get('wt_pct', 0)}" for m in metals)
            return f"{metal_str} on {support}"

        # Generic summary
        keys = list(request.keys())[:3]
        return f"keys: {keys}"

    @staticmethod
    def summarize_output(result: dict[str, Any], max_len: int = 100) -> str:
        """Create a brief summary of result output."""
        if not result:
            return "empty"

        ok = result.get("ok", True)
        if not ok:
            return f"error: {result.get('error', 'unknown')[:50]}"

        # Extract numeric results
        numerics = []
        for k, v in result.items():
            if k in ("ok", "method", "error"):
                continue
            if isinstance(v, (int, float)):
                numerics.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")

        if numerics:
            summary = ", ".join(numerics[:4])
            return summary[:max_len]

        return "ok"


class ActionTracker:
    """Context manager for tracking action execution."""

    def __init__(
        self,
        agent: TrackedAgent,
        action_name: str,
        request: dict[str, Any],
    ):
        self._agent = agent
        self._action_name = action_name
        self._request = request
        self._result: dict[str, Any] | None = None
        self._error: str | None = None
        self._start_time: float = 0.0

    def set_result(self, result: dict[str, Any]) -> None:
        """Set the result of the action."""
        self._result = result

    def __enter__(self) -> "ActionTracker":
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.time() - self._start_time

        if exc_type is not None:
            self._error = str(exc_val)
            success = False
            output_summary = f"exception: {exc_type.__name__}"
        else:
            success = self._result.get("ok", True) if self._result else True
            output_summary = TrackedAgent.summarize_output(self._result or {})
            if not success:
                self._error = self._result.get("error") if self._result else None

        input_summary = TrackedAgent.summarize_input(self._request)

        self._agent._record_action(
            action_name=self._action_name,
            duration_s=duration,
            input_summary=input_summary,
            output_summary=output_summary,
            success=success,
            error=self._error,
        )

        return False  # Don't suppress exceptions
