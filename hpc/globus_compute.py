from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from globus_compute_sdk import Client  # pip install globus-compute-sdk


@dataclass
class SubmittedTask:
    task_id: str
    endpoint_id: str
    function_id: str


class GlobusComputeAdapter:
    def __init__(self, endpoint_id: str):
        self.endpoint_id = endpoint_id
        self.gc = Client()

    def _task_details(self, t: Dict[str, Any]) -> Dict[str, Any]:
        details = t.get("details") or {}
        # Normalize presence of transitions
        trans = details.get("task_transitions") or {}
        details["task_transitions"] = trans
        return details

    def submit(self, function_id: str, payload: Dict[str, Any]) -> SubmittedTask:
        task_id = self.gc.run(payload, endpoint_id=self.endpoint_id, function_id=function_id)
        return SubmittedTask(task_id=task_id, endpoint_id=self.endpoint_id, function_id=function_id)

    def get_task(self, task_id: str) -> Dict[str, Any]:
        return self.gc.get_task(task_id)

def try_result(self, task_id: str) -> Dict[str, Any]:
    t = self.gc.get_task(task_id)

    pending = t.get("pending", None)
    status_raw = (t.get("status") or "").lower()
    details = self._task_details(t)

    # SUCCEEDED
    if pending is False and status_raw in ("success", "succeeded"):
        return {
            "task_id": task_id,
            "status": "SUCCEEDED",
            "result": t.get("result"),
            "details": details,
            "pending": False,
            "raw_status": status_raw,
            "completion_t": t.get("completion_t"),
        }

    # FAILED
    if pending is False and status_raw in ("failed", "failure", "error"):
        return {
            "task_id": task_id,
            "status": "FAILED",
            "error": t.get("exception") or t.get("details") or t,
            "details": details,
            "pending": False,
            "raw_status": status_raw,
            "completion_t": t.get("completion_t"),
        }

    # PENDING / RUNNING
    if pending is True:
        # Some deployments report "pending", "waiting-for-ep", "running", etc.
        return {
            "task_id": task_id,
            "status": "PENDING",
            "details": details,
            "pending": True,
            "raw_status": status_raw,
        }

    # Fallback (unknown shape)
    return {
        "task_id": task_id,
        "status": status_raw.upper() or "UNKNOWN",
        "details": details,
        "pending": pending,
        "raw_status": status_raw,
    }
