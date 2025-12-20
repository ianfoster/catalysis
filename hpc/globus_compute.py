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

    def submit(self, function_id: str, payload: Dict[str, Any]) -> SubmittedTask:
        task_id = self.gc.run(payload, endpoint_id=self.endpoint_id, function_id=function_id)
        return SubmittedTask(task_id=task_id, endpoint_id=self.endpoint_id, function_id=function_id)

    def get_task(self, task_id: str) -> Dict[str, Any]:
        return self.gc.get_task(task_id)

    def try_result(self, task_id: str) -> Dict[str, Any]:
        t = self.gc.get_task(task_id)
    
        # Globus Compute SDK v4 returns "pending": bool and "status": "success"/"failed"/...
        pending = t.get("pending", None)
        status_raw = (t.get("status") or "").lower()
    
        if pending is False and status_raw in ("success", "succeeded"):
            # result is already included in get_task() under key "result" in your output
            return {"status": "SUCCEEDED", "result": t.get("result"), "details": t.get("details")}
    
        if pending is False and status_raw in ("failed", "failure", "error"):
            # Some failures may include error fields in details; keep task payload
            return {"status": "FAILED", "error": t.get("exception") or t.get("details") or t}
    
        # Otherwise, still pending/running
        # Some deployments may report "active" or similar; keep it as RUNNING if not pending False
        if pending is True:
            return {"status": "PENDING"}
    
        # Fallback
        return {"status": status_raw.upper() or "UNKNOWN"}
