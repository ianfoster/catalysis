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
        t = self.get_task(task_id)
        status = t.get("status", "UNKNOWN")

        if status == "SUCCEEDED":
            # result is whatever your function returns (must be serializable)
            return {"status": "SUCCEEDED", "result": self.gc.get_result(task_id)}
        if status == "FAILED":
            return {"status": "FAILED", "error": t.get("exception", "unknown")}

        return {"status": status}
