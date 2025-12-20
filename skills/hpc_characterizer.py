from __future__ import annotations

from academy.agent import Agent, action
from hpc.globus_compute import GlobusComputeAdapter


class HPCCharacterizerSkill(Agent):
    """
    Pure-function Globus Compute dispatcher for characterization functions.

    function_map: maps characterizer name -> Globus Compute function_id
    """

    def __init__(self, endpoint_id: str, function_map: dict[str, str]):
        self.endpoint_id = endpoint_id
        self.function_map = function_map
        self.gc = GlobusComputeAdapter(endpoint_id)

    @action
    async def submit(self, req: dict) -> dict:
        name = req["characterizer"]
        payload = req["payload"]

        if name not in self.function_map:
            raise KeyError(f"Unknown characterizer '{name}'. Known: {sorted(self.function_map.keys())}")

        function_id = self.function_map[name]
        task = self.gc.submit(function_id=function_id, payload=payload)
        return {
            "task_id": task.task_id,
            "characterizer": name,
            "endpoint_id": task.endpoint_id,
            "function_id": task.function_id,
        }

    @action
    async def get(self, req: dict) -> dict:
        task_id = req["task_id"]
        return {"task_id": task_id, **self.gc.try_result(task_id)}
