from __future__ import annotations

from typing import Any, Dict

from academy.agent import Agent, action

# This skill is a thin wrapper; it does not run OpenMM locally.
# It returns whatever payload you pass through (or you can remove it entirely).
class OpenMMSkill(Agent):
    @action
    async def minimize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # This is just a placeholder if you later want to run OpenMM locally under Academy.
        # For Spark execution, you will call Globus Compute via the escalator driver.
        return {"ok": False, "error": "OpenMMSkill not used; use openmm_escalate via Globus Compute", "payload": payload}
