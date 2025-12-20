from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

logger = logging.getLogger(__name__)

async def build_react_agent(model: str, api_key: str, base_url: str | None, tools: List[Any]):
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)

    # Debug: catch tool-shape errors early
    for t in tools:
        name = getattr(t, "name", getattr(t, "__name__", None))
        logger.info("Registered tool: name=%s type=%s", name, type(t))
        if isinstance(t, (tuple, list)):
            raise TypeError(f"Tool list must be flat. Found nested element type={type(t)} value={t}")

    return create_agent(llm, tools=tools)

def _maybe_json(x: Any) -> Any:
    """Try to parse JSON strings; otherwise return as-is."""
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x

def extract_tool_results(result: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Collect ToolMessage outputs keyed by tool name."""
    tool_results: Dict[str, List[Any]] = {}
    for m in result.get("messages", []):
        cls = m.__class__.__name__
        if cls.endswith("ToolMessage"):
            name = getattr(m, "name", None)
            content = getattr(m, "content", None)
            if name is None:
                continue
            tool_results.setdefault(name, []).append(_maybe_json(content))
    return tool_results

def extract_tool_calls(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect tool call intents from AI messages (useful for debugging)."""
    calls: List[Dict[str, Any]] = []
    for m in result.get("messages", []):
        cls = m.__class__.__name__
        if cls.endswith("AIMessage"):
            for tc in getattr(m, "tool_calls", []) or []:
                calls.append(tc)
    return calls

def extract_final_answer(result: Dict[str, Any]) -> str:
    msgs = result.get("messages", [])
    for m in reversed(msgs):
        if m.__class__.__name__.endswith("AIMessage"):
            content = getattr(m, "content", "") or ""
            if content.strip():
                return content
    raise ValueError("No final AIMessage with non-empty content found.")
