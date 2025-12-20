from __future__ import annotations
import logging
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

logger = logging.getLogger(__name__)

async def build_react_agent(model: str, api_key: str, base_url: str | None, tools: List[Any]):
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)

    # Debug: catch the exact tool-shape errors early
    for t in tools:
        logger.info("Registered tool: name=%s type=%s", getattr(t, "name", getattr(t, "__name__", None)), type(t))
        if isinstance(t, (tuple, list)):
            raise TypeError(f"Tool list must be flat. Found nested element type={type(t)} value={t}")

    return create_agent(llm, tools=tools)

def extract_final_answer(result: Dict[str, Any]) -> str:
    # Works for LangGraph/LangChain message dict output
    msgs = result.get("messages", [])
    for m in reversed(msgs):
        # AIMessage has .content; dict messages have 'content'
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else None)
        if content and (role in (None, "ai") or m.__class__.__name__.endswith("AIMessage")):
            return content
    raise ValueError("No final AI answer content found in messages.")
