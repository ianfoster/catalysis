"""LLM client abstraction for vLLM/TGI endpoints."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible client for vLLM/TGI endpoints.

    Provides simple async interface for reasoning tasks with JSON parsing.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize LLM client.

        Args:
            base_url: OpenAI-compatible API endpoint (e.g., http://localhost:8000/v1)
            model: Model name/identifier
            api_key: Optional API key (some local deployments don't require one)
            timeout: Request timeout in seconds
            max_retries: Number of retries for transient failures
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # LangChain requires a non-empty API key even if the server doesn't check it
        effective_api_key = api_key or "not-required"

        self._client = ChatOpenAI(
            model=model,
            api_key=effective_api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def reason(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send prompt to LLM and return response text.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context

        Returns:
            Response text from LLM
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = await self._client.ainvoke(messages)
        return response.content

    async def reason_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        retries: int = 2,
    ) -> dict[str, Any]:
        """Send prompt and parse JSON response.

        Attempts to extract JSON from the response, with retries on parse failure.

        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: Optional system prompt
            retries: Number of retries on JSON parse failure

        Returns:
            Parsed JSON as dict

        Raises:
            ValueError: If JSON cannot be extracted after retries
        """
        last_error: Exception | None = None
        last_response: str = ""

        for attempt in range(retries + 1):
            try:
                response = await self.reason(prompt, system_prompt)
                last_response = response
                return _extract_json(response)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s",
                    attempt + 1,
                    retries + 1,
                    e,
                )
                if attempt < retries:
                    # Retry with a more explicit prompt
                    prompt = (
                        f"{prompt}\n\n"
                        "IMPORTANT: Your response must be valid JSON only. "
                        "Do not include any text before or after the JSON object."
                    )

        raise ValueError(
            f"Failed to parse JSON after {retries + 1} attempts. "
            f"Last response: {last_response[:500]}... "
            f"Last error: {last_error}"
        )


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response text.

    Handles common cases:
    - Pure JSON response
    - JSON in markdown code blocks
    - JSON with surrounding text

    Args:
        text: Raw LLM response

    Returns:
        Parsed JSON dict

    Raises:
        json.JSONDecodeError: If no valid JSON found
    """
    text = text.strip()

    # Try direct parse first
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code block
    code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try finding JSON object in text
    # Look for outermost { ... } pair
    brace_start = text.find("{")
    if brace_start != -1:
        depth = 0
        in_string = False
        escape_next = False
        for i, char in enumerate(text[brace_start:], start=brace_start):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    raise json.JSONDecodeError("No valid JSON object found", text, 0)


def create_llm_client_from_config(config: dict[str, Any]) -> LLMClient:
    """Create LLMClient from shepherd config section.

    Args:
        config: Config dict with 'llm' section containing:
            - mode: "shared" or "local"
            - model: model name
            - shared_url: URL for shared mode
            - local_url: URL for local mode
            - api_key_env: Optional env var name for API key

    Returns:
        Configured LLMClient instance
    """
    import os

    llm_config = config.get("llm", {})
    mode = llm_config.get("mode", "shared")
    model = llm_config.get("model", "meta-llama/Llama-3-8B-Instruct")

    if mode == "shared":
        base_url = llm_config.get("shared_url", "http://localhost:8000/v1")
    else:
        base_url = llm_config.get("local_url", "http://localhost:8000/v1")

    api_key = None
    api_key_env = llm_config.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)

    # Auto-detect common API providers
    if not api_key:
        if "openai.com" in base_url:
            api_key = os.environ.get("OPENAI_API_KEY")
        elif "anthropic.com" in base_url:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif "together" in base_url:
            api_key = os.environ.get("TOGETHER_API_KEY")

    timeout = config.get("timeouts", {}).get("llm_call", 30.0)

    return LLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
