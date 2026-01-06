"""LLMProxyAgent - Proxy for LLM requests with tracking.

Sits in front of vLLM to provide visibility into LLM usage:
- Request/response logging
- Token counting
- Latency tracking
- Error rate monitoring
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from academy.agent import action
from skills.base_agent import TrackedAgent

logger = logging.getLogger(__name__)


class LLMProxyAgent(TrackedAgent):
    """Proxy agent for LLM requests with usage tracking.

    Provides a single point of access for all LLM calls, enabling:
    - Centralized logging and monitoring
    - Token usage tracking
    - Latency metrics
    - Request/response history

    Inherits from TrackedAgent for automatic history tracking.
    """

    def __init__(
        self,
        llm_url: str = "http://localhost:8000/v1",
        model: str = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
    ):
        """Initialize LLMProxyAgent.

        Args:
            llm_url: URL to the vLLM server
            model: Model name to use
        """
        super().__init__(max_history=200)
        self._llm_url = llm_url
        self._model = model
        self._client: Any = None

        # Additional metrics beyond TrackedAgent
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_requests = 0
        self._failed_requests = 0

    async def agent_on_startup(self) -> None:
        """Initialize the OpenAI client for vLLM."""
        logger.info("LLMProxyAgent starting: url=%s model=%s", self._llm_url, self._model)

        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            base_url=self._llm_url,
            api_key="not-needed",
        )

        # Test connection
        try:
            models = await self._client.models.list()
            available = [m.id for m in models.data]
            logger.info("Connected to vLLM. Available models: %s", available)
            if self._model not in available:
                logger.warning("Model %s not in available models", self._model)
        except Exception as e:
            logger.error("Failed to connect to vLLM: %s", e)
            raise

    @action
    async def chat_completion(self, request: dict[str, Any]) -> dict[str, Any]:
        """Proxy a chat completion request to vLLM.

        Args:
            request: Dict with:
                - messages: List of message dicts
                - temperature: Optional temperature (default 0.7)
                - max_tokens: Optional max tokens (default 1024)
                - model: Optional model override

        Returns:
            Dict with:
                - ok: Success flag
                - content: Response content
                - usage: Token usage dict
                - latency_ms: Request latency
        """
        with self.track_action("chat_completion", request) as tracker:
            messages = request.get("messages", [])
            temperature = request.get("temperature", 0.7)
            max_tokens = request.get("max_tokens", 1024)
            model = request.get("model", self._model)

            self._total_requests += 1
            t0 = time.time()

            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                latency_ms = (time.time() - t0) * 1000
                content = response.choices[0].message.content

                # Track token usage
                usage = response.usage
                if usage:
                    self._total_prompt_tokens += usage.prompt_tokens
                    self._total_completion_tokens += usage.completion_tokens

                result = {
                    "ok": True,
                    "content": content,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                    },
                    "latency_ms": round(latency_ms, 1),
                    "model": model,
                }
                tracker.set_result(result)
                return result

            except Exception as e:
                self._failed_requests += 1
                latency_ms = (time.time() - t0) * 1000
                logger.error("LLM request failed: %s", e)
                result = {
                    "ok": False,
                    "error": str(e),
                    "latency_ms": round(latency_ms, 1),
                }
                tracker.set_result(result)
                return result

    @action
    async def chat_completion_json(self, request: dict[str, Any]) -> dict[str, Any]:
        """Proxy a chat completion and parse JSON response.

        Same as chat_completion but attempts to parse the response as JSON.

        Args:
            request: Same as chat_completion

        Returns:
            Dict with:
                - ok: Success flag
                - content: Raw response content
                - parsed: Parsed JSON dict (if successful)
                - usage: Token usage dict
                - latency_ms: Request latency
        """
        with self.track_action("chat_completion_json", request) as tracker:
            # Get the raw completion
            result = await self.chat_completion(request)

            if not result.get("ok"):
                tracker.set_result(result)
                return result

            content = result.get("content", "")

            # Try to parse JSON
            try:
                parsed = self._parse_json_response(content)
                result["parsed"] = parsed
            except ValueError as e:
                result["parse_error"] = str(e)
                logger.warning("Failed to parse JSON from LLM response: %s", e)

            tracker.set_result(result)
            return result

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {content[:200]}...")

    @action
    async def get_status(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get agent status including LLM usage statistics."""
        stats = self._get_statistics()

        return {
            "ok": True,
            "ready": self._client is not None,
            "llm_url": self._llm_url,
            "model": self._model,
            # Request stats
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": round(
                (self._total_requests - self._failed_requests) / max(1, self._total_requests),
                3
            ),
            # Token stats
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            # TrackedAgent stats
            "total_actions": stats["total_actions"],
            "total_time_s": stats["total_time_s"],
            "action_counts": stats["action_counts"],
            "avg_latency_ms": round(
                stats["total_time_s"] * 1000 / max(1, stats["total_actions"]),
                1
            ),
        }

    @action
    async def get_metrics(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get detailed metrics for monitoring dashboards."""
        stats = self._get_statistics()
        action_times = stats.get("action_times", {})

        return {
            "ok": True,
            # Summary
            "total_requests": self._total_requests,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "avg_latency_ms": round(
                stats["total_time_s"] * 1000 / max(1, stats["total_actions"]),
                1
            ),
            # Detailed breakdown
            "requests_by_type": stats["action_counts"],
            "time_by_type_s": action_times,
            "tokens": {
                "prompt": self._total_prompt_tokens,
                "completion": self._total_completion_tokens,
            },
            "errors": {
                "total": self._failed_requests,
                "rate": round(self._failed_requests / max(1, self._total_requests), 3),
            },
        }
