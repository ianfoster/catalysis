"""Tests for LLM client abstraction.

Run with: pytest tests/test_llm_client.py -v
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestration.llm_client import (
    LLMClient,
    _extract_json,
    create_llm_client_from_config,
)


# --- JSON Extraction Tests ---


class TestExtractJson:
    """Tests for _extract_json function."""

    def test_pure_json_object(self):
        """Extract from pure JSON string."""
        text = '{"key": "value", "number": 42}'
        result = _extract_json(text)
        assert result == {"key": "value", "number": 42}

    def test_json_with_whitespace(self):
        """Extract JSON with leading/trailing whitespace."""
        text = '  \n  {"key": "value"}  \n  '
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_json_in_markdown_code_block(self):
        """Extract JSON from markdown code block."""
        text = '''Here's the response:
```json
{"action": "test", "value": 123}
```
That's all.'''
        result = _extract_json(text)
        assert result == {"action": "test", "value": 123}

    def test_json_in_plain_code_block(self):
        """Extract JSON from code block without language tag."""
        text = '''Response:
```
{"data": [1, 2, 3]}
```'''
        result = _extract_json(text)
        assert result == {"data": [1, 2, 3]}

    def test_json_with_surrounding_text(self):
        """Extract JSON embedded in prose."""
        text = '''Based on my analysis, I recommend:
{"action": "stop", "confidence": 0.95, "reasoning": "converged"}
This is my final recommendation.'''
        result = _extract_json(text)
        assert result["action"] == "stop"
        assert result["confidence"] == 0.95

    def test_nested_json(self):
        """Extract nested JSON structure."""
        text = '{"outer": {"inner": {"deep": "value"}}, "list": [{"a": 1}]}'
        result = _extract_json(text)
        assert result["outer"]["inner"]["deep"] == "value"
        assert result["list"][0]["a"] == 1

    def test_json_with_escaped_quotes(self):
        """Extract JSON containing escaped quotes."""
        text = '{"message": "He said \\"hello\\""}'
        result = _extract_json(text)
        assert result["message"] == 'He said "hello"'

    def test_json_with_newlines_in_strings(self):
        """Extract JSON with newlines in string values."""
        text = '{"text": "line1\\nline2"}'
        result = _extract_json(text)
        assert result["text"] == "line1\nline2"

    def test_multiple_json_objects_returns_first(self):
        """When multiple JSON objects present, return the first valid one."""
        text = '''First: {"a": 1}
Second: {"b": 2}'''
        result = _extract_json(text)
        assert result == {"a": 1}

    def test_invalid_json_raises(self):
        """Raise error for completely invalid input."""
        text = "This contains no JSON at all"
        with pytest.raises(json.JSONDecodeError):
            _extract_json(text)

    def test_malformed_json_raises(self):
        """Raise error for malformed JSON."""
        text = '{"key": value_without_quotes}'
        with pytest.raises(json.JSONDecodeError):
            _extract_json(text)

    def test_empty_string_raises(self):
        """Raise error for empty string."""
        with pytest.raises(json.JSONDecodeError):
            _extract_json("")

    def test_json_array_not_supported(self):
        """Arrays at top level are not extracted (we expect objects)."""
        text = '[1, 2, 3]'
        # Current implementation looks for {} not []
        with pytest.raises(json.JSONDecodeError):
            _extract_json(text)

    def test_unicode_in_json(self):
        """Handle unicode characters in JSON."""
        text = '{"name": "Café", "symbol": "→"}'
        result = _extract_json(text)
        assert result["name"] == "Café"
        assert result["symbol"] == "→"

    def test_large_numbers(self):
        """Handle large numbers in JSON."""
        text = '{"big": 12345678901234567890, "float": 1.23e45}'
        result = _extract_json(text)
        assert result["big"] == 12345678901234567890
        assert result["float"] == 1.23e45

    def test_boolean_and_null(self):
        """Handle boolean and null values."""
        text = '{"yes": true, "no": false, "nothing": null}'
        result = _extract_json(text)
        assert result["yes"] is True
        assert result["no"] is False
        assert result["nothing"] is None


# --- LLMClient Tests ---


class TestLLMClient:
    """Tests for LLMClient class."""

    @pytest.fixture
    def mock_chat_openai(self):
        """Mock ChatOpenAI for testing."""
        with patch("orchestration.llm_client.ChatOpenAI") as mock:
            instance = AsyncMock()
            instance.ainvoke = AsyncMock()
            mock.return_value = instance
            yield mock, instance

    def test_init_with_api_key(self, mock_chat_openai):
        """Initialize with explicit API key."""
        mock_class, _ = mock_chat_openai
        client = LLMClient(
            base_url="http://localhost:8000/v1",
            model="test-model",
            api_key="test-key",
        )
        mock_class.assert_called_once()
        call_kwargs = mock_class.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["base_url"] == "http://localhost:8000/v1"
        assert call_kwargs["model"] == "test-model"

    def test_init_without_api_key(self, mock_chat_openai):
        """Initialize without API key (uses placeholder)."""
        mock_class, _ = mock_chat_openai
        client = LLMClient(
            base_url="http://localhost:8000/v1",
            model="test-model",
        )
        call_kwargs = mock_class.call_args[1]
        assert call_kwargs["api_key"] == "not-required"

    @pytest.mark.asyncio
    async def test_reason_basic(self, mock_chat_openai):
        """Test basic reasoning call."""
        _, mock_instance = mock_chat_openai
        mock_response = MagicMock()
        mock_response.content = "This is the response"
        mock_instance.ainvoke = AsyncMock(return_value=mock_response)

        client = LLMClient("http://localhost:8000/v1", "test-model")
        result = await client.reason("What is 2+2?")

        assert result == "This is the response"
        mock_instance.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_reason_with_system_prompt(self, mock_chat_openai):
        """Test reasoning with system prompt."""
        _, mock_instance = mock_chat_openai
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_instance.ainvoke = AsyncMock(return_value=mock_response)

        client = LLMClient("http://localhost:8000/v1", "test-model")
        await client.reason("User prompt", system_prompt="You are a helpful assistant")

        # Check that two messages were sent
        call_args = mock_instance.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # System + User

    @pytest.mark.asyncio
    async def test_reason_json_success(self, mock_chat_openai):
        """Test JSON reasoning with valid response."""
        _, mock_instance = mock_chat_openai
        mock_response = MagicMock()
        mock_response.content = '{"result": "success", "value": 42}'
        mock_instance.ainvoke = AsyncMock(return_value=mock_response)

        client = LLMClient("http://localhost:8000/v1", "test-model")
        result = await client.reason_json("Return JSON")

        assert result == {"result": "success", "value": 42}

    @pytest.mark.asyncio
    async def test_reason_json_retries_on_invalid(self, mock_chat_openai):
        """Test JSON reasoning retries on invalid JSON."""
        _, mock_instance = mock_chat_openai

        # First response is invalid, second is valid
        invalid_response = MagicMock()
        invalid_response.content = "Not valid JSON"
        valid_response = MagicMock()
        valid_response.content = '{"valid": true}'

        mock_instance.ainvoke = AsyncMock(side_effect=[invalid_response, valid_response])

        client = LLMClient("http://localhost:8000/v1", "test-model")
        result = await client.reason_json("Return JSON", retries=2)

        assert result == {"valid": True}
        assert mock_instance.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_reason_json_raises_after_retries(self, mock_chat_openai):
        """Test JSON reasoning raises after all retries fail."""
        _, mock_instance = mock_chat_openai
        mock_response = MagicMock()
        mock_response.content = "Never valid JSON"
        mock_instance.ainvoke = AsyncMock(return_value=mock_response)

        client = LLMClient("http://localhost:8000/v1", "test-model")

        with pytest.raises(ValueError) as exc_info:
            await client.reason_json("Return JSON", retries=2)

        assert "Failed to parse JSON" in str(exc_info.value)
        assert mock_instance.ainvoke.call_count == 3  # Initial + 2 retries


# --- Config Factory Tests ---


class TestCreateLLMClientFromConfig:
    """Tests for create_llm_client_from_config."""

    def test_shared_mode(self):
        """Create client in shared mode."""
        with patch("orchestration.llm_client.ChatOpenAI"):
            config = {
                "llm": {
                    "mode": "shared",
                    "model": "llama3",
                    "shared_url": "http://shared:8000/v1",
                    "local_url": "http://local:8000/v1",
                },
            }
            client = create_llm_client_from_config(config)
            assert client.base_url == "http://shared:8000/v1"
            assert client.model == "llama3"

    def test_local_mode(self):
        """Create client in local mode."""
        with patch("orchestration.llm_client.ChatOpenAI"):
            config = {
                "llm": {
                    "mode": "local",
                    "model": "llama3",
                    "shared_url": "http://shared:8000/v1",
                    "local_url": "http://local:8000/v1",
                },
            }
            client = create_llm_client_from_config(config)
            assert client.base_url == "http://local:8000/v1"

    def test_api_key_from_env(self):
        """Load API key from environment variable."""
        with patch("orchestration.llm_client.ChatOpenAI"):
            with patch.dict("os.environ", {"TEST_API_KEY": "secret-key"}):
                config = {
                    "llm": {
                        "mode": "shared",
                        "model": "llama3",
                        "shared_url": "http://localhost:8000/v1",
                        "api_key_env": "TEST_API_KEY",
                    },
                }
                # Note: The actual API key loading happens internally
                client = create_llm_client_from_config(config)
                assert client is not None

    def test_default_values(self):
        """Use defaults when config is minimal."""
        with patch("orchestration.llm_client.ChatOpenAI"):
            config = {}  # Empty config
            client = create_llm_client_from_config(config)
            assert client.model == "meta-llama/Llama-3-8B-Instruct"
            assert client.base_url == "http://localhost:8000/v1"

    def test_timeout_from_config(self):
        """Load timeout from config."""
        with patch("orchestration.llm_client.ChatOpenAI") as mock:
            config = {
                "llm": {"model": "test"},
                "timeouts": {"llm_call": 60},
            }
            client = create_llm_client_from_config(config)
            assert client.timeout == 60


# --- Edge Cases and Error Handling ---


class TestLLMClientEdgeCases:
    """Edge cases and error handling tests."""

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        """Handle empty response from LLM."""
        with patch("orchestration.llm_client.ChatOpenAI") as mock:
            instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = ""
            instance.ainvoke = AsyncMock(return_value=mock_response)
            mock.return_value = instance

            client = LLMClient("http://localhost:8000/v1", "test-model")
            result = await client.reason("Test")
            assert result == ""

    @pytest.mark.asyncio
    async def test_reason_json_with_markdown_response(self):
        """Handle JSON in markdown code blocks."""
        with patch("orchestration.llm_client.ChatOpenAI") as mock:
            instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = '''Here's my analysis:
```json
{
  "action": "test",
  "test": "fast_surrogate",
  "reasoning": "Need initial data"
}
```
Let me know if you need more details.'''
            instance.ainvoke = AsyncMock(return_value=mock_response)
            mock.return_value = instance

            client = LLMClient("http://localhost:8000/v1", "test-model")
            result = await client.reason_json("Test")

            assert result["action"] == "test"
            assert result["test"] == "fast_surrogate"
