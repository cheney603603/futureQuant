"""
Tests for LLMClient
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from futureQuant.core.llm_client import LLMClient, LLMProvider, LLMResponse


class TestLLMClient:
    def test_provider_parsing(self):
        client = LLMClient(provider="openai", model="gpt-4o")
        assert client.provider == LLMProvider.OPENAI
        assert client.model == "gpt-4o"

        client2 = LLMClient(provider="ollama", model="qwen2.5")
        assert client2.provider == LLMProvider.OLLAMA
        assert client2.model == "qwen2.5"

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError):
            LLMClient(provider="unknown")

    @patch("futureQuant.core.llm_client.httpx.Client")
    def test_chat_openai_success(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = LLMClient(provider="openai", api_key="test-key")
        resp = client.chat([{"role": "user", "content": "Hi"}])

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hello!"
        assert resp.model == "gpt-4o-mini"
        assert resp.usage.total_tokens == 12
        assert resp.finish_reason == "stop"

    @patch("futureQuant.core.llm_client.httpx.Client")
    def test_chat_with_tools(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "web_search", "arguments": json.dumps({"query": "test"})},
        }
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            "model": "gpt-4o",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = LLMClient(provider="openai", api_key="test-key")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search web",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]
        resp = client.chat([{"role": "user", "content": "Search something"}], tools=tools)

        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["function"]["name"] == "web_search"

    @patch("futureQuant.core.llm_client.httpx.Client")
    def test_retry_on_429(self, mock_client_cls):
        fail_response = MagicMock()
        fail_response.status_code = 429
        fail_response.text = "Rate limited"
        fail_response.raise_for_status.side_effect = Exception("429")

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        success_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.side_effect = [fail_response, success_response]
        mock_client_cls.return_value = mock_client

        client = LLMClient(provider="openai", api_key="test-key", max_retries=2)
        resp = client.chat([{"role": "user", "content": "Hi"}])
        assert resp.content == "OK"
