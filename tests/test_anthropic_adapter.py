"""Tests for Anthropic adapter.

All tests use mocked Anthropic responses — no real API calls.
"""

import pytest
from unittest.mock import MagicMock, patch

from pulse.message import PulseMessage
from pulse.adapter import AdapterError, AdapterConnectionError

from pulse_anthropic import AnthropicAdapter


# --- Mock Helpers ---


def make_mock_response(content="Hello!", model="claude-haiku-4-5-20251001",
                       input_tokens=10, output_tokens=5, stop_reason="end_turn"):
    """Create a mock Anthropic Messages response."""
    mock = MagicMock()

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = content
    mock.content = [text_block]

    mock.model = model
    mock.usage.input_tokens = input_tokens
    mock.usage.output_tokens = output_tokens
    mock.stop_reason = stop_reason
    return mock


# --- Fixtures ---


@pytest.fixture
def adapter():
    """Create an AnthropicAdapter with mocked client."""
    a = AnthropicAdapter(api_key="sk-ant-test-key")
    a._client = MagicMock()
    a.connected = True
    return a


@pytest.fixture
def query_message():
    return PulseMessage(
        action="ACT.QUERY.DATA",
        parameters={"query": "What is PULSE Protocol?"},
        sender="test-agent",
    )


@pytest.fixture
def sentiment_message():
    return PulseMessage(
        action="ACT.ANALYZE.SENTIMENT",
        parameters={"text": "I absolutely love this product!"},
        sender="test-agent",
    )


@pytest.fixture
def create_message():
    return PulseMessage(
        action="ACT.CREATE.TEXT",
        parameters={"instructions": "Write a haiku about AI"},
        sender="test-agent",
    )


@pytest.fixture
def translate_message():
    return PulseMessage(
        action="ACT.TRANSFORM.TRANSLATE",
        parameters={"text": "Hello, world!", "target_language": "French"},
        sender="test-agent",
    )


# --- Test Initialization ---


class TestAnthropicAdapterInit:
    """Test adapter initialization."""

    def test_basic_init(self):
        adapter = AnthropicAdapter(api_key="sk-ant-test")
        assert adapter.name == "anthropic"
        assert adapter.default_model == "claude-haiku-4-5-20251001"
        assert adapter.connected is False

    def test_custom_model(self):
        adapter = AnthropicAdapter(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        assert adapter.default_model == "claude-sonnet-4-5-20250929"

    def test_custom_base_url(self):
        adapter = AnthropicAdapter(api_key="sk-ant-test", base_url="https://my-proxy.com")
        assert adapter.base_url == "https://my-proxy.com"

    def test_repr(self):
        adapter = AnthropicAdapter(api_key="sk-ant-test")
        r = repr(adapter)
        assert "claude-haiku" in r
        assert "False" in r


# --- Test Connection ---


class TestAnthropicConnection:

    def test_connect_without_key_raises(self):
        adapter = AnthropicAdapter()
        with pytest.raises(AdapterConnectionError, match="API key required"):
            adapter.connect()

    @patch("pulse_anthropic.adapter.Anthropic")
    def test_connect_creates_client(self, mock_cls):
        adapter = AnthropicAdapter(api_key="sk-ant-test")
        adapter.connect()
        assert adapter.connected is True
        assert adapter._client is not None

    def test_disconnect(self, adapter):
        adapter.disconnect()
        assert adapter.connected is False
        assert adapter._client is None


# --- Test to_native ---


class TestToNative:

    def test_query_message(self, adapter, query_message):
        native = adapter.to_native(query_message)
        assert native["model"] == "claude-haiku-4-5-20251001"
        assert native["messages"][0]["role"] == "user"
        assert "What is PULSE Protocol?" in native["messages"][0]["content"]
        assert native["max_tokens"] == 1024  # Default

    def test_sentiment_has_system_prompt(self, adapter, sentiment_message):
        native = adapter.to_native(sentiment_message)
        assert "system" in native
        assert "sentiment" in native["system"].lower()

    def test_create_uses_sonnet(self, adapter, create_message):
        native = adapter.to_native(create_message)
        assert "sonnet" in native["model"]

    def test_translate_includes_language(self, adapter, translate_message):
        native = adapter.to_native(translate_message)
        assert "French" in native["messages"][0]["content"]

    def test_custom_model_override(self, adapter):
        msg = PulseMessage(
            action="ACT.QUERY.DATA",
            parameters={"query": "test", "model": "claude-opus-4-6"},
        )
        native = adapter.to_native(msg)
        assert native["model"] == "claude-opus-4-6"

    def test_custom_max_tokens(self, adapter):
        msg = PulseMessage(
            action="ACT.QUERY.DATA",
            parameters={"query": "test", "max_tokens": 2048},
        )
        native = adapter.to_native(msg)
        assert native["max_tokens"] == 2048

    def test_temperature_passed(self, adapter):
        msg = PulseMessage(
            action="ACT.QUERY.DATA",
            parameters={"query": "test", "temperature": 0.7},
        )
        native = adapter.to_native(msg)
        assert native["temperature"] == 0.7

    def test_custom_system_prompt(self, adapter):
        msg = PulseMessage(
            action="ACT.ANALYZE.SENTIMENT",
            parameters={"text": "hello", "system_prompt": "You are a pirate."},
        )
        native = adapter.to_native(msg)
        assert native["system"] == "You are a pirate."

    def test_missing_content_raises(self, adapter):
        msg = PulseMessage(
            action="ACT.QUERY.DATA",
            parameters={"temperature": 0.5},
        )
        with pytest.raises(AdapterError, match="Missing content parameter"):
            adapter.to_native(msg)

    def test_query_no_system_prompt(self, adapter, query_message):
        native = adapter.to_native(query_message)
        assert "system" not in native


# --- Test call_api ---


class TestCallAPI:

    def test_successful_call(self, adapter):
        adapter._client.messages.create.return_value = make_mock_response(
            content="PULSE is a semantic protocol.",
            input_tokens=15,
            output_tokens=8,
        )

        result = adapter.call_api({
            "model": "claude-haiku-4-5-20251001",
            "messages": [{"role": "user", "content": "What is PULSE?"}],
            "max_tokens": 1024,
        })

        assert result["content"] == "PULSE is a semantic protocol."
        assert result["usage"]["input_tokens"] == 15
        assert result["usage"]["output_tokens"] == 8
        assert result["usage"]["total_tokens"] == 23
        assert result["stop_reason"] == "end_turn"

    def test_auto_connect(self):
        adapter = AnthropicAdapter(api_key="sk-ant-test")
        with patch("pulse_anthropic.adapter.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = make_mock_response()

            result = adapter.call_api({
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1024,
            })
            assert adapter.connected is True
            assert result["content"] == "Hello!"

    def test_auth_error(self, adapter):
        from anthropic import AuthenticationError
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        adapter._client.messages.create.side_effect = AuthenticationError(
            message="Invalid API key",
            response=mock_resp,
            body=None,
        )
        with pytest.raises(AdapterError, match="authentication failed"):
            adapter.call_api({"model": "x", "messages": [], "max_tokens": 1})

    def test_rate_limit_error(self, adapter):
        from anthropic import RateLimitError
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        adapter._client.messages.create.side_effect = RateLimitError(
            message="Rate limit",
            response=mock_resp,
            body=None,
        )
        with pytest.raises(AdapterError, match="rate limit"):
            adapter.call_api({"model": "x", "messages": [], "max_tokens": 1})

    def test_connection_error(self, adapter):
        from anthropic import APIConnectionError
        adapter._client.messages.create.side_effect = APIConnectionError(
            request=MagicMock(),
        )
        with pytest.raises(AdapterConnectionError, match="Cannot reach"):
            adapter.call_api({"model": "x", "messages": [], "max_tokens": 1})


# --- Test from_native ---


class TestFromNative:

    def test_response_conversion(self, adapter):
        native = {
            "content": "The answer is 42.",
            "model": "claude-haiku-4-5-20251001",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "stop_reason": "end_turn",
        }
        response = adapter.from_native(native)
        assert isinstance(response, PulseMessage)
        assert response.content["parameters"]["result"] == "The answer is 42."
        assert response.content["parameters"]["model"] == "claude-haiku-4-5-20251001"
        assert response.content["parameters"]["stop_reason"] == "end_turn"


# --- Test Full Pipeline ---


class TestFullPipeline:

    def test_query_pipeline(self, adapter, query_message):
        adapter._client.messages.create.return_value = make_mock_response(
            content="PULSE is a semantic protocol for AI communication.",
        )
        response = adapter.send(query_message)

        assert isinstance(response, PulseMessage)
        assert response.type == "RESPONSE"
        assert response.envelope["receiver"] == "test-agent"
        assert response.envelope["sender"] == "adapter:anthropic"
        assert "PULSE is a semantic" in response.content["parameters"]["result"]

    def test_sentiment_pipeline(self, adapter, sentiment_message):
        adapter._client.messages.create.return_value = make_mock_response(
            content='{"sentiment": "positive", "confidence": 0.95}',
        )
        response = adapter.send(sentiment_message)
        assert response.type == "RESPONSE"
        assert "positive" in response.content["parameters"]["result"]

    def test_pipeline_tracks_requests(self, adapter, query_message):
        adapter._client.messages.create.return_value = make_mock_response()
        adapter.send(query_message)
        adapter.send(query_message)
        assert adapter._request_count == 2
        assert adapter._error_count == 0

    def test_pipeline_error_tracking(self, adapter, query_message):
        from anthropic import AuthenticationError
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        adapter._client.messages.create.side_effect = AuthenticationError(
            message="bad key", response=mock_resp, body=None,
        )
        with pytest.raises(AdapterError):
            adapter.send(query_message)
        assert adapter._error_count == 1


# --- Test Supported Actions ---


class TestSupportedActions:

    def test_supported_actions_list(self, adapter):
        actions = adapter.supported_actions
        assert len(actions) == 6
        assert "ACT.QUERY.DATA" in actions
        assert "ACT.ANALYZE.SENTIMENT" in actions

    def test_supports_check(self, adapter):
        assert adapter.supports("ACT.QUERY.DATA") is True
        assert adapter.supports("ACT.TRANSACT.REQUEST") is False


# --- Test Provider Switching ---


class TestProviderSwitching:
    """Prove that switching providers is trivial."""

    def test_same_interface(self):
        """Both adapters have the same supported actions."""
        from pulse_openai import OpenAIAdapter

        openai = OpenAIAdapter(api_key="sk-test")
        anthropic = AnthropicAdapter(api_key="sk-ant-test")

        assert set(openai.supported_actions) == set(anthropic.supported_actions)

    def test_same_message_works(self, adapter, query_message):
        """Same PULSE message works with both adapters."""
        adapter._client.messages.create.return_value = make_mock_response(content="test")

        response = adapter.send(query_message)
        assert response.type == "RESPONSE"
        assert response.content["parameters"]["result"] == "test"
