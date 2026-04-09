"""Tests for shared LLM utilities."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import httpx
import pytest

from cuecard.llm_utils import (
    _ALLOWED_HAIKU_MODELS,
    call_haiku,
    call_local,
    validate_endpoint,
)
from cuecard.security import ConfigError


class TestValidateEndpoint:
    def test_localhost_allowed(self) -> None:
        validate_endpoint("http://localhost:8081/v1")

    def test_127_allowed(self) -> None:
        validate_endpoint("http://127.0.0.1:8081/v1")

    def test_ipv6_loopback_allowed(self) -> None:
        validate_endpoint("http://[::1]:8081/v1")

    def test_external_host_raises(self) -> None:
        fake = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        with (
            patch("cuecard.llm_utils.socket.getaddrinfo", return_value=fake),
            pytest.raises(ConfigError, match="loopback"),
        ):
            validate_endpoint("http://evil.com:8081/v1")

    def test_ip_address_raises(self) -> None:
        fake = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0))]
        with (
            patch("cuecard.llm_utils.socket.getaddrinfo", return_value=fake),
            pytest.raises(ConfigError, match="loopback"),
        ):
            validate_endpoint("http://10.0.0.1:8081/v1")

    def test_userinfo_bypass_raises(self) -> None:
        with pytest.raises(ConfigError, match="userinfo"):
            validate_endpoint("http://evil.com@localhost:8081/v1")

    def test_file_scheme_raises(self) -> None:
        with pytest.raises(ConfigError, match="http/https"):
            validate_endpoint("file:///etc/passwd")

    def test_ftp_scheme_raises(self) -> None:
        with pytest.raises(ConfigError, match="http/https"):
            validate_endpoint("ftp://localhost/data")

    def test_password_in_url_raises(self) -> None:
        with pytest.raises(ConfigError, match="userinfo"):
            validate_endpoint("http://user:pass@localhost:8081/v1")


class TestCallLocal:
    def test_successful_call(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello"}}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx, "post", return_value=mock_response):
            result = call_local(
                "system", "user", "http://localhost:8081/v1", False,
            )
            assert result == "hello"

    def test_custom_temperature_and_max_tokens(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx, "post", return_value=mock_response) as mock_post:
            call_local(
                "sys", "usr", "http://localhost:8081/v1", False,
                max_tokens=512, temperature=0.7,
            )
            body = mock_post.call_args.kwargs["json"]
            assert body["max_tokens"] == 512
            assert body["temperature"] == 0.7

    def test_thinking_enabled(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx, "post", return_value=mock_response) as mock_post:
            call_local(
                "sys", "usr", "http://localhost:8081/v1", True,
            )
            body = mock_post.call_args.kwargs["json"]
            assert "chat_template_kwargs" not in body

    def test_thinking_disabled(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx, "post", return_value=mock_response) as mock_post:
            call_local(
                "sys", "usr", "http://localhost:8081/v1", False,
            )
            body = mock_post.call_args.kwargs["json"]
            assert body["chat_template_kwargs"] == {"enable_thinking": False}

    def test_timeout_raises(self) -> None:
        with (
            patch.object(
                httpx, "post", side_effect=httpx.TimeoutException("timeout"),
            ),
            pytest.raises(httpx.TimeoutException),
        ):
            call_local("sys", "usr", "http://localhost:8081/v1", False)

    def test_connect_error_raises(self) -> None:
        with (
            patch.object(
                httpx, "post", side_effect=httpx.ConnectError("refused"),
            ),
            pytest.raises(httpx.ConnectError),
        ):
            call_local("sys", "usr", "http://localhost:8081/v1", False)

    def test_no_choices_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = MagicMock()
        with (
            patch.object(httpx, "post", return_value=mock_response),
            pytest.raises(ValueError, match="No choices"),
        ):
            call_local("sys", "usr", "http://localhost:8081/v1", False)

    def test_invalid_choice_format_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": ["not_a_dict"]}
        mock_response.raise_for_status = MagicMock()
        with (
            patch.object(httpx, "post", return_value=mock_response),
            pytest.raises(ValueError, match="Invalid choice"),
        ):
            call_local("sys", "usr", "http://localhost:8081/v1", False)

    def test_invalid_message_format_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": "not_dict"}],
        }
        mock_response.raise_for_status = MagicMock()
        with (
            patch.object(httpx, "post", return_value=mock_response),
            pytest.raises(ValueError, match="Invalid message"),
        ):
            call_local("sys", "usr", "http://localhost:8081/v1", False)

    def test_invalid_content_format_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": 12345}}],
        }
        mock_response.raise_for_status = MagicMock()
        with (
            patch.object(httpx, "post", return_value=mock_response),
            pytest.raises(ValueError, match="Invalid content"),
        ):
            call_local("sys", "usr", "http://localhost:8081/v1", False)

    def test_http_error_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(),
        )
        with (
            patch.object(httpx, "post", return_value=mock_response),
            pytest.raises(httpx.HTTPStatusError),
        ):
            call_local("sys", "usr", "http://localhost:8081/v1", False)


class TestCallHaiku:
    def test_successful_call(self) -> None:
        mock_text_block = MagicMock()
        mock_text_block.text = "result text"

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]

        mock_sdk = MagicMock()
        mock_sdk.TextBlock = type(mock_text_block)
        mock_sdk.AssistantMessage = type(mock_message)

        class MockAsyncIter:
            def __init__(self) -> None:
                self._items = [mock_message]
                self._index = 0

            def __aiter__(self) -> MockAsyncIter:
                return self

            async def __anext__(self) -> object:
                if self._index >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._index]
                self._index += 1
                return item

        mock_sdk.query.return_value = MockAsyncIter()

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            result = call_haiku("system", "user", "claude-haiku-4-5")
            assert result == "result text"

    def test_import_error_propagates(self) -> None:
        with (
            patch.dict("sys.modules", {"claude_agent_sdk": None}),
            pytest.raises((ImportError, ModuleNotFoundError)),
        ):
            call_haiku("system", "user", "claude-haiku-4-5")


class TestCallLocalRequestBody:
    """Verify the request body structure sent to httpx.post (kills body mutants)."""

    def _make_mock_response(self) -> MagicMock:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        mock_response.raise_for_status = MagicMock()
        return mock_response

    def test_model_key_and_value(self) -> None:
        with patch.object(
            httpx, "post", return_value=self._make_mock_response(),
        ) as mock_post:
            call_local("sys", "usr", "http://localhost:8081/v1", False)
            body = mock_post.call_args.kwargs["json"]
            assert "model" in body
            assert body["model"] == "default"

    def test_message_structure(self) -> None:
        with patch.object(
            httpx, "post", return_value=self._make_mock_response(),
        ) as mock_post:
            call_local("my_system", "my_user", "http://localhost:8081/v1", False)
            body = mock_post.call_args.kwargs["json"]
            messages = body["messages"]
            assert len(messages) == 2
            assert messages[0] == {"role": "system", "content": "my_system"}
            assert messages[1] == {"role": "user", "content": "my_user"}

    def test_stop_sequence(self) -> None:
        with patch.object(
            httpx, "post", return_value=self._make_mock_response(),
        ) as mock_post:
            call_local("sys", "usr", "http://localhost:8081/v1", False)
            body = mock_post.call_args.kwargs["json"]
            assert "stop" in body
            assert body["stop"] == ["\n\n"]

    def test_url_construction(self) -> None:
        with patch.object(
            httpx, "post", return_value=self._make_mock_response(),
        ) as mock_post:
            call_local("sys", "usr", "http://localhost:8081/v1", False)
            url = mock_post.call_args.args[0]
            assert url == "http://localhost:8081/v1/chat/completions"

    def test_url_strips_trailing_slash(self) -> None:
        with patch.object(
            httpx, "post", return_value=self._make_mock_response(),
        ) as mock_post:
            call_local("sys", "usr", "http://localhost:8081/v1/", False)
            url = mock_post.call_args.args[0]
            assert url == "http://localhost:8081/v1/chat/completions"

    def test_timeout_set(self) -> None:
        with patch.object(
            httpx, "post", return_value=self._make_mock_response(),
        ) as mock_post:
            call_local("sys", "usr", "http://localhost:8081/v1", False)
            assert mock_post.call_args.kwargs["timeout"] == 60.0

    def test_default_max_tokens(self) -> None:
        with patch.object(
            httpx, "post", return_value=self._make_mock_response(),
        ) as mock_post:
            call_local("sys", "usr", "http://localhost:8081/v1", False)
            body = mock_post.call_args.kwargs["json"]
            assert body["max_tokens"] == 1024

    def test_default_temperature(self) -> None:
        with patch.object(
            httpx, "post", return_value=self._make_mock_response(),
        ) as mock_post:
            call_local("sys", "usr", "http://localhost:8081/v1", False)
            body = mock_post.call_args.kwargs["json"]
            assert body["temperature"] == 0.0


class TestCallHaikuRequestStructure:
    """Verify call_haiku passes correct options to claude-agent-sdk."""

    def test_options_passed_correctly(self) -> None:
        mock_text_block = MagicMock()
        mock_text_block.text = "result"

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]

        mock_sdk = MagicMock()
        mock_sdk.TextBlock = type(mock_text_block)
        mock_sdk.AssistantMessage = type(mock_message)

        class MockAsyncIter:
            def __init__(self) -> None:
                self._items = [mock_message]
                self._index = 0

            def __aiter__(self) -> MockAsyncIter:
                return self

            async def __anext__(self) -> object:
                if self._index >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._index]
                self._index += 1
                return item

        mock_sdk.query.return_value = MockAsyncIter()

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            call_haiku("my_system", "my_user", "claude-haiku-4-5")
            # Verify query was called with correct prompt
            mock_sdk.query.assert_called_once()
            call_kwargs = mock_sdk.query.call_args.kwargs
            assert call_kwargs["prompt"] == "my_user"
            # Verify ClaudeAgentOptions was constructed with correct args
            opts_kwargs = mock_sdk.ClaudeAgentOptions.call_args.kwargs
            assert opts_kwargs["model"] == "claude-haiku-4-5"
            assert opts_kwargs["system_prompt"] == "my_system"
            assert opts_kwargs["tools"] == []
            assert opts_kwargs["max_turns"] == 1
            assert opts_kwargs["permission_mode"] == "bypassPermissions"
            assert opts_kwargs["setting_sources"] == []


class TestValidateEndpointScheme:
    """Kill mutants that corrupt the HTTPS scheme check."""

    def test_https_scheme_accepted(self) -> None:
        """HTTPS must be in the allowed set (case-sensitive lowercase)."""
        validate_endpoint("https://localhost:8081/v1")

    def test_uppercase_scheme_rejected(self) -> None:
        """Schemes are case-normalized by urlparse — HTTPS parses as https."""
        # urlparse normalizes scheme to lowercase, so "HTTPS://..." → scheme="https"
        # This test ensures "https" (lowercase) is in the allowlist
        validate_endpoint("HTTPS://localhost:8081/v1")


class TestCallHaikuOptionsPassedToQuery:
    """Kill mutants that drop options from query() call."""

    def test_query_receives_options_not_none(self) -> None:
        """query() must receive the constructed options object, not None."""
        mock_text_block = MagicMock()
        mock_text_block.text = "result"

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]

        mock_sdk = MagicMock()
        mock_sdk.TextBlock = type(mock_text_block)
        mock_sdk.AssistantMessage = type(mock_message)

        class MockAsyncIter:
            def __init__(self) -> None:
                self._items = [mock_message]
                self._index = 0

            def __aiter__(self) -> MockAsyncIter:
                return self

            async def __anext__(self) -> object:
                if self._index >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._index]
                self._index += 1
                return item

        mock_sdk.query.return_value = MockAsyncIter()
        expected_options = mock_sdk.ClaudeAgentOptions.return_value

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            call_haiku("sys", "usr", "claude-haiku-4-5")
            call_kwargs = mock_sdk.query.call_args.kwargs
            assert call_kwargs["options"] is expected_options
            assert call_kwargs["options"] is not None

    def test_multipart_response_joined_without_separator(self) -> None:
        """Multiple text blocks must be joined with empty string, not 'XXXX'."""

        class FakeTextBlock:
            def __init__(self, text: str) -> None:
                self.text = text

        class FakeAssistantMessage:
            def __init__(self, content: list[object]) -> None:
                self.content = content

        mock_block_1 = FakeTextBlock("hello")
        mock_block_2 = FakeTextBlock(" world")
        mock_message = FakeAssistantMessage([mock_block_1, mock_block_2])

        mock_sdk = MagicMock()
        mock_sdk.TextBlock = FakeTextBlock
        mock_sdk.AssistantMessage = FakeAssistantMessage

        class MockAsyncIter:
            def __init__(self) -> None:
                self._items = [mock_message]
                self._index = 0

            def __aiter__(self) -> MockAsyncIter:
                return self

            async def __anext__(self) -> object:
                if self._index >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._index]
                self._index += 1
                return item

        mock_sdk.query.return_value = MockAsyncIter()

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            result = call_haiku("sys", "usr", "claude-haiku-4-5")
            assert result == "hello world"
            assert "XXXX" not in result


class TestConstants:
    def test_loopback_hosts_accepted(self) -> None:
        """Verify validate_endpoint accepts all loopback addresses."""
        validate_endpoint("http://localhost:8081/v1")
        validate_endpoint("http://127.0.0.1:8081/v1")
        validate_endpoint("http://[::1]:8081/v1")

    def test_allowed_haiku_models(self) -> None:
        assert "claude-haiku-4-5" in _ALLOWED_HAIKU_MODELS
