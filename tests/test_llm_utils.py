"""Tests for shared LLM utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from cuecard.llm_utils import (
    _ALLOWED_HAIKU_MODELS,
    _ALLOWED_LLM_HOSTS,
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
        with pytest.raises(ConfigError, match="localhost"):
            validate_endpoint("http://evil.com:8081/v1")

    def test_ip_address_raises(self) -> None:
        with pytest.raises(ConfigError, match="localhost"):
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


class TestConstants:
    def test_allowed_hosts(self) -> None:
        assert "localhost" in _ALLOWED_LLM_HOSTS
        assert "127.0.0.1" in _ALLOWED_LLM_HOSTS
        assert "::1" in _ALLOWED_LLM_HOSTS

    def test_allowed_haiku_models(self) -> None:
        assert "claude-haiku-4-5" in _ALLOWED_HAIKU_MODELS
