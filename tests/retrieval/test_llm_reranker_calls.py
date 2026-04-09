"""Tests for LLM re-ranker backend calls (local, Haiku) and endpoint validation."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import httpx
import pytest

from cuecard.retrieval.llm_utils import call_local, validate_endpoint
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
            patch("cuecard.retrieval.llm_utils.socket.getaddrinfo", return_value=fake),
            pytest.raises(ConfigError, match="loopback"),
        ):
            validate_endpoint("http://evil.com:8081/v1")

    def test_ip_address_raises(self) -> None:
        fake = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0))]
        with (
            patch("cuecard.retrieval.llm_utils.socket.getaddrinfo", return_value=fake),
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
            "choices": [{"message": {"content": '{"rules": [1, 3]}'}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx, "post", return_value=mock_response) as mock_post:
            result = call_local(
                "system", "user", "http://localhost:8081/v1", False
            )
            assert result == '{"rules": [1, 3]}'
            mock_post.assert_called_once()

    def test_timeout_raises(self) -> None:
        with (
            patch.object(
                httpx, "post", side_effect=httpx.TimeoutException("timeout")
            ),
            pytest.raises(httpx.TimeoutException),
        ):
            call_local(
                "system", "user", "http://localhost:8081/v1", False
            )

    def test_connect_error_raises(self) -> None:
        with (
            patch.object(
                httpx, "post", side_effect=httpx.ConnectError("refused")
            ),
            pytest.raises(httpx.ConnectError),
        ):
            call_local(
                "system", "user", "http://localhost:8081/v1", False
            )

    def test_http_error_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        with (
            patch.object(httpx, "post", return_value=mock_response),
            pytest.raises(httpx.HTTPStatusError),
        ):
            call_local(
                "system", "user", "http://localhost:8081/v1", False
            )

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
        mock_response.json.return_value = {"choices": [{"message": "not_dict"}]}
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


class TestCallHaiku:
    """Haiku backend tests -- call_haiku now lives in llm_utils.

    See test_llm_utils.py for full coverage of call_haiku internals.
    These tests verify rerank_llm integration with the haiku backend.
    """
