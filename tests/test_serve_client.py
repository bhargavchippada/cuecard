"""Tests for query_daemon client and adapter fast path."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from cuecard.serve import query_daemon

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# Client (query_daemon)
# ---------------------------------------------------------------------------


class TestQueryDaemon:
    def test_daemon_not_running(self) -> None:
        # Should return None when nothing is listening
        result = query_daemon(
            {"tool_name": "Bash"}, port=19999, timeout=0.1,
        )
        assert result is None

    def test_daemon_timeout(self) -> None:
        # Non-routable address to trigger timeout
        with patch("http.client.HTTPConnection") as mock_conn_cls:
            mock_conn = MagicMock()
            mock_conn.request.side_effect = TimeoutError
            mock_conn_cls.return_value = mock_conn
            result = query_daemon({"tool_name": "Bash"}, timeout=0.1)
        assert result is None

    def test_daemon_non_200(self) -> None:
        with patch("http.client.HTTPConnection") as mock_conn_cls:
            mock_conn = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status = 500
            mock_conn.getresponse.return_value = mock_resp
            mock_conn_cls.return_value = mock_conn
            result = query_daemon({"tool_name": "Bash"})
        assert result is None

    def test_daemon_invalid_json_response(self) -> None:
        with patch("http.client.HTTPConnection") as mock_conn_cls:
            mock_conn = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = b"not json"
            mock_conn.getresponse.return_value = mock_resp
            mock_conn_cls.return_value = mock_conn
            result = query_daemon({"tool_name": "Bash"})
        assert result is None

    def test_close_exception_suppressed(self) -> None:
        with patch("http.client.HTTPConnection") as mock_conn_cls:
            mock_conn = MagicMock()
            mock_conn.request.side_effect = OSError("connection failed")
            mock_conn.close.side_effect = OSError("close failed")
            mock_conn_cls.return_value = mock_conn
            result = query_daemon({"tool_name": "Bash"})
        assert result is None


# ---------------------------------------------------------------------------
# Adapter fast path
# ---------------------------------------------------------------------------


class TestAdapterFastPath:
    def test_daemon_fast_path(self) -> None:
        """When daemon responds, adapter uses fast path and returns early."""
        from cuecard.adapters.claude_code import _try_daemon

        expected = {
            "tool_name": "Bash",
            "hookSpecificOutput": {"additionalContext": "rule"},
        }
        with patch("cuecard.serve.query_daemon", return_value=expected):
            result = _try_daemon({"tool_name": "Bash"})
        assert result == expected

    def test_daemon_fallback(self) -> None:
        """When daemon is unreachable, _try_daemon returns None."""
        from cuecard.adapters.claude_code import _try_daemon

        with patch("cuecard.serve.query_daemon", return_value=None):
            result = _try_daemon({"tool_name": "Bash"})
        assert result is None

    def test_main_fast_path_returns_early(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When daemon responds, main() prints result and returns early."""
        import io

        from cuecard.adapters.claude_code import main

        payload = {"tool_name": "Bash", "tool_input": "ls"}
        daemon_response = {
            "tool_name": "Bash",
            "tool_input": "ls",
            "hookSpecificOutput": {"additionalContext": "rule text"},
        }

        stdin_data = json.dumps(payload)
        with (
            patch("sys.stdin", io.StringIO(stdin_data)),
            patch(
                "cuecard.adapters.claude_code._try_daemon",
                return_value=daemon_response,
            ),
        ):
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["hookSpecificOutput"]["additionalContext"] == "rule text"
