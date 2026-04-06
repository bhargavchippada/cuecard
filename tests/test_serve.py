"""Tests for cuecard serve daemon module."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cuecard.models import (
    Index,
    PipelineConfig,
    PipelineResult,
    Provenance,
    RankedResult,
    ResolvedConfig,
    Rule,
    SourceMeta,
    StageTrace,
)
from cuecard.serve import (
    _Handler,
    _make_handler,
    _process_request,
    daemon_status,
    is_pid_alive,
    query_daemon,
    read_pid,
    remove_pid,
    start_server,
    stop_server,
    write_pid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rule(text: str = "Never commit secrets") -> Rule:
    return Rule(
        text=text,
        provenance=Provenance(file="/tmp/rules.txt", line_start=1, line_end=1),
    )


def _make_index(rules: tuple[Rule, ...] | None = None) -> Index:
    if rules is None:
        rules = (_make_rule(),)
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((len(rules), 384)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    return Index(
        embeddings=emb,
        rules=rules,
        model_name="BAAI/bge-small-en-v1.5",
        dim=384,
        sources={
            "/tmp/rules.txt": SourceMeta(
                mtime=1711929600.0,
                content_hash="sha256:abc123",
                rule_count=len(rules),
            ),
        },
    )


def _make_config() -> ResolvedConfig:
    return ResolvedConfig(
        source_paths=("/tmp/rules.txt",),
        global_source_paths=("/tmp/rules.txt",),
        project_source_paths=(),
        model_name="BAAI/bge-small-en-v1.5",
        top_k=5,
        threshold=0.30,
        dedup_threshold=0.95,
        query_max_length=500,
        hook_events=("PreToolUse",),
        verbose=False,
        redact=True,
        max_log_size_mb=10,
        global_cache_dir="/tmp/cuecard-test/index",
        project_cache_dir=None,
        allowed_dirs=(),
        pipeline=PipelineConfig(mode="embedding"),
    )


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------


class TestPidFile:
    def test_write_and_read_pid(self, tmp_path: Path) -> None:
        path = write_pid(12345, tmp_path)
        assert path.exists()
        assert read_pid(tmp_path) == 12345

    def test_read_pid_missing(self, tmp_path: Path) -> None:
        assert read_pid(tmp_path) is None

    def test_read_pid_invalid(self, tmp_path: Path) -> None:
        pid_path = tmp_path / ".cuecard" / "serve.pid"
        pid_path.parent.mkdir(parents=True)
        pid_path.write_text("not-a-number\n")
        assert read_pid(tmp_path) is None

    def test_remove_pid(self, tmp_path: Path) -> None:
        write_pid(12345, tmp_path)
        remove_pid(tmp_path)
        assert read_pid(tmp_path) is None

    def test_remove_pid_missing(self, tmp_path: Path) -> None:
        # Should not raise
        remove_pid(tmp_path)

    def test_write_pid_permissions(self, tmp_path: Path) -> None:
        path = write_pid(99, tmp_path)
        mode = oct(path.stat().st_mode & 0o777)
        assert mode == "0o600"


class TestIsPidAlive:
    def test_own_pid_alive(self) -> None:
        assert is_pid_alive(os.getpid()) is True

    def test_nonexistent_pid(self) -> None:
        # PID 99999999 should not exist
        assert is_pid_alive(99999999) is False

    def test_permission_error(self) -> None:
        with patch("os.kill", side_effect=PermissionError):
            assert is_pid_alive(1) is True


class TestDaemonStatus:
    def test_no_pid_file(self, tmp_path: Path) -> None:
        running, pid = daemon_status(tmp_path)
        assert running is False
        assert pid is None

    def test_running_pid(self, tmp_path: Path) -> None:
        write_pid(os.getpid(), tmp_path)
        running, pid = daemon_status(tmp_path)
        assert running is True
        assert pid == os.getpid()

    def test_stale_pid_cleaned(self, tmp_path: Path) -> None:
        write_pid(99999999, tmp_path)
        running, pid = daemon_status(tmp_path)
        assert running is False
        assert pid is None
        # PID file should be cleaned up
        assert read_pid(tmp_path) is None


# ---------------------------------------------------------------------------
# Request processing
# ---------------------------------------------------------------------------


class TestProcessRequest:
    def test_pretooluse_query(self) -> None:
        index = _make_index()
        config = _make_config()
        model = MagicMock()

        fake_pipeline = PipelineResult(
            results=(RankedResult(rule=_make_rule(), score=0.85),),
            stages=(StageTrace(
                stage="retrieval", input_count=1,
                output_count=1, latency_ms=1.0,
            ),),
            mode="embedding",
        )

        with patch("cuecard.pipeline.run_pipeline", return_value=fake_pipeline):
            result = _process_request(
                {"tool_name": "Bash", "tool_input": "git status"},
                index, config, model,
            )

        assert "hookSpecificOutput" in result
        hook_output = result["hookSpecificOutput"]
        assert isinstance(hook_output, dict)
        assert "additionalContext" in hook_output

    def test_user_prompt_submit(self) -> None:
        index = _make_index()
        config = _make_config()
        model = MagicMock()

        fake_pipeline = PipelineResult(
            results=(),
            stages=(StageTrace(
                stage="retrieval", input_count=1,
                output_count=0, latency_ms=0.5,
            ),),
            mode="embedding",
        )

        with patch("cuecard.pipeline.run_pipeline", return_value=fake_pipeline):
            result = _process_request(
                {"event": "UserPromptSubmit", "prompt": "add auth"},
                index, config, model,
            )

        # hookEventName always set, no additionalContext when no results
        hook_out = result["hookSpecificOutput"]
        assert hook_out["hookEventName"] == "UserPromptSubmit"
        assert "permissionDecision" not in hook_out
        assert "additionalContext" not in hook_out

    def test_does_not_mutate_input(self) -> None:
        index = _make_index()
        config = _make_config()
        model = MagicMock()

        fake_pipeline = PipelineResult(
            results=(RankedResult(rule=_make_rule(), score=0.9),),
            stages=(StageTrace(
                stage="retrieval", input_count=1,
                output_count=1, latency_ms=1.0,
            ),),
            mode="embedding",
        )

        original = {"tool_name": "Read", "tool_input": "file.py"}
        with patch("cuecard.pipeline.run_pipeline", return_value=fake_pipeline):
            result = _process_request(original, index, config, model)

        # Original should not be mutated
        assert "hookSpecificOutput" not in original
        assert "hookSpecificOutput" in result

    def test_preserves_existing_hook_output(self) -> None:
        index = _make_index()
        config = _make_config()
        model = MagicMock()

        fake_pipeline = PipelineResult(
            results=(RankedResult(rule=_make_rule(), score=0.9),),
            stages=(StageTrace(
                stage="retrieval", input_count=1,
                output_count=1, latency_ms=1.0,
            ),),
            mode="embedding",
        )

        data = {
            "tool_name": "Read",
            "tool_input": "file.py",
            "hookSpecificOutput": {"existing": "value"},
        }
        with patch("cuecard.pipeline.run_pipeline", return_value=fake_pipeline):
            result = _process_request(data, index, config, model)

        hook_out = result["hookSpecificOutput"]
        assert isinstance(hook_out, dict)
        assert hook_out["existing"] == "value"
        assert "additionalContext" in hook_out


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class TestHandler:
    def test_make_handler_binds_attrs(self) -> None:
        index = _make_index()
        config = _make_config()
        model = MagicMock()
        cls = _make_handler(index, config, model)
        assert cls._index is index
        assert cls._config is config
        assert cls._embedding_model is model
        assert issubclass(cls, _Handler)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class TestStartServer:
    def test_start_with_explicit_deps(self, tmp_path: Path) -> None:
        index = _make_index()
        config = _make_config()
        model = MagicMock()

        server = start_server(
            port=0,  # OS picks a free port
            home=tmp_path,
            config=config,
            index=index,
            embedding_model=model,
        )
        try:
            assert isinstance(server, HTTPServer)
            assert read_pid(tmp_path) == os.getpid()
        finally:
            server.server_close()
            remove_pid(tmp_path)

    def test_start_no_index_raises(self, tmp_path: Path) -> None:
        config = _make_config()
        model = MagicMock()

        with (
            patch("cuecard.loader.load_or_build", return_value=None),
            pytest.raises(RuntimeError, match="No rules indexed"),
        ):
            start_server(
                port=0,
                home=tmp_path,
                config=config,
                embedding_model=model,
            )

    def test_start_empty_index_raises(self, tmp_path: Path) -> None:
        config = _make_config()
        model = MagicMock()
        empty_index = Index(
            embeddings=np.zeros((0, 384), dtype=np.float32),
            rules=(),
            model_name="BAAI/bge-small-en-v1.5",
            dim=384,
            sources={},
            rule_map=(),
        )

        with pytest.raises(RuntimeError, match="No rules indexed"):
            start_server(
                port=0,
                home=tmp_path,
                config=config,
                index=empty_index,
                embedding_model=model,
            )

    def test_start_loads_deps_when_not_provided(
        self, tmp_path: Path,
    ) -> None:
        config = _make_config()
        model = MagicMock()
        index = _make_index()

        with (
            patch("cuecard.config.load_config", return_value=config),
            patch("fastembed.TextEmbedding", return_value=model),
            patch("cuecard.loader.load_or_build", return_value=index),
        ):
            server = start_server(port=0, home=tmp_path)
        try:
            assert isinstance(server, HTTPServer)
        finally:
            server.server_close()
            remove_pid(tmp_path)


class TestStopServer:
    def test_stop_no_daemon(self, tmp_path: Path) -> None:
        assert stop_server(tmp_path) is False

    def test_stop_running_daemon(self, tmp_path: Path) -> None:
        write_pid(os.getpid(), tmp_path)
        original_kill = os.kill
        kill_calls: list[tuple[int, int]] = []

        terminated = False

        def _tracking_kill(pid: int, sig: int) -> None:
            nonlocal terminated
            if sig == signal.SIGTERM:
                kill_calls.append((pid, sig))
                terminated = True
                return  # Don't actually send SIGTERM to ourselves
            if sig == 0 and terminated:
                raise ProcessLookupError  # Simulate process exited
            original_kill(pid, sig)

        with patch("os.kill", side_effect=_tracking_kill):
            result = stop_server(tmp_path)
        assert result is True
        assert kill_calls == [(os.getpid(), signal.SIGTERM)]
        assert read_pid(tmp_path) is None

    def test_stop_stale_pid(self, tmp_path: Path) -> None:
        write_pid(99999999, tmp_path)
        assert stop_server(tmp_path) is False

    def test_stop_process_gone_during_kill(self, tmp_path: Path) -> None:
        write_pid(os.getpid(), tmp_path)
        sig0_calls = 0

        def _kill_side_effect(pid: int, sig: int) -> None:
            nonlocal sig0_calls
            if sig == signal.SIGTERM:
                raise ProcessLookupError
            if sig == 0:
                sig0_calls += 1
                if sig0_calls > 1:
                    raise ProcessLookupError  # Gone after SIGTERM
                return  # First call: daemon_status alive check

        with patch("os.kill", side_effect=_kill_side_effect):
            result = stop_server(tmp_path)
        assert result is True


# ---------------------------------------------------------------------------
# HTTP integration (real server on random port)
# ---------------------------------------------------------------------------


class TestHTTPIntegration:
    def _start_test_server(
        self, tmp_path: Path,
    ) -> tuple[HTTPServer, int]:
        index = _make_index()
        config = _make_config()
        model = MagicMock()

        server = start_server(
            port=0,
            home=tmp_path,
            config=config,
            index=index,
            embedding_model=model,
        )
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, port

    def test_post_retrieve(self, tmp_path: Path) -> None:
        server, port = self._start_test_server(tmp_path)

        fake_pipeline = PipelineResult(
            results=(RankedResult(rule=_make_rule(), score=0.85),),
            stages=(StageTrace(
                stage="retrieval", input_count=1,
                output_count=1, latency_ms=1.0,
            ),),
            mode="embedding",
        )

        try:
            with patch(
                "cuecard.pipeline.run_pipeline",
                return_value=fake_pipeline,
            ):
                result = query_daemon(
                    {"tool_name": "Bash", "tool_input": "ls"},
                    port=port,
                    timeout=5.0,
                )

            assert result is not None
            assert "hookSpecificOutput" in result
        finally:
            server.shutdown()
            server.server_close()
            remove_pid(tmp_path)

    def test_get_health(self, tmp_path: Path) -> None:
        server, port = self._start_test_server(tmp_path)

        import http.client

        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data["status"] == "ok"
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            remove_pid(tmp_path)

    def test_get_not_found(self, tmp_path: Path) -> None:
        server, port = self._start_test_server(tmp_path)

        import http.client

        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
            conn.request("GET", "/nonexistent")
            resp = conn.getresponse()
            assert resp.status == 404
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            remove_pid(tmp_path)

    def test_post_invalid_json(self, tmp_path: Path) -> None:
        server, port = self._start_test_server(tmp_path)

        import http.client

        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
            conn.request(
                "POST", "/retrieve",
                body=b"not json",
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": "8",
                },
            )
            resp = conn.getresponse()
            assert resp.status == 400
            data = json.loads(resp.read())
            assert "Invalid JSON" in data["error"]
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            remove_pid(tmp_path)

    def test_post_not_object(self, tmp_path: Path) -> None:
        server, port = self._start_test_server(tmp_path)

        import http.client

        try:
            body = b"[1,2,3]"
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
            conn.request(
                "POST", "/retrieve",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            resp = conn.getresponse()
            assert resp.status == 400
            data = json.loads(resp.read())
            assert "Expected JSON object" in data["error"]
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            remove_pid(tmp_path)

    def test_post_too_large(self, tmp_path: Path) -> None:
        server, port = self._start_test_server(tmp_path)

        import http.client

        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
            # Claim a huge Content-Length but don't actually send it
            conn.request(
                "POST", "/retrieve",
                body=b"{}",
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": "2000000",
                },
            )
            resp = conn.getresponse()
            assert resp.status == 413
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            remove_pid(tmp_path)

    def test_post_invalid_content_length(self, tmp_path: Path) -> None:
        server, port = self._start_test_server(tmp_path)

        import http.client

        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
            conn.request(
                "POST", "/retrieve",
                body=b"{}",
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": "not-a-number",
                },
            )
            resp = conn.getresponse()
            assert resp.status == 400
            data = json.loads(resp.read())
            assert "Invalid Content-Length" in data["error"]
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            remove_pid(tmp_path)

    def test_post_wrong_path(self, tmp_path: Path) -> None:
        server, port = self._start_test_server(tmp_path)

        import http.client

        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
            conn.request("POST", "/admin", body=b"{}", headers={
                "Content-Type": "application/json",
                "Content-Length": "2",
            })
            resp = conn.getresponse()
            assert resp.status == 404
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            remove_pid(tmp_path)

    def test_post_negative_content_length(self, tmp_path: Path) -> None:
        server, port = self._start_test_server(tmp_path)

        import http.client

        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
            conn.request("POST", "/retrieve", body=b"", headers={
                "Content-Type": "application/json",
                "Content-Length": "-1",
            })
            resp = conn.getresponse()
            assert resp.status == 400
            data = json.loads(resp.read())
            assert "Invalid Content-Length" in data["error"]
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            remove_pid(tmp_path)


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
# run_server (partial, since it blocks)
# ---------------------------------------------------------------------------


class TestRunServer:
    def test_already_running_exits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        write_pid(os.getpid(), tmp_path)

        from cuecard.serve import run_server

        with pytest.raises(SystemExit):
            run_server(port=0, home=tmp_path)

        captured = capsys.readouterr()
        assert "already running" in captured.err

    def test_run_server_lifecycle(self, tmp_path: Path) -> None:
        """Test run_server via start_server + serve_forever + cleanup."""
        index = _make_index()
        config = _make_config()
        model = MagicMock()

        server = start_server(
            port=0, home=tmp_path,
            config=config, index=index, embedding_model=model,
        )
        assert read_pid(tmp_path) is not None

        # Run serve_forever in a thread, then shut it down
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.1)
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()
        remove_pid(tmp_path)
        assert read_pid(tmp_path) is None

    def test_run_server_full_path(self, tmp_path: Path) -> None:
        """Test run_server() with a mock server that exits immediately."""
        from cuecard.serve import run_server

        mock_server = MagicMock()
        mock_server.serve_forever.return_value = None

        with patch("cuecard.serve.start_server", return_value=mock_server):
            run_server(port=0, home=tmp_path)

        mock_server.serve_forever.assert_called_once()
        mock_server.server_close.assert_called_once()
        # PID file cleaned up
        assert read_pid(tmp_path) is None

    def test_run_server_signal_handler_idempotent(self, tmp_path: Path) -> None:
        """Signal handler should only trigger shutdown once."""
        from cuecard.serve import run_server

        mock_server = MagicMock()
        mock_server.serve_forever.return_value = None
        captured_handlers: dict[int, object] = {}

        original_signal = signal.signal

        def _capture_signal(signum: int, handler: object) -> object:
            captured_handlers[signum] = handler
            return original_signal(signum, signal.SIG_DFL)

        with (
            patch("cuecard.serve.start_server", return_value=mock_server),
            patch("signal.signal", side_effect=_capture_signal),
        ):
            run_server(port=0, home=tmp_path)

        # Call the SIGTERM handler twice
        handler = captured_handlers.get(signal.SIGTERM)
        assert handler is not None
        handler(signal.SIGTERM, None)  # type: ignore[operator]
        handler(signal.SIGTERM, None)  # type: ignore[operator]
        # shutdown should only be called once by the handler
        # (serve_forever already returned, so handler sets flag)
        mock_server.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


class TestServeCLI:
    def test_serve_stop_no_daemon(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from typer.testing import CliRunner

        from cuecard.cli import app

        monkeypatch.setattr("cuecard.cli._home_dir", lambda: tmp_path)
        cli_runner = CliRunner()
        result = cli_runner.invoke(app, ["serve", "--stop"])
        assert result.exit_code == 0
        assert "No daemon running" in result.output

    def test_serve_stop_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from typer.testing import CliRunner

        from cuecard.cli import app

        monkeypatch.setattr("cuecard.cli._home_dir", lambda: tmp_path)
        write_pid(os.getpid(), tmp_path)

        cli_runner = CliRunner()
        with patch("cuecard.serve.os.kill"):
            result = cli_runner.invoke(app, ["serve", "--stop"])
        assert result.exit_code == 0
        assert "Daemon stopped" in result.output

    def test_serve_already_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from typer.testing import CliRunner

        from cuecard.cli import app

        monkeypatch.setattr("cuecard.cli._home_dir", lambda: tmp_path)
        write_pid(os.getpid(), tmp_path)

        cli_runner = CliRunner()
        result = cli_runner.invoke(app, ["serve"])
        assert result.exit_code == 1
        assert "already running" in result.output

    def test_serve_daemon_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from typer.testing import CliRunner

        from cuecard.cli import app

        monkeypatch.setattr("cuecard.cli._home_dir", lambda: tmp_path)

        cli_runner = CliRunner()
        with patch("cuecard.cli._fork_daemon") as mock_fork:
            result = cli_runner.invoke(app, ["serve", "--daemon"])
        assert result.exit_code == 0
        mock_fork.assert_called_once()

    def test_fork_daemon_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Parent process (fork returns child PID) prints and returns."""
        from cuecard.cli import _fork_daemon

        monkeypatch.setattr("os.fork", lambda: 12345)
        _fork_daemon(8452, tmp_path)  # Should return without error

    def test_fork_daemon_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Child process (fork returns 0) detaches and runs server."""
        from cuecard.cli import _fork_daemon

        monkeypatch.setattr("os.fork", lambda: 0)
        monkeypatch.setattr("os.setsid", lambda: None)
        monkeypatch.setattr("os.dup2", lambda _fd, _target: None)
        monkeypatch.setattr(
            "os.open", lambda _p, _f, *a: 99,
        )
        monkeypatch.setattr("os.close", lambda _fd: None)
        with patch("cuecard.serve.run_server") as mock_run:
            _fork_daemon(8452, tmp_path)
        mock_run.assert_called_once_with(port=8452, home=tmp_path)

    def test_fork_daemon_child_system_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Child process handles SystemExit gracefully."""
        from cuecard.cli import _fork_daemon

        monkeypatch.setattr("os.fork", lambda: 0)
        monkeypatch.setattr("os.setsid", lambda: None)
        monkeypatch.setattr("os.dup2", lambda _fd, _target: None)
        monkeypatch.setattr(
            "os.open", lambda _p, _f, *a: 99,
        )
        monkeypatch.setattr("os.close", lambda _fd: None)
        with patch("cuecard.serve.run_server", side_effect=SystemExit):
            _fork_daemon(8452, tmp_path)  # Should not raise


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


class TestLogPath:
    def test_log_path_default_home(self) -> None:
        from cuecard.serve import _log_path
        result = _log_path()
        assert result == Path.home() / ".cuecard" / "serve.log"

    def test_log_path_custom_home(self, tmp_path: Path) -> None:
        from cuecard.serve import _log_path
        result = _log_path(tmp_path)
        assert result == tmp_path / ".cuecard" / "serve.log"


class TestForkDaemon:
    def test_fork_parent_reports_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from typer.testing import CliRunner

        from cuecard.cli import app

        monkeypatch.setattr("cuecard.cli._home_dir", lambda: tmp_path)

        # Mock os.fork to return child PID (parent path)
        with patch("os.fork", return_value=42):
            cli_runner = CliRunner()
            result = cli_runner.invoke(app, ["serve", "--daemon"])
        assert result.exit_code == 0
        assert "PID 42" in result.output

    def test_fork_child_path(self, tmp_path: Path) -> None:
        """Test the child process path of _fork_daemon."""
        from cuecard.cli import _fork_daemon

        with (
            patch("os.fork", return_value=0),  # child
            patch("os.setsid"),
            patch("os.open", return_value=99),
            patch("os.dup2"),
            patch("os.close"),
            patch("cuecard.serve.run_server"),
            patch(
                "cuecard.serve._log_path",
                return_value=tmp_path / ".cuecard" / "serve.log",
            ),
        ):
            (tmp_path / ".cuecard").mkdir(parents=True, exist_ok=True)
            _fork_daemon(8452, tmp_path)

    def test_fork_child_system_exit(self, tmp_path: Path) -> None:
        """Test the child process path catches SystemExit."""
        from cuecard.cli import _fork_daemon

        with (
            patch("os.fork", return_value=0),
            patch("os.setsid"),
            patch("os.open", return_value=99),
            patch("os.dup2"),
            patch("os.close"),
            patch(
                "cuecard.serve.run_server",
                side_effect=SystemExit(0),
            ),
            patch(
                "cuecard.serve._log_path",
                return_value=tmp_path / ".cuecard" / "serve.log",
            ),
        ):
            (tmp_path / ".cuecard").mkdir(parents=True, exist_ok=True)
            _fork_daemon(8452, tmp_path)

    def test_serve_foreground(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from typer.testing import CliRunner

        from cuecard.cli import app

        monkeypatch.setattr("cuecard.cli._home_dir", lambda: tmp_path)

        cli_runner = CliRunner()
        with patch("cuecard.serve.run_server") as mock_run:
            result = cli_runner.invoke(app, ["serve", "--port", "9999"])
        assert result.exit_code == 0
        assert "Starting cuecard daemon" in result.output
        mock_run.assert_called_once_with(port=9999, home=tmp_path)


# ---------------------------------------------------------------------------
# Status shows daemon
# ---------------------------------------------------------------------------


class TestStatusDaemon:
    def test_status_shows_daemon_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from typer.testing import CliRunner

        from cuecard.cli import app

        monkeypatch.setattr("cuecard.cli._home_dir", lambda: tmp_path)
        monkeypatch.setattr("cuecard.cli_hooks._cli._home_dir", lambda: tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        config_path = cuecard_dir / "config.toml"
        config_path.write_text(
            "[sources]\n"
            'rules = ["rules/global.txt"]\n\n'
            "[embedding]\n"
            'model = "BAAI/bge-small-en-v1.5"\n\n'
            "[retrieval]\n"
            "top_k = 5\n"
            "threshold = 0.30\n",
        )
        rules_dir = cuecard_dir / "rules"
        rules_dir.mkdir()
        (rules_dir / "global.txt").write_text("test rule\n")

        write_pid(os.getpid(), tmp_path)
        monkeypatch.chdir(tmp_path)

        cli_runner = CliRunner()
        result = cli_runner.invoke(app, ["status"])
        assert f"PID {os.getpid()}" in result.output
        assert "Daemon running" in result.output

        remove_pid(tmp_path)

    def test_status_shows_daemon_not_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from typer.testing import CliRunner

        from cuecard.cli import app

        monkeypatch.setattr("cuecard.cli._home_dir", lambda: tmp_path)
        monkeypatch.setattr("cuecard.cli_hooks._cli._home_dir", lambda: tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        config_path = cuecard_dir / "config.toml"
        config_path.write_text(
            "[sources]\n"
            'rules = ["rules/global.txt"]\n\n'
            "[embedding]\n"
            'model = "BAAI/bge-small-en-v1.5"\n\n'
            "[retrieval]\n"
            "top_k = 5\n"
            "threshold = 0.30\n",
        )
        rules_dir = cuecard_dir / "rules"
        rules_dir.mkdir()
        (rules_dir / "global.txt").write_text("test rule\n")
        monkeypatch.chdir(tmp_path)

        cli_runner = CliRunner()
        result = cli_runner.invoke(app, ["status"])
        assert "Daemon not running" in result.output
