"""Tests for server start, stop, and run_server lifecycle."""

from __future__ import annotations

import os
import signal
import threading
import time
from http.server import HTTPServer
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cuecard.models import (
    Index,
    LoadedIndex,
    PipelineConfig,
    Provenance,
    ResolvedConfig,
    Rule,
    SourceMeta,
)
from cuecard.serve import (
    read_pid,
    remove_pid,
    start_server,
    stop_server,
    write_pid,
)

if TYPE_CHECKING:
    from pathlib import Path

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
            patch("cuecard.indexing.loader.load_or_build", return_value=None),
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
            patch(
                "cuecard.indexing.loader.load_or_build",
                return_value=LoadedIndex(index=index),
            ),
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



class TestStopServerForceKill:
    def test_stop_force_kills_stuck_daemon(self, tmp_path: Path) -> None:
        write_pid(os.getpid(), tmp_path)
        calls: list[int] = []

        def _kill_side_effect(pid: int, sig: int) -> None:
            calls.append(sig)
            if sig == signal.SIGTERM:
                return
            if sig == 0:
                return
            if sig == signal.SIGKILL:
                raise ProcessLookupError

        with (
            patch("os.kill", side_effect=_kill_side_effect),
            patch("time.sleep"),
        ):
            result = stop_server(tmp_path)

        assert result is True
        assert signal.SIGTERM in calls
        assert signal.SIGKILL in calls
        assert read_pid(tmp_path) is None
