"""Tests for serve CLI commands, fork daemon, log path, and status."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from cuecard.serve import (
    remove_pid,
    write_pid,
)

if TYPE_CHECKING:
    import pytest

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

        call_count = 0

        def _fake_kill(_pid: int, sig: int) -> None:
            nonlocal call_count
            call_count += 1
            # 1st: is_pid_alive check (sig 0) → alive
            # 2nd: SIGTERM → ok
            # 3rd: liveness poll (sig 0) → dead
            if sig == 0 and call_count > 2:
                raise ProcessLookupError

        with patch("cuecard.serve.os.kill", side_effect=_fake_kill):
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
