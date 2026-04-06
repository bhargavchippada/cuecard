"""Tests for PID file management and daemon status."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

from cuecard.serve import (
    daemon_status,
    is_pid_alive,
    read_pid,
    remove_pid,
    write_pid,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
