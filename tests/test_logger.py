"""Tests for cuecard.logger — structured logging, rotation, stats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from cuecard.logger import (
    _TAIL_CHUNK,
    _build_entry,
    _default_log_dir,
    _tail_lines,
    compute_stats,
    log_retrieval,
    read_log,
    rotate_log,
)
from cuecard.models import Provenance, RankedResult, Rule

if TYPE_CHECKING:
    import pytest


def _make_results(count: int = 2) -> list[RankedResult]:
    results: list[RankedResult] = []
    for i in range(count):
        prov = Provenance(
            file="/tmp/rules.txt", line_start=i + 1, line_end=i + 1,
        )
        rule = Rule(text=f"Rule number {i + 1}", provenance=prov)
        results.append(RankedResult(rule=rule, score=0.9 - i * 0.1))
    return results


class TestBuildEntry:
    def test_basic_entry(self) -> None:
        results = _make_results(2)
        entry = _build_entry(
            "PreToolUse", "Bash", "Bash: git commit",
            results,
            total_rules=50, index_rebuilt=False,
            latency_ms=42.123, model="BAAI/bge-small-en-v1.5",
            redact=False, max_query_length=500,
        )
        assert entry["event"] == "PreToolUse"
        assert entry["tool_name"] == "Bash"
        assert entry["query"] == "Bash: git commit"
        assert entry["query_truncated"] is False
        assert entry["injected_count"] == 2
        assert entry["total_rules"] == 50
        assert entry["index_rebuilt"] is False
        assert entry["latency_ms"] == 42.12
        assert entry["model"] == "BAAI/bge-small-en-v1.5"
        assert "timestamp" in entry
        assert len(entry["results"]) == 2  # type: ignore[arg-type]

    def test_query_truncation(self) -> None:
        long_query = "x" * 600
        entry = _build_entry(
            "PreToolUse", "Bash", long_query,
            [],
            total_rules=10, index_rebuilt=False,
            latency_ms=5.0, model="test",
            redact=False, max_query_length=100,
        )
        assert entry["query_truncated"] is True
        assert len(str(entry["query"])) == 100

    def test_no_truncation_at_limit(self) -> None:
        query = "x" * 500
        entry = _build_entry(
            "PreToolUse", "Bash", query,
            [],
            total_rules=10, index_rebuilt=False,
            latency_ms=5.0, model="test",
            redact=False, max_query_length=500,
        )
        assert entry["query_truncated"] is False

    def test_redact_secrets_in_query(self) -> None:
        secret = "sk_live_abcdefghijklmnopqrstuvwxyz"
        query = f"Bash: git commit -m '{secret}'"
        entry = _build_entry(
            "PreToolUse", "Bash", query,
            [],
            total_rules=10, index_rebuilt=False,
            latency_ms=5.0, model="test",
            redact=True, max_query_length=500,
        )
        assert "[REDACTED]" in str(entry["query"])
        assert "sk_live_" not in str(entry["query"])

    def test_redact_secrets_in_results(self) -> None:
        prov = Provenance(
            file="/tmp/rules.txt", line_start=1, line_end=1,
        )
        secret = "sk_live_abcdefghijklmnopqrstuvwxyz"
        rule = Rule(text=f"Use key {secret}", provenance=prov)
        results = [RankedResult(rule=rule, score=0.9)]
        entry = _build_entry(
            "PreToolUse", "Bash", "test",
            results,
            total_rules=10, index_rebuilt=False,
            latency_ms=5.0, model="test",
            redact=True, max_query_length=500,
        )
        r = entry["results"]
        assert isinstance(r, list)
        assert "[REDACTED]" in str(r[0]["text"])  # type: ignore[index]

    def test_no_redact(self) -> None:
        secret = "sk_live_abcdefghijklmnopqrstuvwxyz"
        query = f"Bash: {secret}"
        entry = _build_entry(
            "PreToolUse", "Bash", query,
            [],
            total_rules=10, index_rebuilt=False,
            latency_ms=5.0, model="test",
            redact=False, max_query_length=500,
        )
        assert "sk_live_" in str(entry["query"])

    def test_result_fields(self) -> None:
        results = _make_results(1)
        entry = _build_entry(
            "PreToolUse", "Bash", "test",
            results,
            total_rules=10, index_rebuilt=False,
            latency_ms=5.0, model="test",
            redact=False, max_query_length=500,
        )
        r = entry["results"]
        assert isinstance(r, list)
        item = r[0]
        assert isinstance(item, dict)
        assert item["text"] == "Rule number 1"
        assert item["score"] == 0.9
        assert item["file"] == "/tmp/rules.txt"
        assert item["line"] == 1

    def test_path_default_is_cold(self) -> None:
        entry = _build_entry(
            "PreToolUse", "Bash", "q", [],
            total_rules=1, index_rebuilt=False,
            latency_ms=1.0, model="test",
            redact=False, max_query_length=500,
        )
        assert entry["path"] == "cold"

    def test_path_daemon(self) -> None:
        entry = _build_entry(
            "PreToolUse", "Bash", "q", [],
            total_rules=1, index_rebuilt=False,
            latency_ms=1.0, model="test",
            redact=False, max_query_length=500,
            path="daemon",
        )
        assert entry["path"] == "daemon"

    def test_error_field_present(self) -> None:
        entry = _build_entry(
            "PreToolUse", "Bash", "q", [],
            total_rules=1, index_rebuilt=False,
            latency_ms=1.0, model="test",
            redact=False, max_query_length=500,
            path="daemon", error="RuntimeError: boom",
        )
        assert entry["error"] == "RuntimeError: boom"

    def test_error_field_absent_by_default(self) -> None:
        entry = _build_entry(
            "PreToolUse", "Bash", "q", [],
            total_rules=1, index_rebuilt=False,
            latency_ms=1.0, model="test",
            redact=False, max_query_length=500,
        )
        assert "error" not in entry


class TestLogRetrieval:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        results = _make_results(1)
        log_retrieval(
            "PreToolUse", "Bash", "Bash: test",
            results,
            total_rules=10, index_rebuilt=False,
            latency_ms=42.0, model="test",
            redact=False, log_dir=tmp_path,
        )
        log_path = tmp_path / "log.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "PreToolUse"
        assert entry["injected_count"] == 1

    def test_appends_multiple(self, tmp_path: Path) -> None:
        for i in range(3):
            log_retrieval(
                "PreToolUse", "Bash", f"query {i}",
                [],
                total_rules=10, index_rebuilt=False,
                latency_ms=5.0, model="test",
                redact=False, log_dir=tmp_path,
            )
        log_path = tmp_path / "log.jsonl"
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_verbose_output(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        results = _make_results(2)
        log_retrieval(
            "PreToolUse", "Bash", "test",
            results,
            total_rules=10, index_rebuilt=False,
            latency_ms=42.5, model="test",
            redact=False, log_dir=tmp_path, verbose=True,
        )
        captured = capsys.readouterr()
        assert "[cuecard] PreToolUse:Bash" in captured.err
        assert "2 rules injected" in captured.err
        assert "42ms" in captured.err

    def test_no_verbose_by_default(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        log_retrieval(
            "PreToolUse", "Bash", "test",
            [],
            total_rules=10, index_rebuilt=False,
            latency_ms=5.0, model="test",
            redact=False, log_dir=tmp_path,
        )
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_creates_log_dir(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "nested" / "dir"
        log_retrieval(
            "PreToolUse", "Bash", "test",
            [],
            total_rules=10, index_rebuilt=False,
            latency_ms=5.0, model="test",
            redact=False, log_dir=log_dir,
        )
        assert (log_dir / "log.jsonl").exists()

    def test_rotation_triggered(self, tmp_path: Path) -> None:
        """Write data then trigger rotation at 0 MB threshold."""
        log_retrieval(
            "PreToolUse", "Bash", "test",
            _make_results(2),
            total_rules=10, index_rebuilt=False,
            latency_ms=5.0, model="test",
            redact=False, log_dir=tmp_path,
            max_log_size_mb=0,
        )
        rotated = tmp_path / "log.jsonl.1"
        assert rotated.exists()


class TestRotateLog:
    def test_no_rotation_under_limit(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        log_path.write_text("small\n")
        rotate_log(log_path, max_size_mb=10)
        assert log_path.exists()
        assert not (tmp_path / "log.jsonl.1").exists()

    def test_rotation_creates_dot_1(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        log_path.write_text("x" * (2 * 1024 * 1024))
        rotate_log(log_path, max_size_mb=1)
        assert not log_path.exists()
        assert (tmp_path / "log.jsonl.1").exists()

    def test_rotation_shifts_1_to_2(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        rotated_1 = tmp_path / "log.jsonl.1"
        log_path.write_text("x" * (2 * 1024 * 1024))
        rotated_1.write_text("old content")
        rotate_log(log_path, max_size_mb=1)
        assert not log_path.exists()
        assert (tmp_path / "log.jsonl.1").exists()
        assert (tmp_path / "log.jsonl.2").exists()
        r2 = tmp_path / "log.jsonl.2"
        assert r2.read_text() == "old content"

    def test_rotation_drops_old_2(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        rotated_1 = tmp_path / "log.jsonl.1"
        rotated_2 = tmp_path / "log.jsonl.2"
        log_path.write_text("x" * (2 * 1024 * 1024))
        rotated_1.write_text("middle")
        rotated_2.write_text("oldest")
        rotate_log(log_path, max_size_mb=1)
        r2 = tmp_path / "log.jsonl.2"
        assert r2.read_text() == "middle"

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nonexistent.jsonl"
        rotate_log(log_path, max_size_mb=1)

    def test_race_file_removed_after_lock(self, tmp_path: Path) -> None:
        """Another process rotated (removed) the file before we got the lock."""
        log_path = tmp_path / "log.jsonl"
        # Create a big file so the pre-lock size check passes
        log_path.write_text("x" * (2 * 1024 * 1024))
        rotated_1 = tmp_path / "log.jsonl.1"

        import fcntl as _fcntl

        orig_flock = _fcntl.flock

        def _flock_side_effect(fd: int, op: int) -> None:
            # On LOCK_EX, remove the log file to simulate race
            if op == _fcntl.LOCK_EX and log_path.exists():
                log_path.unlink()
            orig_flock(fd, op)

        with patch("cuecard.logger.fcntl.flock", side_effect=_flock_side_effect):
            rotate_log(log_path, max_size_mb=1)

        # File was removed by "another process" — no rotation happened
        assert not rotated_1.exists()

    def test_race_file_shrunk_after_lock(self, tmp_path: Path) -> None:
        """Another process already rotated — file is now small."""
        log_path = tmp_path / "log.jsonl"
        log_path.write_text("x" * (2 * 1024 * 1024))

        import fcntl as _fcntl

        orig_flock = _fcntl.flock

        def _flock_side_effect(fd: int, op: int) -> None:
            if op == _fcntl.LOCK_EX:
                # Simulate another process rotating: replace big file with small
                log_path.write_text("tiny\n")
            orig_flock(fd, op)

        with patch("cuecard.logger.fcntl.flock", side_effect=_flock_side_effect):
            rotate_log(log_path, max_size_mb=1)

        # File is still there (small), not rotated
        assert log_path.exists()
        assert not (tmp_path / "log.jsonl.1").exists()


class TestTailLines:
    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_bytes(b"")
        assert _tail_lines(p, 10) == []

    def test_small_file(self, tmp_path: Path) -> None:
        p = tmp_path / "small.jsonl"
        p.write_text("line1\nline2\nline3\n")
        result = _tail_lines(p, 2)
        assert result == ["line2", "line3"]

    def test_small_file_all_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "small.jsonl"
        p.write_text("a\nb\nc\n")
        result = _tail_lines(p, 100)
        assert result == ["a", "b", "c"]

    def test_large_file_tail(self, tmp_path: Path) -> None:
        """File larger than _TAIL_CHUNK exercises seek-backward path."""
        p = tmp_path / "large.jsonl"
        line_count = (_TAIL_CHUNK // 50) + 100
        lines = [json.dumps({"i": i}) for i in range(line_count)]
        p.write_text("\n".join(lines) + "\n")
        result = _tail_lines(p, 5)
        assert len(result) == 5

    def test_large_file_preserves_order(self, tmp_path: Path) -> None:
        p = tmp_path / "large.jsonl"
        line_count = (_TAIL_CHUNK // 20) + 50
        lines = [f"line-{i}" for i in range(line_count)]
        p.write_text("\n".join(lines) + "\n")
        result = _tail_lines(p, 3)
        assert len(result) == 3
        assert result[-1] == f"line-{line_count - 1}"
        assert result[-2] == f"line-{line_count - 2}"
        assert result[-3] == f"line-{line_count - 3}"

    def test_multi_chunk_spanning(self, tmp_path: Path) -> None:
        """File spanning >2 chunks to exercise the while loop fully."""
        p = tmp_path / "huge.jsonl"
        line_count = (_TAIL_CHUNK * 3) // 20
        lines = [f"row-{i}" for i in range(line_count)]
        p.write_text("\n".join(lines) + "\n")
        result = _tail_lines(p, 10)
        assert len(result) == 10


class TestReadLog:
    def test_empty_dir(self, tmp_path: Path) -> None:
        entries = read_log(log_dir=tmp_path)
        assert entries == []

    def test_reads_entries(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        data = [
            {"event": "PreToolUse", "tool_name": "Bash"},
            {"event": "PreToolUse", "tool_name": "Edit"},
        ]
        lines = [json.dumps(d) for d in data]
        log_path.write_text("\n".join(lines) + "\n")
        entries = read_log(log_dir=tmp_path)
        assert len(entries) == 2
        assert entries[0]["tool_name"] == "Bash"

    def test_limit(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        lines = [json.dumps({"i": i}) for i in range(50)]
        log_path.write_text("\n".join(lines) + "\n")
        entries = read_log(log_dir=tmp_path, limit=5)
        assert len(entries) == 5
        assert entries[0]["i"] == 45

    def test_skips_malformed(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        log_path.write_text('{"good": 1}\nnot json\n{"good": 2}\n')
        entries = read_log(log_dir=tmp_path)
        assert len(entries) == 2

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        log_path.write_text('{"a": 1}\n\n{"b": 2}\n')
        entries = read_log(log_dir=tmp_path)
        assert len(entries) == 2

    def test_default_log_dir(self) -> None:
        d = _default_log_dir()
        assert d.name == ".cuecard"

    def test_large_file_uses_tail(self, tmp_path: Path) -> None:
        """Verify read_log works on files larger than _TAIL_CHUNK."""
        log_path = tmp_path / "log.jsonl"
        line_count = (_TAIL_CHUNK // 20) + 100
        lines = [json.dumps({"i": i}) for i in range(line_count)]
        log_path.write_text("\n".join(lines) + "\n")
        entries = read_log(log_dir=tmp_path, limit=10)
        assert len(entries) == 10
        # Should be the last 10 entries
        assert entries[-1]["i"] == line_count - 1


class TestComputeStats:
    def test_empty_entries(self) -> None:
        s = compute_stats([])
        assert s["total_events"] == 0
        assert s["latency_p50"] == 0.0
        assert s["top_rules"] == []
        assert s["coverage"] == 0.0

    def test_basic_stats(self) -> None:
        entries = [
            {
                "latency_ms": 10,
                "results": [
                    {"text": "Rule A", "score": 0.9},
                ],
            },
            {
                "latency_ms": 20,
                "results": [
                    {"text": "Rule A", "score": 0.8},
                    {"text": "Rule B", "score": 0.7},
                ],
            },
            {
                "latency_ms": 30,
                "results": [],
            },
        ]
        s = compute_stats(entries)
        assert s["total_events"] == 3
        assert s["avg_injected"] == 1.0
        assert s["coverage"] == round(2 / 3, 4)
        top = s["top_rules"]
        assert isinstance(top, list)
        assert top[0] == ("Rule A", 2)

    def test_latency_percentiles(self) -> None:
        entries = [
            {"latency_ms": float(i), "results": []}
            for i in range(100)
        ]
        s = compute_stats(entries)
        assert s["latency_p50"] == 49.5
        assert s["latency_p95"] == 94.05
        assert s["latency_p99"] == 98.01

    def test_handles_missing_fields(self) -> None:
        entries: list[dict[str, object]] = [
            {"some_field": "value"},
            {"latency_ms": "not_a_number", "results": "bad"},
        ]
        s = compute_stats(entries)
        assert s["total_events"] == 2
        assert s["latency_p50"] == 0.0

    def test_top_rules_limit(self) -> None:
        entries = []
        for i in range(15):
            entries.append({
                "latency_ms": 1.0,
                "results": [{"text": f"Rule {i}", "score": 0.5}],
            })
        s = compute_stats(entries)
        top = s["top_rules"]
        assert isinstance(top, list)
        assert len(top) <= 10

    def test_results_with_empty_text(self) -> None:
        entries: list[dict[str, object]] = [
            {"latency_ms": 1.0, "results": [{"text": ""}]},
        ]
        s = compute_stats(entries)
        assert s["top_rules"] == []

    def test_results_with_non_dict_items(self) -> None:
        entries: list[dict[str, object]] = [
            {"latency_ms": 1.0, "results": ["not_a_dict"]},
        ]
        s = compute_stats(entries)
        assert s["total_events"] == 1


class TestDefaultLogDir:
    def test_uses_home(self) -> None:
        with patch("cuecard.logger.Path") as mock_path:
            mock_path.home.return_value = Path("/mock/home")
            d = _default_log_dir()
            assert str(d) == "/mock/home/.cuecard"
