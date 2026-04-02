"""Structured logging with secrets scrubbing and rotation."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cuecard.security import ensure_directory, scrub_secrets

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cuecard.models import RankedResult

logger = logging.getLogger(__name__)

_LOG_FILENAME = "log.jsonl"
_MAX_ROTATED = 2


def _default_log_dir() -> Path:
    return Path.home() / ".cuecard"


def _build_entry(
    event: str,
    tool_name: str,
    query: str,
    results: Sequence[RankedResult],
    *,
    total_rules: int,
    index_rebuilt: bool,
    latency_ms: float,
    model: str,
    redact: bool,
    max_query_length: int,
) -> dict[str, object]:
    """Build a structured log entry dict."""
    query_truncated = len(query) > max_query_length
    if query_truncated:
        query = query[:max_query_length]

    if redact:
        query = scrub_secrets(query)

    result_dicts: list[dict[str, object]] = []
    for r in results:
        text = r.rule.text
        if redact:
            text = scrub_secrets(text)
        result_dicts.append({
            "text": text,
            "score": round(r.score, 4),
            "file": r.rule.provenance.file,
            "line": r.rule.provenance.line_start,
        })

    return {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "event": event,
        "tool_name": tool_name,
        "query": query,
        "query_truncated": query_truncated,
        "results": result_dicts,
        "injected_count": len(results),
        "total_rules": total_rules,
        "index_rebuilt": index_rebuilt,
        "latency_ms": round(latency_ms, 2),
        "model": model,
    }


def log_retrieval(
    event: str,
    tool_name: str,
    query: str,
    results: Sequence[RankedResult],
    *,
    total_rules: int,
    index_rebuilt: bool,
    latency_ms: float,
    model: str,
    redact: bool = True,
    max_query_length: int = 500,
    log_dir: Path | None = None,
    max_log_size_mb: int = 10,
    verbose: bool = False,
) -> None:
    """Append a structured log entry to log.jsonl.

    - Truncates query to max_query_length
    - If redact=True, scrubs secrets from query AND results[].text
    - Appends JSON line to log_dir/log.jsonl
    - Checks rotation after writing
    - If verbose, prints to stderr
    """
    log_dir = log_dir if log_dir is not None else _default_log_dir()
    ensure_directory(log_dir)

    entry = _build_entry(
        event,
        tool_name,
        query,
        results,
        total_rules=total_rules,
        index_rebuilt=index_rebuilt,
        latency_ms=latency_ms,
        model=model,
        redact=redact,
        max_query_length=max_query_length,
    )

    log_path = log_dir / _LOG_FILENAME
    line = json.dumps(entry, separators=(",", ":")) + "\n"

    # Append with secure permissions and flock for concurrent safety
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(str(log_path), flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line.encode())
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

    rotate_log(log_path, max_log_size_mb)

    if verbose:
        count = len(results)
        ms = round(latency_ms)
        print(
            f"[cuecard] {event}:{tool_name}"
            f" \u2192 {count} rules injected ({ms}ms)",
            file=sys.stderr,
        )


def rotate_log(log_path: Path, max_size_mb: int) -> None:
    """Rotate log.jsonl if over max_size_mb.

    Rename to log.jsonl.1, start fresh.
    Keep at most 2 rotated files (log.jsonl.1, log.jsonl.2).
    If log.jsonl.1 exists, move it to .2 first (drop .2 if exists).

    Uses fcntl.flock to prevent concurrent writers from losing entries
    during the rename gap.
    """
    if not log_path.exists():
        return

    size_mb = log_path.stat().st_size / (1024 * 1024)
    if size_mb <= max_size_mb:
        return

    lock_path = log_path.parent / ".log.lock"
    lock_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    lock_fd = os.open(str(lock_path), lock_flags, 0o600)
    with os.fdopen(lock_fd, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Re-check after acquiring lock — another process may have
            # already rotated.
            if not log_path.exists():
                return
            size_mb = log_path.stat().st_size / (1024 * 1024)
            if size_mb <= max_size_mb:
                return

            rotated_1 = log_path.parent / f"{log_path.name}.1"
            rotated_2 = log_path.parent / f"{log_path.name}.2"

            # Shift existing rotated files
            if rotated_1.exists():
                os.replace(str(rotated_1), str(rotated_2))

            os.replace(str(log_path), str(rotated_1))
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


_TAIL_CHUNK = 8192


def _tail_lines(path: Path, limit: int) -> list[str]:
    """Read the last *limit* non-empty lines from *path*.

    Uses a seek-backward approach for large files so the entire file
    is never loaded into memory.
    """
    size = path.stat().st_size
    if size == 0:
        return []

    with open(path, "rb") as fh:
        if size <= _TAIL_CHUNK:
            all_lines = fh.read().decode(errors="replace").split("\n")
        else:
            # Seek backwards in chunks, collecting lines
            all_lines_parts: list[str] = []
            remaining = size
            leftover = b""

            while remaining > 0 and len(all_lines_parts) < limit + 1:
                chunk_size = min(_TAIL_CHUNK, remaining)
                remaining -= chunk_size
                fh.seek(remaining)
                chunk = fh.read(chunk_size) + leftover
                parts = chunk.split(b"\n")
                leftover = parts[0]
                all_lines_parts = [
                    p.decode(errors="replace") for p in parts[1:]
                ] + all_lines_parts

            if leftover:
                all_lines_parts = [
                    leftover.decode(errors="replace"),
                ] + all_lines_parts

            all_lines = all_lines_parts

    # Filter out empty strings (blank lines / trailing newline)
    non_empty = [ln for ln in all_lines if ln.strip()]
    return non_empty[-limit:]


def read_log(
    log_dir: Path | None = None,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Read recent log entries from log.jsonl.

    Returns up to *limit* most recent entries.  Uses a tail-like
    approach that seeks from the end of the file, avoiding loading the
    entire file into memory.
    """
    log_dir = log_dir if log_dir is not None else _default_log_dir()
    log_path = log_dir / _LOG_FILENAME

    if not log_path.exists():
        return []

    recent = _tail_lines(log_path, limit)

    entries: list[dict[str, object]] = []
    for line in recent:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed log line")
    return entries


def compute_stats(entries: list[dict[str, object]]) -> dict[str, object]:
    """Compute aggregate stats from log entries.

    Returns dict with:
    - total_events: int
    - latency_p50, latency_p95, latency_p99: float (ms)
    - top_rules: list of (text, count) tuples, sorted desc
    - avg_injected: float
    - coverage: float (fraction of events with >=1 result)
    """
    if not entries:
        return {
            "total_events": 0,
            "latency_p50": 0.0,
            "latency_p95": 0.0,
            "latency_p99": 0.0,
            "top_rules": [],
            "avg_injected": 0.0,
            "coverage": 0.0,
        }

    latencies: list[float] = []
    rule_counts: dict[str, int] = {}
    total_injected = 0
    events_with_results = 0

    for entry in entries:
        lat = entry.get("latency_ms")
        if isinstance(lat, (int, float)):
            latencies.append(float(lat))

        results = entry.get("results")
        if isinstance(results, list):
            total_injected += len(results)
            if len(results) > 0:
                events_with_results += 1
            for r in results:
                if isinstance(r, dict):
                    text = r.get("text", "")
                    if isinstance(text, str) and text:
                        rule_counts[text] = rule_counts.get(text, 0) + 1

    latencies.sort()
    n = len(entries)

    def _percentile(sorted_vals: list[float], p: float) -> float:
        if not sorted_vals:
            return 0.0
        rank = p / 100.0 * (len(sorted_vals) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = rank - lo
        return round(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]), 2)

    top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_events": n,
        "latency_p50": _percentile(latencies, 50),
        "latency_p95": _percentile(latencies, 95),
        "latency_p99": _percentile(latencies, 99),
        "top_rules": top_rules,
        "avg_injected": round(total_injected / n, 2) if n > 0 else 0.0,
        "coverage": round(events_with_results / n, 4) if n > 0 else 0.0,
    }
