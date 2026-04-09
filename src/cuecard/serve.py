"""Lightweight HTTP daemon that keeps the embedding model warm.

Loads config, embedding model, and index on startup.  Serves POST
requests with the same JSON payload the hook adapter receives and
returns the same JSON response, cutting per-call latency from ~1.7s
to ~50ms by avoiding repeated model loads.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

from cuecard.models import MAX_REQUEST_BYTES

if TYPE_CHECKING:
    from cuecard.models import AffinityIndex, Index, ResolvedConfig

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8452
_PID_FILE = "serve.pid"
_LOG_FILE = "serve.log"


class _ReuseHTTPServer(HTTPServer):
    """HTTPServer with SO_REUSEADDR set before bind."""

    allow_reuse_address = True


# -- PID file management --


def _pid_path(home: Path | None = None) -> Path:
    """Return the path to the PID file."""
    base = home if home is not None else Path.home()
    return base / ".cuecard" / _PID_FILE


def _log_path(home: Path | None = None) -> Path:
    """Return the path to the server log file."""
    base = home if home is not None else Path.home()
    return base / ".cuecard" / _LOG_FILE


def write_pid(pid: int, home: Path | None = None) -> Path:
    """Write the server PID to the PID file.  Returns the path written."""
    path = _pid_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, f"{pid}\n".encode())
    finally:
        os.close(fd)
    return path


def read_pid(home: Path | None = None) -> int | None:
    """Read the PID from the PID file.  Returns None if missing or invalid."""
    path = _pid_path(home)
    if not path.exists():
        return None
    try:
        text = path.read_text().strip()
        return int(text)
    except (ValueError, OSError):
        return None


def remove_pid(home: Path | None = None) -> None:
    """Remove the PID file if it exists."""
    path = _pid_path(home)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is running."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it
        return True
    return True


def daemon_status(home: Path | None = None) -> tuple[bool, int | None]:
    """Return (is_running, pid) for the daemon.

    Cleans up stale PID files when the process is no longer alive.
    """
    pid = read_pid(home)
    if pid is None:
        return False, None
    if is_pid_alive(pid):
        return True, pid
    # Stale PID file
    remove_pid(home)
    return False, None


# -- Request handler --


class _Handler(BaseHTTPRequestHandler):
    """Handle POST /retrieve requests."""

    # Injected via functools.partial in _make_handler
    _index: Index
    _config: ResolvedConfig
    _embedding_model: object  # fastembed.TextEmbedding

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Route HTTP logs to the Python logger instead of stderr."""
        logger.info(format, *args)

    def do_POST(self) -> None:  # noqa: N802
        """Process a hook JSON payload and return results."""
        if self.path != "/retrieve":
            self._send_error(404, "Not found")
            return

        content_length_raw = self.headers.get("Content-Length", "0")
        try:
            content_length = int(content_length_raw)
        except ValueError:
            self._send_error(400, "Invalid Content-Length")
            return

        if content_length < 0:
            self._send_error(400, "Invalid Content-Length")
            return

        if content_length > MAX_REQUEST_BYTES:
            self._send_error(413, "Request too large")
            return

        raw = self.rfile.read(content_length)
        try:
            data: dict[str, object] = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error(400, "Invalid JSON")
            return

        if not isinstance(data, dict):
            self._send_error(400, "Expected JSON object")
            return

        result = _process_request(
            data, self._index, self._config, self._embedding_model,
            affinity=getattr(self, "_affinity", None),
        )
        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        """Health check endpoint."""
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_error(404, "Not found")

    def _send_error(self, code: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _make_handler(
    index: Index,
    config: ResolvedConfig,
    embedding_model: object,
    affinity: AffinityIndex | None = None,
) -> type[_Handler]:
    """Create a handler class with the index/config/model/affinity bound."""
    return type(
        "_BoundHandler",
        (_Handler,),
        {
            "_index": index,
            "_config": config,
            "_embedding_model": embedding_model,
            "_affinity": affinity,
        },
    )


# -- Request processing (same logic as adapter) --


def _process_request(
    data: dict[str, object],
    index: Index,
    config: ResolvedConfig,
    embedding_model: object,
    affinity: AffinityIndex | None = None,
) -> dict[str, object]:
    """Process a hook payload and return the modified data dict."""
    from cuecard.adapters.claude_code import (
        _EVENT_HANDLERS,
        _EVENT_LABELS,
        _LABEL_PREVENT,
        _detect_event,
        _handle_pre_tool_use,
    )
    from cuecard.retrieval.formatter import format_rules
    from cuecard.retrieval.pipeline import run_pipeline

    event = _detect_event(data)
    handler = _EVENT_HANDLERS.get(event, _handle_pre_tool_use)
    query, tool_name, event = handler(data)

    pipeline_mode = getattr(
        getattr(config, "pipeline", None), "mode", "embedding",
    )

    start = time.monotonic()
    pipeline_result = run_pipeline(
        query, index, config,
        embedding_model=embedding_model,  # type: ignore[arg-type]
        mode=pipeline_mode,
        event=event,
        tool_name=tool_name,
        affinity=affinity,
    )
    results = pipeline_result.results
    latency_ms = (time.monotonic() - start) * 1000

    # Build a fresh output dict (don't mutate the input)
    output: dict[str, object] = dict(data)

    raw_hook_output = data.get("hookSpecificOutput")
    hook_output: dict[str, object] = (
        dict(raw_hook_output) if isinstance(raw_hook_output, dict) else {}
    )
    hook_output["hookEventName"] = event
    if event == "PreToolUse":
        hook_output["permissionDecision"] = "allow"

    if results:
        label = _EVENT_LABELS.get(event, _LABEL_PREVENT)
        context = format_rules(results, label=label)
        hook_output["additionalContext"] = context

    output["hookSpecificOutput"] = hook_output

    logger.info(
        "Served query in %.1fms: %d results for %s",
        latency_ms,
        len(results),
        event,
    )

    return output


# -- Server lifecycle --


def start_server(
    port: int = DEFAULT_PORT,
    home: Path | None = None,
    *,
    config: ResolvedConfig | None = None,
    index: Index | None = None,
    embedding_model: object | None = None,
) -> HTTPServer:
    """Start the cuecard daemon server.

    If config/index/embedding_model are not provided, they are loaded
    from the standard locations (used in production).  Pass them
    explicitly for testing.

    Returns the HTTPServer instance (call serve_forever() to block).
    """
    effective_home = home if home is not None else Path.home()

    if config is None:
        from cuecard.config import load_config
        config = load_config(project_dir=Path.cwd(), home_dir=effective_home)

    if embedding_model is None:
        from fastembed import TextEmbedding
        embedding_model = TextEmbedding(model_name=config.model_name)

    affinity: AffinityIndex | None = None
    if index is None:
        from cuecard.indexing.loader import load_or_build
        loaded = load_or_build(config, embedding_model)  # type: ignore[arg-type]
        index = loaded.index if loaded is not None else None
        affinity = loaded.affinity if loaded is not None else None

    if index is None or index.size == 0:
        msg = "No rules indexed. Run 'cuecard setup' first."
        raise RuntimeError(msg)

    handler_cls = _make_handler(index, config, embedding_model, affinity=affinity)
    server = _ReuseHTTPServer(("127.0.0.1", port), handler_cls)

    # Write PID file
    write_pid(os.getpid(), effective_home)

    logger.info(
        "cuecard serve listening on 127.0.0.1:%d (%d rules, model=%s)",
        port,
        index.size,
        config.model_name,
    )

    return server


def stop_server(home: Path | None = None) -> bool:
    """Stop a running daemon by sending SIGTERM.  Returns True if stopped."""
    running, pid = daemon_status(home)
    if not running or pid is None:
        return False
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    # Wait for process to actually exit (up to 5s)
    for _ in range(50):
        try:
            os.kill(pid, 0)  # Check if alive
        except (ProcessLookupError, PermissionError):
            break
        time.sleep(0.1)
    else:
        # Still alive — force kill
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)
    remove_pid(home)
    return True


def run_server(
    port: int = DEFAULT_PORT,
    home: Path | None = None,
) -> None:
    """Start and run the server until interrupted.

    This is the main entry point for `cuecard serve`.
    """
    effective_home = home if home is not None else Path.home()

    # Check for already-running daemon
    running, existing_pid = daemon_status(effective_home)
    if running:
        print(
            f"Daemon already running (PID {existing_pid})."
            " Use 'cuecard serve --stop' to stop it.",
            file=sys.stderr,
        )
        sys.exit(1)

    server = start_server(port=port, home=effective_home)
    shutdown_requested = False

    def _shutdown(signum: int, frame: object) -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        logger.info("Shutting down (signal %d)...", signum)
        server.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        server.serve_forever()
    finally:
        server.server_close()
        remove_pid(effective_home)
        logger.info("Server stopped.")


# -- Client (used by adapter for fast path) --


def query_daemon(
    payload: dict[str, object],
    port: int = DEFAULT_PORT,
    timeout: float | None = None,
) -> dict[str, object] | None:
    """Send a hook payload to the running daemon.

    Returns the response dict on success, or None if the daemon is
    unreachable or times out.  Timeout defaults to 5s for LLM modes
    (llm-local/llm-haiku) and 0.5s for embedding mode.
    """
    if timeout is None:
        from cuecard.config import load_config

        cfg = load_config(project_dir=Path.cwd())
        mode = cfg.pipeline.mode
        timeout = 5.0 if mode in {"llm-local", "llm-haiku"} else 0.5

    body = json.dumps(payload).encode()
    conn: http.client.HTTPConnection | None = None
    try:
        conn = http.client.HTTPConnection(
            "127.0.0.1", port, timeout=timeout,
        )
        conn.request(
            "POST", "/retrieve",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        raw = resp.read()
        result: dict[str, object] = json.loads(raw)
        return result
    except (
        OSError,
        TimeoutError,
        http.client.HTTPException,
        json.JSONDecodeError,
    ):
        return None
    finally:
        if conn is not None:
            with contextlib.suppress(OSError):
                conn.close()
