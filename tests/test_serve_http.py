"""Tests for HTTP integration (real server on random port)."""

from __future__ import annotations

import http.client
import json
import threading
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np

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
    query_daemon,
    remove_pid,
    start_server,
)

if TYPE_CHECKING:
    from http.server import HTTPServer
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


def _start_test_server(
    tmp_path: Path,
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


# ---------------------------------------------------------------------------
# HTTP integration (real server on random port)
# ---------------------------------------------------------------------------


class TestHTTPIntegration:
    def test_post_retrieve(self, tmp_path: Path) -> None:
        server, port = _start_test_server(tmp_path)

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
        server, port = _start_test_server(tmp_path)

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
        server, port = _start_test_server(tmp_path)

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
        server, port = _start_test_server(tmp_path)

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
        server, port = _start_test_server(tmp_path)

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
        server, port = _start_test_server(tmp_path)

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
        server, port = _start_test_server(tmp_path)

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
        server, port = _start_test_server(tmp_path)

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
        server, port = _start_test_server(tmp_path)

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
