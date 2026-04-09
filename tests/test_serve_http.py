"""Tests for HTTP integration (real server on random port)."""

from __future__ import annotations

import http.client
import json
import threading
from typing import TYPE_CHECKING
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
    query_daemon,
    remove_pid,
    start_server,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
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
    @pytest.fixture(autouse=True, scope="class")
    def _server(
        self, tmp_path_factory: pytest.TempPathFactory,
    ) -> Iterator[tuple[HTTPServer, int]]:
        """Shared server for all HTTP integration tests."""
        tmp_path = tmp_path_factory.mktemp("serve_http")
        server, port = _start_test_server(tmp_path)
        self.__class__._port = port  # type: ignore[attr-defined]
        self.__class__._tmp_path = tmp_path  # type: ignore[attr-defined]
        yield server, port
        server.shutdown()
        server.server_close()
        remove_pid(tmp_path)

    def _conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self._port, timeout=5.0)

    def test_post_retrieve(self) -> None:
        fake_pipeline = PipelineResult(
            results=(RankedResult(rule=_make_rule(), score=0.85),),
            stages=(StageTrace(
                stage="retrieval", input_count=1,
                output_count=1, latency_ms=1.0,
            ),),
            mode="embedding",
        )

        with patch(
            "cuecard.retrieval.pipeline.run_pipeline",
            return_value=fake_pipeline,
        ):
            result = query_daemon(
                {"tool_name": "Bash", "tool_input": "ls"},
                port=self._port,
                timeout=5.0,
            )

        assert result is not None
        assert "hookSpecificOutput" in result

    def test_get_health(self) -> None:
        conn = self._conn()
        conn.request("GET", "/health")
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["status"] == "ok"
        conn.close()

    def test_get_not_found(self) -> None:
        conn = self._conn()
        conn.request("GET", "/nonexistent")
        resp = conn.getresponse()
        assert resp.status == 404
        conn.close()

    def test_post_invalid_json(self) -> None:
        conn = self._conn()
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

    def test_post_not_object(self) -> None:
        body = b"[1,2,3]"
        conn = self._conn()
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

    def test_post_too_large(self) -> None:
        conn = self._conn()
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

    def test_post_invalid_content_length(self) -> None:
        conn = self._conn()
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

    def test_post_wrong_path(self) -> None:
        conn = self._conn()
        conn.request("POST", "/admin", body=b"{}", headers={
            "Content-Type": "application/json",
            "Content-Length": "2",
        })
        resp = conn.getresponse()
        assert resp.status == 404
        conn.close()

    def test_post_negative_content_length(self) -> None:
        conn = self._conn()
        conn.request("POST", "/retrieve", body=b"", headers={
            "Content-Type": "application/json",
            "Content-Length": "-1",
        })
        resp = conn.getresponse()
        assert resp.status == 400
        data = json.loads(resp.read())
        assert "Invalid Content-Length" in data["error"]
        conn.close()
