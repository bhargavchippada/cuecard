"""Tests for request processing and handler factory."""

from __future__ import annotations

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
    _Handler,
    _make_handler,
    _process_request,
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

        with patch(
                "cuecard.retrieval.pipeline.run_pipeline",
                return_value=fake_pipeline,
            ):
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

        with patch(
                "cuecard.retrieval.pipeline.run_pipeline",
                return_value=fake_pipeline,
            ):
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
        with patch(
                "cuecard.retrieval.pipeline.run_pipeline",
                return_value=fake_pipeline,
            ):
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
        with patch(
                "cuecard.retrieval.pipeline.run_pipeline",
                return_value=fake_pipeline,
            ):
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
