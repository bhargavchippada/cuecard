"""Tests for cuecard.adapters.claude_code — PreToolUse hook."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np

from cuecard.adapters.claude_code import _format_tool_input, main
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

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _make_config(
    tmp_path: Path,
    *,
    pipeline_mode: str = "embedding",
) -> ResolvedConfig:
    return ResolvedConfig(
        source_paths=(),
        model_name="BAAI/bge-small-en-v1.5",
        top_k=5,
        threshold=0.30,
        dedup_threshold=0.95,
        query_max_length=500,
        hook_events=("PreToolUse",),
        verbose=False,
        redact=True,
        max_log_size_mb=10,
        global_cache_dir=str(tmp_path / "index"),
        project_cache_dir=None,
        allowed_dirs=(),
        pipeline=PipelineConfig(mode=pipeline_mode),
    )


def _make_index() -> Index:
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((2, 384)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    return Index(
        embeddings=emb,
        rules=(
            Rule(
                text="Never commit secrets",
                provenance=Provenance(
                    file="/tmp/rules.txt",
                    line_start=1, line_end=1,
                ),
            ),
            Rule(
                text="Validate all input",
                provenance=Provenance(
                    file="/tmp/rules.txt",
                    line_start=2, line_end=2,
                ),
            ),
        ),
        model_name="BAAI/bge-small-en-v1.5",
        dim=384,
        sources={
            "/tmp/rules.txt": SourceMeta(
                mtime=1.0,
                content_hash="sha256:abc",
                rule_count=2,
            ),
        },
    )


def _make_results() -> list[RankedResult]:
    prov = Provenance(
        file="/tmp/rules.txt", line_start=1, line_end=1,
    )
    rule = Rule(text="Never commit secrets", provenance=prov)
    return [RankedResult(rule=rule, score=0.87)]


_MOD = "cuecard.adapters.claude_code"


class TestAdapterMain:
    def test_with_results(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        hook_input = {
            "tool_name": "Bash",
            "tool_input": "git commit -m 'fix'",
        }
        config = _make_config(tmp_path)
        index = _make_index()
        results = _make_results()

        with (
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch(f"{_MOD}.load_index", return_value=index),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.retrieve", return_value=results),
            patch(f"{_MOD}.log_retrieval") as mock_log,
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        hook_out = output["hookSpecificOutput"]
        assert "additionalContext" in hook_out
        assert "Never commit secrets" in hook_out["additionalContext"]
        mock_log.assert_called_once()

    def test_with_no_results(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        hook_input = {
            "tool_name": "Read",
            "tool_input": "/path/to/file",
        }
        config = _make_config(tmp_path)
        index = _make_index()

        with (
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch(f"{_MOD}.load_index", return_value=index),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.retrieve", return_value=[]),
            patch(f"{_MOD}.log_retrieval"),
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        hook_out = output.get("hookSpecificOutput", {})
        assert "additionalContext" not in hook_out

    def test_with_no_index(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        hook_input = {"tool_name": "Bash", "tool_input": "ls"}
        config = _make_config(tmp_path)

        with (
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch(f"{_MOD}.load_index", return_value=None),
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["tool_name"] == "Bash"

    def test_with_empty_index(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        hook_input = {"tool_name": "Bash", "tool_input": "ls"}
        config = _make_config(tmp_path)
        empty_index = Index(
            embeddings=np.zeros((0, 384), dtype=np.float32),
            rules=(),
            model_name="BAAI/bge-small-en-v1.5",
            dim=384,
            sources={},
        )

        with (
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch(f"{_MOD}.load_index", return_value=empty_index),
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["tool_name"] == "Bash"

    def test_error_passes_through(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        hook_input = {"tool_name": "Bash", "tool_input": "ls"}

        with (
            patch("sys.stdin") as mock_stdin,
            patch(
                f"{_MOD}.load_config",
                side_effect=RuntimeError("config broken"),
            ),
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["tool_name"] == "Bash"
        assert "[cuecard] Error:" in captured.err

    def test_preserves_existing_hook_output(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        hook_input = {
            "tool_name": "Bash",
            "tool_input": "git status",
            "hookSpecificOutput": {"existingKey": "val"},
        }
        config = _make_config(tmp_path)
        index = _make_index()
        results = _make_results()

        with (
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch(f"{_MOD}.load_index", return_value=index),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.retrieve", return_value=results),
            patch(f"{_MOD}.log_retrieval"),
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        hook_out = output["hookSpecificOutput"]
        assert hook_out["existingKey"] == "val"
        assert "additionalContext" in hook_out

    def test_with_pipeline_mode(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        hook_input = {
            "tool_name": "Bash",
            "tool_input": "git commit -m 'fix'",
        }
        config = _make_config(tmp_path, pipeline_mode="rerank")
        index = _make_index()
        results = _make_results()

        fake_pipeline = PipelineResult(
            results=results,
            stages=(
                StageTrace(
                    stage="embedding", input_count=2,
                    output_count=1, latency_ms=1.0,
                ),
            ),
            mode="rerank",
        )

        with (
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch(f"{_MOD}.load_index", return_value=index),
            patch("fastembed.TextEmbedding"),
            patch(
                "cuecard.pipeline.run_pipeline",
                return_value=fake_pipeline,
            ) as mock_pipe,
            patch(f"{_MOD}.log_retrieval"),
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        hook_out = output["hookSpecificOutput"]
        assert "Never commit secrets" in hook_out["additionalContext"]
        mock_pipe.assert_called_once()

    def test_tool_input_truncation(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        long_input = "x" * 1000
        hook_input = {
            "tool_name": "Bash",
            "tool_input": long_input,
        }
        config = _make_config(tmp_path)
        index = _make_index()

        with (
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch(f"{_MOD}.load_index", return_value=index),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.retrieve", return_value=[]),
            patch(f"{_MOD}.log_retrieval"),
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["tool_name"] == "Bash"


class TestFormatToolInput:
    def test_dict_uses_json_dumps(self) -> None:
        result = _format_tool_input({"key": "value"})
        assert result == '{"key": "value"}'

    def test_dict_truncated_at_500(self) -> None:
        big = {"k": "x" * 600}
        result = _format_tool_input(big)
        assert len(result) == 500

    def test_string_passthrough(self) -> None:
        assert _format_tool_input("hello") == "hello"

    def test_int_passthrough(self) -> None:
        assert _format_tool_input(42) == "42"

    def test_empty_string(self) -> None:
        assert _format_tool_input("") == ""


class TestMalformedStdin:
    def test_empty_stdin_outputs_empty_json(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = ""
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output == {}
        assert "[cuecard] Error:" in captured.err

    def test_malformed_json_outputs_empty_json(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = "not valid json{{"
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output == {}
        assert "[cuecard] Error:" in captured.err


class TestMainGuard:
    def test_main_module_entry(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Cover the if __name__ == '__main__' block."""
        import runpy

        hook_input = {"tool_name": "Bash", "tool_input": "ls"}

        with (
            patch("sys.stdin") as mock_stdin,
            patch(
                f"{_MOD}.load_config",
                side_effect=RuntimeError("skip"),
            ),
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            runpy.run_module(
                "cuecard.adapters.claude_code",
                run_name="__main__",
            )

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["tool_name"] == "Bash"
