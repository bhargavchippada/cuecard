"""Tests for cuecard.adapters.claude_code — all hook events."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np

from cuecard.adapters.claude_code import (
    _EVENT_HANDLERS,
    _EVENT_LABELS,
    _LABEL_AUDIT,
    _LABEL_PREVENT,
    _LABEL_PROPAGATE,
    _LABEL_VERIFY,
    _detect_event,
    _format_tool_input,
    _handle_post_tool_use,
    _handle_pre_tool_use,
    _handle_stop,
    _handle_subagent_start,
    _handle_user_prompt_submit,
    main,
)
from cuecard.models import (
    Index,
    LoadedIndex,
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
        global_source_paths=(),
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


def _make_pipeline_result(
    results: list[RankedResult] | None = None,
    mode: str = "embedding",
) -> PipelineResult:
    if results is None:
        results = _make_results()
    return PipelineResult(
        results=tuple(results),
        stages=(
            StageTrace(
                stage="retrieval", input_count=2,
                output_count=len(results), latency_ms=1.0,
            ),
        ),
        mode=mode,
    )


_MOD = "cuecard.adapters.claude_code"
_PIPELINE = "cuecard.retrieval.pipeline.run_pipeline"


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
            patch(f"{_MOD}._try_daemon", return_value=None),
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.load_or_build", return_value=LoadedIndex(index=index)),
            patch(_PIPELINE, return_value=_make_pipeline_result(results)),
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
            patch(f"{_MOD}._try_daemon", return_value=None),
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.load_or_build", return_value=LoadedIndex(index=index)),
            patch(_PIPELINE, return_value=_make_pipeline_result([])),
            patch(f"{_MOD}.log_retrieval"),
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        hook_out = output.get("hookSpecificOutput", {})
        assert hook_out.get("permissionDecision") == "allow"
        assert "additionalContext" not in hook_out

    def test_with_no_index(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        hook_input = {"tool_name": "Bash", "tool_input": "ls"}
        config = _make_config(tmp_path)

        with (
            patch(f"{_MOD}._try_daemon", return_value=None),
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.load_or_build", return_value=None),
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
            patch(f"{_MOD}._try_daemon", return_value=None),
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.load_or_build", return_value=LoadedIndex(index=empty_index)),
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
            patch(f"{_MOD}._try_daemon", return_value=None),
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
            patch(f"{_MOD}._try_daemon", return_value=None),
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.load_or_build", return_value=LoadedIndex(index=index)),
            patch(_PIPELINE, return_value=_make_pipeline_result(results)),
            patch(f"{_MOD}.log_retrieval"),
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        hook_out = output["hookSpecificOutput"]
        assert hook_out["existingKey"] == "val"
        assert "additionalContext" in hook_out

    def test_user_prompt_submit_event(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """UserPromptSubmit events use prompt field, not tool_name/tool_input."""
        hook_input = {
            "event": "UserPromptSubmit",
            "prompt": "Add authentication to the API endpoints",
        }
        config = _make_config(tmp_path)
        index = _make_index()
        results = _make_results()

        with (
            patch(f"{_MOD}._try_daemon", return_value=None),
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.load_or_build", return_value=LoadedIndex(index=index)),
            patch(
                _PIPELINE, return_value=_make_pipeline_result(results),
            ) as mock_pipe,
            patch(f"{_MOD}.log_retrieval") as mock_log,
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        # Verify query has UserPromptSubmit prefix
        pipe_args = mock_pipe.call_args
        assert "UserPromptSubmit:" in pipe_args[0][0]

        # Verify log uses empty tool_name for UserPromptSubmit (not a tool)
        log_kwargs = mock_log.call_args
        assert log_kwargs[1].get("tool_name", "") == ""

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        hook_out = output["hookSpecificOutput"]
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

        with (
            patch(f"{_MOD}._try_daemon", return_value=None),
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.load_or_build", return_value=LoadedIndex(index=index)),
            patch(
                _PIPELINE,
                return_value=_make_pipeline_result(results, mode="rerank"),
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
            patch(f"{_MOD}._try_daemon", return_value=None),
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.load_or_build", return_value=LoadedIndex(index=index)),
            patch(_PIPELINE, return_value=_make_pipeline_result([])),
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


class TestHandlePreToolUse:
    def test_basic_query(self) -> None:
        data: dict[str, object] = {
            "tool_name": "Bash",
            "tool_input": "git commit -m 'fix'",
        }
        query, tool_name, event = _handle_pre_tool_use(data)
        assert query == "Bash: git commit -m 'fix'"
        assert tool_name == "Bash"
        assert event == "PreToolUse"

    def test_dict_tool_input(self) -> None:
        data: dict[str, object] = {
            "tool_name": "Edit",
            "tool_input": {"file": "a.py", "content": "x"},
        }
        query, tool_name, event = _handle_pre_tool_use(data)
        assert "Edit:" in query
        assert '"file": "a.py"' in query
        assert tool_name == "Edit"
        assert event == "PreToolUse"

    def test_missing_fields(self) -> None:
        query, tool_name, event = _handle_pre_tool_use({})
        assert query == ": "
        assert tool_name == ""
        assert event == "PreToolUse"


class TestHandlePostToolUse:
    def test_basic_query(self) -> None:
        data: dict[str, object] = {
            "tool_name": "Bash",
            "tool_input": "git status",
            "tool_output": "On branch master\nnothing to commit",
        }
        query, tool_name, event = _handle_post_tool_use(data)
        assert query.startswith("PostToolUse:Bash:")
        assert "git status" in query
        assert "\u2192" in query
        assert "On branch master" in query
        assert tool_name == "Bash"
        assert event == "PostToolUse"

    def test_scrubs_secrets_in_output(self) -> None:
        data: dict[str, object] = {
            "tool_name": "Bash",
            "tool_input": "cat .env",
            "tool_output": "API_KEY=sk-ant-abcdefghijklmnopqrstuvwxyz",
        }
        query, _, _ = _handle_post_tool_use(data)
        assert "sk-ant-" not in query
        assert "[REDACTED]" in query

    def test_large_output_truncated_at_2000_then_500(self) -> None:
        big_output = "x" * 5000
        data: dict[str, object] = {
            "tool_name": "Bash",
            "tool_input": "cat big.log",
            "tool_output": big_output,
        }
        query, _, _ = _handle_post_tool_use(data)
        # The output portion should be at most 500 chars
        # Total query has prefix + tool_input + arrow + output
        arrow_idx = query.index("\u2192")
        output_part = query[arrow_idx + 2:]  # skip "→ "
        assert len(output_part) <= 500

    def test_non_string_output(self) -> None:
        data: dict[str, object] = {
            "tool_name": "Read",
            "tool_input": "/tmp/f.txt",
            "tool_output": 12345,
        }
        query, _, _ = _handle_post_tool_use(data)
        assert "12345" in query

    def test_tool_input_truncated_at_200(self) -> None:
        long_input = "a" * 400
        data: dict[str, object] = {
            "tool_name": "Bash",
            "tool_input": long_input,
            "tool_output": "ok",
        }
        query, _, _ = _handle_post_tool_use(data)
        # tool_input should be truncated to 200
        prefix = "PostToolUse:Bash: "
        arrow_idx = query.index(" \u2192 ")
        input_part = query[len(prefix):arrow_idx]
        assert len(input_part) <= 200

    def test_missing_output(self) -> None:
        data: dict[str, object] = {"tool_name": "Bash", "tool_input": "ls"}
        query, _, _ = _handle_post_tool_use(data)
        assert "PostToolUse:Bash:" in query

    def test_newlines_stripped_from_output(self) -> None:
        data: dict[str, object] = {
            "tool_name": "Bash",
            "tool_input": "ls",
            "tool_output": "line1\nline2\rline3",
        }
        query, _, _ = _handle_post_tool_use(data)
        assert "\n" not in query
        assert "\r" not in query


class TestHandleUserPromptSubmit:
    def test_basic_query(self) -> None:
        data: dict[str, object] = {
            "prompt": "Add authentication to the API",
        }
        query, tool_name, event = _handle_user_prompt_submit(data)
        assert query == "UserPromptSubmit: Add authentication to the API"
        assert tool_name == ""
        assert event == "UserPromptSubmit"

    def test_missing_prompt(self) -> None:
        query, tool_name, event = _handle_user_prompt_submit({})
        assert query == "UserPromptSubmit: "
        assert tool_name == ""

    def test_prompt_truncated(self) -> None:
        long_prompt = "z" * 1000
        data: dict[str, object] = {"prompt": long_prompt}
        query, _, _ = _handle_user_prompt_submit(data)
        # "UserPromptSubmit: " + 500 chars max
        assert len(query) <= len("UserPromptSubmit: ") + 500


class TestHandleSubagentStart:
    def test_basic_query(self) -> None:
        data: dict[str, object] = {
            "agent_type": "security-reviewer",
            "tool_input": {"prompt": "Review auth module for vulnerabilities"},
        }
        query, tool_name, event = _handle_subagent_start(data)
        assert query.startswith("SubagentStart:security-reviewer:")
        assert "Review auth module" in query
        assert tool_name == ""
        assert event == "SubagentStart"

    def test_agent_type_sanitized(self) -> None:
        data: dict[str, object] = {
            "agent_type": "bad<script>type&foo",
            "tool_input": {"prompt": "test"},
        }
        query, _, _ = _handle_subagent_start(data)
        assert "SubagentStart:badscripttypefoo:" in query
        assert "<" not in query
        assert "&" not in query

    def test_agent_type_truncated(self) -> None:
        data: dict[str, object] = {
            "agent_type": "a" * 200,
            "tool_input": {"prompt": "test"},
        }
        query, _, _ = _handle_subagent_start(data)
        # agent_type capped at 100
        prefix = "SubagentStart:"
        colon_idx = query.index(":", len(prefix))
        agent_part = query[len(prefix):colon_idx]
        assert len(agent_part) <= 100

    def test_missing_agent_type(self) -> None:
        data: dict[str, object] = {
            "tool_input": {"prompt": "do something"},
        }
        query, _, _ = _handle_subagent_start(data)
        assert query.startswith("SubagentStart::")

    def test_prompt_scrubbed(self) -> None:
        data: dict[str, object] = {
            "agent_type": "reviewer",
            "tool_input": {
                "prompt": "Check sk-ant-abcdefghijklmnopqrstuvwxyz in code",
            },
        }
        query, _, _ = _handle_subagent_start(data)
        assert "sk-ant-" not in query
        assert "[REDACTED]" in query

    def test_non_dict_tool_input(self) -> None:
        """When tool_input is not a dict, prompt should be empty."""
        data: dict[str, object] = {
            "agent_type": "reviewer",
            "tool_input": "not a dict",
        }
        query, _, _ = _handle_subagent_start(data)
        assert query == "SubagentStart:reviewer: "

    def test_large_prompt_truncated(self) -> None:
        data: dict[str, object] = {
            "agent_type": "reviewer",
            "tool_input": {"prompt": "x" * 5000},
        }
        query, _, _ = _handle_subagent_start(data)
        # After "SubagentStart:reviewer: ", prompt part <= 500
        prefix = "SubagentStart:reviewer: "
        prompt_part = query[len(prefix):]
        assert len(prompt_part) <= 500


class TestHandleStop:
    def test_basic_query(self) -> None:
        data: dict[str, object] = {"stop_reason": "user_interrupt"}
        query, tool_name, event = _handle_stop(data)
        assert query == "Stop: user_interrupt"
        assert tool_name == ""
        assert event == "Stop"

    def test_fallback_to_turn_completed(self) -> None:
        query, _, _ = _handle_stop({})
        assert query == "Stop: turn completed"

    def test_stop_reason_scrubbed(self) -> None:
        data: dict[str, object] = {
            "stop_reason": "leaked sk-ant-abcdefghijklmnopqrstuvwxyz",
        }
        query, _, _ = _handle_stop(data)
        assert "sk-ant-" not in query
        assert "[REDACTED]" in query

    def test_large_stop_reason_truncated(self) -> None:
        data: dict[str, object] = {"stop_reason": "r" * 5000}
        query, _, _ = _handle_stop(data)
        # "Stop: " + max 500 chars
        assert len(query) <= len("Stop: ") + 500


class TestDetectEvent:
    def test_known_events(self) -> None:
        for event_name in (
            "PreToolUse", "PostToolUse", "UserPromptSubmit",
            "SubagentStart", "Stop",
        ):
            data: dict[str, object] = {"hook_event_name": event_name}
            assert _detect_event(data) == event_name

    def test_fallback_to_pre_tool_use(self) -> None:
        assert _detect_event({"hook_event_name": "Unknown"}) == "PreToolUse"
        assert _detect_event({}) == "PreToolUse"

    def test_legacy_event_field(self) -> None:
        data: dict[str, object] = {"event": "UserPromptSubmit"}
        assert _detect_event(data) == "UserPromptSubmit"

    def test_hook_event_name_takes_priority(self) -> None:
        data: dict[str, object] = {
            "hook_event_name": "PostToolUse",
            "event": "PreToolUse",
        }
        assert _detect_event(data) == "PostToolUse"


class TestEventHandlersDict:
    def test_all_five_events_registered(self) -> None:
        expected = {"PreToolUse", "PostToolUse", "UserPromptSubmit",
                    "SubagentStart", "Stop"}
        assert set(_EVENT_HANDLERS.keys()) == expected

    def test_handlers_callable(self) -> None:
        for handler in _EVENT_HANDLERS.values():
            assert callable(handler)


class TestEventLabels:
    def test_pre_tool_use_label(self) -> None:
        assert _EVENT_LABELS["PreToolUse"] == _LABEL_PREVENT

    def test_post_tool_use_label(self) -> None:
        assert _EVENT_LABELS["PostToolUse"] == _LABEL_VERIFY

    def test_user_prompt_submit_label(self) -> None:
        assert _EVENT_LABELS["UserPromptSubmit"] == _LABEL_PREVENT

    def test_subagent_start_label(self) -> None:
        assert _EVENT_LABELS["SubagentStart"] == _LABEL_PROPAGATE

    def test_stop_label(self) -> None:
        assert _EVENT_LABELS["Stop"] == _LABEL_AUDIT

    def test_label_contents(self) -> None:
        assert "VERIFY compliance" in _LABEL_VERIFY
        assert "RULES this agent" in _LABEL_PROPAGATE
        assert "AUDIT" in _LABEL_AUDIT
        assert "RULES you must follow" in _LABEL_PREVENT


class TestInjectionLabelInOutput:
    """Verify that each event type uses its correct injection label."""

    def _run_main_with_event(
        self,
        hook_input: dict[str, object],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> dict[str, object]:
        config = _make_config(tmp_path)
        index = _make_index()
        results = _make_results()

        with (
            patch(f"{_MOD}._try_daemon", return_value=None),
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.load_or_build", return_value=LoadedIndex(index=index)),
            patch(_PIPELINE, return_value=_make_pipeline_result(results)),
            patch(f"{_MOD}.log_retrieval"),
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        captured = capsys.readouterr()
        return json.loads(captured.out)

    def test_pre_tool_use_label(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = self._run_main_with_event(
            {"tool_name": "Bash", "tool_input": "ls"},
            tmp_path, capsys,
        )
        ctx = output["hookSpecificOutput"]["additionalContext"]
        assert ctx.startswith(_LABEL_PREVENT)

    def test_post_tool_use_label(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = self._run_main_with_event(
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_input": "ls",
                "tool_output": "file.txt",
            },
            tmp_path, capsys,
        )
        ctx = output["hookSpecificOutput"]["additionalContext"]
        assert ctx.startswith(_LABEL_VERIFY)

    def test_user_prompt_submit_label(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = self._run_main_with_event(
            {
                "hook_event_name": "UserPromptSubmit",
                "prompt": "add auth",
            },
            tmp_path, capsys,
        )
        ctx = output["hookSpecificOutput"]["additionalContext"]
        assert ctx.startswith(_LABEL_PREVENT)

    def test_subagent_start_label(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = self._run_main_with_event(
            {
                "hook_event_name": "SubagentStart",
                "agent_type": "reviewer",
                "tool_input": {"prompt": "review code"},
            },
            tmp_path, capsys,
        )
        ctx = output["hookSpecificOutput"]["additionalContext"]
        assert ctx.startswith(_LABEL_PROPAGATE)

    def test_stop_label(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = self._run_main_with_event(
            {"hook_event_name": "Stop", "stop_reason": "done"},
            tmp_path, capsys,
        )
        ctx = output["hookSpecificOutput"]["additionalContext"]
        assert ctx.startswith(_LABEL_AUDIT)


class TestHookOutputFormat:
    """Verify permissionDecision is only set for PreToolUse."""

    def _run_event(
        self,
        hook_input: dict[str, object],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> dict[str, object]:
        config = _make_config(tmp_path)
        index = _make_index()

        with (
            patch(f"{_MOD}._try_daemon", return_value=None),
            patch("sys.stdin") as mock_stdin,
            patch(f"{_MOD}.load_config", return_value=config),
            patch("fastembed.TextEmbedding"),
            patch(f"{_MOD}.load_or_build", return_value=LoadedIndex(index=index)),
            patch(_PIPELINE, return_value=_make_pipeline_result([])),
            patch(f"{_MOD}.log_retrieval"),
        ):
            mock_stdin.read.return_value = json.dumps(hook_input)
            main()

        captured = capsys.readouterr()
        return json.loads(captured.out)

    def test_pre_tool_use_has_permission_decision(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = self._run_event(
            {"tool_name": "Bash", "tool_input": "ls"},
            tmp_path, capsys,
        )
        hook_out = output["hookSpecificOutput"]
        assert hook_out["hookEventName"] == "PreToolUse"
        assert hook_out["permissionDecision"] == "allow"

    def test_post_tool_use_no_permission_decision(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = self._run_event(
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_input": "ls",
                "tool_output": "ok",
            },
            tmp_path, capsys,
        )
        hook_out = output["hookSpecificOutput"]
        assert hook_out["hookEventName"] == "PostToolUse"
        assert "permissionDecision" not in hook_out

    def test_user_prompt_submit_no_permission_decision(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = self._run_event(
            {"hook_event_name": "UserPromptSubmit", "prompt": "hello"},
            tmp_path, capsys,
        )
        hook_out = output["hookSpecificOutput"]
        assert hook_out["hookEventName"] == "UserPromptSubmit"
        assert "permissionDecision" not in hook_out

    def test_subagent_start_no_permission_decision(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = self._run_event(
            {
                "hook_event_name": "SubagentStart",
                "agent_type": "reviewer",
                "tool_input": {"prompt": "review"},
            },
            tmp_path, capsys,
        )
        hook_out = output["hookSpecificOutput"]
        assert hook_out["hookEventName"] == "SubagentStart"
        assert "permissionDecision" not in hook_out

    def test_stop_no_permission_decision(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = self._run_event(
            {"hook_event_name": "Stop"},
            tmp_path, capsys,
        )
        hook_out = output["hookSpecificOutput"]
        assert hook_out["hookEventName"] == "Stop"
        assert "permissionDecision" not in hook_out


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
            patch(f"{_MOD}._try_daemon", return_value=None),
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


class TestEntryPoint:
    """Tests for cuecard._entry fast dispatch."""

    def test_hook_dispatches_to_adapter(self) -> None:
        with (
            patch("sys.argv", ["cuecard", "hook"]),
            patch(
                "cuecard.adapters.claude_code.main",
            ) as mock_hook,
        ):
            from cuecard._entry import main as entry_main

            entry_main()
        mock_hook.assert_called_once()

    def test_non_hook_dispatches_to_typer(self) -> None:
        with (
            patch("sys.argv", ["cuecard", "status"]),
            patch("cuecard.cli.main.app") as mock_app,
        ):
            from cuecard._entry import main as entry_main

            entry_main()
        mock_app.assert_called_once()

    def test_no_args_dispatches_to_typer(self) -> None:
        with (
            patch("sys.argv", ["cuecard"]),
            patch("cuecard.cli.main.app") as mock_app,
        ):
            from cuecard._entry import main as entry_main

            entry_main()
        mock_app.assert_called_once()
