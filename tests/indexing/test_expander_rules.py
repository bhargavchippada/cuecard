"""Tests for cuecard.expander — expand_rules function."""

from __future__ import annotations

import secrets
import socket
from unittest.mock import MagicMock, patch

import pytest

from cuecard.indexing.expander import (
    _TOOL_STYLE,
    _VALID_EVENT_TYPES,
    _WORKFLOW_STYLE,
    _build_expansion_prompt,
    _expand_single_rule,
    _expansion_styles_for_rule,
    expand_rules,
)
from cuecard.models import (
    AffinityIndex,
    Provenance,
    Rule,
    RuleAffinity,
    _hash_rule_text,
)
from cuecard.security import ConfigError


def _make_rule(text: str = "Never commit secrets", **kwargs: object) -> Rule:
    return Rule(
        text=text,
        provenance=Provenance(
            file="/tmp/rules.txt", line_start=1, line_end=1,
        ),
        **kwargs,  # type: ignore[arg-type]
    )


class TestExpandRules:
    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid backend"):
            expand_rules(
                [_make_rule()], "openai", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)

    def test_haiku_model_not_in_allowlist_raises(self) -> None:
        with pytest.raises(ValueError, match="not in allowlist"):
            expand_rules(
                [_make_rule()], "haiku", "http://localhost:8081/v1",
                "gpt-4o", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)

    def test_local_validates_endpoint(self) -> None:
        fake = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        with (
            patch("cuecard.retrieval.llm_utils.socket.getaddrinfo", return_value=fake),
            pytest.raises(ConfigError, match="loopback"),
        ):
            expand_rules(
                [_make_rule()], "local", "http://evil.com:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)

    def test_dry_run_does_not_call_llm(self) -> None:
        rules = [_make_rule()]
        with patch("cuecard.indexing.expander.call_local") as mock_call:
            result = expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dry_run=True, dedup_threshold=0.80,
                    max_tokens=1024, timeout=60.0,
                )
            mock_call.assert_not_called()
        assert len(result) == 1
        assert result[0].expansions == ()

    def test_missing_only_skips_existing(self) -> None:
        rules = [
            _make_rule(expansions=("existing",)),
            _make_rule(text="No expansions yet"),
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '{"expansions": ["new one"]}'}},
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cuecard.indexing.expander.call_local",
            return_value='{"expansions": ["new one"]}',
        ):
            result = expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", missing_only=True,
                dedup_threshold=0.80, max_tokens=1024, timeout=60.0)

        assert result[0].expansions == ("existing",)
        assert result[1].expansions == ("new one",)

    def test_local_backend_success(self) -> None:
        rules = [_make_rule()]

        with patch(
            "cuecard.indexing.expander.call_local",
            return_value=(
                '{"expansions": ["docker build image", "run pytest coverage"]}'
            ),
        ):
            result = expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)

        assert len(result) == 1
        assert result[0].expansions == ("docker build image", "run pytest coverage")

    def test_haiku_backend_success(self) -> None:
        rules = [_make_rule()]

        with patch(
            "cuecard.indexing.expander.call_haiku",
            return_value='{"expansions": ["haiku phrase"]}',
        ):
            result = expand_rules(
                rules, "haiku", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)

        assert len(result) == 1
        assert result[0].expansions == ("haiku phrase",)

    def test_llm_failure_keeps_original(self) -> None:
        rules = [_make_rule()]

        with patch(
            "cuecard.indexing.expander.call_local",
            side_effect=RuntimeError("connection failed"),
        ):
            result = expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)

        assert len(result) == 1
        assert result[0].expansions == ()

    def test_single_rule_retries_once_on_json_parse_failure(self) -> None:
        rule = _make_rule()

        with patch(
            "cuecard.indexing.expander.call_local",
            side_effect=[
                "not json at all",
                '{"expansions": ["retry worked"]}',
            ],
        ) as mock_call:
            expanded, count = _expand_single_rule(
                rule,
                "local",
                "http://localhost:8081/v1",
                "claude-haiku-4-5",
                "PreToolUse",
                0.80, max_tokens=1024, timeout=60.0)

        assert mock_call.call_count == 2
        assert expanded.expansions == ("retry worked",)
        assert count == 1

    def test_haiku_backend_retries_once_on_json_parse_failure(self) -> None:
        rule = _make_rule()

        with patch(
            "cuecard.indexing.expander.call_haiku",
            side_effect=[
                "not json at all",
                '{"expansions": ["haiku retry worked"]}',
            ],
        ) as mock_call:
            expanded, count = _expand_single_rule(
                rule,
                "haiku",
                "http://localhost:8081/v1",
                "claude-haiku-4-5",
                "PreToolUse",
                0.80, max_tokens=1024, timeout=60.0)

        assert mock_call.call_count == 2
        assert expanded.expansions == ("haiku retry worked",)
        assert count == 1

    def test_nonce_unique_per_rule(self) -> None:
        rules = [_make_rule(text="Rule A"), _make_rule(text="Rule B")]
        nonces: list[str] = []

        original_token_hex = secrets.token_hex

        def capture_nonce(n: int) -> str:
            result = original_token_hex(n)
            nonces.append(result)
            return result

        with (
            patch(
                "cuecard.indexing.expander.call_local",
                return_value='{"expansions": ["x"]}',
            ),
            patch(
                "cuecard.indexing.expander.secrets.token_hex",
                side_effect=capture_nonce,
            ),
        ):
            expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)

        assert len(nonces) == 2
        assert nonces[0] != nonces[1]

    def test_empty_rules_list(self) -> None:
        result = expand_rules(
            [], "local", "http://localhost:8081/v1",
            "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)
        assert result == []

    def test_config_error_propagates(self) -> None:
        """ConfigError from validate_endpoint should propagate, not be caught."""
        fake = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        with (
            patch("cuecard.retrieval.llm_utils.socket.getaddrinfo", return_value=fake),
            pytest.raises(ConfigError),
        ):
            expand_rules(
                [_make_rule()], "local", "http://evil.com/v1",
                "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)

    def test_value_error_propagates(self) -> None:
        """ValueError for invalid backend should propagate."""
        with pytest.raises(ValueError):
            expand_rules(
                [_make_rule()], "invalid", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)

    def test_replaces_existing_expansions(self) -> None:
        """Without missing_only, existing expansions are overwritten."""
        rules = [_make_rule(expansions=("old",))]

        with patch(
            "cuecard.indexing.expander.call_local",
            return_value='{"expansions": ["new"]}',
        ):
            result = expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)

        assert result[0].expansions == ("new",)

    def test_value_error_from_llm_call_propagates(self) -> None:
        """ValueError raised during LLM call propagates through loop."""
        rules = [_make_rule()]

        with (
            patch(
                "cuecard.indexing.expander.call_local",
                side_effect=ValueError("bad response format"),
            ),
            pytest.raises(ValueError, match="bad response format"),
        ):
            expand_rules(
                    rules, "local", "http://localhost:8081/v1",
                    "claude-haiku-4-5", dedup_threshold=0.80,
                        max_tokens=1024, timeout=60.0,
                    )

    def test_returns_new_rule_objects(self) -> None:
        """expand_rules returns new Rule objects, not mutated originals."""
        original = _make_rule()

        with patch(
            "cuecard.indexing.expander.call_local",
            return_value='{"expansions": ["x"]}',
        ):
            result = expand_rules(
                [original], "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)

        assert result[0] is not original
        assert original.expansions == ()
        assert result[0].expansions == ("x",)

    def test_invalid_event_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid event_type"):
            expand_rules(
                [_make_rule()], "local", "http://localhost:8081/v1",
                "claude-haiku-4-5",
                event_type="InvalidEvent", dedup_threshold=0.80,
                    max_tokens=1024, timeout=60.0,
                )

    def test_event_type_passed_to_prompt(self) -> None:
        """event_type is forwarded to _build_expansion_prompt."""
        with (
            patch(
                "cuecard.indexing.expander.call_local",
                return_value='{"expansions": ["x"]}',
            ),
            patch(
                "cuecard.indexing.expander._build_expansion_prompt",
                wraps=_build_expansion_prompt,
            ) as mock_build,
            patch(
                "cuecard.indexing.expander._semantic_dedup",
                side_effect=lambda x, **kw: x,
            ),
        ):
            expand_rules(
                [_make_rule()], "local", "http://localhost:8081/v1",
                "claude-haiku-4-5",
                event_type="UserPromptSubmit", dedup_threshold=0.80,
                    max_tokens=1024, timeout=60.0,
                )
            _, kwargs = mock_build.call_args
            assert kwargs.get("event_type") == "UserPromptSubmit"

    def test_semantic_dedup_called(self) -> None:
        """expand_rules calls _semantic_dedup on parsed expansions."""
        with (
            patch(
                "cuecard.indexing.expander.call_local",
                return_value='{"expansions": ["a", "b", "c"]}',
            ),
            patch(
                "cuecard.indexing.expander._semantic_dedup",
                return_value=["a", "c"],
            ) as mock_dedup,
        ):
            result = expand_rules(
                [_make_rule()], "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)
            mock_dedup.assert_called_once_with(["a", "b", "c"], threshold=0.8)
            assert result[0].expansions == ("a", "c")

    def test_valid_event_types_constant(self) -> None:
        assert "PreToolUse" in _VALID_EVENT_TYPES
        assert "UserPromptSubmit" in _VALID_EVENT_TYPES

    def test_on_progress_called_for_each_rule(self) -> None:
        """on_progress callback is invoked once per rule."""
        rules = [_make_rule(text="Rule A"), _make_rule(text="Rule B")]
        updates: list[object] = []

        with patch(
            "cuecard.indexing.expander.call_local",
            return_value='{"expansions": ["x"]}',
        ):
            expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80,
                on_progress=updates.append, max_tokens=1024, timeout=60.0)

        assert len(updates) == 2
        assert updates[0].rule_index == 0  # type: ignore[union-attr]
        assert updates[0].total_rules == 2  # type: ignore[union-attr]
        assert updates[0].skipped is False  # type: ignore[union-attr]
        assert updates[1].rule_index == 1  # type: ignore[union-attr]
        assert updates[1].expansions_generated == 2  # type: ignore[union-attr]

    def test_on_progress_skipped_for_missing_only(self) -> None:
        """on_progress reports skipped=True for rules with existing expansions."""
        rules = [
            _make_rule(text="Has expansions", expansions=("a", "b")),
            _make_rule(text="No expansions"),
        ]
        updates: list[object] = []

        with patch(
            "cuecard.indexing.expander.call_local",
            return_value='{"expansions": ["new"]}',
        ):
            expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", missing_only=True,
                dedup_threshold=0.80,
                on_progress=updates.append, max_tokens=1024, timeout=60.0)

        assert len(updates) == 2
        assert updates[0].skipped is True  # type: ignore[union-attr]
        assert updates[0].expansions_generated == 2  # type: ignore[union-attr]
        assert updates[1].skipped is False  # type: ignore[union-attr]
        assert updates[1].expansions_generated == 3  # type: ignore[union-attr]

    def test_on_progress_not_called_when_none(self) -> None:
        """No error when on_progress is None (default)."""
        with patch(
            "cuecard.indexing.expander.call_local",
            return_value='{"expansions": ["x"]}',
        ):
            result = expand_rules(
                [_make_rule()], "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80, max_tokens=1024, timeout=60.0)
        assert len(result) == 1

    def test_affinity_tool_use_gets_tool_style(self) -> None:
        """Rules with tool_use affinity get PreToolUse expansion style."""
        rule = _make_rule()

        with (
            patch(
                "cuecard.indexing.expander._build_expansion_prompt",
                wraps=_build_expansion_prompt,
            ) as mock_prompt,
            patch(
                "cuecard.indexing.expander.call_local",
                return_value='{"expansions": ["x"]}',
            ),
            patch(
                "cuecard.indexing.expander._semantic_dedup",
                side_effect=lambda x, **kw: x,
            ),
        ):
            aff = _make_affinity(rule, events=frozenset({"PreToolUse"}))
            expand_rules(
                [rule], "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80,
                affinity=aff, max_tokens=1024, timeout=60.0)
            _, kwargs = mock_prompt.call_args
            assert kwargs["event_type"] == "PreToolUse"

    def test_affinity_workflow_gets_workflow_style(self) -> None:
        """Rules with workflow affinity get UserPromptSubmit expansion style."""
        rule = _make_rule()

        with (
            patch(
                "cuecard.indexing.expander._build_expansion_prompt",
                wraps=_build_expansion_prompt,
            ) as mock_prompt,
            patch(
                "cuecard.indexing.expander.call_local",
                return_value='{"expansions": ["x"]}',
            ),
            patch(
                "cuecard.indexing.expander._semantic_dedup",
                side_effect=lambda x, **kw: x,
            ),
        ):
            aff = _make_affinity(
                rule, events=frozenset({"UserPromptSubmit", "Stop"}),
            )
            expand_rules(
                [rule], "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80,
                affinity=aff, max_tokens=1024, timeout=60.0)
            _, kwargs = mock_prompt.call_args
            assert kwargs["event_type"] == "UserPromptSubmit"

    def test_affinity_both_gets_two_calls(self) -> None:
        """Rules with both tool+workflow affinity expand twice."""
        rule = _make_rule()
        call_count = 0

        def _fake_call(*args: object, **kwargs: object) -> str:
            nonlocal call_count
            call_count += 1
            return '{"expansions": ["exp' + str(call_count) + '"]}'

        with (
            patch("cuecard.indexing.expander.call_local", side_effect=_fake_call),
            patch(
                "cuecard.indexing.expander._semantic_dedup",
                side_effect=lambda x, **kw: x,
            ),
        ):
            aff = _make_affinity(
                rule,
                events=frozenset({"PreToolUse", "UserPromptSubmit"}),
            )
            result = expand_rules(
                [rule], "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80,
                affinity=aff, max_tokens=1024, timeout=60.0)
        assert call_count == 2
        assert len(result[0].expansions) == 2

    def test_affinity_none_uses_fallback(self) -> None:
        """When affinity is None, falls back to event_type."""
        rule = _make_rule()

        with (
            patch(
                "cuecard.indexing.expander._build_expansion_prompt",
                wraps=_build_expansion_prompt,
            ) as mock_prompt,
            patch(
                "cuecard.indexing.expander.call_local",
                return_value='{"expansions": ["x"]}',
            ),
            patch(
                "cuecard.indexing.expander._semantic_dedup",
                side_effect=lambda x, **kw: x,
            ),
        ):
            expand_rules(
                [rule], "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80,
                event_type="UserPromptSubmit",
                affinity=None, max_tokens=1024, timeout=60.0)
            _, kwargs = mock_prompt.call_args
            assert kwargs["event_type"] == "UserPromptSubmit"

    def test_affinity_rule_not_found_uses_fallback(self) -> None:
        """When rule not in affinity index, falls back to event_type."""
        rule = _make_rule()
        # Empty affinity — rule won't be found
        aff = AffinityIndex(version=1, mode="infer", model="test", affinities=())

        with (
            patch(
                "cuecard.indexing.expander._build_expansion_prompt",
                wraps=_build_expansion_prompt,
            ) as mock_prompt,
            patch(
                "cuecard.indexing.expander.call_local",
                return_value='{"expansions": ["x"]}',
            ),
            patch(
                "cuecard.indexing.expander._semantic_dedup",
                side_effect=lambda x, **kw: x,
            ),
        ):
            expand_rules(
                [rule], "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80,
                event_type="UserPromptSubmit",
                affinity=aff, max_tokens=1024, timeout=60.0)
            _, kwargs = mock_prompt.call_args
            assert kwargs["event_type"] == "UserPromptSubmit"

    def test_parallel_produces_same_results(self) -> None:
        """Parallel mode produces the same rules as sequential."""
        rules = [
            _make_rule(text="Rule A"),
            _make_rule(text="Rule B"),
            _make_rule(text="Rule C"),
        ]

        def _fake_call(*args: object, **kwargs: object) -> str:
            return '{"expansions": ["expanded"]}'

        with patch("cuecard.indexing.expander.call_local", side_effect=_fake_call):
            sequential = expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80,
                max_workers=1, max_tokens=1024, timeout=60.0)

        with patch("cuecard.indexing.expander.call_local", side_effect=_fake_call):
            parallel = expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80,
                max_workers=3, max_tokens=1024, timeout=60.0)

        assert len(sequential) == len(parallel)
        for s, p in zip(sequential, parallel, strict=True):
            assert s.text == p.text
            assert s.expansions == p.expansions

    def test_parallel_skips_missing_only(self) -> None:
        """Parallel mode respects missing_only flag."""
        rules = [
            _make_rule(text="Has exp", expansions=("existing",)),
            _make_rule(text="Needs exp"),
        ]

        with patch(
            "cuecard.indexing.expander.call_local",
            return_value='{"expansions": ["new"]}',
        ):
            result = expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80,
                missing_only=True, max_workers=2, max_tokens=1024, timeout=60.0)

        assert result[0].expansions == ("existing",)
        assert result[1].expansions == ("new",)

    def test_parallel_with_affinity(self) -> None:
        """Parallel mode uses affinity for per-rule style selection."""
        tool_rule = _make_rule(text="tool rule")
        workflow_rule = _make_rule(text="workflow rule")
        rules = [tool_rule, workflow_rule]

        prompt_styles: list[str] = []
        original_build = _build_expansion_prompt

        def _capture_prompt(
            text: str, nonce: str, event_type: str = "PreToolUse",
        ) -> tuple:
            prompt_styles.append(event_type)
            return original_build(text, nonce, event_type=event_type)

        aff = AffinityIndex(
            version=1, mode="infer", model="test",
            affinities=(
                (_hash_rule_text("tool rule"), RuleAffinity(
                    events=frozenset({"PreToolUse"}),
                    tools=frozenset(), source="inferred",
                )),
                (_hash_rule_text("workflow rule"), RuleAffinity(
                    events=frozenset({"UserPromptSubmit"}),
                    tools=frozenset(), source="inferred",
                )),
            ),
        )

        with (
            patch(
                "cuecard.indexing.expander._build_expansion_prompt",
                side_effect=_capture_prompt,
            ),
            patch(
                "cuecard.indexing.expander.call_local",
                return_value='{"expansions": ["x"]}',
            ),
            patch(
                "cuecard.indexing.expander._semantic_dedup",
                side_effect=lambda x, **kw: x,
            ),
        ):
            expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80,
                affinity=aff, max_workers=2, max_tokens=1024, timeout=60.0)

        assert "PreToolUse" in prompt_styles
        assert "UserPromptSubmit" in prompt_styles

    def test_on_progress_disables_parallel(self) -> None:
        """on_progress forces sequential mode even with max_workers > 1."""
        rules = [_make_rule(text="A"), _make_rule(text="B")]
        updates: list[object] = []

        with patch(
            "cuecard.indexing.expander.call_local",
            return_value='{"expansions": ["x"]}',
        ):
            expand_rules(
                rules, "local", "http://localhost:8081/v1",
                "claude-haiku-4-5", dedup_threshold=0.80,
                max_workers=5, on_progress=updates.append,
                    max_tokens=1024, timeout=60.0,
                )

        # on_progress was called (sequential mode) — parallel would skip it
        assert len(updates) == 2
        assert updates[0].rule_index == 0  # type: ignore[union-attr]


class TestExpansionStylesForRule:
    """Tests for _expansion_styles_for_rule."""

    def test_no_affinity_returns_fallback(self) -> None:
        rule = _make_rule()
        assert _expansion_styles_for_rule(rule, None, "PreToolUse") == ("PreToolUse",)

    def test_rule_not_in_affinity_returns_fallback(self) -> None:
        rule = _make_rule()
        empty_aff = AffinityIndex(
            version=1, mode="infer", model="test", affinities=(),
        )
        result = _expansion_styles_for_rule(rule, empty_aff, "PreToolUse")
        assert result == ("PreToolUse",)

    def test_tool_use_events_return_tool_style(self) -> None:
        rule = _make_rule()
        aff = _make_affinity(rule, events=frozenset({"PreToolUse"}))
        result = _expansion_styles_for_rule(rule, aff, "PreToolUse")
        assert result == (_TOOL_STYLE,)

    def test_workflow_events_return_workflow_style(self) -> None:
        rule = _make_rule()
        aff = _make_affinity(rule, events=frozenset({"UserPromptSubmit", "Stop"}))
        result = _expansion_styles_for_rule(rule, aff, "PreToolUse")
        assert result == (_WORKFLOW_STYLE,)

    def test_both_events_return_both_styles(self) -> None:
        rule = _make_rule()
        aff = _make_affinity(
            rule,
            events=frozenset({"PreToolUse", "UserPromptSubmit", "Stop"}),
        )
        result = _expansion_styles_for_rule(rule, aff, "PreToolUse")
        assert result == (_TOOL_STYLE, _WORKFLOW_STYLE)

    def test_empty_events_returns_fallback(self) -> None:
        rule = _make_rule()
        aff = _make_affinity(rule, events=frozenset())
        result = _expansion_styles_for_rule(rule, aff, "UserPromptSubmit")
        assert result == ("UserPromptSubmit",)

    def test_subagent_start_is_workflow(self) -> None:
        rule = _make_rule()
        aff = _make_affinity(rule, events=frozenset({"SubagentStart"}))
        result = _expansion_styles_for_rule(rule, aff, "PreToolUse")
        assert result == (_WORKFLOW_STYLE,)


def _make_affinity(
    rule: Rule,
    *,
    events: frozenset[str] = frozenset(),
) -> AffinityIndex:
    """Create a minimal AffinityIndex with one rule."""
    text_hash = _hash_rule_text(rule.text)
    return AffinityIndex(
        version=1, mode="infer", model="test",
        affinities=(
            (text_hash, RuleAffinity(
                events=events, tools=frozenset(), source="inferred",
            )),
        ),
    )



class TestExpandRulesParallelCoverage:
    def test_parallel_multistyle_uses_multi_style_helper(self) -> None:
        rules = [_make_rule(text="Both rule")]
        aff = AffinityIndex(
            version=1,
            mode="strict",
            model="",
            affinities=((
                _hash_rule_text("Both rule"),
                RuleAffinity(
                    events=frozenset({"PreToolUse", "UserPromptSubmit"}),
                    tools=frozenset(),
                    source="explicit",
                ),
            ),),
        )

        with patch(
            "cuecard.indexing.expander._expand_single_rule_multi_style",
            return_value=(
                _make_rule(text="Both rule", expansions=("x",)),
                1,
            ),
        ) as mock_multi:
            result = expand_rules(
                rules,
                "local",
                "http://localhost:8081/v1",
                "claude-haiku-4-5",
                dedup_threshold=0.80,
                affinity=aff,
                max_workers=2, max_tokens=1024, timeout=60.0)

        assert result[0].expansions == ("x",)
        mock_multi.assert_called_once()
