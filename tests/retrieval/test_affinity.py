"""Tests for cuecard.affinity."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from cuecard.models import (
    KNOWN_HOOK_EVENTS,
    AffinityIndex,
    Provenance,
    ResolvedConfig,
    Rule,
    RuleAffinity,
)
from cuecard.retrieval.affinity import (
    _build_affinity_prompt,
    _compute_checksum,
    _hash_rule_text,
    _parse_affinity_response,
    _strict_affinity,
    build_strict_affinity,
    infer_affinities,
    load_affinity,
    save_affinity,
)

if TYPE_CHECKING:
    from pathlib import Path


# --- Helpers ---


def _make_rule(
    text: str,
    events: frozenset[str] | None = None,
    tools: frozenset[str] | None = None,
) -> Rule:
    return Rule(
        text=text,
        provenance=Provenance(file="test.txt", line_start=1, line_end=1),
        events=events or frozenset(),
        tools=tools or frozenset(),
    )


def _make_config(
    affinity_mode: str = "infer",
    pipeline_mode: str = "llm-local",
) -> ResolvedConfig:
    from cuecard.models import PipelineConfig

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
        global_cache_dir="/tmp/test-cache",
        project_cache_dir=None,
        allowed_dirs=(),
        pipeline=PipelineConfig(mode=pipeline_mode),
        affinity_mode=affinity_mode,
    )


# --- _hash_rule_text ---


class TestHashRuleText:
    def test_deterministic(self) -> None:
        h1 = _hash_rule_text("Never commit secrets")
        h2 = _hash_rule_text("Never commit secrets")
        assert h1 == h2

    def test_different_text_different_hash(self) -> None:
        h1 = _hash_rule_text("Rule A")
        h2 = _hash_rule_text("Rule B")
        assert h1 != h2

    def test_returns_hex_string(self) -> None:
        h = _hash_rule_text("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_no_prefix(self) -> None:
        h = _hash_rule_text("test")
        assert not h.startswith("sha256:")


# --- _build_affinity_prompt ---


class TestBuildAffinityPrompt:
    def test_nonce_in_system_prompt(self) -> None:
        system, user = _build_affinity_prompt(
            "Use uv", "abc123", frozenset(), frozenset(),
        )
        assert "abc123" in system
        assert "abc123" in user

    def test_scrubs_secrets(self) -> None:
        system, user = _build_affinity_prompt(
            "key=sk-ant-abcdefghijklmnopqrst",
            "nonce1",
            frozenset(),
            frozenset(),
        )
        assert "sk-ant-" not in user
        assert "[REDACTED]" in user

    def test_nonce_stripped_from_rule_text(self) -> None:
        _, user = _build_affinity_prompt(
            "Rule with nonce1 in it",
            "nonce1",
            frozenset(),
            frozenset(),
        )
        # The nonce should be stripped from the rule text inside tags
        assert "<rule_data_nonce1>Rule with  in it</rule_data_nonce1>" in user

    def test_explicit_annotations_in_user_prompt(self) -> None:
        _, user = _build_affinity_prompt(
            "Use uv",
            "abc",
            frozenset({"PreToolUse"}),
            frozenset({"Bash"}),
        )
        assert "PreToolUse" in user
        assert "Bash" in user

    def test_categories_in_system_prompt(self) -> None:
        system, _ = _build_affinity_prompt(
            "test", "nonce", frozenset(), frozenset(),
        )
        assert "tool_use" in system
        assert "workflow" in system
        assert "category" in system


# --- _parse_affinity_response ---


class TestParseAffinityResponse:
    def test_valid_json(self) -> None:
        response = json.dumps({
            "events": ["PreToolUse", "Stop"],
            "tools": ["Bash"],
            "reasoning": "Git commit is a Bash operation",
        })
        events, tools, reasoning = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert events == frozenset({"PreToolUse", "Stop"})
        assert tools == frozenset({"Bash"})
        assert "Git commit" in reasoning

    def test_extends_explicit_annotations(self) -> None:
        response = json.dumps({
            "events": ["Stop"],
            "tools": ["Edit"],
            "reasoning": "",
        })
        events, tools, _ = _parse_affinity_response(
            response,
            frozenset({"PreToolUse"}),
            frozenset({"Bash"}),
        )
        assert "PreToolUse" in events
        assert "Stop" in events
        assert "Bash" in tools
        assert "Edit" in tools

    def test_invalid_json_returns_explicit(self) -> None:
        events, tools, reasoning = _parse_affinity_response(
            "not json",
            frozenset({"PreToolUse"}),
            frozenset({"Bash"}),
        )
        assert events == frozenset({"PreToolUse"})
        assert tools == frozenset({"Bash"})
        assert reasoning == ""

    def test_drops_unknown_events(self) -> None:
        response = json.dumps({
            "events": ["PreToolUse", "FakeEvent"],
            "tools": [],
            "reasoning": "",
        })
        events, _, _ = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert events == frozenset({"PreToolUse"})
        assert "FakeEvent" not in events

    def test_strips_thinking_tags(self) -> None:
        response = (
            "<think>Let me think...</think>"
            '{"events": ["Stop"], "tools": [], "reasoning": "audit"}'
        )
        events, _, reasoning = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert events == frozenset({"Stop"})
        assert reasoning == "audit"

    def test_json_in_code_block(self) -> None:
        response = (
            "```json\n"
            '{"events": ["PreToolUse"], "tools": ["Bash"], '
            '"reasoning": "test"}\n'
            "```"
        )
        events, tools, _ = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert events == frozenset({"PreToolUse"})
        assert tools == frozenset({"Bash"})

    def test_generic_code_block(self) -> None:
        response = (
            "```\n"
            '{"events": ["Stop"], "tools": [], '
            '"reasoning": "audit"}\n'
            "```"
        )
        events, _, reasoning = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert events == frozenset({"Stop"})
        assert reasoning == "audit"

    def test_non_dict_response(self) -> None:
        events, tools, _ = _parse_affinity_response(
            '["not", "a", "dict"]',
            frozenset({"PreToolUse"}),
            frozenset(),
        )
        assert events == frozenset({"PreToolUse"})

    def test_non_list_events_field(self) -> None:
        response = json.dumps({
            "events": "PreToolUse",
            "tools": [],
            "reasoning": "",
        })
        events, _, _ = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert events == frozenset()

    def test_non_string_event_values(self) -> None:
        response = json.dumps({
            "events": ["PreToolUse", 42, None],
            "tools": [],
            "reasoning": "",
        })
        events, _, _ = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert events == frozenset({"PreToolUse"})

    def test_reasoning_truncated(self) -> None:
        response = json.dumps({
            "events": [],
            "tools": [],
            "reasoning": "x" * 600,
        })
        _, _, reasoning = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert len(reasoning) == 500

    def test_non_string_reasoning(self) -> None:
        response = json.dumps({
            "events": [],
            "tools": [],
            "reasoning": 42,
        })
        _, _, reasoning = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert reasoning == ""

    def test_tool_whitespace_stripped(self) -> None:
        response = json.dumps({
            "events": [],
            "tools": ["  Bash  ", "Edit"],
            "reasoning": "",
        })
        _, tools, _ = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert "Bash" in tools
        assert "  Bash  " not in tools

    def test_empty_tool_string_dropped(self) -> None:
        response = json.dumps({
            "events": [],
            "tools": ["", "  ", "Bash"],
            "reasoning": "",
        })
        _, tools, _ = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert tools == frozenset({"Bash"})

    def test_singular_category_tool_use(self) -> None:
        """New prompt format: singular 'category' field."""
        response = json.dumps({
            "reasoning": "Constrains Bash commands",
            "category": "tool_use",
        })
        events, _, reasoning = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert events == frozenset({"PreToolUse"})
        assert "Constrains" in reasoning

    def test_singular_category_workflow(self) -> None:
        response = json.dumps({
            "reasoning": "Process decision",
            "category": "workflow",
        })
        events, _, _ = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert events == frozenset({"UserPromptSubmit", "SubagentStart", "Stop"})

    def test_singular_category_both(self) -> None:
        response = json.dumps({
            "reasoning": "Both moments",
            "category": "both",
        })
        events, _, _ = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert events == frozenset({
            "PreToolUse",
            "UserPromptSubmit", "SubagentStart", "Stop",
        })

    def test_singular_category_overrides_categories_list(self) -> None:
        """Singular 'category' takes precedence over legacy 'categories' list."""
        response = json.dumps({
            "reasoning": "",
            "category": "tool_use",
            "categories": ["workflow"],
        })
        events, _, _ = _parse_affinity_response(
            response, frozenset(), frozenset(),
        )
        assert events == frozenset({"PreToolUse"})

    def test_invalid_explicit_event_dropped(self) -> None:
        """Typo'd explicit event is stripped before merge."""
        response = json.dumps({
            "events": ["PreToolUse"],
            "tools": [],
            "reasoning": "",
        })
        events, _, _ = _parse_affinity_response(
            response,
            frozenset({"PreToolUes"}),  # typo in explicit
            frozenset(),
        )
        # Typo is dropped, only valid inferred event survives
        assert events == frozenset({"PreToolUse"})
        assert "PreToolUes" not in events


# --- _strict_affinity ---


class TestStrictAffinity:
    def test_empty_events_defaults_to_all(self) -> None:
        rule = _make_rule("No events set")
        aff = _strict_affinity(rule)
        assert aff.events == KNOWN_HOOK_EVENTS
        assert aff.source == "default"

    def test_explicit_events_preserved(self) -> None:
        rule = _make_rule(
            "Has events",
            events=frozenset({"PreToolUse", "Stop"}),
        )
        aff = _strict_affinity(rule)
        assert aff.events == frozenset({"PreToolUse", "Stop"})
        assert aff.source == "explicit"

    def test_empty_tools_means_all(self) -> None:
        rule = _make_rule("No tools")
        aff = _strict_affinity(rule)
        assert aff.tools == frozenset()

    def test_explicit_tools_preserved(self) -> None:
        rule = _make_rule("Has tools", tools=frozenset({"Bash"}))
        aff = _strict_affinity(rule)
        assert aff.tools == frozenset({"Bash"})

    def test_explicit_fields_tracked(self) -> None:
        rule = _make_rule(
            "test",
            events=frozenset({"Stop"}),
            tools=frozenset({"Edit"}),
        )
        aff = _strict_affinity(rule)
        assert aff.explicit_events == frozenset({"Stop"})
        assert aff.explicit_tools == frozenset({"Edit"})


# --- build_strict_affinity ---


class TestBuildStrictAffinity:
    def test_returns_affinity_index(self) -> None:
        rules = [_make_rule("Rule A"), _make_rule("Rule B")]
        idx = build_strict_affinity(rules)
        assert isinstance(idx, AffinityIndex)
        assert idx.mode == "strict"
        assert idx.model == ""
        assert idx.version == 1

    def test_all_rules_indexed(self) -> None:
        rules = [_make_rule("Rule A"), _make_rule("Rule B")]
        idx = build_strict_affinity(rules)
        assert len(idx.items) == 2

    def test_lookup_works(self) -> None:
        rule = _make_rule(
            "Use uv",
            events=frozenset({"PreToolUse"}),
        )
        idx = build_strict_affinity([rule])
        aff = idx.get(rule)
        assert aff is not None
        assert aff.events == frozenset({"PreToolUse"})
        assert aff.source == "explicit"

    def test_unannotated_rule_gets_all_events(self) -> None:
        rule = _make_rule("No annotations")
        idx = build_strict_affinity([rule])
        aff = idx.get(rule)
        assert aff is not None
        assert aff.events == KNOWN_HOOK_EVENTS
        assert aff.source == "default"


# --- save_affinity / load_affinity ---


class TestSaveLoadAffinity:
    def test_round_trip(self, tmp_path: Path) -> None:
        rule = _make_rule("Never commit secrets")
        text_hash = _hash_rule_text(rule.text)
        ra = RuleAffinity(
            events=frozenset({"PreToolUse", "Stop"}),
            tools=frozenset({"Bash"}),
            source="explicit+inferred",
            explicit_events=frozenset({"PreToolUse"}),
            explicit_tools=frozenset({"Bash"}),
            reasoning="Git operations",
        )
        idx = AffinityIndex(
            version=1,
            mode="infer",
            model="local",
            affinities=((text_hash, ra),),
        )

        save_affinity(idx, str(tmp_path))
        loaded = load_affinity(str(tmp_path))

        assert loaded is not None
        assert loaded.mode == "infer"
        assert loaded.model == "local"
        assert loaded.version == 1

        loaded_ra = loaded.get_by_hash(text_hash)
        assert loaded_ra is not None
        assert loaded_ra.events == frozenset({"PreToolUse", "Stop"})
        assert loaded_ra.tools == frozenset({"Bash"})
        assert loaded_ra.source == "explicit+inferred"
        assert loaded_ra.explicit_events == frozenset({"PreToolUse"})
        assert loaded_ra.explicit_tools == frozenset({"Bash"})
        assert loaded_ra.reasoning == "Git operations"

    def test_file_permissions(self, tmp_path: Path) -> None:
        rule = _make_rule("test")
        idx = build_strict_affinity([rule])
        save_affinity(idx, str(tmp_path))

        path = tmp_path / "affinity.json"
        mode = os.stat(path).st_mode & 0o777
        assert mode == 0o600

    def test_atomic_write(self, tmp_path: Path) -> None:
        """No .tmp files remain after successful write."""
        rule = _make_rule("test")
        idx = build_strict_affinity([rule])
        save_affinity(idx, str(tmp_path))

        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "affinity.json"

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        result = load_affinity(str(tmp_path))
        assert result is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "affinity.json"
        path.write_text("not json{{{")
        result = load_affinity(str(tmp_path))
        assert result is None

    def test_unknown_version_returns_none(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "affinity.json"
        path.write_text(json.dumps({
            "version": 99,
            "mode": "infer",
            "model": "local",
            "checksum": "abc",
            "rules": [],
        }))
        result = load_affinity(str(tmp_path))
        assert result is None

    def test_checksum_mismatch_returns_none(
        self, tmp_path: Path,
    ) -> None:
        rule = _make_rule("test")
        idx = build_strict_affinity([rule])
        save_affinity(idx, str(tmp_path))

        # Tamper with checksum
        path = tmp_path / "affinity.json"
        data = json.loads(path.read_text())
        data["checksum"] = "tampered"
        path.write_text(json.dumps(data))

        result = load_affinity(str(tmp_path))
        assert result is None

    def test_non_dict_top_level_returns_none(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "affinity.json"
        path.write_text("[]")
        result = load_affinity(str(tmp_path))
        assert result is None

    def test_malformed_rules_returns_none(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "affinity.json"
        path.write_text(json.dumps({
            "version": 1,
            "mode": "strict",
            "model": "",
            "checksum": "",
            "rules": "not a list",
        }))
        result = load_affinity(str(tmp_path))
        assert result is None

    def test_skips_invalid_entries(self, tmp_path: Path) -> None:
        """Entries with missing text_hash are skipped."""
        text_hash = _hash_rule_text("valid rule")
        rules_data = [
            {"not_text_hash": "missing"},
            {
                "text_hash": text_hash,
                "events": ["PreToolUse"],
                "tools": [],
                "source": "explicit",
                "explicit_events": [],
                "explicit_tools": [],
                "reasoning": "",
            },
            "not a dict",
            {"text_hash": ""},
        ]
        checksum = _compute_checksum(
            ((text_hash, RuleAffinity(
                events=frozenset({"PreToolUse"}),
                tools=frozenset(),
                source="explicit",
            )),),
        )
        path = tmp_path / "affinity.json"
        path.write_text(json.dumps({
            "version": 1,
            "mode": "strict",
            "model": "",
            "checksum": checksum,
            "rules": rules_data,
        }))

        result = load_affinity(str(tmp_path))
        assert result is not None
        assert len(result.items) == 1

    def test_save_cleans_up_on_error(self, tmp_path: Path) -> None:
        """Temp file is cleaned up if os.replace fails."""
        rule = _make_rule("test")
        idx = build_strict_affinity([rule])

        with (
            patch("cuecard.retrieval.affinity.os.replace", side_effect=OSError("fail")),
            pytest.raises(OSError, match="fail"),
        ):
            save_affinity(idx, str(tmp_path))

        # No .tmp files left behind
        tmp_files = [
            f for f in tmp_path.iterdir() if f.suffix == ".tmp"
        ]
        assert len(tmp_files) == 0

    def test_creates_cache_dir(self, tmp_path: Path) -> None:
        cache_dir = str(tmp_path / "nested" / "dir")
        rule = _make_rule("test")
        idx = build_strict_affinity([rule])
        save_affinity(idx, cache_dir)
        assert os.path.exists(
            os.path.join(cache_dir, "affinity.json"),
        )

    def test_non_string_reasoning_in_json(
        self, tmp_path: Path,
    ) -> None:
        """Non-string reasoning in JSON defaults to empty string."""
        text_hash = _hash_rule_text("test")
        checksum = _compute_checksum(
            ((text_hash, RuleAffinity(
                events=frozenset(),
                tools=frozenset(),
                source="inferred",
            )),),
        )
        path = tmp_path / "affinity.json"
        path.write_text(json.dumps({
            "version": 1,
            "mode": "infer",
            "model": "local",
            "checksum": checksum,
            "rules": [{
                "text_hash": text_hash,
                "events": [],
                "tools": [],
                "source": "inferred",
                "explicit_events": [],
                "explicit_tools": [],
                "reasoning": 42,
            }],
        }))
        result = load_affinity(str(tmp_path))
        assert result is not None
        aff = result.get_by_hash(text_hash)
        assert aff is not None
        assert aff.reasoning == ""


# --- _compute_checksum ---


class TestComputeChecksum:
    def test_deterministic(self) -> None:
        items: tuple[tuple[str, RuleAffinity], ...] = (
            ("hash_a", RuleAffinity(
                events=frozenset(), tools=frozenset(),
                source="default",
            )),
            ("hash_b", RuleAffinity(
                events=frozenset(), tools=frozenset(),
                source="default",
            )),
        )
        c1 = _compute_checksum(items)
        c2 = _compute_checksum(items)
        assert c1 == c2

    def test_order_independent(self) -> None:
        """Checksum sorts hashes, so order doesn't matter."""
        ra = RuleAffinity(
            events=frozenset(), tools=frozenset(),
            source="default",
        )
        items_ab = (("aaa", ra), ("bbb", ra))
        items_ba = (("bbb", ra), ("aaa", ra))
        assert _compute_checksum(items_ab) == _compute_checksum(items_ba)

    def test_different_hashes_different_checksum(self) -> None:
        ra = RuleAffinity(
            events=frozenset(), tools=frozenset(),
            source="default",
        )
        c1 = _compute_checksum((("aaa", ra),))
        c2 = _compute_checksum((("bbb", ra),))
        assert c1 != c2


# --- infer_affinities ---


class TestInferAffinities:
    def test_calls_llm_for_each_rule(self) -> None:
        rules = [
            _make_rule("Rule A"),
            _make_rule("Rule B"),
        ]
        config = _make_config()
        llm_response = json.dumps({
            "events": ["PreToolUse"],
            "tools": ["Bash"],
            "reasoning": "test reasoning",
        })

        with patch(
            "cuecard.retrieval.affinity.call_local", return_value=llm_response,
        ) as mock_call:
            idx = infer_affinities(rules, config)

        assert mock_call.call_count == 2
        assert idx.mode == "infer"
        assert len(idx.items) == 2

    def test_extends_explicit_annotations(self) -> None:
        rule = _make_rule(
            "Use uv",
            events=frozenset({"PreToolUse"}),
            tools=frozenset({"Bash"}),
        )
        config = _make_config()
        llm_response = json.dumps({
            "events": ["Stop"],
            "tools": ["Edit"],
            "reasoning": "",
        })

        with patch(
            "cuecard.retrieval.affinity.call_local", return_value=llm_response,
        ):
            idx = infer_affinities([rule], config)

        aff = idx.get(rule)
        assert aff is not None
        assert "PreToolUse" in aff.events
        assert "Stop" in aff.events
        assert "Bash" in aff.tools
        assert "Edit" in aff.tools
        assert aff.source == "explicit+inferred"

    def test_inferred_source_when_no_explicit(self) -> None:
        rule = _make_rule("No annotations")
        config = _make_config()
        llm_response = json.dumps({
            "events": ["UserPromptSubmit"],
            "tools": [],
            "reasoning": "",
        })

        with patch(
            "cuecard.retrieval.affinity.call_local", return_value=llm_response,
        ):
            idx = infer_affinities([rule], config)

        aff = idx.get(rule)
        assert aff is not None
        assert aff.source == "inferred"

    def test_fallback_on_endpoint_validation_failure(self) -> None:
        rules = [_make_rule("test")]
        config = _make_config()

        with patch(
            "cuecard.retrieval.affinity.validate_endpoint",
            side_effect=__import__(
                "cuecard.security", fromlist=["ConfigError"],
            ).ConfigError("bad endpoint"),
        ):
            idx = infer_affinities(rules, config)

        assert idx.mode == "strict-fallback"

    def test_fallback_on_llm_exception(self) -> None:
        rules = [_make_rule("test")]
        config = _make_config()

        with patch(
            "cuecard.retrieval.affinity.call_local",
            side_effect=RuntimeError("LLM down"),
        ):
            idx = infer_affinities(rules, config)

        assert idx.mode == "infer"
        aff = idx.get(rules[0])
        assert aff is not None
        # Falls back to strict for this rule
        assert aff.events == KNOWN_HOOK_EVENTS
        assert aff.source == "default"

    def test_nonce_in_prompt(self) -> None:
        rule = _make_rule("test")
        config = _make_config()

        prompts_captured: list[str] = []

        def capture_call(
            system: str, user: str, *args: object, **kwargs: object,
        ) -> str:
            prompts_captured.append(user)
            return json.dumps({
                "events": [],
                "tools": [],
                "reasoning": "",
            })

        with patch("cuecard.retrieval.affinity.call_local", side_effect=capture_call):
            infer_affinities([rule], config)

        assert len(prompts_captured) == 1
        assert "<rule_data_" in prompts_captured[0]

    def test_scrub_secrets_on_rule_text(self) -> None:
        rule = _make_rule("key=sk-ant-abcdefghijklmnopqrst")
        config = _make_config()

        prompts_captured: list[str] = []

        def capture_call(
            system: str, user: str, *args: object, **kwargs: object,
        ) -> str:
            prompts_captured.append(user)
            return json.dumps({
                "events": [],
                "tools": [],
                "reasoning": "",
            })

        with patch("cuecard.retrieval.affinity.call_local", side_effect=capture_call):
            infer_affinities([rule], config)

        assert "sk-ant-" not in prompts_captured[0]

    def test_haiku_backend(self) -> None:
        rules = [_make_rule("test")]
        config = _make_config(pipeline_mode="llm-haiku")
        llm_response = json.dumps({
            "events": ["PreToolUse"],
            "tools": [],
            "reasoning": "",
        })

        with patch(
            "cuecard.retrieval.affinity.call_haiku",
            return_value=llm_response,
        ) as mock_haiku:
            idx = infer_affinities(rules, config)

        assert mock_haiku.call_count == 1
        assert idx.mode == "infer"

    def test_config_error_reraises(self) -> None:
        from cuecard.security import ConfigError

        rules = [_make_rule("test")]
        config = _make_config()

        with (
            patch(
                "cuecard.retrieval.affinity.call_local",
                side_effect=ConfigError("bad"),
            ),
            pytest.raises(ConfigError),
        ):
            infer_affinities(rules, config)

    def test_value_error_reraises(self) -> None:
        rules = [_make_rule("test")]
        config = _make_config()

        with (
            patch(
                "cuecard.retrieval.affinity.call_local",
                side_effect=ValueError("bad"),
            ),
            pytest.raises(ValueError, match="bad"),
        ):
            infer_affinities(rules, config)

    def test_parallel_produces_correct_ordered_results(self) -> None:
        """max_workers > 1 uses ThreadPoolExecutor and preserves order."""
        rules = [_make_rule(f"rule {i}") for i in range(4)]
        config = _make_config()

        llm_response = '{"reasoning": "test", "category": "tool_use"}'

        with patch(
            "cuecard.retrieval.affinity.call_local",
            return_value=llm_response,
        ) as mock_call:
            idx = infer_affinities(rules, config, max_workers=3)

        assert mock_call.call_count == 4
        assert idx.mode == "infer"
        assert len(idx.items) == 4
        # Verify each rule is in the index with correct events
        for rule in rules:
            aff = idx.get(rule)
            assert aff is not None
            assert "PreToolUse" in aff.events
