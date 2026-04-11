"""Tests for cuecard.expander — prompt building and response parsing."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from cuecard.indexing.expander import (
    _build_expansion_prompt,
    _parse_expansion_response,
)
from cuecard.models import (
    MAX_EXPANSION_LENGTH,
    MAX_EXPANSIONS_PER_RULE,
)


class TestBuildExpansionPrompt:
    def test_contains_nonce_delimiters(self) -> None:
        system, user = _build_expansion_prompt("test rule", "abc123")
        assert "rule_data_abc123" in system
        assert "rule_data_abc123" in user

    def test_contains_reasoning_principles(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "REASONING PRINCIPLES:" in system
        assert "vocabulary gap" in system

    def test_contains_golden_examples_pretooluse(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "When adding new dependencies" in system
        assert "Always close file handles" in system
        assert "Run quality checks before every commit" in system

    def test_contains_golden_examples_workflow(self) -> None:
        system, _ = _build_expansion_prompt(
            "test rule", "nonce1", event_type="UserPromptSubmit",
        )
        assert "Classify every task as SIMPLE" in system
        assert "Save task state to artifacts/" in system
        assert "Update README when user-facing behavior changes" in system

    def test_workflow_prompt_has_user_message_language(self) -> None:
        system, _ = _build_expansion_prompt(
            "test rule", "nonce1", event_type="UserPromptSubmit",
        )
        assert "natural language messages" in system
        assert "USER MESSAGES" in system

    def test_pretooluse_prompt_has_tool_language(self) -> None:
        system, _ = _build_expansion_prompt(
            "test rule", "nonce1", event_type="PreToolUse",
        )
        assert "TOOL CALLS" in system
        assert "Use tool prefixes (Bash:, Edit:, Write:)" in system

    def test_anti_template_instruction(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "Use CONSISTENT vocabulary" in system
        assert "same concept should always produce" in system

    def test_trigger_direction_instruction(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "TRIGGER" in system
        assert "not the response" in system.lower()

    def test_indirect_trigger_instruction(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "indirect triggers" in system.lower()

    def test_variable_count_instruction(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "ABSTRACT (3-8)" in system
        assert "SPECIFIC (3-8)" in system

    def test_workflow_cross_domain_dont(self) -> None:
        system, _ = _build_expansion_prompt(
            "test rule", "nonce1", event_type="UserPromptSubmit",
        )
        assert "DOMAIN BOUNDARY" in system
        assert "USER MESSAGES" in system

    def test_pretooluse_cross_domain_dont(self) -> None:
        system, _ = _build_expansion_prompt(
            "test rule", "nonce1", event_type="PreToolUse",
        )
        assert "DOMAIN BOUNDARY" in system
        assert "TOOL CALLS" in system

    def test_workflow_bad_examples_show_false_positives(self) -> None:
        system, _ = _build_expansion_prompt(
            "test rule", "nonce1", event_type="UserPromptSubmit",
        )
        assert "Bad abstract tags" in system
        assert "too broad" in system or "paraphrase" in system

    def test_scrubs_secrets_from_rule(self) -> None:
        system, user = _build_expansion_prompt(
            "Use sk_live_abc123456789012345678901 key", "nonce1",
        )
        assert "sk_live_" not in user
        assert "[REDACTED]" in user

    def test_nonce_stripped_from_rule_text(self) -> None:
        nonce = "deadbeef1234"
        _, user = _build_expansion_prompt(
            f"Rule about {nonce} topic", nonce,
        )
        tag = f"<rule_data_{nonce}>"
        parts = user.split(tag)
        assert len(parts) >= 2
        content = parts[1].split(f"</rule_data_{nonce}>")[0]
        assert nonce not in content

    def test_user_prompt_has_json_format(self) -> None:
        _, user = _build_expansion_prompt("test", "nonce1")
        assert '"abstract"' in user
        assert '"specific"' in user
        assert '"reasoning"' in user
        assert "Return JSON" in user

    def test_reasoning_instruction_in_system(self) -> None:
        system, _ = _build_expansion_prompt("test", "nonce1")
        assert '"reasoning"' in system
        assert "vocabulary gap" in system

    def test_examples_contain_reasoning_field(self) -> None:
        system, _ = _build_expansion_prompt("test", "nonce1")
        assert '"reasoning":' in system
        # Both event types should have reasoning in examples
        system_wf, _ = _build_expansion_prompt(
            "test", "nonce1", event_type="UserPromptSubmit",
        )
        assert '"reasoning":' in system_wf


class TestParseExpansionResponse:
    def test_abstract_specific_format_balances_results(self) -> None:
        response = json.dumps({
            "abstract": ["abs 1", "abs 2", "abs 3"],
            "specific": ["spec 1", "spec 2", "spec 3"],
        })
        result = _parse_expansion_response(response, max_per_rule=5)
        assert result == ["abs 1", "abs 2", "abs 3", "spec 1", "spec 2"]

    def test_valid_json(self) -> None:
        response = '{"expansions": ["phrase 1", "phrase 2"]}'
        result = _parse_expansion_response(response)
        assert result == ["phrase 1", "phrase 2"]

    def test_json_in_code_block(self) -> None:
        response = '```json\n{"expansions": ["a", "b"]}\n```'
        result = _parse_expansion_response(response)
        assert result == ["a", "b"]

    def test_json_in_plain_code_block(self) -> None:
        response = '```\n{"expansions": ["a", "b"]}\n```'
        result = _parse_expansion_response(response)
        assert result == ["a", "b"]

    def test_strips_thinking_tags(self) -> None:
        response = '<think>hmm</think>{"expansions": ["a"]}'
        result = _parse_expansion_response(response)
        assert result == ["a"]

    def test_empty_expansions(self) -> None:
        response = '{"expansions": []}'
        result = _parse_expansion_response(response)
        assert result == []

    def test_invalid_json_returns_empty(self) -> None:
        result = _parse_expansion_response("not json at all")
        assert result == []

    def test_non_object_returns_empty(self) -> None:
        result = _parse_expansion_response("[1, 2, 3]")
        assert result == []

    def test_missing_expansions_key(self) -> None:
        result = _parse_expansion_response('{"data": []}')
        assert result == []

    def test_expansions_not_list(self) -> None:
        result = _parse_expansion_response('{"expansions": "string"}')
        assert result == []

    def test_non_string_items_filtered(self) -> None:
        response = '{"expansions": ["valid", 42, null, "also valid"]}'
        result = _parse_expansion_response(response)
        assert result == ["valid", "also valid"]

    def test_empty_strings_filtered(self) -> None:
        response = '{"expansions": ["", "   ", "valid"]}'
        result = _parse_expansion_response(response)
        assert result == ["valid"]

    def test_secrets_scrubbed(self) -> None:
        response = (
            '{"expansions": '
            '["use sk_live_abc123456789012345678901 key"]}'
        )
        result = _parse_expansion_response(response)
        assert len(result) == 1
        assert "sk_live_" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_max_length_enforced(self) -> None:
        long_text = "x" * 600
        response = json.dumps({"expansions": [long_text]})
        result = _parse_expansion_response(response)
        assert len(result) == 1
        assert len(result[0]) == MAX_EXPANSION_LENGTH

    def test_max_count_enforced(self) -> None:
        items = [f"phrase {i}" for i in range(20)]
        response = json.dumps({"expansions": items})
        result = _parse_expansion_response(response)
        assert len(result) == MAX_EXPANSIONS_PER_RULE

    def test_exact_text_dedup(self) -> None:
        response = '{"expansions": ["dup", "dup", "unique"]}'
        result = _parse_expansion_response(response)
        assert result == ["dup", "unique"]

    def test_whitespace_stripped(self) -> None:
        response = '{"expansions": ["  hello  ", "  world  "]}'
        result = _parse_expansion_response(response)
        assert result == ["hello", "world"]

    def test_reasoning_field_ignored_for_expansions(self) -> None:
        response = json.dumps({
            "reasoning": "This rule has a wide vocabulary gap.",
            "expansions": ["phrase 1", "phrase 2"],
        })
        result = _parse_expansion_response(response)
        assert result == ["phrase 1", "phrase 2"]

    def test_reasoning_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger="cuecard.indexing.expander"):
            response = json.dumps({
                "reasoning": "Wide gap between abstract rule and code.",
                "expansions": ["phrase 1"],
            })
            _parse_expansion_response(response)
        assert "Expansion reasoning:" in caplog.text
