"""Tests for LLM expansion generation."""

from __future__ import annotations

import json
import secrets
from unittest.mock import MagicMock, patch

import pytest

from cuecard.expander import (
    _build_expansion_prompt,
    _parse_expansion_response,
    expand_rules,
)
from cuecard.models import (
    MAX_EXPANSION_LENGTH,
    MAX_EXPANSIONS_PER_RULE,
    Provenance,
    Rule,
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


class TestBuildExpansionPrompt:
    def test_contains_nonce_delimiters(self) -> None:
        system, user = _build_expansion_prompt("test rule", "abc123")
        assert "rule_data_abc123" in system
        assert "rule_data_abc123" in user

    def test_contains_do_dont_guidelines(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "DO:" in system
        assert "DON'T:" in system

    def test_contains_golden_examples(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "Always close file handles" in system
        assert "Run quality checks before every commit" in system
        assert "Never trust small sample benchmark results" in system

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
        assert '"expansions"' in user
        assert "8-10 retrieval expansion" in user


class TestParseExpansionResponse:
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
        long_text = "x" * 300
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


class TestExpandRules:
    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid backend"):
            expand_rules([_make_rule()], backend="openai")

    def test_haiku_model_not_in_allowlist_raises(self) -> None:
        with pytest.raises(ValueError, match="not in allowlist"):
            expand_rules(
                [_make_rule()],
                backend="haiku",
                haiku_model="gpt-4o",
            )

    def test_local_validates_endpoint(self) -> None:
        with pytest.raises(ConfigError, match="localhost"):
            expand_rules(
                [_make_rule()],
                backend="local",
                endpoint="http://evil.com:8081/v1",
            )

    def test_dry_run_does_not_call_llm(self) -> None:
        rules = [_make_rule()]
        with patch("cuecard.expander.call_local") as mock_call:
            result = expand_rules(
                rules,
                backend="local",
                endpoint="http://localhost:8081/v1",
                dry_run=True,
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
            "cuecard.expander.call_local",
            return_value='{"expansions": ["new one"]}',
        ):
            result = expand_rules(
                rules,
                backend="local",
                endpoint="http://localhost:8081/v1",
                missing_only=True,
            )

        assert result[0].expansions == ("existing",)
        assert result[1].expansions == ("new one",)

    def test_local_backend_success(self) -> None:
        rules = [_make_rule()]

        with patch(
            "cuecard.expander.call_local",
            return_value='{"expansions": ["phrase a", "phrase b"]}',
        ):
            result = expand_rules(
                rules,
                backend="local",
                endpoint="http://localhost:8081/v1",
            )

        assert len(result) == 1
        assert result[0].expansions == ("phrase a", "phrase b")

    def test_haiku_backend_success(self) -> None:
        rules = [_make_rule()]

        with patch(
            "cuecard.expander.call_haiku",
            return_value='{"expansions": ["haiku phrase"]}',
        ):
            result = expand_rules(
                rules, backend="haiku",
            )

        assert len(result) == 1
        assert result[0].expansions == ("haiku phrase",)

    def test_llm_failure_keeps_original(self) -> None:
        rules = [_make_rule()]

        with patch(
            "cuecard.expander.call_local",
            side_effect=RuntimeError("connection failed"),
        ):
            result = expand_rules(
                rules,
                backend="local",
                endpoint="http://localhost:8081/v1",
            )

        assert len(result) == 1
        assert result[0].expansions == ()

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
                "cuecard.expander.call_local",
                return_value='{"expansions": ["x"]}',
            ),
            patch(
                "cuecard.expander.secrets.token_hex",
                side_effect=capture_nonce,
            ),
        ):
            expand_rules(
                rules,
                backend="local",
                endpoint="http://localhost:8081/v1",
            )

        assert len(nonces) == 2
        assert nonces[0] != nonces[1]

    def test_empty_rules_list(self) -> None:
        result = expand_rules(
            [], backend="local", endpoint="http://localhost:8081/v1",
        )
        assert result == []

    def test_config_error_propagates(self) -> None:
        """ConfigError from validate_endpoint should propagate, not be caught."""
        with pytest.raises(ConfigError):
            expand_rules(
                [_make_rule()],
                backend="local",
                endpoint="http://evil.com/v1",
            )

    def test_value_error_propagates(self) -> None:
        """ValueError for invalid backend should propagate."""
        with pytest.raises(ValueError):
            expand_rules([_make_rule()], backend="invalid")

    def test_replaces_existing_expansions(self) -> None:
        """Without missing_only, existing expansions are overwritten."""
        rules = [_make_rule(expansions=("old",))]

        with patch(
            "cuecard.expander.call_local",
            return_value='{"expansions": ["new"]}',
        ):
            result = expand_rules(
                rules,
                backend="local",
                endpoint="http://localhost:8081/v1",
            )

        assert result[0].expansions == ("new",)

    def test_value_error_from_llm_call_propagates(self) -> None:
        """ValueError raised during LLM call propagates through loop."""
        rules = [_make_rule()]

        with (
            patch(
                "cuecard.expander.call_local",
                side_effect=ValueError("bad response format"),
            ),
            pytest.raises(ValueError, match="bad response format"),
        ):
            expand_rules(
                    rules,
                    backend="local",
                    endpoint="http://localhost:8081/v1",
                )

    def test_returns_new_rule_objects(self) -> None:
        """expand_rules returns new Rule objects, not mutated originals."""
        original = _make_rule()

        with patch(
            "cuecard.expander.call_local",
            return_value='{"expansions": ["x"]}',
        ):
            result = expand_rules(
                [original],
                backend="local",
                endpoint="http://localhost:8081/v1",
            )

        assert result[0] is not original
        assert original.expansions == ()
        assert result[0].expansions == ("x",)
