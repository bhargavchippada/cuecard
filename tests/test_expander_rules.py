"""Tests for cuecard.expander — expand_rules function."""

from __future__ import annotations

import secrets
import socket
from unittest.mock import MagicMock, patch

import pytest

from cuecard.indexing.expander import (
    _VALID_EVENT_TYPES,
    _build_expansion_prompt,
    expand_rules,
)
from cuecard.models import Provenance, Rule
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
            expand_rules([_make_rule()], backend="openai")

    def test_haiku_model_not_in_allowlist_raises(self) -> None:
        with pytest.raises(ValueError, match="not in allowlist"):
            expand_rules(
                [_make_rule()],
                backend="haiku",
                haiku_model="gpt-4o",
            )

    def test_local_validates_endpoint(self) -> None:
        fake = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        with (
            patch("cuecard.retrieval.llm_utils.socket.getaddrinfo", return_value=fake),
            pytest.raises(ConfigError, match="loopback"),
        ):
            expand_rules(
                [_make_rule()],
                backend="local",
                endpoint="http://evil.com:8081/v1",
            )

    def test_dry_run_does_not_call_llm(self) -> None:
        rules = [_make_rule()]
        with patch("cuecard.indexing.expander.call_local") as mock_call:
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
            "cuecard.indexing.expander.call_local",
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
            "cuecard.indexing.expander.call_local",
            return_value=(
                '{"expansions": ["docker build image", "run pytest coverage"]}'
            ),
        ):
            result = expand_rules(
                rules,
                backend="local",
                endpoint="http://localhost:8081/v1",
            )

        assert len(result) == 1
        assert result[0].expansions == ("docker build image", "run pytest coverage")

    def test_haiku_backend_success(self) -> None:
        rules = [_make_rule()]

        with patch(
            "cuecard.indexing.expander.call_haiku",
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
            "cuecard.indexing.expander.call_local",
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
                "cuecard.indexing.expander.call_local",
                return_value='{"expansions": ["x"]}',
            ),
            patch(
                "cuecard.indexing.expander.secrets.token_hex",
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
        fake = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        with (
            patch("cuecard.retrieval.llm_utils.socket.getaddrinfo", return_value=fake),
            pytest.raises(ConfigError),
        ):
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
            "cuecard.indexing.expander.call_local",
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
                "cuecard.indexing.expander.call_local",
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
            "cuecard.indexing.expander.call_local",
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

    def test_invalid_event_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid event_type"):
            expand_rules(
                [_make_rule()],
                backend="local",
                endpoint="http://localhost:8081/v1",
                event_type="InvalidEvent",
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
                [_make_rule()],
                backend="local",
                endpoint="http://localhost:8081/v1",
                event_type="UserPromptSubmit",
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
                [_make_rule()],
                backend="local",
                endpoint="http://localhost:8081/v1",
            )
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
                rules,
                backend="local",
                endpoint="http://localhost:8081/v1",
                on_progress=updates.append,
            )

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
                rules,
                backend="local",
                endpoint="http://localhost:8081/v1",
                missing_only=True,
                on_progress=updates.append,
            )

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
                [_make_rule()],
                backend="local",
                endpoint="http://localhost:8081/v1",
            )
        assert len(result) == 1
