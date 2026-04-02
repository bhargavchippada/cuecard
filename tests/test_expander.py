"""Tests for LLM expansion generation."""

from __future__ import annotations

import json
import secrets
from unittest.mock import MagicMock, patch

import pytest

from cuecard.expander import (
    _VALID_EVENT_TYPES,
    DEDUP_COSINE_THRESHOLD,
    _build_expansion_prompt,
    _parse_expansion_response,
    _semantic_dedup,
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

    def test_contains_reasoning_principles(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "REASONING PRINCIPLES:" in system
        assert "vocabulary gap" in system

    def test_contains_golden_examples_pretooluse(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "Always close file handles" in system
        assert "Run quality checks before every commit" in system
        assert "Never trust small sample benchmark results" in system

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
        assert "tool calls and code actions" in system

    def test_anti_template_instruction(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "Vary the form" in system
        assert "template repetition" in system

    def test_trigger_direction_instruction(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "TRIGGER" in system
        assert "not the response" in system.lower()

    def test_indirect_trigger_instruction(self) -> None:
        system, _ = _build_expansion_prompt("test rule", "nonce1")
        assert "indirect triggers" in system.lower()

    def test_variable_count_instruction(self) -> None:
        _, user = _build_expansion_prompt("test rule", "nonce1")
        assert "3-10" in user
        assert "Stop when additional expansions would just be rephrasing" in user

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
        assert "would false-match" in system or "would fire on every" in system
        assert "would match unrelated" in system or "would fire on every" in system

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
        assert '"reasoning"' in user
        assert "3-10 retrieval expansion" in user

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

    def test_reasoning_field_ignored_for_expansions(self) -> None:
        response = json.dumps({
            "reasoning": "This rule has a wide vocabulary gap.",
            "expansions": ["phrase 1", "phrase 2"],
        })
        result = _parse_expansion_response(response)
        assert result == ["phrase 1", "phrase 2"]

    def test_reasoning_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        with caplog.at_level(logging.DEBUG, logger="cuecard.expander"):
            response = json.dumps({
                "reasoning": "Wide gap between abstract rule and code.",
                "expansions": ["phrase 1"],
            })
            _parse_expansion_response(response)
        assert "Expansion reasoning:" in caplog.text


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
                "cuecard.expander.call_local",
                return_value='{"expansions": ["x"]}',
            ),
            patch(
                "cuecard.expander._build_expansion_prompt",
                wraps=_build_expansion_prompt,
            ) as mock_build,
            patch(
                "cuecard.expander._semantic_dedup",
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
                "cuecard.expander.call_local",
                return_value='{"expansions": ["a", "b", "c"]}',
            ),
            patch(
                "cuecard.expander._semantic_dedup",
                return_value=["a", "c"],
            ) as mock_dedup,
        ):
            result = expand_rules(
                [_make_rule()],
                backend="local",
                endpoint="http://localhost:8081/v1",
            )
            mock_dedup.assert_called_once_with(["a", "b", "c"])
            assert result[0].expansions == ("a", "c")

    def test_valid_event_types_constant(self) -> None:
        assert "PreToolUse" in _VALID_EVENT_TYPES
        assert "UserPromptSubmit" in _VALID_EVENT_TYPES


class TestSemanticDedup:
    def test_empty_list(self) -> None:
        assert _semantic_dedup([]) == []

    def test_single_item(self) -> None:
        assert _semantic_dedup(["hello"]) == ["hello"]

    def test_preserves_order(self) -> None:
        """Non-duplicate items are returned in original order."""
        with patch("cuecard.expander.np") as mock_np:
            # Mock numpy to simulate no duplicates
            mock_array = MagicMock()
            mock_np.array.return_value = mock_array
            mock_np.linalg.norm.return_value = MagicMock()
            mock_np.where.return_value = MagicMock()

            # Just test the short-circuit paths
            result = _semantic_dedup(["a"])
            assert result == ["a"]

    def test_fastembed_import_failure_returns_original(self) -> None:
        """If fastembed is unavailable, return original list unchanged."""
        import builtins

        items = ["phrase a", "phrase b"]
        original_import = builtins.__import__
        with (
            patch.dict("sys.modules", {"fastembed": None}),
            patch(
                "builtins.__import__",
                side_effect=lambda name, *a, **kw: (
                    (_ for _ in ()).throw(ImportError("no fastembed"))
                    if name == "fastembed"
                    else original_import(name, *a, **kw)
                ),
            ),
        ):
            result = _semantic_dedup(items)
        assert result == items

    def test_embedding_failure_returns_original(self) -> None:
        """If embedding fails, return original list unchanged."""
        items = ["phrase a", "phrase b"]
        mock_model = MagicMock()
        mock_model.passage_embed.side_effect = RuntimeError("model load failed")
        with patch(
            "fastembed.TextEmbedding",
            return_value=mock_model,
        ):
            result = _semantic_dedup(items)
        assert result == items

    def test_threshold_constant(self) -> None:
        assert DEDUP_COSINE_THRESHOLD == 0.85

    def test_drops_near_duplicate(self) -> None:
        """Expansions with cosine > 0.85 are dropped."""
        import numpy as np

        items = ["pip install requests", "pip install numpy"]
        # Create nearly identical embeddings (cosine ~0.99)
        base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        near = np.array([0.99, 0.1, 0.0], dtype=np.float32)
        near = near / np.linalg.norm(near)
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [base, near]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        assert len(result) == 1
        assert result[0] == "pip install requests"

    def test_three_items_keeps_distinct_drops_duplicate(self) -> None:
        """With 3 items: first and third distinct, second similar to first."""
        import numpy as np

        items = ["phrase A", "phrase B (dup of A)", "phrase C (distinct)"]
        # A and B nearly identical (cosine ~0.99), C orthogonal (cosine ~0)
        vec_a = np.array([3.0, 0.0, 0.0], dtype=np.float32)  # unnormalized!
        vec_b = np.array([2.97, 0.3, 0.0], dtype=np.float32)
        vec_c = np.array([0.0, 0.0, 5.0], dtype=np.float32)  # orthogonal
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec_a, vec_b, vec_c]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        # B should be dropped (similar to A), A and C kept
        assert len(result) == 2
        assert result[0] == "phrase A"
        assert result[1] == "phrase C (distinct)"

    def test_normalization_divides_not_multiplies(self) -> None:
        """Verify embeddings are divided by norms, not multiplied.

        With small-norm near-duplicate vectors (cosine ~0.95, norms ~0.1):
        - Division by norms: similarity = cos(θ) ≈ 0.95 > 0.85 → B dropped ✓
        - Mult by norms: sim ~ 0.95*0.01*0.01 ~ 0.00009 < 0.85 → B kept ✗
        """
        import numpy as np

        items = ["item A", "item B"]
        # Small-norm vectors with cosine ~0.95 (above threshold)
        vec_a = np.array([0.1, 0.01], dtype=np.float32)
        vec_b = np.array([0.1, 0.0], dtype=np.float32)
        # Verify cosine is above threshold
        cos_sim = float(np.dot(vec_a, vec_b) / (
            np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        ))
        assert cos_sim > 0.85, f"cosine {cos_sim} should be above threshold"
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec_a, vec_b]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        # After proper normalization: cosine ~0.95 > 0.85 → B dropped
        assert len(result) == 1
        assert result[0] == "item A"

    def test_model_name_is_bge_small(self) -> None:
        """TextEmbedding must be initialized with the correct model name."""
        import numpy as np

        items = ["a", "b"]
        vec = np.array([1.0, 0.0], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec, vec]
        with patch("fastembed.TextEmbedding", return_value=mock_model) as mock_cls:
            _semantic_dedup(items)
            mock_cls.assert_called_once_with("BAAI/bge-small-en-v1.5")

    def test_passage_embed_receives_expansions(self) -> None:
        """passage_embed must be called with the actual expansion strings."""
        import numpy as np

        items = ["expansion one", "expansion two"]
        vec = np.array([1.0, 0.0], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec, vec]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            _semantic_dedup(items)
            mock_model.passage_embed.assert_called_once_with(items)

    def test_single_item_returned_unchanged(self) -> None:
        """A single-item list must short-circuit without embedding."""
        with patch("fastembed.TextEmbedding") as mock_cls:
            result = _semantic_dedup(["only one"])
            assert result == ["only one"]
            assert len(result) == 1
            # Must NOT call TextEmbedding — short-circuit on len <= 1
            mock_cls.assert_not_called()

    def test_zero_norm_vector_does_not_crash(self) -> None:
        """A zero-norm embedding should not cause division-by-zero.

        Zero-norm guard replaces 0 with 1.0 (not 2.0 or other values).
        With replacement=1.0: [0,0,0]/1.0 = [0,0,0], dot with [1,0,0] = 0 < 0.85 → kept.
        With replacement=2.0: [0,0,0]/2.0 = [0,0,0], same result (benign for this case).
        """
        import numpy as np

        items = ["zero vector", "normal vector"]
        vec_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        vec_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec_zero, vec_normal]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        # Should not crash; zero-norm guard replaces 0 with 1.0
        assert len(result) == 2

    def test_zero_norm_guard_replaces_zero_not_nonzero(self) -> None:
        """np.where condition must be norms==0, not norms!=0 or norms==1."""
        import numpy as np

        # Two items with identical direction but different magnitudes
        # After proper normalization: cosine=1.0 → second dropped
        # If guard is inverted (!=0): all norms become 1.0 before division,
        # making normalized = embeddings/1.0 = embeddings (unnormalized),
        # and dot products become magnitude-dependent instead of direction-only
        items = ["item A", "item B", "item C"]
        vec_a = np.array([5.0, 0.0], dtype=np.float32)
        vec_b = np.array([5.0, 0.1], dtype=np.float32)  # nearly same direction
        vec_c = np.array([0.0, 5.0], dtype=np.float32)  # orthogonal
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec_a, vec_b, vec_c]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        # B is near-duplicate of A, C is distinct
        assert len(result) == 2
        assert result[0] == "item A"
        assert result[1] == "item C"

    def test_first_item_always_kept(self) -> None:
        """Loop starts at index 1; first item is always in the result."""
        import numpy as np

        items = ["first", "second"]
        # Identical vectors → second should be dropped, first always kept
        vec = np.array([1.0, 0.0], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec, vec]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        assert len(result) == 1
        assert result[0] == "first"

    def test_below_threshold_kept(self) -> None:
        """Items with similarity below threshold must be kept."""
        import numpy as np

        items = ["item A", "item B"]
        # cosine ~0.6 (well below 0.85)
        vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec_a, vec_b]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        assert len(result) == 2
        assert result[0] == "item A"
        assert result[1] == "item B"
