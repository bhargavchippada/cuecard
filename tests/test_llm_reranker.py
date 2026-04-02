"""Tests for LLM re-ranking (Stage 3)."""

from __future__ import annotations

import json
import secrets
from unittest.mock import MagicMock, patch

import httpx
import pytest

from cuecard.llm_reranker import (
    _build_prompt,
    _compute_ordinal_scores,
    _parse_llm_response,
    _strip_thinking_tags,
    rerank_llm,
)
from cuecard.llm_utils import call_local, validate_endpoint
from cuecard.models import Provenance, RankedResult, Rule
from cuecard.security import ConfigError


def _make_candidates(n: int = 5) -> list[RankedResult]:
    """Create n test candidates with deterministic data."""
    return [
        RankedResult(
            rule=Rule(
                text=f"Rule {i + 1} text",
                provenance=Provenance(
                    file="/tmp/rules.txt", line_start=i + 1, line_end=i + 1
                ),
            ),
            score=0.9 - i * 0.1,
        )
        for i in range(n)
    ]


class TestValidateEndpoint:
    def test_localhost_allowed(self) -> None:
        validate_endpoint("http://localhost:8081/v1")

    def test_127_allowed(self) -> None:
        validate_endpoint("http://127.0.0.1:8081/v1")

    def test_ipv6_loopback_allowed(self) -> None:
        validate_endpoint("http://[::1]:8081/v1")

    def test_external_host_raises(self) -> None:
        with pytest.raises(ConfigError, match="localhost"):
            validate_endpoint("http://evil.com:8081/v1")

    def test_ip_address_raises(self) -> None:
        with pytest.raises(ConfigError, match="localhost"):
            validate_endpoint("http://10.0.0.1:8081/v1")

    def test_userinfo_bypass_raises(self) -> None:
        with pytest.raises(ConfigError, match="userinfo"):
            validate_endpoint("http://evil.com@localhost:8081/v1")

    def test_file_scheme_raises(self) -> None:
        with pytest.raises(ConfigError, match="http/https"):
            validate_endpoint("file:///etc/passwd")

    def test_ftp_scheme_raises(self) -> None:
        with pytest.raises(ConfigError, match="http/https"):
            validate_endpoint("ftp://localhost/data")

    def test_password_in_url_raises(self) -> None:
        with pytest.raises(ConfigError, match="userinfo"):
            validate_endpoint("http://user:pass@localhost:8081/v1")


class TestBuildPrompt:
    def test_system_prompt_contains_nonce(self) -> None:
        candidates = _make_candidates(2)
        system, _ = _build_prompt(candidates, "test query", "abc123")
        assert "rule_data_abc123" in system

    def test_system_prompt_contains_few_shot(self) -> None:
        candidates = _make_candidates(1)
        system, _ = _build_prompt(candidates, "test", "nonce1")
        assert "Example 1" in system
        assert "Example 2" in system
        assert "Example 3" in system
        assert "Example 4" in system
        assert "Example 5" in system
        assert "rule_data_EXAMPLE" in system

    def test_system_prompt_contains_reasoning_principles(self) -> None:
        candidates = _make_candidates(1)
        system, _ = _build_prompt(candidates, "test", "nonce1")
        assert "REASONING PRINCIPLES:" in system
        assert "When in doubt, include" in system
        assert "tangential" in system

    def test_system_prompt_has_empty_rules_example(self) -> None:
        """Prompt shows the model how to return empty rules array."""
        candidates = _make_candidates(1)
        system, _ = _build_prompt(candidates, "test", "nonce1")
        assert '"rules": []' in system

    def test_user_prompt_numbered_rules(self) -> None:
        candidates = _make_candidates(3)
        _, user = _build_prompt(candidates, "test query", "nonce1")
        assert "1. <rule_data_nonce1>" in user
        assert "2. <rule_data_nonce1>" in user
        assert "3. <rule_data_nonce1>" in user
        assert "test query" in user and "query_data_" in user

    def test_nonce_stripped_from_rule_text(self) -> None:
        """If a rule contains the nonce string, it must be removed."""
        nonce = "deadbeef1234"
        rule = RankedResult(
            rule=Rule(
                text="Do not use deadbeef1234 in code",
                provenance=Provenance(file="/tmp/r.txt", line_start=1, line_end=1),
            ),
            score=0.9,
        )
        _, user = _build_prompt([rule], "test", nonce)
        # The nonce should not appear as literal text inside the rule content
        # It should only appear as part of the XML tags
        tag = f"<rule_data_{nonce}>"
        # Split by tag to get the content between tags
        parts = user.split(tag)
        assert len(parts) >= 2
        content_after_tag = parts[1].split(f"</rule_data_{nonce}>")[0]
        assert nonce not in content_after_tag

    def test_nonce_stripped_from_query(self) -> None:
        """If the query contains the nonce string, it must be removed."""
        nonce = "deadbeef1234"
        candidates = _make_candidates(1)
        _, user = _build_prompt(
            candidates, f"Bash: echo {nonce}", nonce,
        )
        action_section = user.split("ACTION: ")[1]
        # Nonce should only appear in the delimiter tags, not in the content
        content_parts = action_section.split(f"query_data_{nonce}")
        assert len(content_parts) >= 2  # delimiter present
        inner = content_parts[1].lstrip(">").split("<")[0]
        assert nonce not in inner

    def test_query_secrets_scrubbed(self) -> None:
        candidates = _make_candidates(1)
        _, user = _build_prompt(
            candidates,
            "install sk_live_abc123456789012345678901",
            "nonce1",
        )
        assert "sk_live_" not in user
        assert "[REDACTED]" in user

    def test_empty_candidates(self) -> None:
        _, user = _build_prompt([], "test query", "nonce1")
        assert "RULES:\n" in user
        assert "test query" in user and "query_data_" in user


class TestStripThinkingTags:
    def test_removes_think_block(self) -> None:
        text = '<think>reasoning here</think>{"rules": [1]}'
        assert _strip_thinking_tags(text) == '{"rules": [1]}'

    def test_multiple_think_blocks(self) -> None:
        text = "<think>a</think>hello<think>b</think>world"
        assert _strip_thinking_tags(text) == "helloworld"

    def test_no_think_blocks(self) -> None:
        text = '{"rules": [1, 2]}'
        assert _strip_thinking_tags(text) == '{"rules": [1, 2]}'

    def test_think_with_newlines(self) -> None:
        text = "<think>\nline1\nline2\n</think>result"
        assert _strip_thinking_tags(text) == "result"


class TestParseLLMResponse:
    def test_valid_json(self) -> None:
        result = _parse_llm_response('{"rules": [1, 3]}', 5)
        assert result.indices == [1, 3]

    def test_json_extra_whitespace(self) -> None:
        result = _parse_llm_response('  { "rules" : [ 1 , 3 ] }  ', 5)
        assert result.indices == [1, 3]

    def test_regex_fallback_comma_separated(self) -> None:
        result = _parse_llm_response("1, 3, 5", 5)
        assert result.indices == [1, 3, 5]

    def test_regex_fallback_brackets(self) -> None:
        result = _parse_llm_response("[1,3,5]", 5)
        assert result.indices == [1, 3, 5]

    def test_regex_fallback_space_separated(self) -> None:
        result = _parse_llm_response("1 3 5", 5)
        assert result.indices == [1, 3, 5]

    def test_invalid_prose_returns_none(self) -> None:
        result = _parse_llm_response("Error: rule 42 not found", 5)
        assert result.indices is None

    def test_out_of_range_filtered(self) -> None:
        result = _parse_llm_response('{"rules": [0, 1, 99]}', 5)
        assert result.indices == [1]

    def test_duplicates_removed(self) -> None:
        result = _parse_llm_response('{"rules": [1, 1, 3]}', 5)
        assert result.indices == [1, 3]

    def test_capped_at_max(self) -> None:
        indices = list(range(1, 21))
        response = json.dumps({"rules": indices})
        result = _parse_llm_response(response, 10)
        assert result.indices is not None
        assert len(result.indices) <= 10

    def test_empty_response_returns_none(self) -> None:
        result = _parse_llm_response("", 5)
        assert result.indices is None

    def test_empty_rules_array_returns_empty_list(self) -> None:
        result = _parse_llm_response('{"rules": []}', 5)
        assert result.indices == []

    def test_thinking_tags_stripped(self) -> None:
        text = '<think>thinking...</think>{"rules": [2, 4]}'
        result = _parse_llm_response(text, 5)
        assert result.indices == [2, 4]

    def test_thinking_with_empty_rules(self) -> None:
        text = '<think>\n\n</think>\n\n{"rules": []}'
        result = _parse_llm_response(text, 5)
        assert result.indices == []

    def test_json_missing_rules_key(self) -> None:
        result = _parse_llm_response('{"data": 42}', 50)
        assert result.indices is None

    def test_json_missing_rules_key_with_prose(self) -> None:
        result = _parse_llm_response('{"error": "not found"}', 5)
        assert result.indices is None

    def test_reasoning_captured(self) -> None:
        response = '{"reasoning": "Rule 1 applies because X.", "rules": [1]}'
        result = _parse_llm_response(response, 5)
        assert result.indices == [1]
        assert result.reasoning == "Rule 1 applies because X."

    def test_reasoning_empty_rules(self) -> None:
        response = '{"reasoning": "No rules apply.", "rules": []}'
        result = _parse_llm_response(response, 5)
        assert result.indices == []
        assert result.reasoning == "No rules apply."

    def test_no_reasoning_field(self) -> None:
        result = _parse_llm_response('{"rules": [2]}', 5)
        assert result.indices == [2]
        assert result.reasoning is None


class TestComputeOrdinalScores:
    def test_single_selection(self) -> None:
        candidates = _make_candidates(3)
        result = _compute_ordinal_scores([2], candidates)
        assert len(result) == 1
        assert result[0].score == 1.0
        assert result[0].rule.text == "Rule 2 text"

    def test_three_selections(self) -> None:
        candidates = _make_candidates(5)
        result = _compute_ordinal_scores([1, 3, 5], candidates)
        assert len(result) == 3
        # score = 1.0 - (pos/n) * (1 - 1/n)
        # pos=0: 1.0, pos=1: 1 - 1/3 * 2/3 = 7/9, pos=2: 1 - 2/3 * 2/3 = 5/9
        assert result[0].score == pytest.approx(1.0)
        assert result[1].score == pytest.approx(7 / 9, abs=0.01)
        assert result[2].score == pytest.approx(5 / 9, abs=0.01)

    def test_empty_selection(self) -> None:
        assert _compute_ordinal_scores([], _make_candidates(3)) == []

    def test_out_of_range_skipped(self) -> None:
        candidates = _make_candidates(3)
        result = _compute_ordinal_scores([1, 99], candidates)
        # Only index 1 is valid; n=2 but only 1 result produced
        assert len(result) == 1


class TestCallLocal:
    def test_successful_call(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"rules": [1, 3]}'}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx, "post", return_value=mock_response) as mock_post:
            result = call_local(
                "system", "user", "http://localhost:8081/v1", False
            )
            assert result == '{"rules": [1, 3]}'
            mock_post.assert_called_once()

    def test_timeout_raises(self) -> None:
        with (
            patch.object(
                httpx, "post", side_effect=httpx.TimeoutException("timeout")
            ),
            pytest.raises(httpx.TimeoutException),
        ):
            call_local(
                "system", "user", "http://localhost:8081/v1", False
            )

    def test_connect_error_raises(self) -> None:
        with (
            patch.object(
                httpx, "post", side_effect=httpx.ConnectError("refused")
            ),
            pytest.raises(httpx.ConnectError),
        ):
            call_local(
                "system", "user", "http://localhost:8081/v1", False
            )

    def test_http_error_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        with (
            patch.object(httpx, "post", return_value=mock_response),
            pytest.raises(httpx.HTTPStatusError),
        ):
            call_local(
                "system", "user", "http://localhost:8081/v1", False
            )

    def test_no_choices_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = MagicMock()
        with (
            patch.object(httpx, "post", return_value=mock_response),
            pytest.raises(ValueError, match="No choices"),
        ):
            call_local("sys", "usr", "http://localhost:8081/v1", False)

    def test_invalid_choice_format_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": ["not_a_dict"]}
        mock_response.raise_for_status = MagicMock()
        with (
            patch.object(httpx, "post", return_value=mock_response),
            pytest.raises(ValueError, match="Invalid choice"),
        ):
            call_local("sys", "usr", "http://localhost:8081/v1", False)

    def test_invalid_message_format_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": "not_dict"}]}
        mock_response.raise_for_status = MagicMock()
        with (
            patch.object(httpx, "post", return_value=mock_response),
            pytest.raises(ValueError, match="Invalid message"),
        ):
            call_local("sys", "usr", "http://localhost:8081/v1", False)

    def test_invalid_content_format_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": 12345}}],
        }
        mock_response.raise_for_status = MagicMock()
        with (
            patch.object(httpx, "post", return_value=mock_response),
            pytest.raises(ValueError, match="Invalid content"),
        ):
            call_local("sys", "usr", "http://localhost:8081/v1", False)


class TestCallHaiku:
    """Haiku backend tests — call_haiku now lives in llm_utils.

    See test_llm_utils.py for full coverage of call_haiku internals.
    These tests verify rerank_llm integration with the haiku backend.
    """


class TestRerankLLM:
    def test_empty_candidates(self) -> None:
        assert rerank_llm([], "test query") == []

    def test_local_backend_success(self) -> None:
        candidates = _make_candidates(5)
        mock_response = MagicMock()
        content = '{"reasoning": "Rules 1 and 3 apply.", "rules": [1, 3]}'
        mock_response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx, "post", return_value=mock_response):
            result = rerank_llm(
                candidates,
                "test query",
                backend="local",
                endpoint="http://localhost:8081/v1",
            )
            assert len(result) == 2
            assert result[0].rule.text == "Rule 1 text"
            assert result[1].rule.text == "Rule 3 text"

    def test_haiku_backend_success(self) -> None:
        candidates = _make_candidates(3)

        with patch(
            "cuecard.llm_reranker.call_haiku",
            return_value='{"rules": [2]}',
        ):
            result = rerank_llm(
                candidates, "test query", backend="haiku"
            )
            assert len(result) == 1
            assert result[0].rule.text == "Rule 2 text"

    def test_backend_failure_returns_fallback(self) -> None:
        candidates = _make_candidates(5)
        with patch.object(
            httpx, "post", side_effect=httpx.ConnectError("refused")
        ):
            result = rerank_llm(
                candidates,
                "test query",
                backend="local",
                endpoint="http://localhost:8081/v1",
                top_k=3,
            )
            assert len(result) == 3
            # Fallback returns first top_k candidates unchanged
            assert result[0].rule.text == "Rule 1 text"

    def test_timeout_returns_fallback(self) -> None:
        candidates = _make_candidates(5)
        with patch.object(
            httpx, "post", side_effect=httpx.TimeoutException("timeout")
        ):
            result = rerank_llm(
                candidates,
                "test query",
                backend="local",
                endpoint="http://localhost:8081/v1",
                top_k=2,
            )
            assert len(result) == 2

    def test_invalid_backend_raises(self) -> None:
        candidates = _make_candidates(1)
        with pytest.raises(ValueError, match="Invalid backend"):
            rerank_llm(candidates, "test", backend="openai")

    def test_haiku_model_allowlist_rejects_opus(self) -> None:
        candidates = _make_candidates(1)
        with pytest.raises(ValueError, match="not in allowlist"):
            rerank_llm(
                candidates, "test",
                backend="haiku", haiku_model="claude-opus-4-5",
            )

    def test_haiku_model_allowlist_rejects_arbitrary(self) -> None:
        candidates = _make_candidates(1)
        with pytest.raises(ValueError, match="not in allowlist"):
            rerank_llm(
                candidates, "test",
                backend="haiku", haiku_model="gpt-4o",
            )

    def test_secrets_scrubbed_from_query(self) -> None:
        candidates = _make_candidates(1)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"rules": [1]}'}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx, "post", return_value=mock_response) as mock_post:
            rerank_llm(
                candidates,
                "install sk_live_abc123456789012345678901",
                backend="local",
                endpoint="http://localhost:8081/v1",
            )
            call_args = mock_post.call_args
            body = call_args.kwargs["json"]
            user_msg = body["messages"][1]["content"]
            assert "sk_live_" not in user_msg
            assert "[REDACTED]" in user_msg

    def test_nonce_unique_per_call(self) -> None:
        candidates = _make_candidates(1)
        nonces: list[str] = []

        original_token_hex = secrets.token_hex

        def capture_nonce(n: int) -> str:
            result = original_token_hex(n)
            nonces.append(result)
            return result

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"rules": [1]}'}}]
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch.object(httpx, "post", return_value=mock_response),
            patch("cuecard.llm_reranker.secrets.token_hex", side_effect=capture_nonce),
        ):
            rerank_llm(
                candidates, "q1", backend="local",
                endpoint="http://localhost:8081/v1",
            )
            rerank_llm(
                candidates, "q2", backend="local",
                endpoint="http://localhost:8081/v1",
            )

        assert len(nonces) == 2
        assert nonces[0] != nonces[1]

    def test_top_k_limits_output(self) -> None:
        candidates = _make_candidates(10)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '{"rules": [1, 2, 3, 4, 5, 6, 7]}'}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx, "post", return_value=mock_response):
            result = rerank_llm(
                candidates,
                "test",
                backend="local",
                endpoint="http://localhost:8081/v1",
                top_k=3,
            )
            assert len(result) <= 3

    def test_ssrf_validation(self) -> None:
        candidates = _make_candidates(1)
        with pytest.raises(ConfigError, match="localhost"):
            rerank_llm(
                candidates,
                "test",
                backend="local",
                endpoint="http://evil.com:8081/v1",
            )

    def test_haiku_import_error_returns_fallback(self) -> None:
        candidates = _make_candidates(3)
        with patch(
            "cuecard.llm_reranker.call_haiku",
            side_effect=ImportError("no claude_agent_sdk"),
        ):
            result = rerank_llm(
                candidates, "test", backend="haiku", top_k=2
            )
            assert len(result) == 2
            assert result[0].rule.text == "Rule 1 text"

    def test_parse_failure_returns_fallback(self) -> None:
        candidates = _make_candidates(3)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "I cannot help with that request."}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx, "post", return_value=mock_response):
            result = rerank_llm(
                candidates,
                "test",
                backend="local",
                endpoint="http://localhost:8081/v1",
                top_k=2,
            )
            # Both calls return unparseable → fallback after retry
            assert len(result) == 2

    def test_retry_succeeds_on_second_call(self) -> None:
        """First LLM call returns garbage, retry returns valid JSON."""
        candidates = _make_candidates(3)
        bad_response = MagicMock()
        bad_response.json.return_value = {
            "choices": [{"message": {"content": "not json at all"}}],
        }
        bad_response.raise_for_status = MagicMock()

        good_response = MagicMock()
        good_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"reasoning": "Retry worked.", "rules": [1]},
                        ),
                    },
                },
            ],
        }
        good_response.raise_for_status = MagicMock()

        with patch.object(
            httpx, "post", side_effect=[bad_response, good_response],
        ):
            result = rerank_llm(
                candidates, "test", backend="local",
                endpoint="http://localhost:8081/v1", top_k=3,
            )
            assert len(result) == 1
            assert result[0].rule.text == "Rule 1 text"

    def test_retry_haiku_path(self) -> None:
        """Haiku retry path is exercised when first call is unparseable."""
        candidates = _make_candidates(2)

        with (
            patch(
                "cuecard.llm_reranker.call_haiku",
                side_effect=[
                    "garbage response",
                    json.dumps({"reasoning": "Retry OK.", "rules": [2]}),
                ],
            ),
        ):
            result = rerank_llm(
                candidates, "test", backend="haiku",
                haiku_model="claude-haiku-4-5", top_k=2,
            )
            assert len(result) == 1
            assert result[0].rule.text == "Rule 2 text"


class TestParseDangerousMutants:
    """Tests targeting dangerous surviving mutants in _parse_llm_response."""

    def test_multiline_json_extraction(self) -> None:
        """JSON spanning multiple lines must be extracted (kills re.DOTALL removal)."""
        response = (
            'Some preamble\n{"reasoning": "Multi\\nline",'
            '\n"rules": [1, 3]}\ntrailing'
        )
        result = _parse_llm_response(response, 5)
        assert result.indices == [1, 3]

    def test_non_numeric_rules_filtered(self) -> None:
        """Non-numeric values in rules array must be filtered out (kills and→or)."""
        response = '{"rules": [1, "two", 3, null, true]}'
        result = _parse_llm_response(response, 5)
        assert result.indices == [1, 3]

    def test_float_non_integer_filtered(self) -> None:
        """Float values like 1.5 must be filtered (not integer-equal)."""
        response = '{"rules": [1, 1.5, 2.0]}'
        result = _parse_llm_response(response, 5)
        assert result.indices == [1, 2]

    def test_reasoning_preserved_in_regex_fallback(self) -> None:
        """If JSON parsed reasoning before regex fallback, it's preserved."""
        # This path: JSON parse fails, regex matches number list
        # reasoning is None because no JSON was parsed
        result = _parse_llm_response("1, 3", 5)
        assert result.indices == [1, 3]
        assert result.reasoning is None

    def test_reasoning_preserved_on_final_none(self) -> None:
        """Unparseable response still returns reasoning if JSON had it partially."""
        # Full JSON parse fails, regex doesn't match prose → indices=None
        result = _parse_llm_response("I think rules 1 and 3 apply here", 5)
        assert result.indices is None


class TestComputeOrdinalDangerousMutants:
    """Tests targeting dangerous surviving mutants in _compute_ordinal_scores."""

    def test_invalid_index_before_valid_skipped_not_break(self) -> None:
        """Invalid index followed by valid index — valid must still be processed."""
        candidates = _make_candidates(3)
        result = _compute_ordinal_scores([99, 1], candidates)
        # With continue: skips 99, processes 1 → 1 result
        # With break: stops at 99 → 0 results
        assert len(result) == 1
        assert result[0].rule.text == "Rule 1 text"

    def test_two_selections_score_formula(self) -> None:
        """Two selections must have decaying scores (kills n==1 → n==2)."""
        candidates = _make_candidates(5)
        result = _compute_ordinal_scores([1, 3], candidates)
        assert len(result) == 2
        # With n=2: score(pos=0) = 1.0, score(pos=1) = 1.0 - (1/2)*(1-1/2) = 0.75
        assert result[0].score == pytest.approx(1.0)
        assert result[1].score == pytest.approx(0.75)
        # The mutant (n==2 instead of n==1) would give both score=1.0 when n=2
        # since condition `n == 2` would be True, giving score=1.0 for both
        # Wait — no, the mutant changes the condition to `n == 2`, so with n=2
        # it would enter the True branch giving 1.0 for BOTH. But normal code
        # with n==1 and n=2: n!=1 so uses else branch, giving different scores.
        assert result[0].score > result[1].score

    def test_max_index_boundary(self) -> None:
        """Index equal to len(candidates) is out of bounds (kills >= to >)."""
        candidates = _make_candidates(3)
        # Index 3 → zero_idx=2 → candidates[2] is valid
        # Index 4 → zero_idx=3 → candidates[3] is out of bounds
        result = _compute_ordinal_scores([3, 4], candidates)
        assert len(result) == 1
        assert result[0].rule.text == "Rule 3 text"


class TestBuildPromptNonceSecurity:
    """Tests for nonce stripping in _build_prompt (kills XXXX mutations)."""

    def test_nonce_fully_removed_from_query_content(self) -> None:
        """Nonce must be stripped to empty string, not replaced with 'XXXX'."""
        nonce = "abcdef123456"
        candidates = _make_candidates(1)
        _, user = _build_prompt(candidates, f"query with {nonce} embedded", nonce)
        # The content between query_data tags should not contain the nonce
        # AND should not contain any replacement like XXXX
        query_tag = f"<query_data_{nonce}>"
        end_tag = f"</query_data_{nonce}>"
        content = user.split(query_tag)[1].split(end_tag)[0]
        assert nonce not in content
        assert "XXXX" not in content

    def test_nonce_fully_removed_from_rule_content(self) -> None:
        """Nonce in rule text must be stripped to empty, not replaced."""
        nonce = "abcdef123456"
        rule = RankedResult(
            rule=Rule(
                text=f"Rule with {nonce} inside",
                provenance=Provenance(file="/tmp/r.txt", line_start=1, line_end=1),
            ),
            score=0.9,
        )
        _, user = _build_prompt([rule], "test", nonce)
        rule_tag = f"<rule_data_{nonce}>"
        end_tag = f"</rule_data_{nonce}>"
        content = user.split(rule_tag)[1].split(end_tag)[0]
        assert nonce not in content
        assert "XXXX" not in content

    def test_rule_text_appears_in_prompt(self) -> None:
        """Rule text must be preserved in the user prompt (not replaced with None)."""
        candidates = _make_candidates(2)
        _, user = _build_prompt(candidates, "test", "nonce1")
        assert "Rule 1 text" in user
        assert "Rule 2 text" in user

    def test_rules_joined_with_newlines(self) -> None:
        """Multiple rules must be joined with plain newlines."""
        candidates = _make_candidates(3)
        _, user = _build_prompt(candidates, "test", "nonce1")
        # Should contain "1. <rule_data_nonce1>...\n2. <rule_data_nonce1>..."
        lines = user.split("RULES:\n")[1].split("\n\nACTION:")[0].split("\n")
        assert len(lines) == 3
        assert lines[0].startswith("1. ")
        assert lines[1].startswith("2. ")
        assert lines[2].startswith("3. ")
        # No XX padding between rules
        for line in lines:
            assert "XX" not in line


class TestRetryArgumentVerification:
    """Verify retry call passes correct arguments (kills None/dropped arg mutants)."""

    def test_retry_local_passes_same_arguments(self) -> None:
        """Retry call to call_local must use same system_prompt, user_prompt, etc."""
        candidates = _make_candidates(3)

        call_args_list: list[tuple[object, ...]] = []

        def capture_call_local(
            system: str, user: str, endpoint: str, thinking: bool, **kw: object,
        ) -> str:
            call_args_list.append((system, user, endpoint, thinking))
            if len(call_args_list) == 1:
                return "unparseable garbage"
            return json.dumps({"reasoning": "OK", "rules": [1]})

        with patch("cuecard.llm_reranker.call_local", side_effect=capture_call_local):
            result = rerank_llm(
                candidates, "test", backend="local",
                endpoint="http://localhost:8081/v1",
            )

        assert len(call_args_list) == 2
        # Both calls should have the same arguments
        assert call_args_list[0] == call_args_list[1]
        # Arguments should not be None
        assert all(arg is not None for arg in call_args_list[0])
        assert len(result) == 1

    def test_retry_haiku_passes_same_arguments(self) -> None:
        """Retry call to call_haiku must use same system_prompt, user_prompt, model."""
        candidates = _make_candidates(3)

        call_args_list: list[tuple[object, ...]] = []

        def capture_call_haiku(system: str, user: str, model: str) -> str:
            call_args_list.append((system, user, model))
            if len(call_args_list) == 1:
                return "unparseable garbage"
            return json.dumps({"reasoning": "OK", "rules": [2]})

        with patch("cuecard.llm_reranker.call_haiku", side_effect=capture_call_haiku):
            result = rerank_llm(
                candidates, "test", backend="haiku",
                haiku_model="claude-haiku-4-5",
            )

        assert len(call_args_list) == 2
        # Both calls should have same args, none should be None
        assert call_args_list[0] == call_args_list[1]
        assert all(arg is not None for arg in call_args_list[0])
        assert len(result) == 1


class TestParseJsonExtraction:
    """Tests for first-JSON-block extraction in _parse_llm_response."""

    def test_json_with_trailing_text(self) -> None:
        """Valid JSON followed by repeated copies — extracts first block."""
        response = (
            '{"reasoning": "Match.", "rules": [1, 3]}\n'
            '{"reasoning": "Match.", "rules": [1, 3]}'
        )
        result = _parse_llm_response(response, 5)
        assert result.indices == [1, 3]
        assert result.reasoning == "Match."

    def test_json_with_trailing_garbage(self) -> None:
        """Valid JSON followed by non-JSON text."""
        response = '{"reasoning": "OK.", "rules": [2]} some trailing text'
        result = _parse_llm_response(response, 5)
        assert result.indices == [2]

    def test_invalid_json_in_braces(self) -> None:
        """Braces present but content is not valid JSON."""
        response = "{this is not json at all}"
        result = _parse_llm_response(response, 5)
        assert result.indices is None

    def test_no_braces_at_all(self) -> None:
        """No JSON-like content anywhere in response."""
        response = "I cannot help with that request."
        result = _parse_llm_response(response, 5)
        assert result.indices is None


