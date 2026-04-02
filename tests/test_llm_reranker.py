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

    def test_system_prompt_contains_matching_guidelines(self) -> None:
        candidates = _make_candidates(1)
        system, _ = _build_prompt(candidates, "test", "nonce1")
        assert "MATCHING GUIDELINES:" in system
        assert "DO match" in system
        assert "DO NOT match" in system

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
            # Parse returns None (unparseable), so fallback is used
            assert len(result) == 2


