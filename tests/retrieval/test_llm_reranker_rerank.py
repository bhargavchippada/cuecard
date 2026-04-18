"""Tests for LLM re-ranker orchestration, ordinal scoring, and retry logic."""

from __future__ import annotations

import json
import secrets
import socket
from unittest.mock import MagicMock, patch

import httpx
import pytest

from cuecard.models import Provenance, RankedResult, Rule
from cuecard.retrieval.llm_reranker import (
    _compute_ordinal_scores,
    rerank_llm,
)
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


class TestComputeOrdinalDangerousMutants:
    """Tests targeting dangerous surviving mutants in _compute_ordinal_scores."""

    def test_invalid_index_before_valid_skipped_not_break(self) -> None:
        """Invalid index followed by valid index -- valid must still be processed."""
        candidates = _make_candidates(3)
        result = _compute_ordinal_scores([99, 1], candidates)
        # With continue: skips 99, processes 1 -> 1 result
        # With break: stops at 99 -> 0 results
        assert len(result) == 1
        assert result[0].rule.text == "Rule 1 text"

    def test_two_selections_score_formula(self) -> None:
        """Two selections must have decaying scores (kills n==1 -> n==2)."""
        candidates = _make_candidates(5)
        result = _compute_ordinal_scores([1, 3], candidates)
        assert len(result) == 2
        # With n=2: score(pos=0) = 1.0, score(pos=1) = 1.0 - (1/2)*(1-1/2) = 0.75
        assert result[0].score == pytest.approx(1.0)
        assert result[1].score == pytest.approx(0.75)
        # The mutant (n==2 instead of n==1) would give both score=1.0 when n=2
        # since condition `n == 2` would be True, giving 1.0 for BOTH. But normal code
        # with n==1 and n=2: n!=1 so uses else branch, giving different scores.
        assert result[0].score > result[1].score

    def test_max_index_boundary(self) -> None:
        """Index equal to len(candidates) is out of bounds (kills >= to >)."""
        candidates = _make_candidates(3)
        # Index 3 -> zero_idx=2 -> candidates[2] is valid
        # Index 4 -> zero_idx=3 -> candidates[3] is out of bounds
        result = _compute_ordinal_scores([3, 4], candidates)
        assert len(result) == 1
        assert result[0].rule.text == "Rule 3 text"


class TestRerankLLM:
    def test_empty_candidates(self) -> None:
        assert rerank_llm(
            [], "test query",
            backend="local", endpoint="http://localhost:8081/v1",
            haiku_model="claude-haiku-4-5", top_k=5,
            max_tokens=1024, timeout=60.0,
        ) == []

    def test_max_tokens_and_timeout_forwarded_to_httpx(self) -> None:
        """Config values must reach httpx.post — not hardcoded defaults."""
        candidates = _make_candidates(5)
        mock_response = MagicMock()
        content = '{"reasoning": "ok", "rules": [1]}'
        mock_response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx, "post", return_value=mock_response) as mock_post:
            rerank_llm(
                candidates,
                "test query",
                backend="local",
                endpoint="http://localhost:8081/v1",
                haiku_model="claude-haiku-4-5",
                top_k=5,
                max_tokens=2048,
                timeout=12.5,
            )
        kwargs = mock_post.call_args.kwargs
        assert kwargs["json"]["max_tokens"] == 2048
        assert kwargs["timeout"] == 12.5

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
                haiku_model="claude-haiku-4-5",
                top_k=5, max_tokens=1024, timeout=60.0)
            assert len(result) == 2
            assert result[0].rule.text == "Rule 1 text"
            assert result[1].rule.text == "Rule 3 text"

    def test_haiku_backend_success(self) -> None:
        candidates = _make_candidates(3)

        with patch(
            "cuecard.retrieval.llm_reranker.call_haiku",
            return_value='{"rules": [2]}',
        ):
            result = rerank_llm(
                candidates, "test query", backend="haiku",
                endpoint="http://localhost:8081/v1",
                haiku_model="claude-haiku-4-5", top_k=5, max_tokens=1024, timeout=60.0)
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
                haiku_model="claude-haiku-4-5",
                top_k=3, max_tokens=1024, timeout=60.0)
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
                haiku_model="claude-haiku-4-5",
                top_k=2, max_tokens=1024, timeout=60.0)
            assert len(result) == 2

    def test_invalid_backend_raises(self) -> None:
        candidates = _make_candidates(1)
        with pytest.raises(ValueError, match="Invalid backend"):
            rerank_llm(
                candidates, "test", backend="openai",
                endpoint="http://localhost:8081/v1",
                haiku_model="claude-haiku-4-5", top_k=5, max_tokens=1024, timeout=60.0)

    def test_haiku_model_allowlist_rejects_opus(self) -> None:
        candidates = _make_candidates(1)
        with pytest.raises(ValueError, match="not in allowlist"):
            rerank_llm(
                candidates, "test",
                backend="haiku", haiku_model="claude-opus-4-5",
                endpoint="http://localhost:8081/v1", top_k=5,
                    max_tokens=1024, timeout=60.0,
                )

    def test_haiku_model_allowlist_rejects_arbitrary(self) -> None:
        candidates = _make_candidates(1)
        with pytest.raises(ValueError, match="not in allowlist"):
            rerank_llm(
                candidates, "test",
                backend="haiku", haiku_model="gpt-4o",
                endpoint="http://localhost:8081/v1", top_k=5,
                    max_tokens=1024, timeout=60.0,
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
                haiku_model="claude-haiku-4-5",
                top_k=5, max_tokens=1024, timeout=60.0)
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
            patch(
                "cuecard.retrieval.llm_reranker.secrets.token_hex",
                side_effect=capture_nonce,
            ),
        ):
            rerank_llm(
                candidates, "q1", backend="local",
                endpoint="http://localhost:8081/v1",
                haiku_model="claude-haiku-4-5", top_k=5, max_tokens=1024, timeout=60.0)
            rerank_llm(
                candidates, "q2", backend="local",
                endpoint="http://localhost:8081/v1",
                haiku_model="claude-haiku-4-5", top_k=5, max_tokens=1024, timeout=60.0)

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
                haiku_model="claude-haiku-4-5",
                top_k=3, max_tokens=1024, timeout=60.0)
            assert len(result) <= 3

    def test_ssrf_validation(self) -> None:
        candidates = _make_candidates(1)
        fake = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        with (
            patch("cuecard.retrieval.llm_utils.socket.getaddrinfo", return_value=fake),
            pytest.raises(ConfigError, match="loopback"),
        ):
            rerank_llm(
                candidates,
                "test",
                backend="local",
                endpoint="http://evil.com:8081/v1",
                haiku_model="claude-haiku-4-5",
                top_k=5, max_tokens=1024, timeout=60.0)

    def test_haiku_import_error_returns_fallback(self) -> None:
        candidates = _make_candidates(3)
        with patch(
            "cuecard.retrieval.llm_reranker.call_haiku",
            side_effect=ImportError("no claude_agent_sdk"),
        ):
            result = rerank_llm(
                candidates, "test", backend="haiku", top_k=2,
                endpoint="http://localhost:8081/v1",
                haiku_model="claude-haiku-4-5", max_tokens=1024, timeout=60.0)
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
                haiku_model="claude-haiku-4-5",
                top_k=2, max_tokens=1024, timeout=60.0)
            # Both calls return unparseable -> fallback after retry
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
                endpoint="http://localhost:8081/v1",
                haiku_model="claude-haiku-4-5", top_k=3, max_tokens=1024, timeout=60.0)
            assert len(result) == 1
            assert result[0].rule.text == "Rule 1 text"

    def test_retry_haiku_path(self) -> None:
        """Haiku retry path is exercised when first call is unparseable."""
        candidates = _make_candidates(2)

        with (
            patch(
                "cuecard.retrieval.llm_reranker.call_haiku",
                side_effect=[
                    "garbage response",
                    json.dumps({"reasoning": "Retry OK.", "rules": [2]}),
                ],
            ),
        ):
            result = rerank_llm(
                candidates, "test", backend="haiku",
                endpoint="http://localhost:8081/v1",
                haiku_model="claude-haiku-4-5", top_k=2, max_tokens=1024, timeout=60.0)
            assert len(result) == 1
            assert result[0].rule.text == "Rule 2 text"


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

        with patch(
                "cuecard.retrieval.llm_reranker.call_local",
                side_effect=capture_call_local,
            ):
            result = rerank_llm(
                candidates, "test", backend="local",
                endpoint="http://localhost:8081/v1",
                haiku_model="claude-haiku-4-5", top_k=5, max_tokens=1024, timeout=60.0)

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

        with patch(
                "cuecard.retrieval.llm_reranker.call_haiku",
                side_effect=capture_call_haiku,
            ):
            result = rerank_llm(
                candidates, "test", backend="haiku",
                endpoint="http://localhost:8081/v1",
                haiku_model="claude-haiku-4-5", top_k=5, max_tokens=1024, timeout=60.0)

        assert len(call_args_list) == 2
        # Both calls should have same args, none should be None
        assert call_args_list[0] == call_args_list[1]
        assert all(arg is not None for arg in call_args_list[0])
        assert len(result) == 1
