"""Tests for query-side expansion."""

from __future__ import annotations

from unittest.mock import patch

import httpx

from cuecard.retrieval.query_expander import expand_query


class TestExpandQuery:
    def test_empty_query_short_circuits(self) -> None:
        with patch("cuecard.retrieval.query_expander.call_local") as mock_call:
            result = expand_query("   ", endpoint="http://localhost:8081/v1")
        assert result == ()
        mock_call.assert_not_called()

    def test_forwards_timeout_and_parse_limit(self) -> None:
        with (
            patch(
                "cuecard.retrieval.query_expander._build_expansion_prompt",
                return_value=("system", "user"),
            ) as mock_prompt,
            patch(
                "cuecard.retrieval.query_expander.call_local",
                return_value='{"abstract":["a"],"specific":["b"]}',
            ) as mock_call,
            patch(
                "cuecard.retrieval.query_expander._parse_expansion_response",
                return_value=["a", "b"],
            ) as mock_parse,
        ):
            result = expand_query(
                "Bash: cargo add serde",
                endpoint="http://localhost:8081/v1",
                event="Stop",
                max_per_query=3,
                timeout=1.25,
            )

        assert result == ("a", "b")
        mock_prompt.assert_called_once()
        assert mock_prompt.call_args.kwargs["event_type"] == "Stop"
        assert mock_call.call_args.kwargs["timeout"] == 1.25
        assert mock_call.call_args.kwargs["temperature"] == 0.7
        assert mock_parse.call_args.kwargs["max_per_rule"] == 3

    def test_timeout_falls_back_to_empty(self) -> None:
        with patch(
            "cuecard.retrieval.query_expander.call_local",
            side_effect=httpx.TimeoutException("timed out"),
        ):
            result = expand_query(
                "Bash: git commit",
                endpoint="http://localhost:8081/v1",
            )

        assert result == ()
