"""Tests for LLM re-ranker prompt building."""

from __future__ import annotations

from cuecard.llm_reranker import _build_prompt
from cuecard.models import Provenance, RankedResult, Rule


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
        assert "When in doubt about CONCRETE rules" in system
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
