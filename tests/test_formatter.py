"""Tests for cuecard.formatter."""

from __future__ import annotations

from cuecard.formatter import format_rules, format_rules_verbose
from cuecard.models import Provenance, RankedResult, Rule

_BOUNDARY_LABEL = "[cuecard \u2014 user-defined guidelines relevant to this action]"


def _make_result(
    text: str,
    score: float,
    *,
    file: str = "rules.txt",
    line_start: int = 1,
    line_end: int = 1,
    summary: str | None = None,
) -> RankedResult:
    prov = Provenance(file=file, line_start=line_start, line_end=line_end)
    rule = Rule(text=text, provenance=prov, summary=summary)
    return RankedResult(rule=rule, score=score)


class TestFormatRules:
    def test_single_result(self) -> None:
        results = [_make_result("Never commit secrets to git", 0.87)]
        output = format_rules(results)
        assert output == (
            f"{_BOUNDARY_LABEL}\n"
            "- Never commit secrets to git"
        )

    def test_multiple_results(self) -> None:
        results = [
            _make_result("Never commit secrets to git", 0.87),
            _make_result("Run quality checks before every commit", 0.72),
            _make_result("Use frozen dataclasses for immutable data", 0.65),
        ]
        output = format_rules(results)
        lines = output.split("\n")
        assert len(lines) == 4
        assert lines[0] == _BOUNDARY_LABEL
        assert lines[1] == "- Never commit secrets to git"
        assert lines[2] == "- Run quality checks before every commit"
        assert lines[3] == "- Use frozen dataclasses for immutable data"

    def test_empty_results_returns_empty_string(self) -> None:
        assert format_rules([]) == ""

    def test_uses_summary_when_available(self) -> None:
        results = [
            _make_result(
                "Very long rule text that goes on and on",
                0.90,
                summary="Short summary",
            ),
        ]
        output = format_rules(results)
        assert "Short summary" in output
        assert "Very long rule text" not in output

    def test_uses_text_when_summary_is_none(self) -> None:
        results = [_make_result("Actual rule text", 0.80, summary=None)]
        output = format_rules(results)
        assert "Actual rule text" in output

    def test_boundary_label_present(self) -> None:
        results = [_make_result("Any rule", 0.50)]
        output = format_rules(results)
        assert output.startswith(_BOUNDARY_LABEL)

    def test_no_scores_in_output(self) -> None:
        results = [_make_result("Rule text", 0.87)]
        output = format_rules(results)
        assert "0.87" not in output

    def test_no_provenance_in_output(self) -> None:
        results = [
            _make_result("Rule text", 0.87, file="/path/to/rules.txt", line_start=5),
        ]
        output = format_rules(results)
        assert "rules.txt" not in output
        assert ":5" not in output

    def test_no_trailing_newline(self) -> None:
        results = [_make_result("Rule text", 0.50)]
        output = format_rules(results)
        assert not output.endswith("\n")


class TestFormatRulesVerbose:
    def test_single_result_with_score_and_provenance(self) -> None:
        results = [
            _make_result("Never commit secrets to git", 0.87, file="/tmp/rules.txt"),
        ]
        output = format_rules_verbose(results)
        assert "[1] (0.87) Never commit secrets to git" in output
        assert "\u2514\u2500 rules.txt:1" in output

    def test_multiple_results_numbered_correctly(self) -> None:
        results = [
            _make_result("First rule", 0.90, file="a.txt", line_start=1),
            _make_result("Second rule", 0.75, file="b.txt", line_start=10),
            _make_result("Third rule", 0.60, file="c.txt", line_start=20),
        ]
        output = format_rules_verbose(results)
        assert "[1] (0.90) First rule" in output
        assert "[2] (0.75) Second rule" in output
        assert "[3] (0.60) Third rule" in output

    def test_empty_results(self) -> None:
        assert format_rules_verbose([]) == "No matching rules found."

    def test_provenance_shows_basename_not_full_path(self) -> None:
        results = [
            _make_result(
                "Rule text",
                0.80,
                file="/home/user/.claude/rules/security.md",
                line_start=5,
            ),
        ]
        output = format_rules_verbose(results)
        assert "security.md:5" in output
        assert "/home/user" not in output

    def test_multiline_provenance_shows_range(self) -> None:
        results = [
            _make_result("Rule text", 0.80, file="rules.md", line_start=5, line_end=8),
        ]
        output = format_rules_verbose(results)
        assert "rules.md:5-8" in output

    def test_single_line_provenance_no_range(self) -> None:
        results = [
            _make_result("Rule text", 0.80, file="rules.md", line_start=5, line_end=5),
        ]
        output = format_rules_verbose(results)
        assert "rules.md:5" in output
        assert "5-5" not in output

    def test_uses_summary_when_available(self) -> None:
        results = [
            _make_result("Long text", 0.80, summary="Short"),
        ]
        output = format_rules_verbose(results)
        assert "Short" in output
        assert "Long text" not in output

    def test_lines_joined_with_newline(self) -> None:
        """Lines must be joined with plain newline, not 'XX\\nXX'."""
        results = [
            _make_result("Rule A", 0.90, file="a.txt", line_start=1),
            _make_result("Rule B", 0.80, file="b.txt", line_start=2),
        ]
        output = format_rules_verbose(results)
        assert "XX" not in output
        # Verify proper newline separation
        lines = output.split("\n")
        assert len(lines) == 4  # 2 results × 2 lines each


class TestFormatRulesScrubDefault:
    """Kill mutant that changes scrub default from True to False."""

    def test_scrub_defaults_to_true(self) -> None:
        """format_rules() must scrub secrets by default (no explicit scrub=)."""
        results = [_make_result("key=AKIA1234567890ABCDEF", 0.80)]
        output = format_rules(results)
        assert "AKIA1234567890ABCDEF" not in output
        assert "[REDACTED]" in output
