"""Tests for cuecard.parser."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from cuecard.parser import parse_rules

if TYPE_CHECKING:
    from pathlib import Path


class TestParseTxtHappyPath:
    def test_single_rule(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.txt"
        f.write_text("Never commit secrets\n")

        rules = parse_rules((str(f),))

        assert len(rules) == 1
        assert rules[0].text == "Never commit secrets"

    def test_multiple_rules(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.txt"
        f.write_text("Rule one\nRule two\nRule three\n")

        rules = parse_rules((str(f),))

        assert len(rules) == 3
        assert rules[0].text == "Rule one"
        assert rules[2].text == "Rule three"


class TestParseTxtSkipping:
    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.txt"
        f.write_text("Rule one\n\n\nRule two\n")

        rules = parse_rules((str(f),))

        assert len(rules) == 2

    def test_comments_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.txt"
        f.write_text("# This is a comment\nActual rule\n# Another comment\n")

        rules = parse_rules((str(f),))

        assert len(rules) == 1
        assert rules[0].text == "Actual rule"

    def test_whitespace_only_lines_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.txt"
        f.write_text("  \n\t\nRule one\n")

        rules = parse_rules((str(f),))

        assert len(rules) == 1


class TestRuleTruncation:
    def test_long_rule_truncated(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        f = tmp_path / "rules.txt"
        long_text = "x" * 600
        f.write_text(long_text + "\n")

        with caplog.at_level(logging.WARNING, logger="cuecard.parser"):
            rules = parse_rules((str(f),))

        assert len(rules) == 1
        assert len(rules[0].text) == 500
        assert rules[0].text == "x" * 500
        assert "exceeds 500 chars" in caplog.text

    def test_exactly_500_not_truncated(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        f = tmp_path / "rules.txt"
        text = "y" * 500
        f.write_text(text + "\n")

        with caplog.at_level(logging.WARNING, logger="cuecard.parser"):
            rules = parse_rules((str(f),))

        assert len(rules) == 1
        assert len(rules[0].text) == 500
        assert "exceeds" not in caplog.text


class TestMultipleFiles:
    def test_rules_from_two_files(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("Rule from A\n")
        f2.write_text("Rule from B\n")

        rules = parse_rules((str(f1), str(f2)))

        assert len(rules) == 2
        assert rules[0].text == "Rule from A"
        assert rules[1].text == "Rule from B"


class TestUnsupportedFormats:
    def test_md_raises_not_implemented(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.md"
        f.write_text("# Heading\n")

        with pytest.raises(
            NotImplementedError,
            match="Markdown parsing not yet implemented",
        ):
            parse_rules((str(f),))

    def test_unsupported_suffix_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.yaml"
        f.write_text("key: value\n")

        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_rules((str(f),))


class TestEmptyFile:
    def test_empty_file_produces_empty_list(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")

        rules = parse_rules((str(f),))

        assert rules == []


class TestProvenance:
    def test_provenance_fields(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.txt"
        f.write_text("# comment\nFirst rule\n\nSecond rule\n")

        rules = parse_rules((str(f),))

        assert len(rules) == 2

        p0 = rules[0].provenance
        assert p0.file == str(f.resolve())
        assert p0.line_start == 2
        assert p0.line_end == 2
        assert p0.chunk_type == "rule"

        p1 = rules[1].provenance
        assert p1.line_start == 4
        assert p1.line_end == 4

    def test_provenance_file_is_resolved(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.txt"
        f.write_text("A rule\n")

        rules = parse_rules((str(f),))

        assert rules[0].provenance.file == str(f.resolve())
