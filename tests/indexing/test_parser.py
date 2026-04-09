"""Tests for cuecard.parser."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

from cuecard.indexing.parser import parse_rules

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

    def test_max_rules_cap(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        f = tmp_path / "rules.txt"
        lines = "\n".join(f"Rule {i}" for i in range(510))
        f.write_text(lines + "\n")

        with caplog.at_level(logging.WARNING, logger="cuecard.parser"):
            rules = parse_rules((str(f),))

        assert len(rules) == 500
        assert "capped" in caplog.text


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


class TestParseJson:
    def test_valid_json_with_expansions(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.json"
        data = {
            "version": 1,
            "rules": [
                {
                    "text": "Never commit secrets",
                    "expansions": ["hardcoded API key", "AKIA in source"],
                    "source": {
                        "file": "/tmp/rules.txt",
                        "line_start": 1,
                        "line_end": 1,
                        "chunk_type": "rule",
                    },
                },
            ],
        }
        f.write_text(json.dumps(data))

        rules = parse_rules((str(f),))

        assert len(rules) == 1
        assert rules[0].text == "Never commit secrets"
        assert rules[0].expansions == ("hardcoded API key", "AKIA in source")
        # provenance.file uses the JSON file's own path (not embedded source.file)
        assert rules[0].provenance.file == str(f.resolve())
        assert rules[0].provenance.line_start == 1

    def test_empty_expansions(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.json"
        data = {
            "version": 1,
            "rules": [{"text": "A rule", "expansions": [], "source": {}}],
        }
        f.write_text(json.dumps(data))

        rules = parse_rules((str(f),))

        assert len(rules) == 1
        assert rules[0].expansions == ()

    def test_expansion_truncation(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        f = tmp_path / "rules.json"
        long_exp = "x" * 600
        data = {
            "version": 1,
            "rules": [
                {"text": "A rule", "expansions": [long_exp], "source": {}},
            ],
        }
        f.write_text(json.dumps(data))

        with caplog.at_level(logging.WARNING, logger="cuecard.parser"):
            rules = parse_rules((str(f),))

        assert len(rules[0].expansions) == 1
        assert len(rules[0].expansions[0]) == 500
        assert "exceeds 500 chars" in caplog.text

    def test_max_expansions_enforced(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        f = tmp_path / "rules.json"
        data = {
            "version": 1,
            "rules": [
                {
                    "text": "A rule",
                    "expansions": [f"exp{i}" for i in range(15)],
                    "source": {},
                },
            ],
        }
        f.write_text(json.dumps(data))

        with caplog.at_level(logging.WARNING, logger="cuecard.parser"):
            rules = parse_rules((str(f),))

        assert len(rules[0].expansions) == 10
        assert "dropping extras" in caplog.text

    def test_empty_and_nonstring_expansions_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.json"
        data = {
            "version": 1,
            "rules": [
                {
                    "text": "A rule",
                    "expansions": ["", "  ", None, 42, "valid"],
                    "source": {},
                },
            ],
        }
        f.write_text(json.dumps(data))

        rules = parse_rules((str(f),))

        assert rules[0].expansions == ("valid",)

    def test_empty_text_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.json"
        data = {
            "version": 1,
            "rules": [
                {"text": "", "expansions": [], "source": {}},
                {"text": "Real rule", "expansions": [], "source": {}},
            ],
        }
        f.write_text(json.dumps(data))

        rules = parse_rules((str(f),))

        assert len(rules) == 1
        assert rules[0].text == "Real rule"

    def test_wrong_version_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.json"
        data = {"version": 99, "rules": []}
        f.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="Unsupported rules.json version"):
            parse_rules((str(f),))

    def test_rule_text_truncated(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        f = tmp_path / "rules.json"
        data = {
            "version": 1,
            "rules": [
                {"text": "z" * 600, "expansions": [], "source": {}},
            ],
        }
        f.write_text(json.dumps(data))

        with caplog.at_level(logging.WARNING, logger="cuecard.parser"):
            rules = parse_rules((str(f),))

        assert len(rules[0].text) == 500

    def test_missing_source_uses_defaults(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.json"
        data = {"version": 1, "rules": [{"text": "A rule"}]}
        f.write_text(json.dumps(data))

        rules = parse_rules((str(f),))

        assert rules[0].provenance.file == str(f.resolve())
        assert rules[0].provenance.line_start == 0

    def test_max_rules_cap(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        f = tmp_path / "rules.json"
        data = {
            "version": 1,
            "rules": [
                {"text": f"Rule {i}", "expansions": [], "source": {}}
                for i in range(510)
            ],
        }
        f.write_text(json.dumps(data))

        with caplog.at_level(logging.WARNING, logger="cuecard.parser"):
            rules = parse_rules((str(f),))

        assert len(rules) == 500
        assert "capped at 500" in caplog.text


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


class TestParseToml:
    def test_single_rule(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.toml"
        f.write_text('[[rules]]\ntext = "Never commit secrets"\n')

        rules = parse_rules((str(f),))

        assert len(rules) == 1
        assert rules[0].text == "Never commit secrets"

    def test_events_and_tools(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.toml"
        f.write_text(
            '[[rules]]\n'
            'text = "Use uv"\n'
            'events = ["PreToolUse"]\n'
            'tools = ["Bash"]\n',
        )

        rules = parse_rules((str(f),))

        assert rules[0].events == frozenset({"PreToolUse"})
        assert rules[0].tools == frozenset({"Bash"})

    def test_empty_events_tools(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.toml"
        f.write_text('[[rules]]\ntext = "A rule"\n')

        rules = parse_rules((str(f),))

        assert rules[0].events == frozenset()
        assert rules[0].tools == frozenset()

    def test_multiple_rules(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.toml"
        f.write_text(
            '[[rules]]\ntext = "Rule one"\n\n'
            '[[rules]]\ntext = "Rule two"\n',
        )

        rules = parse_rules((str(f),))

        assert len(rules) == 2
        assert rules[0].text == "Rule one"
        assert rules[1].text == "Rule two"

    def test_empty_text_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.toml"
        f.write_text(
            '[[rules]]\ntext = ""\n\n'
            '[[rules]]\ntext = "Real rule"\n',
        )

        rules = parse_rules((str(f),))

        assert len(rules) == 1
        assert rules[0].text == "Real rule"

    def test_long_rule_truncated(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        f = tmp_path / "rules.toml"
        f.write_text(f'[[rules]]\ntext = "{"x" * 600}"\n')

        with caplog.at_level(logging.WARNING, logger="cuecard.parser"):
            rules = parse_rules((str(f),))

        assert len(rules[0].text) == 500
        assert "truncating" in caplog.text

    def test_unknown_event_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        f = tmp_path / "rules.toml"
        f.write_text(
            '[[rules]]\n'
            'text = "A rule"\n'
            'events = ["FakeEvent"]\n',
        )

        with caplog.at_level(logging.WARNING, logger="cuecard.parser"):
            rules = parse_rules((str(f),))

        assert len(rules) == 1
        assert "Unknown event" in caplog.text
        # Still stores the event (forward compat)
        assert "FakeEvent" in rules[0].events

    def test_malformed_toml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.toml"
        f.write_text("this is not valid toml {{{")

        with pytest.raises(ValueError, match="Malformed TOML"):
            parse_rules((str(f),))

    def test_chunk_type_is_toml_rule(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.toml"
        f.write_text('[[rules]]\ntext = "A rule"\n')

        rules = parse_rules((str(f),))

        assert rules[0].provenance.chunk_type == "toml_rule"

    def test_provenance_ordinal(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.toml"
        f.write_text(
            '[[rules]]\ntext = "First"\n\n'
            '[[rules]]\ntext = "Second"\n',
        )

        rules = parse_rules((str(f),))

        assert rules[0].provenance.line_start == 1
        assert rules[1].provenance.line_start == 2

    def test_max_rules_cap(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        f = tmp_path / "rules.toml"
        entries = "\n".join(
            f'[[rules]]\ntext = "Rule {i}"' for i in range(510)
        )
        f.write_text(entries)

        with caplog.at_level(logging.WARNING, logger="cuecard.parser"):
            rules = parse_rules((str(f),))

        assert len(rules) == 500
        assert "capped at 500" in caplog.text

    def test_no_rules_key(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.toml"
        f.write_text('[metadata]\ntitle = "my rules"\n')

        rules = parse_rules((str(f),))

        assert rules == []


class TestParseJsonV2:
    def test_v2_with_events_tools(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.json"
        data = {
            "version": 2,
            "rules": [
                {
                    "text": "Never commit secrets",
                    "expansions": [],
                    "events": ["PreToolUse", "PostToolUse"],
                    "tools": ["Bash"],
                    "source": {},
                },
            ],
        }
        f.write_text(json.dumps(data))

        rules = parse_rules((str(f),))

        assert len(rules) == 1
        assert rules[0].events == frozenset({"PreToolUse", "PostToolUse"})
        assert rules[0].tools == frozenset({"Bash"})

    def test_v2_empty_events_tools(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.json"
        data = {
            "version": 2,
            "rules": [
                {"text": "A rule", "expansions": [], "source": {}},
            ],
        }
        f.write_text(json.dumps(data))

        rules = parse_rules((str(f),))

        assert rules[0].events == frozenset()
        assert rules[0].tools == frozenset()

    def test_v1_defaults_empty_events_tools(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.json"
        data = {
            "version": 1,
            "rules": [
                {"text": "A rule", "expansions": [], "source": {}},
            ],
        }
        f.write_text(json.dumps(data))

        rules = parse_rules((str(f),))

        assert rules[0].events == frozenset()
        assert rules[0].tools == frozenset()
