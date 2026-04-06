"""Tests for cuecard.indexer — rules.json save/load/merge."""

from __future__ import annotations

import json
import stat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from cuecard.indexer import (
    load_rules_json,
    merge_rules_json,
    save_rules_json,
)
from cuecard.models import Provenance, Rule


class TestSaveRulesJson:
    def test_round_trip(self, tmp_path: Path) -> None:
        rules = [
            Rule(
                text="Never commit secrets",
                provenance=Provenance(
                    file="/tmp/rules.txt", line_start=1, line_end=1,
                ),
                expansions=("hardcoded API key", "AKIA in source"),
            ),
            Rule(
                text="Use uv not pip",
                provenance=Provenance(
                    file="/tmp/rules.txt", line_start=2, line_end=2,
                ),
            ),
        ]

        save_rules_json(rules, str(tmp_path))
        loaded = load_rules_json(str(tmp_path))

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].text == "Never commit secrets"
        assert loaded[0].expansions == ("hardcoded API key", "AKIA in source")
        assert loaded[1].text == "Use uv not pip"
        assert loaded[1].expansions == ()

    def test_file_permissions(self, tmp_path: Path) -> None:
        rules = [
            Rule(
                text="A rule",
                provenance=Provenance(
                    file="/tmp/rules.txt", line_start=1, line_end=1,
                ),
            ),
        ]

        save_rules_json(rules, str(tmp_path))
        json_path = tmp_path / "rules.json"

        mode = stat.S_IMODE(json_path.stat().st_mode)
        assert mode == 0o600

    def test_atomic_write(self, tmp_path: Path) -> None:
        """Verify the file exists after save (no temp file left)."""
        rules = [
            Rule(
                text="A rule",
                provenance=Provenance(
                    file="/tmp/rules.txt", line_start=1, line_end=1,
                ),
            ),
        ]

        save_rules_json(rules, str(tmp_path))

        assert (tmp_path / "rules.json").exists()
        # No temp files should remain
        temp_files = [f for f in tmp_path.iterdir() if f.suffix != ".json"]
        assert len(temp_files) == 0


class TestLoadRulesJson:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        result = load_rules_json(str(tmp_path))
        assert result is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "rules.json").write_text("not json{{{")
        result = load_rules_json(str(tmp_path))
        assert result is None

    def test_wrong_version_returns_none(self, tmp_path: Path) -> None:
        data = {"version": 99, "rules": []}
        (tmp_path / "rules.json").write_text(json.dumps(data))
        result = load_rules_json(str(tmp_path))
        assert result is None

    def test_expansion_length_capped(self, tmp_path: Path) -> None:
        data = {
            "version": 1,
            "rules": [
                {
                    "text": "A rule",
                    "expansions": ["x" * 300],
                    "source": {"file": "/tmp/r.txt", "line_start": 1, "line_end": 1},
                },
            ],
        }
        (tmp_path / "rules.json").write_text(json.dumps(data))

        rules = load_rules_json(str(tmp_path))
        assert rules is not None
        assert len(rules[0].expansions[0]) == 200

    def test_expansion_count_capped(self, tmp_path: Path) -> None:
        data = {
            "version": 1,
            "rules": [
                {
                    "text": "A rule",
                    "expansions": [f"exp{i}" for i in range(15)],
                    "source": {"file": "/tmp/r.txt", "line_start": 1, "line_end": 1},
                },
            ],
        }
        (tmp_path / "rules.json").write_text(json.dumps(data))

        rules = load_rules_json(str(tmp_path))
        assert rules is not None
        assert len(rules[0].expansions) == 10

    def test_empty_text_skipped(self, tmp_path: Path) -> None:
        data = {
            "version": 1,
            "rules": [
                {"text": "", "expansions": [], "source": {}},
                {"text": "Real rule", "expansions": [], "source": {}},
            ],
        }
        (tmp_path / "rules.json").write_text(json.dumps(data))

        rules = load_rules_json(str(tmp_path))
        assert rules is not None
        assert len(rules) == 1
        assert rules[0].text == "Real rule"

    def test_nonstring_expansions_skipped(self, tmp_path: Path) -> None:
        data = {
            "version": 1,
            "rules": [
                {
                    "text": "A rule",
                    "expansions": [42, None, "valid"],
                    "source": {},
                },
            ],
        }
        (tmp_path / "rules.json").write_text(json.dumps(data))

        rules = load_rules_json(str(tmp_path))
        assert rules is not None
        assert rules[0].expansions == ("valid",)

    def test_empty_expansion_strings_skipped(self, tmp_path: Path) -> None:
        data = {
            "version": 1,
            "rules": [
                {
                    "text": "A rule",
                    "expansions": ["", "  ", "valid"],
                    "source": {},
                },
            ],
        }
        (tmp_path / "rules.json").write_text(json.dumps(data))

        rules = load_rules_json(str(tmp_path))
        assert rules is not None
        assert rules[0].expansions == ("valid",)


class TestMergeRulesJson:
    def test_preserves_expansions_for_unchanged_text(self) -> None:
        prov = Provenance(file="/tmp/r.txt", line_start=1, line_end=1)
        fresh = [Rule(text="Rule A", provenance=prov)]
        cached = [
            Rule(
                text="Rule A",
                provenance=prov,
                expansions=("exp1", "exp2"),
            ),
        ]

        merged = merge_rules_json(fresh, cached)

        assert len(merged) == 1
        assert merged[0].expansions == ("exp1", "exp2")

    def test_new_rules_get_empty_expansions(self) -> None:
        prov = Provenance(file="/tmp/r.txt", line_start=1, line_end=1)
        fresh = [Rule(text="New rule", provenance=prov)]
        cached: list[Rule] = []

        merged = merge_rules_json(fresh, cached)

        assert len(merged) == 1
        assert merged[0].expansions == ()

    def test_changed_text_loses_expansions(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        prov = Provenance(file="/tmp/r.txt", line_start=1, line_end=1)
        fresh = [Rule(text="Rule A v2", provenance=prov)]
        cached = [
            Rule(
                text="Rule A v1",
                provenance=prov,
                expansions=("exp1",),
            ),
        ]

        with caplog.at_level("WARNING", logger="cuecard.indexer"):
            merged = merge_rules_json(fresh, cached)

        assert len(merged) == 1
        assert merged[0].expansions == ()
        assert "expansions lost" in caplog.text

    def test_preserves_fresh_provenance(self) -> None:
        old_prov = Provenance(file="/tmp/old.txt", line_start=1, line_end=1)
        new_prov = Provenance(file="/tmp/new.txt", line_start=5, line_end=5)
        fresh = [Rule(text="Rule A", provenance=new_prov)]
        cached = [
            Rule(
                text="Rule A",
                provenance=old_prov,
                expansions=("exp1",),
            ),
        ]

        merged = merge_rules_json(fresh, cached)

        assert merged[0].provenance.file == "/tmp/new.txt"
        assert merged[0].provenance.line_start == 5
        assert merged[0].expansions == ("exp1",)
