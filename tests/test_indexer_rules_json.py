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
        result = load_rules_json(str(tmp_path))

        assert result is not None
        loaded, _aff = result
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
                    "expansions": ["x" * 600],
                    "source": {"file": "/tmp/r.txt", "line_start": 1, "line_end": 1},
                },
            ],
        }
        (tmp_path / "rules.json").write_text(json.dumps(data))

        result = load_rules_json(str(tmp_path))
        assert result is not None
        rules, _aff = result
        assert len(rules[0].expansions[0]) == 500

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

        result = load_rules_json(str(tmp_path))
        assert result is not None
        rules, _aff = result
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

        result = load_rules_json(str(tmp_path))
        assert result is not None
        rules, _aff = result
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

        result = load_rules_json(str(tmp_path))
        assert result is not None
        rules, _aff = result
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

        result = load_rules_json(str(tmp_path))
        assert result is not None
        rules, _aff = result
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


class TestRulesJsonV2:
    def test_round_trip_with_events_tools(self, tmp_path: Path) -> None:
        rules = [
            Rule(
                text="Never commit secrets",
                provenance=Provenance(
                    file="/tmp/rules.toml", line_start=1, line_end=1,
                    chunk_type="toml_rule",
                ),
                events=frozenset({"PreToolUse", "PostToolUse"}),
                tools=frozenset({"Bash"}),
            ),
            Rule(
                text="Use uv not pip",
                provenance=Provenance(
                    file="/tmp/rules.toml", line_start=2, line_end=2,
                    chunk_type="toml_rule",
                ),
            ),
        ]

        save_rules_json(rules, str(tmp_path))
        result = load_rules_json(str(tmp_path))

        assert result is not None
        loaded, _aff = result
        assert len(loaded) == 2
        assert loaded[0].events == frozenset({"PreToolUse", "PostToolUse"})
        assert loaded[0].tools == frozenset({"Bash"})
        assert loaded[1].events == frozenset()
        assert loaded[1].tools == frozenset()

    def test_v2_json_has_events_tools_keys(self, tmp_path: Path) -> None:
        rules = [
            Rule(
                text="A rule",
                provenance=Provenance(
                    file="/tmp/r.txt", line_start=1, line_end=1,
                ),
                events=frozenset({"Stop"}),
                tools=frozenset({"Edit", "Write"}),
            ),
        ]

        save_rules_json(rules, str(tmp_path))

        import json
        raw = json.loads((tmp_path / "rules.json").read_text())
        assert raw["version"] == 2
        assert raw["rules"][0]["events"] == ["Stop"]
        assert sorted(raw["rules"][0]["tools"]) == ["Edit", "Write"]

    def test_inline_affinity_roundtrip(self, tmp_path: Path) -> None:
        """Affinity saved inline in rules.json is loaded back correctly."""
        from cuecard.models import AffinityIndex, RuleAffinity

        rules = [
            Rule(
                text="Use uv not pip",
                provenance=Provenance(file="/tmp/r.txt", line_start=1, line_end=1),
            ),
            Rule(
                text="Classify tasks",
                provenance=Provenance(file="/tmp/r.txt", line_start=2, line_end=2),
            ),
        ]
        import hashlib

        h0 = hashlib.sha256(rules[0].text.encode()).hexdigest()
        h1 = hashlib.sha256(rules[1].text.encode()).hexdigest()
        affinity = AffinityIndex(
            version=1, mode="strict", model="test",
            affinities=(
                (h0, RuleAffinity(
                    events=frozenset({"PreToolUse", "PostToolUse"}),
                    tools=frozenset(), source="explicit", reasoning="tool rule",
                )),
                (h1, RuleAffinity(
                    events=frozenset({"UserPromptSubmit", "Stop"}),
                    tools=frozenset(), source="inferred", reasoning="workflow",
                )),
            ),
        )

        save_rules_json(rules, str(tmp_path), affinity=affinity)

        result = load_rules_json(str(tmp_path))
        assert result is not None
        loaded_rules, loaded_aff = result
        assert len(loaded_rules) == 2
        assert loaded_aff is not None
        assert len(loaded_aff.items) == 2
        assert loaded_aff.mode == "strict"

        # Verify affinity content
        aff_lookup = dict(loaded_aff.items)
        assert aff_lookup[h0].events == frozenset({"PreToolUse", "PostToolUse"})
        assert aff_lookup[h0].source == "explicit"
        assert aff_lookup[h1].events == frozenset({"UserPromptSubmit", "Stop"})

    def test_load_without_inline_affinity_returns_none(self, tmp_path: Path) -> None:
        """rules.json without affinity dicts returns None for affinity."""
        rules = [
            Rule(
                text="A rule",
                provenance=Provenance(file="/tmp/r.txt", line_start=1, line_end=1),
            ),
        ]
        save_rules_json(rules, str(tmp_path))

        result = load_rules_json(str(tmp_path))
        assert result is not None
        _rules, aff = result
        assert aff is None

    def test_load_v1_gets_empty_events_tools(self, tmp_path: Path) -> None:
        """V1 rules.json without events/tools defaults to empty."""
        import json
        data = {
            "version": 1,
            "rules": [
                {
                    "text": "Old rule",
                    "expansions": [],
                    "source": {
                        "file": "/tmp/r.txt",
                        "line_start": 1,
                        "line_end": 1,
                    },
                },
            ],
        }
        (tmp_path / "rules.json").write_text(json.dumps(data))

        result = load_rules_json(str(tmp_path))

        assert result is not None
        loaded, _aff = result
        assert loaded[0].events == frozenset()
        assert loaded[0].tools == frozenset()

    def test_merge_preserves_fresh_events_tools(self) -> None:
        prov = Provenance(file="/tmp/r.txt", line_start=1, line_end=1)
        fresh = [
            Rule(
                text="Rule A",
                provenance=prov,
                events=frozenset({"PreToolUse"}),
                tools=frozenset({"Bash"}),
            ),
        ]
        cached = [
            Rule(
                text="Rule A",
                provenance=prov,
                expansions=("exp1", "exp2"),
            ),
        ]

        merged = merge_rules_json(fresh, cached)

        assert merged[0].expansions == ("exp1", "exp2")
        assert merged[0].events == frozenset({"PreToolUse"})
        assert merged[0].tools == frozenset({"Bash"})
