"""Tests for cuecard CLI — rules subcommands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import cuecard.security as security_module
from cuecard.cli.main import app
from cuecard.models import Provenance, Rule

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config_toml(
    cuecard_dir: Path, *, model: str = "BAAI/bge-small-en-v1.5",
) -> Path:
    config_path = cuecard_dir / "config.toml"
    config_path.write_text(
        "[sources]\n"
        f'rules = ["rules/global.txt"]\n\n'
        "[embedding]\n"
        f'model = "{model}"\n\n'
        "[retrieval]\n"
        "top_k = 5\n"
        "threshold = 0.30\n"
    )
    return config_path


def _make_rules_file(cuecard_dir: Path, rules: list[str] | None = None) -> Path:
    rules_dir = cuecard_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    rules_path = rules_dir / "global.txt"
    if rules is None:
        rules = [
            "Never commit secrets to git",
            "Always validate user input",
        ]
    rules_path.write_text("\n".join(rules) + "\n")
    return rules_path


def _setup_home(tmp_path: Path) -> Path:
    cuecard_dir = tmp_path / ".cuecard"
    cuecard_dir.mkdir()
    _make_config_toml(cuecard_dir)
    _make_rules_file(cuecard_dir)
    index_dir = cuecard_dir / "index"
    index_dir.mkdir()
    (index_dir / "embeddings.npz").write_bytes(b"fake")
    (index_dir / "metadata.json").write_text("{}")
    return cuecard_dir


def _patch_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("cuecard.cli.main._home_dir", lambda: tmp_path)


@pytest.fixture(autouse=True)
def _ensure_dir_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add ensure_dir to cuecard.security so the lazy import in CLI works."""
    def _ensure_dir(path_str: str) -> None:
        Path(path_str).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(security_module, "ensure_dir", _ensure_dir, raising=False)


# ---------------------------------------------------------------------------
# rules list (callback)
# ---------------------------------------------------------------------------


class TestRulesList:
    def test_rules_list_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules"])
        assert result.exit_code == 0
        assert "Never commit secrets" in result.output
        assert "2 rules" in result.output

    def test_rules_list_global_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--global with mixed global+project rules skips project rules."""
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)

        # Add project rules so the skip branch fires
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        (project_dir / "cuecard.toml").write_text(
            "[sources]\n"
            'rules = ["rules.txt"]\n'
        )
        (project_dir / "rules.txt").write_text("Project-only rule\n")

        result = runner.invoke(app, ["rules", "--global"])
        assert result.exit_code == 0
        assert "Global" in result.output
        # Project rule should NOT appear
        assert "Project-only rule" not in result.output

    def test_rules_list_project_only_shows_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "--project"])
        assert result.exit_code == 0
        assert "0 rules" in result.output

    def test_rules_list_with_project_rules(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        (project_dir / "cuecard.toml").write_text(
            "[sources]\n"
            'rules = ["rules.txt"]\n'
        )
        (project_dir / "rules.txt").write_text("Project rule one\n")
        result = runner.invoke(app, ["rules"])
        assert result.exit_code == 0
        assert "Global" in result.output
        assert "Project" in result.output

    def test_rules_list_no_rules(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        rules_dir = cuecard_dir / "rules"
        rules_dir.mkdir()
        (rules_dir / "global.txt").write_text("# comment\n")
        _make_config_toml(cuecard_dir)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules"])
        assert result.exit_code == 0
        assert "No rules found" in result.output


# ---------------------------------------------------------------------------
# rules add
# ---------------------------------------------------------------------------


class TestRulesAdd:
    def test_add_project_rule(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "add", "New project rule"])
        assert result.exit_code == 0
        assert "Added to" in result.output
        assert "New project rule" in (tmp_path / "rules.txt").read_text()

    def test_add_global_rule(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        (cuecard_dir / "rules").mkdir()
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "add", "--global", "Global rule"])
        assert result.exit_code == 0
        assert "Global rule" in (cuecard_dir / "rules" / "global.txt").read_text()

    def test_add_rule_exceeds_500_chars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "add", "x" * 501])
        assert result.exit_code == 1
        assert "500 character limit" in result.output

    def test_add_rule_creates_parent_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "add", "--global", "Another global rule"])
        assert result.exit_code == 0

    def test_add_appends_to_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rules.txt").write_text("Existing rule\n")
        result = runner.invoke(app, ["rules", "add", "Second rule"])
        assert result.exit_code == 0
        content = (tmp_path / "rules.txt").read_text()
        assert "Existing rule" in content
        assert "Second rule" in content


# ---------------------------------------------------------------------------
# rules remove
# ---------------------------------------------------------------------------


class TestRulesRemove:
    def test_remove_rule(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "remove", "1"])
        assert result.exit_code == 0
        assert "Removed from" in result.output

    def test_remove_invalid_number_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "remove", "0"])
        assert result.exit_code == 1
        assert "Invalid rule number" in result.output

    def test_remove_invalid_number_too_high(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "remove", "999"])
        assert result.exit_code == 1
        assert "Invalid rule number" in result.output

    def test_remove_from_non_txt_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        md_rule = Rule(
            text="MD rule",
            provenance=Provenance(file="/some/file.md", line_start=1, line_end=1),
        )
        with patch("cuecard.indexing.parser.parse_rules", return_value=[md_rule]):
            result = runner.invoke(app, ["rules", "remove", "1"])

        assert result.exit_code == 1
        assert "Edit the file directly" in result.output

    def test_remove_outside_source_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        # Rule with provenance pointing outside configured sources
        outside_rule = Rule(
            text="malicious rule",
            provenance=Provenance(
                file="/etc/important.txt",
                line_start=1,
                line_end=1,
            ),
        )
        with patch("cuecard.indexing.parser.parse_rules", return_value=[outside_rule]):
            result = runner.invoke(app, ["rules", "remove", "1"])

        assert result.exit_code == 1
        assert "not in configured source paths" in result.output

    def test_remove_line_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        rules_path = tmp_path / ".cuecard" / "rules" / "global.txt"
        bad_rule = Rule(
            text="ghost",
            provenance=Provenance(
                file=str(rules_path.resolve()),
                line_start=999,
                line_end=999,
            ),
        )
        with patch("cuecard.indexing.parser.parse_rules", return_value=[bad_rule]):
            result = runner.invoke(app, ["rules", "remove", "1"])

        assert result.exit_code == 1
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# rules search
# ---------------------------------------------------------------------------


class TestRulesSearch:
    def test_search_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "search", "secrets"])
        assert result.exit_code == 0
        assert "secrets" in result.output.lower()

    def test_search_no_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "search", "zzzznonexistent"])
        assert result.exit_code == 0
        assert "No rules matching" in result.output

    def test_search_case_insensitive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "search", "SECRETS"])
        assert result.exit_code == 0
        assert "secrets" in result.output.lower()


# ---------------------------------------------------------------------------
# rules sources
# ---------------------------------------------------------------------------


class TestRulesSources:
    def test_sources_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "sources"])
        assert result.exit_code == 0
        assert "2 rules" in result.output

    def test_sources_missing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        (cuecard_dir / "config.toml").write_text(
            "[sources]\n"
            'rules = ["rules/missing.txt"]\n\n'
            "[embedding]\n"
            'model = "BAAI/bge-small-en-v1.5"\n'
        )
        (cuecard_dir / "rules").mkdir()
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "sources"])
        assert result.exit_code == 0
        assert "not found" in result.output

    def test_sources_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        (cuecard_dir / "config.toml").write_text(
            "[embedding]\n"
            'model = "BAAI/bge-small-en-v1.5"\n'
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "sources"])
        assert result.exit_code == 0
        assert "No sources configured" in result.output


# ---------------------------------------------------------------------------
# rules expand
# ---------------------------------------------------------------------------


class TestRulesExpand:
    def test_expand_dry_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "expand", "--dry-run"])
        assert result.exit_code == 0
        assert "would expand" in result.output

    def test_expand_no_rules(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        (cuecard_dir / "config.toml").write_text(
            "[sources]\n"
            'rules = ["rules/empty.txt"]\n\n'
            "[embedding]\n"
            'model = "BAAI/bge-small-en-v1.5"\n'
        )
        (cuecard_dir / "rules").mkdir()
        (cuecard_dir / "rules" / "empty.txt").write_text("# no rules\n")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rules", "expand"])
        assert result.exit_code == 0
        assert "no rules" in result.output

    def test_expand_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        with patch(
            "cuecard.indexing.expander.call_local",
            return_value='{"expansions": ["expansion A"]}',
        ):
            result = runner.invoke(
                app,
                ["rules", "expand", "--endpoint", "http://localhost:8081/v1"],
            )
        assert result.exit_code == 0
        assert "total expansions" in result.output
        assert "new" in result.output

    def test_expand_llm_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        with patch(
            "cuecard.indexing.expander.call_local",
            side_effect=ValueError("bad backend"),
        ):
            result = runner.invoke(
                app,
                ["rules", "expand", "--endpoint", "http://localhost:8081/v1"],
            )
        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "error" in result.output.lower()

    def test_expand_empty_source_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Source files have zero rules -> skip."""
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        with patch(
            "cuecard.indexing.parser.parse_rules", return_value=[],
        ):
            result = runner.invoke(
                app,
                ["rules", "expand", "--endpoint", "http://localhost:8081/v1"],
            )
        assert result.exit_code == 0
        assert "no rules" in result.output

    def test_expand_missing_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        with patch(
            "cuecard.indexing.expander.call_local",
            return_value='{"expansions": ["new exp"]}',
        ):
            result = runner.invoke(
                app,
                [
                    "rules", "expand",
                    "--missing-only",
                    "--endpoint", "http://localhost:8081/v1",
                ],
            )
        assert result.exit_code == 0

    def test_expand_with_progress_skipped_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_expand_with_progress shows skipped count for --missing-only."""
        from cuecard.cli.rules import _expand_with_progress
        from cuecard.models import Provenance, Rule

        rules = [
            Rule(
                text="Has expansions",
                provenance=Provenance(file="a.txt", line_start=1, line_end=1),
                expansions=("existing",),
            ),
        ]

        with patch(
            "cuecard.indexing.expander.call_local",
            return_value='{"expansions": ["new"]}',
        ):
            result = _expand_with_progress(
                rules,
                backend="local",
                endpoint="http://localhost:8081/v1",
                haiku_model="claude-haiku-4-5",
                missing_only=True,
                dedup_threshold=0.80,
            )
        assert len(result) == 1
        assert result[0].expansions == ("existing",)


class TestTruncateRule:
    def test_short_text_unchanged(self) -> None:
        from cuecard.cli.rules import _truncate_rule
        assert _truncate_rule("short text") == "short text"

    def test_exact_length_unchanged(self) -> None:
        from cuecard.cli.rules import _truncate_rule
        text = "a" * 60
        assert _truncate_rule(text) == text

    def test_long_text_truncated(self) -> None:
        from cuecard.cli.rules import _truncate_rule
        text = "a" * 80
        result = _truncate_rule(text)
        assert len(result) == 60
        assert result.endswith("...")

    def test_custom_max_len(self) -> None:
        from cuecard.cli.rules import _truncate_rule
        result = _truncate_rule("a" * 20, max_len=10)
        assert len(result) == 10
        assert result.endswith("...")
