"""Tests for cuecard CLI — migrate command."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from cuecard.cli.main import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


class TestMigrateHappyPath:
    def test_converts_txt_to_toml(self, tmp_path: Path) -> None:
        txt = tmp_path / "rules.txt"
        txt.write_text(
            "Never commit secrets\n"
            "Use uv not pip\n",
        )

        result = runner.invoke(app, ["migrate", str(txt)])

        assert result.exit_code == 0
        toml_path = tmp_path / "rules.toml"
        assert toml_path.exists()

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        assert len(data["rules"]) == 2
        assert data["rules"][0]["text"] == "Never commit secrets"
        assert data["rules"][1]["text"] == "Use uv not pip"

    def test_events_tools_empty(self, tmp_path: Path) -> None:
        txt = tmp_path / "rules.txt"
        txt.write_text("A rule\n")

        runner.invoke(app, ["migrate", str(txt)])

        toml_path = tmp_path / "rules.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        assert data["rules"][0]["events"] == []
        assert data["rules"][0]["tools"] == []

    def test_custom_output_path(self, tmp_path: Path) -> None:
        txt = tmp_path / "rules.txt"
        txt.write_text("A rule\n")
        out = tmp_path / "custom.toml"

        result = runner.invoke(
            app, ["migrate", str(txt), "--output", str(out)],
        )

        assert result.exit_code == 0
        assert out.exists()

    def test_comments_not_preserved(self, tmp_path: Path) -> None:
        txt = tmp_path / "rules.txt"
        txt.write_text(
            "# This is a comment\n"
            "Actual rule\n"
            "# Another comment\n",
        )

        runner.invoke(app, ["migrate", str(txt)])

        toml_path = tmp_path / "rules.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        assert len(data["rules"]) == 1
        assert data["rules"][0]["text"] == "Actual rule"

    def test_original_txt_preserved(self, tmp_path: Path) -> None:
        txt = tmp_path / "rules.txt"
        txt.write_text("A rule\n")

        runner.invoke(app, ["migrate", str(txt)])

        assert txt.exists()

    def test_quotes_escaped(self, tmp_path: Path) -> None:
        txt = tmp_path / "rules.txt"
        txt.write_text('Use "uv" not "pip"\n')

        runner.invoke(app, ["migrate", str(txt)])

        toml_path = tmp_path / "rules.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        assert data["rules"][0]["text"] == 'Use "uv" not "pip"'


class TestMigrateErrors:
    def test_file_not_found(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["migrate", str(tmp_path / "nope.txt")],
        )
        assert result.exit_code == 1
        assert "File not found" in result.output

    def test_not_txt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.json"
        f.write_text("{}")

        result = runner.invoke(app, ["migrate", str(f)])

        assert result.exit_code == 1
        assert "Expected a .txt file" in result.output

    def test_output_exists(self, tmp_path: Path) -> None:
        txt = tmp_path / "rules.txt"
        txt.write_text("A rule\n")
        toml = tmp_path / "rules.toml"
        toml.write_text("existing")

        result = runner.invoke(app, ["migrate", str(txt)])

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_empty_file(self, tmp_path: Path) -> None:
        txt = tmp_path / "rules.txt"
        txt.write_text("# only comments\n")

        result = runner.invoke(app, ["migrate", str(txt)])

        assert result.exit_code == 1
        assert "No rules found" in result.output
