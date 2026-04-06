"""Tests for cuecard CLI — setup command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

import cuecard.security as security_module
from cuecard.cli import _home_dir, app
from cuecard.freshness import FreshnessResult
from cuecard.models import SourceMeta

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
    monkeypatch.setattr("cuecard.cli._home_dir", lambda: tmp_path)


@pytest.fixture(autouse=True)
def _ensure_dir_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add ensure_dir to cuecard.security so the lazy import in CLI works."""
    def _ensure_dir(path_str: str) -> None:
        Path(path_str).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(security_module, "ensure_dir", _ensure_dir, raising=False)


# ---------------------------------------------------------------------------
# setup command
# ---------------------------------------------------------------------------


class TestSetup:
    def test_first_run_creates_config_and_rules(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert "Created config" in result.output
        assert "Created rules" in result.output
        assert (tmp_path / ".cuecard" / "config.toml").exists()
        assert (tmp_path / ".cuecard" / "rules" / "global.txt").exists()

    def test_second_run_downloads_model_and_builds_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        _make_config_toml(cuecard_dir)
        rules_path = _make_rules_file(cuecard_dir)

        mock_model = MagicMock()
        emb = np.random.default_rng(42).standard_normal((2, 384)).astype(np.float32)
        mock_model.passage_embed.return_value = [emb[0], emb[1]]

        fake_sources: dict[str, SourceMeta] = {
            str(rules_path.resolve()): SourceMeta(
                mtime=1711929600.0,
                content_hash="sha256:abc",
                rule_count=2,
            ),
        }
        fake_freshness = FreshnessResult(
            is_stale=True,
            updated_sources=fake_sources,
            changed_files=(),
            removed_files=(),
            new_files=tuple(fake_sources.keys()),
        )

        with (
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch("cuecard.freshness.check_freshness", return_value=fake_freshness),
        ):
            result = runner.invoke(app, ["setup"])

        assert result.exit_code == 0
        assert "Setup complete" in result.output
        assert "Indexed 2 rules" in result.output

    def test_second_run_no_rules_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        _make_config_toml(cuecard_dir)
        rules_dir = cuecard_dir / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "global.txt").write_text("# Only comments\n")

        mock_model = MagicMock()
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = runner.invoke(app, ["setup"])

        assert result.exit_code == 0
        assert "No rules found" in result.output

    def test_setup_custom_model_rejected_without_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        result = runner.invoke(app, ["setup", "--model", "custom/model"])
        assert result.exit_code == 1
        assert "not in the allowed model list" in result.output

    def test_setup_custom_model_allowed_with_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        result = runner.invoke(
            app, ["setup", "--model", "custom/model", "--allow-custom-model"],
        )
        assert result.exit_code == 0
        assert "Warning" in result.output
        config_text = (tmp_path / ".cuecard" / "config.toml").read_text()
        assert "custom/model" in config_text

    def test_setup_allowed_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        result = runner.invoke(app, ["setup", "--model", "BAAI/bge-base-en-v1.5"])
        assert result.exit_code == 0
        config_text = (tmp_path / ".cuecard" / "config.toml").read_text()
        assert "BAAI/bge-base-en-v1.5" in config_text

    def test_config_exists_but_no_rules_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        (cuecard_dir / "rules").mkdir()
        _make_config_toml(cuecard_dir)

        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert "Created rules" in result.output
        assert (cuecard_dir / "rules" / "global.txt").exists()


class TestSetupCheck:
    def test_check_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        result = runner.invoke(app, ["setup", "--check"])
        assert result.exit_code == 0
        assert "Setup OK" in result.output

    def test_check_no_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        result = runner.invoke(app, ["setup", "--check"])
        assert result.exit_code == 1

    def test_check_no_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        _make_config_toml(cuecard_dir)
        result = runner.invoke(app, ["setup", "--check"])
        assert result.exit_code == 1

    def test_check_partial_index_no_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        _make_config_toml(cuecard_dir)
        index_dir = cuecard_dir / "index"
        index_dir.mkdir()
        (index_dir / "embeddings.npz").write_bytes(b"fake")
        result = runner.invoke(app, ["setup", "--check"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# no_args_is_help
# ---------------------------------------------------------------------------


class TestNoArgs:
    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # no_args_is_help causes exit code 0 with help text
        assert "cuecard" in result.output.lower() or "Usage" in result.output


# ---------------------------------------------------------------------------
# _home_dir
# ---------------------------------------------------------------------------


class TestHomeDir:
    def test_home_dir_returns_path(self) -> None:
        result = _home_dir()
        assert isinstance(result, Path)
        assert result == Path.home()
