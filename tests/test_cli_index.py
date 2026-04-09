"""Tests for cuecard CLI — parse and index commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

import cuecard.security as security_module
from cuecard.cli import app
from cuecard.freshness import FreshnessResult
from cuecard.models import (
    Index,
    Provenance,
    Rule,
    SourceMeta,
)

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


def _make_sample_index(
    rules: tuple[Rule, ...] | None = None,
    cache_dir: str = "/tmp/test-index",
) -> Index:
    if rules is None:
        rules = (
            Rule(
                text="Never commit secrets",
                provenance=Provenance(file="/tmp/rules.txt", line_start=1, line_end=1),
            ),
        )
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((len(rules), 384)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    return Index(
        embeddings=emb,
        rules=rules,
        model_name="BAAI/bge-small-en-v1.5",
        dim=384,
        sources={
            "/tmp/rules.txt": SourceMeta(
                mtime=1711929600.0,
                content_hash="sha256:abc123",
                rule_count=len(rules),
            ),
        },
    )


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
# parse command
# ---------------------------------------------------------------------------


class TestParse:
    def test_parse_shows_rules(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["parse"])
        assert result.exit_code == 0
        assert "Never commit secrets" in result.output

    def test_parse_no_rules(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        rules_dir = cuecard_dir / "rules"
        rules_dir.mkdir()
        (rules_dir / "global.txt").write_text("# comment only\n")
        _make_config_toml(cuecard_dir)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["parse"])
        assert result.exit_code == 0
        assert "No rules found" in result.output


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------


class TestIndex:
    def test_index_rebuild(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_model = MagicMock()
        emb = np.random.default_rng(42).standard_normal((2, 384)).astype(np.float32)
        mock_model.passage_embed.return_value = [emb[0], emb[1]]

        rules_path = str((tmp_path / ".cuecard" / "rules" / "global.txt").resolve())
        fake_sources: dict[str, SourceMeta] = {
            rules_path: SourceMeta(mtime=1.0, content_hash="sha256:abc", rule_count=2),
        }
        fake_freshness = FreshnessResult(
            is_stale=True,
            updated_sources=fake_sources,
            changed_files=(),
            removed_files=(),
            new_files=(rules_path,),
        )

        with (
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch("cuecard.freshness.check_freshness", return_value=fake_freshness),
        ):
            result = runner.invoke(app, ["index"])

        assert result.exit_code == 0
        assert "index rebuilt" in result.output.lower()

    def test_index_rebuild_merges_cached_rules_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When rules.json exists with expansions, index rebuild merges them."""
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        rng = np.random.default_rng(42)
        mock_model = MagicMock()

        def _fake_embed(texts: object, **kw: object) -> list[np.ndarray]:
            n = len(list(texts))
            emb = rng.standard_normal((n, 384)).astype(np.float32)
            return [emb[i] for i in range(n)]

        mock_model.passage_embed.side_effect = _fake_embed

        rules_path = str((tmp_path / ".cuecard" / "rules" / "global.txt").resolve())
        fake_sources: dict[str, SourceMeta] = {
            rules_path: SourceMeta(
                mtime=1.0, content_hash="sha256:abc", rule_count=2,
            ),
        }
        fake_freshness = FreshnessResult(
            is_stale=True,
            updated_sources=fake_sources,
            changed_files=(),
            removed_files=(),
            new_files=(rules_path,),
        )

        # Pre-populate rules.json with expansions for a matching rule
        from cuecard.indexer import save_rules_json
        from cuecard.models import Provenance, Rule

        cached_rules = [
            Rule(
                text="Never commit secrets to git",
                provenance=Provenance(file=rules_path, line_start=1, line_end=1),
                expansions=("AKIA in source", "hardcoded API key"),
            ),
        ]
        cache_dir = str(tmp_path / ".cuecard" / "index")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        save_rules_json(cached_rules, cache_dir)

        with (
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch("cuecard.freshness.check_freshness", return_value=fake_freshness),
        ):
            result = runner.invoke(app, ["index"])

        assert result.exit_code == 0
        assert "index rebuilt" in result.output.lower()

        # Verify rules.json was saved with merged content
        from cuecard.indexer import load_rules_json

        result = load_rules_json(cache_dir)
        assert result is not None
        merged, _aff = result
        # Both source rules should be present
        texts = [r.text for r in merged]
        assert "Never commit secrets to git" in texts
        assert "Always validate user input" in texts
        # Expansions preserved for the matching rule
        for r in merged:
            if r.text == "Never commit secrets to git":
                assert r.expansions == ("AKIA in source", "hardcoded API key")

    def test_index_rebuild_no_rules(
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

        with patch("fastembed.TextEmbedding"):
            result = runner.invoke(app, ["index"])

        assert result.exit_code == 0
        assert "No rules found" in result.output

    def test_index_status_with_valid_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        idx = _make_sample_index()
        with patch("cuecard.indexer.load_index", return_value=idx):
            result = runner.invoke(app, ["index", "--status"])

        assert result.exit_code == 0
        assert "Model:" in result.output
        assert "Rules:" in result.output
        assert "Dim:" in result.output

    def test_index_status_no_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        with patch("cuecard.indexer.load_index", return_value=None):
            result = runner.invoke(app, ["index", "--status"])

        assert result.exit_code == 0
        assert "no valid index" in result.output.lower()


class TestIndexWithProjectScope:
    """Exercise project-scope branches in _iter_cache_dirs/_iter_scoped_sources."""

    def test_index_rebuild_with_project_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)

        # Set up project with its own rules
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        (project_dir / "cuecard.toml").write_text(
            "[sources]\n"
            'rules = ["rules.txt"]\n',
        )
        (project_dir / "rules.txt").write_text("Project specific rule\n")

        rng = np.random.default_rng(42)
        mock_model = MagicMock()

        def _fake_passage_embed(texts: list[str], **kw: object) -> list[np.ndarray]:
            n = len(list(texts))
            emb = rng.standard_normal((n, 384)).astype(np.float32)
            return [emb[i] for i in range(n)]

        mock_model.passage_embed.side_effect = _fake_passage_embed

        fake_freshness = FreshnessResult(
            is_stale=True,
            updated_sources={},
            changed_files=(),
            removed_files=(),
            new_files=("fake",),
        )

        with (
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch("cuecard.freshness.check_freshness", return_value=fake_freshness),
        ):
            result = runner.invoke(app, ["index"])

        assert result.exit_code == 0
        # Both scopes should be built
        output_lower = result.output.lower()
        assert "global" in output_lower
        assert "project" in output_lower
        assert "total:" in output_lower

    def test_index_status_with_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)

        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        (project_dir / "cuecard.toml").write_text(
            "[sources]\n"
            'rules = ["rules.txt"]\n',
        )
        (project_dir / "rules.txt").write_text("Project rule\n")

        idx = _make_sample_index()

        with patch("cuecard.indexer.load_index", return_value=idx):
            result = runner.invoke(app, ["index", "--status"])

        assert result.exit_code == 0
        assert "Global" in result.output
        assert "Project" in result.output
