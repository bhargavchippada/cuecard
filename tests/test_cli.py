"""Tests for cuecard CLI (100% coverage target)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

import cuecard.security as security_module
from cuecard.cli import _home_dir, app
from cuecard.freshness import FreshnessResult
from cuecard.models import (
    Index,
    Provenance,
    RankedResult,
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
# config command
# ---------------------------------------------------------------------------


class TestConfig:
    def test_config_shows_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "model" in result.output
        assert "top_k" in result.output

    def test_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        (cuecard_dir / "config.toml").write_text("[retrieval]\ntop_k = -5\n")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 1
        assert "Config error" in result.output


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


# ---------------------------------------------------------------------------
# retrieve command
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_retrieve_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        idx = _make_sample_index()
        results = [RankedResult(rule=idx.rules[0], score=0.87)]

        mock_model = MagicMock()
        with (
            patch("cuecard.loader.load_or_build", return_value=idx),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch("cuecard.retriever.retrieve", return_value=results),
        ):
            result = runner.invoke(app, ["retrieve", "secrets"])

        assert result.exit_code == 0

    def test_retrieve_custom_top_k_and_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        idx = _make_sample_index()
        mock_model = MagicMock()

        with (
            patch("cuecard.loader.load_or_build", return_value=idx),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch("cuecard.retriever.retrieve", return_value=[]) as mock_ret,
        ):
            result = runner.invoke(
                app,
                ["retrieve", "test", "--top-k", "3", "--threshold", "0.5"],
            )

        assert result.exit_code == 0
        call_kwargs = mock_ret.call_args[1]
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["threshold"] == 0.5

    def test_retrieve_no_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        with patch("cuecard.loader.load_or_build", return_value=None):
            result = runner.invoke(app, ["retrieve", "test"])

        assert result.exit_code == 1
        assert "Not set up" in result.output

    def test_retrieve_with_mode_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        idx = _make_sample_index()
        mock_model = MagicMock()

        from cuecard.models import PipelineResult, StageTrace

        fake_pipeline = PipelineResult(
            results=(RankedResult(rule=idx.rules[0], score=0.9),),
            stages=(StageTrace(
                stage="embedding", input_count=1,
                output_count=1, latency_ms=1.0,
            ),),
            mode="rerank",
        )

        with (
            patch("cuecard.loader.load_or_build", return_value=idx),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch(
                "cuecard.pipeline.run_pipeline",
                return_value=fake_pipeline,
            ) as mock_pipe,
        ):
            result = runner.invoke(
                app, ["retrieve", "test", "--mode", "rerank"],
            )

        assert result.exit_code == 0
        mock_pipe.assert_called_once()

    def test_retrieve_invalid_mode_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        idx = _make_sample_index()
        mock_model = MagicMock()

        with (
            patch("cuecard.loader.load_or_build", return_value=idx),
            patch("fastembed.TextEmbedding", return_value=mock_model),
        ):
            result = runner.invoke(
                app, ["retrieve", "test", "--mode", "invalid-mode"],
            )

        assert result.exit_code == 1
        assert "Invalid mode" in result.output

    def test_retrieve_with_config_pipeline_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Config pipeline.mode triggers pipeline when no --mode flag."""
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        # Write project config with pipeline mode
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text('[pipeline]\nmode = "rerank"\n')
        monkeypatch.chdir(project)

        idx = _make_sample_index()
        mock_model = MagicMock()

        from cuecard.models import PipelineResult, StageTrace

        fake_pipeline = PipelineResult(
            results=(),
            stages=(StageTrace(
                stage="embedding", input_count=0,
                output_count=0, latency_ms=0.5,
            ),),
            mode="rerank",
        )

        with (
            patch("cuecard.loader.load_or_build", return_value=idx),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch(
                "cuecard.pipeline.run_pipeline",
                return_value=fake_pipeline,
            ) as mock_pipe,
        ):
            result = runner.invoke(app, ["retrieve", "test"])

        assert result.exit_code == 0
        mock_pipe.assert_called_once()


# ---------------------------------------------------------------------------
# format command
# ---------------------------------------------------------------------------


class TestFormat:
    def test_format_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        idx = _make_sample_index()
        results = [RankedResult(rule=idx.rules[0], score=0.87)]
        mock_model = MagicMock()

        with (
            patch("cuecard.loader.load_or_build", return_value=idx),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch("cuecard.retriever.retrieve", return_value=results),
        ):
            result = runner.invoke(app, ["format", "secrets"])

        assert result.exit_code == 0

    def test_format_no_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        idx = _make_sample_index()
        mock_model = MagicMock()

        with (
            patch("cuecard.loader.load_or_build", return_value=idx),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch("cuecard.retriever.retrieve", return_value=[]),
        ):
            result = runner.invoke(app, ["format", "nonexistent"])

        assert result.exit_code == 0
        assert "No matching rules" in result.output

    def test_format_no_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        with patch("cuecard.loader.load_or_build", return_value=None):
            result = runner.invoke(app, ["format", "test"])

        assert result.exit_code == 1
        assert "Not set up" in result.output


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
        with patch("cuecard.parser.parse_rules", return_value=[md_rule]):
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
        with patch("cuecard.parser.parse_rules", return_value=[outside_rule]):
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
        with patch("cuecard.parser.parse_rules", return_value=[bad_rule]):
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
            "cuecard.expander.call_local",
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
            "cuecard.expander.call_local",
            side_effect=ValueError("bad backend"),
        ):
            result = runner.invoke(
                app,
                ["rules", "expand", "--endpoint", "http://localhost:8081/v1"],
            )
        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "error" in result.output.lower()

    def test_expand_empty_rules_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rules.json exists but has zero rules."""
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        # Write an empty rules.json
        cache_dir = tmp_path / ".cuecard" / "index"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "rules.json").write_text(
            '{"version": 1, "rules": []}',
        )

        with patch(
            "cuecard.indexer.load_rules_json", return_value=[],
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
            "cuecard.expander.call_local",
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


# ---------------------------------------------------------------------------
# embed command
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_embed_shows_stats(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        idx = _make_sample_index()
        index_dir = tmp_path / ".cuecard" / "index"
        np.savez_compressed(
            str(index_dir / "embeddings.npz"),
            embeddings=idx.embeddings,
        )

        with patch("cuecard.indexer.load_index", return_value=idx):
            result = runner.invoke(app, ["embed"])

        assert result.exit_code == 0
        assert "Model:" in result.output
        assert "Dimensions:" in result.output
        assert "Rules embedded:" in result.output
        assert "Index size:" in result.output

    def test_embed_no_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        with patch("cuecard.indexer.load_index", return_value=None):
            result = runner.invoke(app, ["embed"])

        assert result.exit_code == 1
        assert "Not set up" in result.output

    def test_embed_npz_missing_shows_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        idx = _make_sample_index()
        npz_path = tmp_path / ".cuecard" / "index" / "embeddings.npz"
        if npz_path.exists():
            npz_path.unlink()

        with patch("cuecard.indexer.load_index", return_value=idx):
            result = runner.invoke(app, ["embed"])

        assert result.exit_code == 0
        assert "0.00 MB" in result.output


# ---------------------------------------------------------------------------
# no_args_is_help
# ---------------------------------------------------------------------------


class TestNoArgs:
    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # no_args_is_help causes exit code 0 with help text
        assert "cuecard" in result.output.lower() or "Usage" in result.output


# ---------------------------------------------------------------------------
# _load_config_or_exit error path
# ---------------------------------------------------------------------------


class TestHomeDir:
    def test_home_dir_returns_path(self) -> None:
        result = _home_dir()
        assert isinstance(result, Path)
        assert result == Path.home()


# ---------------------------------------------------------------------------
# eval command
# ---------------------------------------------------------------------------


class TestEval:
    def test_eval_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fixture_file = tmp_path / "fixtures.json"
        fixture_file.write_text("[]")

        mock_model = MagicMock()
        fake_fixtures = [{"query": "test", "expected": ["rule1"]}]
        fake_summary = MagicMock()

        with (
            patch(
                "cuecard.eval.load_fixtures",
                return_value=fake_fixtures,
            ) as mock_load,
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch(
                "cuecard.eval.run_eval",
                return_value=fake_summary,
            ) as mock_run,
            patch(
                "cuecard.eval.format_eval_report",
                return_value="Report OK",
            ) as mock_fmt,
        ):
            result = runner.invoke(app, ["eval", str(fixture_file)])

        assert result.exit_code == 0
        assert "Report OK" in result.output
        mock_load.assert_called_once_with(str(fixture_file))
        mock_run.assert_called_once()
        mock_fmt.assert_called_once_with(fake_summary)

    def test_eval_custom_model_and_corpus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fixture_file = tmp_path / "fixtures.json"
        fixture_file.write_text("[]")
        corpus = tmp_path / "corpora"
        corpus.mkdir()

        mock_model = MagicMock()
        fake_summary = MagicMock()

        with (
            patch("cuecard.eval.load_fixtures", return_value=[]),
            patch("fastembed.TextEmbedding", return_value=mock_model) as mock_te,
            patch("cuecard.eval.run_eval", return_value=fake_summary) as mock_run,
            patch("cuecard.eval.format_eval_report", return_value="OK"),
        ):
            result = runner.invoke(app, [
                "eval", str(fixture_file),
                "--model", "custom/model",
                "--corpus-dir", str(corpus),
                "--top-k", "10",
                "--threshold", "0.1",
                "--dedup-threshold", "0.9",
            ])

        assert result.exit_code == 0
        mock_te.assert_called_once_with(model_name="custom/model")
        call_kwargs = mock_run.call_args
        assert call_kwargs[0][1] == str(corpus)
        assert call_kwargs[0][2] == "custom/model"
        assert call_kwargs[1]["top_k"] == 10
        assert call_kwargs[1]["threshold"] == 0.1
        assert call_kwargs[1]["dedup_threshold"] == 0.9

    def test_eval_default_corpus_dir_is_fixture_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sub = tmp_path / "eval_data"
        sub.mkdir()
        fixture_file = sub / "fixtures.json"
        fixture_file.write_text("[]")

        mock_model = MagicMock()
        fake_summary = MagicMock()

        with (
            patch("cuecard.eval.load_fixtures", return_value=[]),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch("cuecard.eval.run_eval", return_value=fake_summary) as mock_run,
            patch("cuecard.eval.format_eval_report", return_value="OK"),
        ):
            result = runner.invoke(app, ["eval", str(fixture_file)])

        assert result.exit_code == 0
        assert mock_run.call_args[0][1] == str(sub)

    def test_eval_fixture_load_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fixture_file = tmp_path / "bad.json"
        fixture_file.write_text("not json")

        with patch(
            "cuecard.eval.load_fixtures",
            side_effect=ValueError("Invalid JSON"),
        ):
            result = runner.invoke(app, ["eval", str(fixture_file)])

        assert result.exit_code == 1
        assert "Failed to load fixtures" in result.output

    def test_eval_model_load_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fixture_file = tmp_path / "fixtures.json"
        fixture_file.write_text("[]")

        with (
            patch("cuecard.eval.load_fixtures", return_value=[]),
            patch(
                "fastembed.TextEmbedding",
                side_effect=RuntimeError("Model not found"),
            ),
        ):
            result = runner.invoke(app, ["eval", str(fixture_file)])

        assert result.exit_code == 1
        assert "Failed to load model" in result.output

    def test_eval_run_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fixture_file = tmp_path / "fixtures.json"
        fixture_file.write_text("[]")

        mock_model = MagicMock()

        with (
            patch("cuecard.eval.load_fixtures", return_value=[]),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch(
                "cuecard.eval.run_eval",
                side_effect=RuntimeError("Eval crashed"),
            ),
        ):
            result = runner.invoke(app, ["eval", str(fixture_file)])

        assert result.exit_code == 1
        assert "Eval failed" in result.output

    def test_eval_with_mode_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fixture_file = tmp_path / "fixtures.json"
        fixture_file.write_text("[]")

        mock_model = MagicMock()
        fake_summary = MagicMock()

        with (
            patch("cuecard.eval.load_fixtures", return_value=[]),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch(
                "cuecard.eval.run_eval",
                return_value=fake_summary,
            ) as mock_run,
            patch("cuecard.eval.format_eval_report", return_value="OK"),
        ):
            result = runner.invoke(app, [
                "eval", str(fixture_file), "--mode", "rerank",
            ])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["mode"] == "rerank"


    def test_eval_with_corpus_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fixture_file = tmp_path / "fixtures.json"
        fixture_file.write_text("[]")

        mock_model = MagicMock()
        fake_summary = MagicMock()

        with (
            patch("cuecard.eval.load_fixtures", return_value=[]),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch(
                "cuecard.eval.run_eval",
                return_value=fake_summary,
            ) as mock_run,
            patch("cuecard.eval.format_eval_report", return_value="OK"),
        ):
            result = runner.invoke(app, [
                "eval", str(fixture_file),
                "--corpus-override", "rules_a.txt,rules_b.txt",
            ])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["corpus_override"] is not None
        assert len(call_kwargs["corpus_override"]) == 2


class TestLoadConfigOrExit:
    def test_config_error_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        (cuecard_dir / "config.toml").write_text(
            "[retrieval]\nthreshold = 5.0\n"
            '[embedding]\nmodel = "test"\n'
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["parse"])
        assert result.exit_code == 1
        assert "Config error" in result.output


# ---------------------------------------------------------------------------
# install command
# ---------------------------------------------------------------------------


class TestInstall:
    def test_install_unknown_target(self) -> None:
        result = runner.invoke(app, ["install", "vscode"])
        assert result.exit_code == 1
        assert "Unknown target" in result.output

    def test_install_claude_code_fresh(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        result = runner.invoke(app, ["install", "claude-code"])
        assert result.exit_code == 0
        assert "Installed" in result.output
        settings = json.loads((claude_dir / "settings.json").read_text())
        hooks = settings["hooks"]["PreToolUse"]
        assert any("cuecard" in h.get("command", "") for h in hooks)

    def test_install_already_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {
            "hooks": {
                "PreToolUse": [
                    {
                        "type": "command",
                        "command": "uv run python -m cuecard.adapters.claude_code",
                    },
                ],
            },
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings))
        result = runner.invoke(app, ["install", "claude-code"])
        assert result.exit_code == 0
        assert "already installed" in result.output

    def test_install_existing_settings_no_hooks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({"other": "data"}))
        result = runner.invoke(app, ["install", "claude-code"])
        assert result.exit_code == 0
        assert "Installed" in result.output

    def test_install_hooks_not_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({"hooks": "invalid"}))
        result = runner.invoke(app, ["install", "claude-code"])
        assert result.exit_code == 0
        assert "Installed" in result.output

    def test_install_pretool_not_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(
            json.dumps({"hooks": {"PreToolUse": "invalid"}}),
        )
        result = runner.invoke(app, ["install", "claude-code"])
        assert result.exit_code == 0
        assert "Installed" in result.output

    def test_install_no_existing_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        result = runner.invoke(app, ["install", "claude-code"])
        assert result.exit_code == 0
        assert "Installed" in result.output

    def test_install_preserves_existing_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(
            json.dumps({"other_key": "value"}),
        )
        result = runner.invoke(app, ["install", "claude-code"])
        assert result.exit_code == 0
        settings = json.loads(
            (claude_dir / "settings.json").read_text(),
        )
        assert settings["other_key"] == "value"
        assert "PreToolUse" in settings["hooks"]


# ---------------------------------------------------------------------------
# uninstall command
# ---------------------------------------------------------------------------


class TestUninstall:
    def test_uninstall_unknown_target(self) -> None:
        result = runner.invoke(app, ["uninstall", "vscode"])
        assert result.exit_code == 1
        assert "Unknown target" in result.output

    def test_uninstall_not_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        result = runner.invoke(app, ["uninstall", "claude-code"])
        assert result.exit_code == 0
        assert "not found" in result.output

    def test_uninstall_removes_hook(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {
            "hooks": {
                "PreToolUse": [
                    {
                        "type": "command",
                        "command": "uv run python -m cuecard.adapters.claude_code",
                    },
                ],
            },
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings))
        result = runner.invoke(app, ["uninstall", "claude-code"])
        assert result.exit_code == 0
        assert "Uninstalled" in result.output
        updated = json.loads((claude_dir / "settings.json").read_text())
        assert "hooks" not in updated

    def test_uninstall_preserves_other_hooks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {
            "hooks": {
                "PreToolUse": [
                    {"type": "command", "command": "other-tool"},
                    {
                        "type": "command",
                        "command": "uv run python -m cuecard.adapters.claude_code",
                    },
                ],
            },
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings))
        result = runner.invoke(app, ["uninstall", "claude-code"])
        assert result.exit_code == 0
        updated = json.loads((claude_dir / "settings.json").read_text())
        assert len(updated["hooks"]["PreToolUse"]) == 1
        assert "other-tool" in updated["hooks"]["PreToolUse"][0]["command"]


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_hook_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir(exist_ok=True)
        settings = {
            "hooks": {
                "PreToolUse": [
                    {
                        "type": "command",
                        "command": "uv run python -m cuecard.adapters.claude_code",
                    },
                ],
            },
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings))

        idx = _make_sample_index()
        with patch("cuecard.indexer.load_index", return_value=idx):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "hook installed" in result.output
        assert "rules" in result.output.lower()

    def test_status_hook_not_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        idx = _make_sample_index()
        with patch("cuecard.indexer.load_index", return_value=idx):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "not installed" in result.output

    def test_status_no_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        with patch("cuecard.indexer.load_index", return_value=None):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "No valid index" in result.output

    def test_status_with_log_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / ".cuecard" / "log.jsonl"
        log_path.write_text('{"event":"test"}\n')

        with patch("cuecard.indexer.load_index", return_value=None):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Log:" in result.output


# ---------------------------------------------------------------------------
# log command
# ---------------------------------------------------------------------------


class TestLogCmd:
    def test_log_no_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        result = runner.invoke(app, ["log"])
        assert result.exit_code == 0
        assert "No log entries" in result.output

    def test_log_recent_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        entries = [
            {
                "timestamp": "2024-01-01T12:00:00",
                "event": "PreToolUse",
                "tool_name": "Bash",
                "injected_count": 3,
                "latency_ms": 12.5,
            },
        ]
        with patch("cuecard.logger.read_log", return_value=entries):
            result = runner.invoke(app, ["log"])

        assert result.exit_code == 0
        assert "PreToolUse" in result.output
        assert "Bash" in result.output

    def test_log_stats(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        entries = [
            {
                "timestamp": "2024-01-01T12:00:00",
                "event": "PreToolUse",
                "tool_name": "Bash",
                "injected_count": 3,
                "latency_ms": 12.5,
            },
        ]
        stats = {
            "total_events": 1,
            "avg_injected": 3,
            "coverage": 1.0,
            "latency_p50": 12,
            "latency_p95": 12,
            "latency_p99": 12,
            "top_rules": [("Never commit secrets", 5)],
        }
        with (
            patch("cuecard.logger.read_log", return_value=entries),
            patch("cuecard.logger.compute_stats", return_value=stats),
        ):
            result = runner.invoke(app, ["log", "--stats"])

        assert result.exit_code == 0
        assert "Log Statistics" in result.output
        assert "Top Rules" in result.output

    def test_log_stats_no_top_rules(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        entries = [{"timestamp": "2024-01-01T12:00:00"}]
        stats = {
            "total_events": 1,
            "avg_injected": 0,
            "coverage": 0.0,
            "latency_p50": 0,
            "latency_p95": 0,
            "latency_p99": 0,
            "top_rules": [],
        }
        with (
            patch("cuecard.logger.read_log", return_value=entries),
            patch("cuecard.logger.compute_stats", return_value=stats),
        ):
            result = runner.invoke(app, ["log", "--stats"])

        assert result.exit_code == 0
        assert "Log Statistics" in result.output

    def test_log_custom_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        with patch("cuecard.logger.read_log", return_value=[]) as mock_read:
            result = runner.invoke(app, ["log", "--limit", "5"])
        assert result.exit_code == 0
        mock_read.assert_called_once()
        assert mock_read.call_args[1]["limit"] == 5


# ---------------------------------------------------------------------------
# adapter __main__ guard
# ---------------------------------------------------------------------------


class TestAdapterMainGuard:
    def test_main_guard(self) -> None:
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "-m", "cuecard.adapters.claude_code"],
            input='{"tool_name":"Bash","tool_input":"ls"}',
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = json.loads(result.stdout)
        assert output["tool_name"] == "Bash"


class TestStatusDetailed:
    def test_status_no_hook_no_index(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        _make_config_toml(cuecard_dir)
        _make_rules_file(cuecard_dir)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "not installed" in result.output
        assert "No valid index" in result.output

    def test_status_with_hook_and_index(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        _make_config_toml(cuecard_dir)
        _make_rules_file(cuecard_dir)
        monkeypatch.chdir(tmp_path)

        # Install hook
        runner.invoke(app, ["install", "claude-code"])

        # Create index
        idx = _make_sample_index()
        with patch(
            "cuecard.indexer.load_index", return_value=idx,
        ):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "hook installed" in result.output

    def test_status_with_log(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        _make_config_toml(cuecard_dir)
        _make_rules_file(cuecard_dir)
        monkeypatch.chdir(tmp_path)

        # Create a log file
        (cuecard_dir / "log.jsonl").write_text(
            '{"event": "test"}\n',
        )

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Log:" in result.output


class TestLogCmdDetailed:
    def test_log_shows_entries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        import json

        entries = [
            {
                "timestamp": "2026-04-01T12:00:00",
                "event": "PreToolUse",
                "tool_name": "Bash",
                "injected_count": 2,
                "latency_ms": 42,
            },
        ]
        (cuecard_dir / "log.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
        )

        result = runner.invoke(app, ["log"])
        assert result.exit_code == 0
        assert "PreToolUse:Bash" in result.output

    def test_log_stats_from_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        import json

        entries = [
            {
                "latency_ms": 42.0,
                "results": [
                    {"text": "Rule A", "score": 0.9},
                ],
            },
            {
                "latency_ms": 100.0,
                "results": [],
            },
        ]
        (cuecard_dir / "log.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
        )

        result = runner.invoke(app, ["log", "--stats"])
        assert result.exit_code == 0
        assert "Total events" in result.output
        assert "Latency p50" in result.output

    def test_log_stats_with_top_rules(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        import json

        entries = [
            {
                "latency_ms": 10.0,
                "results": [
                    {"text": "Popular rule", "score": 0.9},
                ],
            },
        ]
        (cuecard_dir / "log.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
        )

        result = runner.invoke(app, ["log", "--stats"])
        assert result.exit_code == 0
        assert "Top Rules" in result.output
        assert "Popular rule" in result.output

    def test_log_limit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        import json

        entries = [
            {
                "timestamp": f"2026-04-01T{i:02d}:00:00",
                "event": "PreToolUse",
                "tool_name": "Bash",
                "injected_count": 0,
                "latency_ms": 1,
            }
            for i in range(10)
        ]
        (cuecard_dir / "log.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
        )

        result = runner.invoke(app, ["log", "--limit", "3"])
        assert result.exit_code == 0


# --- helpers coverage ---


class TestClaudeSettingsHelpers:
    def test_load_missing(self, tmp_path: Path) -> None:
        from cuecard.cli import _load_claude_settings

        path = tmp_path / "nonexistent.json"
        assert _load_claude_settings(path) == {}

    def test_has_hook_bad_hooks_type(self) -> None:
        from cuecard.cli import _has_cuecard_hook

        assert _has_cuecard_hook({"hooks": "not_a_dict"}) is False

    def test_has_hook_bad_pretool_type(self) -> None:
        from cuecard.cli import _has_cuecard_hook

        settings: dict[str, object] = {
            "hooks": {"PreToolUse": "not_a_list"},
        }
        assert _has_cuecard_hook(settings) is False

    def test_claude_settings_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cuecard.cli import _claude_settings_path

        _patch_home(monkeypatch, tmp_path)
        p = _claude_settings_path()
        assert p == tmp_path / ".claude" / "settings.json"
