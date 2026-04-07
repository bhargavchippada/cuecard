"""Tests for cuecard CLI — retrieve, format, and embed commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

import cuecard.security as security_module
from cuecard.cli import app
from cuecard.models import (
    Index,
    LoadedIndex,
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

        from cuecard.models import PipelineResult, StageTrace

        fake_pipeline = PipelineResult(
            results=(RankedResult(rule=idx.rules[0], score=0.87),),
            stages=(StageTrace(
                stage="retrieval", input_count=1,
                output_count=1, latency_ms=1.0,
            ),),
            mode="embedding",
        )

        mock_model = MagicMock()
        with (
            patch("cuecard.loader.load_or_build", return_value=LoadedIndex(index=idx)),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch("cuecard.pipeline.run_pipeline", return_value=fake_pipeline),
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

        from cuecard.models import PipelineResult, StageTrace

        fake_pipeline = PipelineResult(
            results=(),
            stages=(StageTrace(
                stage="retrieval", input_count=1,
                output_count=0, latency_ms=1.0,
            ),),
            mode="embedding",
        )

        with (
            patch("cuecard.loader.load_or_build", return_value=LoadedIndex(index=idx)),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch(
                "cuecard.pipeline.run_pipeline",
                return_value=fake_pipeline,
            ) as mock_pipe,
        ):
            result = runner.invoke(
                app,
                ["retrieve", "test", "--top-k", "3", "--threshold", "0.5"],
            )

        assert result.exit_code == 0
        mock_pipe.assert_called_once()

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
            patch("cuecard.loader.load_or_build", return_value=LoadedIndex(index=idx)),
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
            patch("cuecard.loader.load_or_build", return_value=LoadedIndex(index=idx)),
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
            patch("cuecard.loader.load_or_build", return_value=LoadedIndex(index=idx)),
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
        results = (RankedResult(rule=idx.rules[0], score=0.87),)
        mock_model = MagicMock()

        from cuecard.models import PipelineResult, StageTrace

        fake_pipeline = PipelineResult(
            results=results,
            stages=(StageTrace(
                stage="embedding", input_count=1,
                output_count=1, latency_ms=1.0,
            ),),
            mode="embedding",
        )

        with (
            patch("cuecard.loader.load_or_build", return_value=LoadedIndex(index=idx)),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch("cuecard.pipeline.run_pipeline", return_value=fake_pipeline),
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

        from cuecard.models import PipelineResult, StageTrace

        fake_pipeline = PipelineResult(
            results=(),
            stages=(StageTrace(
                stage="embedding", input_count=0,
                output_count=0, latency_ms=1.0,
            ),),
            mode="embedding",
        )

        with (
            patch("cuecard.loader.load_or_build", return_value=LoadedIndex(index=idx)),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch("cuecard.pipeline.run_pipeline", return_value=fake_pipeline),
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
