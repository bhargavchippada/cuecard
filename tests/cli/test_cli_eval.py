"""Tests for cuecard CLI — eval command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import cuecard.security as security_module
from cuecard.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("cuecard.cli.main._home_dir", lambda: tmp_path)


@pytest.fixture(autouse=True)
def _ensure_dir_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add ensure_dir to cuecard.security so the lazy import in CLI works."""
    def _ensure_dir(path_str: str) -> None:
        Path(path_str).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(security_module, "ensure_dir", _ensure_dir, raising=False)


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
                "cuecard.eval.harness.load_fixtures",
                return_value=fake_fixtures,
            ) as mock_load,
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch(
                "cuecard.eval.harness.run_eval",
                return_value=fake_summary,
            ) as mock_run,
            patch(
                "cuecard.eval.harness.format_eval_report",
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
            patch("cuecard.eval.harness.load_fixtures", return_value=[]),
            patch("fastembed.TextEmbedding", return_value=mock_model) as mock_te,
            patch(
                "cuecard.eval.harness.run_eval",
                return_value=fake_summary,
            ) as mock_run,
            patch("cuecard.eval.harness.format_eval_report", return_value="OK"),
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
            patch("cuecard.eval.harness.load_fixtures", return_value=[]),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch(
                "cuecard.eval.harness.run_eval",
                return_value=fake_summary,
            ) as mock_run,
            patch("cuecard.eval.harness.format_eval_report", return_value="OK"),
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
            "cuecard.eval.harness.load_fixtures",
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
            patch("cuecard.eval.harness.load_fixtures", return_value=[]),
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
            patch("cuecard.eval.harness.load_fixtures", return_value=[]),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch(
                "cuecard.eval.harness.run_eval",
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
            patch("cuecard.eval.harness.load_fixtures", return_value=[]),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch(
                "cuecard.eval.harness.run_eval",
                return_value=fake_summary,
            ) as mock_run,
            patch("cuecard.eval.harness.format_eval_report", return_value="OK"),
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
            patch("cuecard.eval.harness.load_fixtures", return_value=[]),
            patch("fastembed.TextEmbedding", return_value=mock_model),
            patch(
                "cuecard.eval.harness.run_eval",
                return_value=fake_summary,
            ) as mock_run,
            patch("cuecard.eval.harness.format_eval_report", return_value="OK"),
        ):
            result = runner.invoke(app, [
                "eval", str(fixture_file),
                "--corpus-override", "rules_a.txt,rules_b.txt",
            ])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["corpus_override"] is not None
        assert len(call_kwargs["corpus_override"]) == 2
