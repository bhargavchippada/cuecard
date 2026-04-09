"""Tests for cuecard CLI — config and configure commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import cuecard.security as security_module
from cuecard.cli.main import app

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
# _load_config_or_exit error path
# ---------------------------------------------------------------------------


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
# configure command
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_configure_fresh_embedding_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fresh configure with all defaults (embedding mode)."""
        _patch_home(monkeypatch, tmp_path)
        result = runner.invoke(
            app,
            ["configure"],
            input="embedding\n5\n0.30\ny\n",
        )
        assert result.exit_code == 0
        config_path = tmp_path / ".cuecard" / "config.toml"
        assert config_path.exists()
        content = config_path.read_text()
        assert 'mode = "embedding"' in content
        assert "top_k = 5" in content
        assert "threshold = 0.30" in content
        assert "sparse_enabled = true" in content
        assert "Next steps" in result.output

    def test_configure_llm_local_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configure with llm-local mode prompts for endpoint."""
        _patch_home(monkeypatch, tmp_path)
        result = runner.invoke(
            app,
            ["configure"],
            input="llm-local\nhttp://localhost:9999/v1\n3\n0.25\nn\n",
        )
        assert result.exit_code == 0
        content = (tmp_path / ".cuecard" / "config.toml").read_text()
        assert 'mode = "llm-local"' in content
        assert 'local_endpoint = "http://localhost:9999/v1"' in content
        assert "top_k = 3" in content
        assert "sparse_enabled = false" in content

    def test_configure_llm_haiku_with_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configure with llm-haiku mode when API key is present."""
        _patch_home(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        result = runner.invoke(
            app,
            ["configure"],
            input="llm-haiku\n5\n0.30\ny\n",
        )
        assert result.exit_code == 0
        content = (tmp_path / ".cuecard" / "config.toml").read_text()
        assert 'mode = "llm-haiku"' in content
        assert "Anthropic API key found" in result.output

    def test_configure_llm_haiku_no_key_confirmed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configure with llm-haiku, no env key, user confirms they have one."""
        _patch_home(monkeypatch, tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = runner.invoke(
            app,
            ["configure"],
            input="llm-haiku\ny\n5\n0.30\ny\n",
        )
        assert result.exit_code == 0
        content = (tmp_path / ".cuecard" / "config.toml").read_text()
        assert 'mode = "llm-haiku"' in content

    def test_configure_llm_haiku_no_key_denied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configure with llm-haiku, no key, user says no -- exits."""
        _patch_home(monkeypatch, tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = runner.invoke(
            app,
            ["configure"],
            input="llm-haiku\nn\n",
        )
        assert result.exit_code == 1
        assert "Set ANTHROPIC_API_KEY" in result.output

    def test_configure_invalid_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invalid pipeline mode exits with error."""
        _patch_home(monkeypatch, tmp_path)
        result = runner.invoke(
            app,
            ["configure"],
            input="invalid-mode\n",
        )
        assert result.exit_code == 1
        assert "Invalid mode" in result.output

    def test_configure_preserves_existing_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Re-running configure uses previous config as defaults."""
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        (cuecard_dir / "config.toml").write_text(
            "[sources]\n"
            'rules = ["rules/custom.txt"]\n\n'
            "[embedding]\n"
            'model = "BAAI/bge-base-en-v1.5"\n\n'
            "[retrieval]\n"
            "top_k = 10\n"
            "threshold = 0.40\n"
            "sparse_enabled = false\n\n"
            "[pipeline]\n"
            'mode = "llm-local"\n\n'
            "[pipeline.llm]\n"
            'local_endpoint = "http://localhost:9000/v1"\n',
        )

        # Accept all defaults (just press enter for each prompt)
        result = runner.invoke(
            app,
            ["configure"],
            input="\n\n\n\n\n",
        )
        assert result.exit_code == 0
        content = (cuecard_dir / "config.toml").read_text()
        # Previous values should be preserved as defaults
        assert 'mode = "llm-local"' in content
        assert 'local_endpoint = "http://localhost:9000/v1"' in content
        assert "top_k = 10" in content
        assert "threshold = 0.40" in content
        assert "sparse_enabled = false" in content
        assert 'model = "BAAI/bge-base-en-v1.5"' in content
        assert "rules/custom.txt" in content

    def test_configure_overwrites_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Running configure again overwrites the config file."""
        _patch_home(monkeypatch, tmp_path)
        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        (cuecard_dir / "config.toml").write_text("[retrieval]\ntop_k = 5\n")

        result = runner.invoke(
            app,
            ["configure"],
            input="embedding\n8\n0.35\ny\n",
        )
        assert result.exit_code == 0
        content = (cuecard_dir / "config.toml").read_text()
        assert "top_k = 8" in content
        assert "threshold = 0.35" in content


class TestBuildConfigToml:
    def test_embedding_mode_no_pipeline_llm(self) -> None:
        from cuecard.cli.setup import _build_config_toml

        content = _build_config_toml(
            mode="embedding",
            model="BAAI/bge-small-en-v1.5",
            top_k=5,
            threshold=0.30,
            sparse_enabled=True,
            local_endpoint="http://localhost:8081/v1",
            rules=["rules/global.txt"],
        )
        assert "[pipeline.llm]" not in content
        assert 'mode = "embedding"' in content

    def test_llm_local_has_endpoint(self) -> None:
        from cuecard.cli.setup import _build_config_toml

        content = _build_config_toml(
            mode="llm-local",
            model="BAAI/bge-small-en-v1.5",
            top_k=5,
            threshold=0.30,
            sparse_enabled=True,
            local_endpoint="http://localhost:9999/v1",
            rules=["rules/global.txt"],
        )
        assert "[pipeline.llm]" in content
        assert 'local_endpoint = "http://localhost:9999/v1"' in content
        assert "thinking = false" in content

    def test_llm_haiku_no_endpoint(self) -> None:
        from cuecard.cli.setup import _build_config_toml

        content = _build_config_toml(
            mode="llm-haiku",
            model="BAAI/bge-small-en-v1.5",
            top_k=5,
            threshold=0.30,
            sparse_enabled=True,
            local_endpoint="http://localhost:8081/v1",
            rules=["rules/global.txt"],
        )
        assert "[pipeline.llm]" in content
        assert "local_endpoint" not in content
        assert "thinking = false" in content


class TestFormatRulesToml:
    def test_single_rule(self) -> None:
        from cuecard.cli.setup import _format_rules_toml

        assert _format_rules_toml(["a.txt"]) == '["a.txt"]'

    def test_multiple_rules(self) -> None:
        from cuecard.cli.setup import _format_rules_toml

        result = _format_rules_toml(["a.txt", "b.txt"])
        assert result == '["a.txt", "b.txt"]'


class TestLoadExistingGlobalConfig:
    def test_missing_config(self, tmp_path: Path) -> None:
        from cuecard.cli.setup import _load_existing_global_config

        result = _load_existing_global_config(tmp_path)
        assert result == {}

    def test_loads_all_fields(self, tmp_path: Path) -> None:
        from cuecard.cli.setup import _load_existing_global_config

        cuecard_dir = tmp_path / ".cuecard"
        cuecard_dir.mkdir()
        (cuecard_dir / "config.toml").write_text(
            "[pipeline]\n"
            'mode = "llm-local"\n\n'
            "[pipeline.llm]\n"
            'local_endpoint = "http://localhost:9000/v1"\n'
            "thinking = true\n\n"
            "[retrieval]\n"
            "top_k = 10\n"
            "threshold = 0.40\n"
            "sparse_enabled = false\n\n"
            "[embedding]\n"
            'model = "BAAI/bge-base-en-v1.5"\n\n'
            "[sources]\n"
            'rules = ["rules/custom.txt"]\n',
        )
        result = _load_existing_global_config(tmp_path)
        assert result["mode"] == "llm-local"
        assert result["local_endpoint"] == "http://localhost:9000/v1"
        assert result["thinking"] is True
        assert result["top_k"] == 10
        assert result["threshold"] == 0.40
        assert result["sparse_enabled"] is False
        assert result["model"] == "BAAI/bge-base-en-v1.5"
