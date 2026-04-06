"""Tests for cuecard CLI — install, uninstall, status, log, hook commands."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

import cuecard.security as security_module
from cuecard.cli import app
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
        for event in ("PreToolUse", "UserPromptSubmit"):
            hooks = settings["hooks"][event]
            assert any(
                "cuecard" in str(h.get("hooks", []))
                for h in hooks
            )

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
                        "command": "cuecard hook 2>/dev/null",
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
        assert "UserPromptSubmit" in settings["hooks"]


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
                    {"matcher": "", "hooks": [
                        {"type": "command", "command": "cuecard hook 2>/dev/null"},
                    ]},
                ],
                "UserPromptSubmit": [
                    {"matcher": "", "hooks": [
                        {"type": "command", "command": "cuecard hook 2>/dev/null"},
                    ]},
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
                    {"matcher": "Bash", "hooks": [
                        {"type": "command", "command": "other-tool"},
                    ]},
                    {"matcher": "", "hooks": [
                        {"type": "command", "command": "cuecard hook 2>/dev/null"},
                    ]},
                ],
            },
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings))
        result = runner.invoke(app, ["uninstall", "claude-code"])
        assert result.exit_code == 0
        updated = json.loads((claude_dir / "settings.json").read_text())
        assert len(updated["hooks"]["PreToolUse"]) == 1
        assert "other-tool" in str(updated["hooks"]["PreToolUse"][0])


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
                    {"matcher": "", "hooks": [
                        {"type": "command", "command": "cuecard hook 2>/dev/null"},
                    ]},
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

    def test_status_global_index_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_home(monkeypatch, tmp_path)
        _setup_home(tmp_path)
        monkeypatch.chdir(tmp_path)

        idx = _make_sample_index()
        with patch("cuecard.indexer.load_index", return_value=idx):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Global index" in result.output

    def test_status_project_index_only(
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

        def _load_by_path(cache_dir: str) -> Index | None:
            if "myproject" in cache_dir:
                return idx
            return None

        with patch("cuecard.indexer.load_index", side_effect=_load_by_path):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Project index" in result.output
        assert "No valid index" not in result.output

    def test_status_both_indexes(
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
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Global index" in result.output
        assert "Project index" in result.output


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


class TestHookCommand:
    def test_hook_invokes_adapter(self) -> None:
        with patch("cuecard.adapters.claude_code.main") as mock_main:
            runner.invoke(app, ["hook"])
            mock_main.assert_called_once()


class TestAdapterMainGuard:
    def test_main_guard(self) -> None:
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

    def test_has_hook_nested_hooks_format(self) -> None:
        """Settings.json uses nested {matcher, hooks: [...]} format."""
        from cuecard.cli import _has_cuecard_hook

        settings: dict[str, object] = {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "",
                        "hooks": [
                            {"type": "command", "command": "cuecard hook 2>/dev/null"},
                        ],
                    },
                ],
            },
        }
        assert _has_cuecard_hook(settings) is True

    def test_entry_has_cuecard_non_dict(self) -> None:
        from cuecard.cli_hooks import _entry_has_cuecard

        assert _entry_has_cuecard("not a dict") is False

    def test_entry_has_cuecard_hooks_not_list(self) -> None:
        from cuecard.cli_hooks import _entry_has_cuecard

        assert _entry_has_cuecard({"hooks": "not_a_list"}) is False

    def test_claude_settings_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cuecard.cli import _claude_settings_path

        _patch_home(monkeypatch, tmp_path)
        p = _claude_settings_path()
        assert p == tmp_path / ".claude" / "settings.json"
