"""Tests for cuecard.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from cuecard.config import (
    _ALLOWED_MODELS,
    _VALID_HOOK_EVENTS,
    _extract_flat,
    _load_toml,
    _validate_bool,
    _validate_field,
    _validate_str,
    load_config,
)
from cuecard.models import ResolvedConfig
from cuecard.security import ConfigError


class TestValidateField:
    def test_top_k_valid(self) -> None:
        _validate_field("top_k", 5)

    def test_top_k_min(self) -> None:
        with pytest.raises(ConfigError, match=">="):
            _validate_field("top_k", 0)

    def test_top_k_max(self) -> None:
        with pytest.raises(ConfigError, match="<="):
            _validate_field("top_k", 51)

    def test_top_k_wrong_type(self) -> None:
        with pytest.raises(ConfigError, match="must be int"):
            _validate_field("top_k", 5.0)

    def test_threshold_valid(self) -> None:
        _validate_field("threshold", 0.5)

    def test_threshold_int_accepted(self) -> None:
        """Int values should be accepted for float fields."""
        _validate_field("threshold", 0)
        _validate_field("threshold", 1)

    def test_threshold_below_range(self) -> None:
        with pytest.raises(ConfigError, match=">="):
            _validate_field("threshold", -0.1)

    def test_threshold_above_range(self) -> None:
        with pytest.raises(ConfigError, match="<="):
            _validate_field("threshold", 1.1)

    def test_dedup_threshold_valid(self) -> None:
        _validate_field("dedup_threshold", 0.95)

    def test_query_max_length_valid(self) -> None:
        _validate_field("query_max_length", 500)

    def test_query_max_length_below(self) -> None:
        with pytest.raises(ConfigError, match=">="):
            _validate_field("query_max_length", 49)

    def test_max_log_size_mb_valid(self) -> None:
        _validate_field("max_log_size_mb", 10)

    def test_unknown_field_passes(self) -> None:
        _validate_field("unknown", "anything")


class TestValidateBool:
    def test_valid(self) -> None:
        _validate_bool("verbose", True)
        _validate_bool("verbose", False)

    def test_invalid(self) -> None:
        with pytest.raises(ConfigError, match="must be bool"):
            _validate_bool("verbose", "true")


class TestValidateStr:
    def test_valid(self) -> None:
        _validate_str("model", "BAAI/bge-small-en-v1.5")

    def test_empty(self) -> None:
        with pytest.raises(ConfigError, match="non-empty string"):
            _validate_str("model", "")

    def test_whitespace_only(self) -> None:
        with pytest.raises(ConfigError, match="non-empty string"):
            _validate_str("model", "   ")

    def test_not_string(self) -> None:
        with pytest.raises(ConfigError, match="non-empty string"):
            _validate_str("model", 42)


class TestLoadToml:
    def test_nonexistent(self, tmp_path: Path) -> None:
        assert _load_toml(tmp_path / "missing.toml") == {}

    def test_valid(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text('[retrieval]\ntop_k = 10\n')
        result = _load_toml(config)
        assert result["retrieval"]["top_k"] == 10


class TestExtractFlat:
    def test_full_config(self) -> None:
        raw = {
            "retrieval": {"top_k": 10, "threshold": 0.5, "dedup_threshold": 0.9},
            "hooks": {"query_max_length": 300, "events": ["PreToolUse", "PostToolUse"]},
            "logging": {"max_log_size_mb": 50, "verbose": True, "redact": False},
            "embedding": {"model": "custom/model"},
            "sources": {"rules": ["a.txt"], "allowed_dirs": ["/ext"]},
        }
        flat = _extract_flat(raw)
        assert flat["top_k"] == 10
        assert flat["threshold"] == 0.5
        assert flat["dedup_threshold"] == 0.9
        assert flat["query_max_length"] == 300
        assert flat["hook_events"] == ["PreToolUse", "PostToolUse"]
        assert flat["max_log_size_mb"] == 50
        assert flat["verbose"] is True
        assert flat["redact"] is False
        assert flat["model"] == "custom/model"
        assert flat["source_rules"] == ["a.txt"]
        assert flat["allowed_dirs"] == ["/ext"]

    def test_empty(self) -> None:
        assert _extract_flat({}) == {}

    def test_partial(self) -> None:
        raw = {"retrieval": {"top_k": 3}}
        flat = _extract_flat(raw)
        assert flat == {"top_k": 3}


class TestLoadConfig:
    def test_defaults_no_config_files(self, tmp_path: Path) -> None:
        """When no config files exist, all defaults should be used."""
        home = tmp_path / "home"
        home.mkdir()
        config = load_config(home_dir=home)
        assert isinstance(config, ResolvedConfig)
        assert config.top_k == 5
        assert config.threshold == 0.35
        assert config.dedup_threshold == 0.95
        assert config.query_max_length == 500
        assert config.max_log_size_mb == 10
        assert config.verbose is False
        assert config.redact is True
        assert config.model_name == "BAAI/bge-small-en-v1.5"
        assert config.hook_events == ("PreToolUse",)
        assert config.source_paths == ()
        assert config.global_cache_dir == str(home / ".cuecard" / "index")
        assert config.project_cache_dir is None

    def test_global_config_only(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        rules = cuecard_dir / "global.txt"
        rules.write_text("rule1\n")
        config_file = cuecard_dir / "config.toml"
        config_file.write_text(
            '[sources]\nrules = ["global.txt"]\n'
            '[retrieval]\ntop_k = 10\n'
        )
        config = load_config(home_dir=home)
        assert config.top_k == 10
        assert len(config.source_paths) == 1
        assert config.source_paths[0] == str(rules.resolve())

    def test_project_overrides_global(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        (cuecard_dir / "config.toml").write_text(
            '[retrieval]\ntop_k = 3\nthreshold = 0.2\n'
        )
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            '[retrieval]\ntop_k = 15\n'
        )
        config = load_config(project_dir=project, home_dir=home)
        assert config.top_k == 15
        assert config.threshold == 0.2  # inherited from global

    def test_redact_global_only(self, tmp_path: Path) -> None:
        """Project cannot override redact."""
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        (cuecard_dir / "config.toml").write_text(
            '[logging]\nredact = true\n'
        )
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            '[logging]\nredact = false\n'
        )
        config = load_config(project_dir=project, home_dir=home)
        assert config.redact is True

    def test_source_union(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        global_rules = cuecard_dir / "global.txt"
        global_rules.write_text("rule\n")
        (cuecard_dir / "config.toml").write_text(
            '[sources]\nrules = ["global.txt"]\n'
        )
        project = tmp_path / "project"
        project.mkdir()
        proj_rules = project / "rules.txt"
        proj_rules.write_text("rule\n")
        (project / "cuecard.toml").write_text(
            '[sources]\nrules = ["rules.txt"]\n'
        )
        config = load_config(project_dir=project, home_dir=home)
        assert len(config.source_paths) == 2

    def test_source_dedup(self, tmp_path: Path) -> None:
        """Same resolved path from both configs should appear once."""
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        shared = cuecard_dir / "shared.txt"
        shared.write_text("rule\n")

        project = tmp_path / "project"
        project.mkdir()

        # Both configs reference the same file
        (cuecard_dir / "config.toml").write_text(
            f'[sources]\nrules = ["{shared}"]\n'
        )
        (project / "cuecard.toml").write_text(
            f'[sources]\nrules = ["{shared}"]\nallowed_dirs = ["{cuecard_dir}"]\n'
        )
        config = load_config(project_dir=project, home_dir=home)
        assert len(config.source_paths) == 1

    def test_project_cache_dir(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        config = load_config(project_dir=project, home_dir=home)
        assert config.project_cache_dir == str(project / ".cuecard" / "index")

    def test_invalid_top_k_raises(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        (cuecard_dir / "config.toml").write_text(
            '[retrieval]\ntop_k = 100\n'
        )
        with pytest.raises(ConfigError, match="<="):
            load_config(home_dir=home)

    def test_invalid_verbose_raises(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        (cuecard_dir / "config.toml").write_text(
            '[logging]\nverbose = "yes"\n'
        )
        with pytest.raises(ConfigError, match="must be bool"):
            load_config(home_dir=home)

    def test_invalid_model_raises(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        (cuecard_dir / "config.toml").write_text(
            '[embedding]\nmodel = ""\n'
        )
        with pytest.raises(ConfigError, match="non-empty string"):
            load_config(home_dir=home)

    def test_glob_expansion(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        (cuecard_dir / "a.txt").write_text("rule\n")
        (cuecard_dir / "b.txt").write_text("rule\n")
        (cuecard_dir / "config.toml").write_text(
            '[sources]\nrules = ["*.txt"]\n'
        )
        config = load_config(home_dir=home)
        assert len(config.source_paths) == 2

    def test_glob_zero_matches_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        (cuecard_dir / "config.toml").write_text(
            '[sources]\nrules = ["*.nonexistent"]\n'
        )
        import logging

        with caplog.at_level(logging.WARNING):
            config = load_config(home_dir=home)
        assert len(config.source_paths) == 0
        assert "zero files" in caplog.text

    def test_allowed_dirs_merged(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        ext1 = tmp_path / "ext1"
        ext1.mkdir()
        ext2 = tmp_path / "ext2"
        ext2.mkdir()
        (cuecard_dir / "config.toml").write_text(
            f'[sources]\nallowed_dirs = ["{ext1}"]\n'
        )
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            f'[sources]\nallowed_dirs = ["{ext2}"]\n'
        )
        config = load_config(project_dir=project, home_dir=home)
        assert str(ext1) in config.allowed_dirs
        assert str(ext2) in config.allowed_dirs

    def test_hook_events_override(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            '[hooks]\nevents = ["PreToolUse", "PostToolUse"]\n'
        )
        config = load_config(project_dir=project, home_dir=home)
        assert config.hook_events == ("PreToolUse", "PostToolUse")

    def test_model_override(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            '[embedding]\nmodel = "BAAI/bge-base-en-v1.5"\n'
        )
        config = load_config(project_dir=project, home_dir=home)
        assert config.model_name == "BAAI/bge-base-en-v1.5"

    def test_model_not_in_allowlist_raises(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            '[embedding]\nmodel = "custom/model"\n'
        )
        with pytest.raises(ConfigError, match="not in the allowed model list"):
            load_config(project_dir=project, home_dir=home)

    def test_no_project_dir(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        config = load_config(project_dir=None, home_dir=home)
        assert config.project_cache_dir is None
        assert config.source_paths == ()

    def test_hook_events_not_a_list_raises(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            '[hooks]\nevents = "PreToolUse"\n'
        )
        with pytest.raises(ConfigError, match="must be a list"):
            load_config(project_dir=project, home_dir=home)

    def test_hook_events_unknown_raises(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            '[hooks]\nevents = ["PreToolUse", "InvalidEvent"]\n'
        )
        with pytest.raises(ConfigError, match="Unknown hook event"):
            load_config(project_dir=project, home_dir=home)

    def test_hook_events_all_valid(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        all_events = sorted(_VALID_HOOK_EVENTS)
        events_str = ", ".join(f'"{e}"' for e in all_events)
        (project / "cuecard.toml").write_text(
            f'[hooks]\nevents = [{events_str}]\n'
        )
        config = load_config(project_dir=project, home_dir=home)
        assert set(config.hook_events) == _VALID_HOOK_EVENTS


class TestAllowedModels:
    def test_all_defaults_in_allowlist(self) -> None:
        assert "BAAI/bge-small-en-v1.5" in _ALLOWED_MODELS

    def test_allowlist_is_frozenset(self) -> None:
        assert isinstance(_ALLOWED_MODELS, frozenset)

    def test_valid_hook_events_is_frozenset(self) -> None:
        assert isinstance(_VALID_HOOK_EVENTS, frozenset)
