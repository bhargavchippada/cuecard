"""Tests for cuecard.config."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from pathlib import Path


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
            "hooks": {"query_max_length": 300, "events": ["PreToolUse", "Stop"]},
            "logging": {"max_log_size_mb": 50, "verbose": True, "redact": False},
            "embedding": {"model": "custom/model"},
            "sources": {"rules": ["a.txt"], "allowed_dirs": ["/ext"]},
            "serve": {"port": 9000},
            "pipeline": {
                "mode": "llm-local",
                "llm": {"max_tokens": 256, "timeout": 12.5},
            },
        }
        flat = _extract_flat(raw)
        assert flat["top_k"] == 10
        assert flat["threshold"] == 0.5
        assert flat["dedup_threshold"] == 0.9
        assert flat["query_max_length"] == 300
        assert flat["hook_events"] == ["PreToolUse", "Stop"]
        assert flat["max_log_size_mb"] == 50
        assert flat["verbose"] is True
        assert flat["redact"] is False
        assert flat["model"] == "custom/model"
        assert flat["source_rules"] == ["a.txt"]
        assert flat["allowed_dirs"] == ["/ext"]
        assert flat["serve_port"] == 9000
        assert flat["pipeline_mode"] == "llm-local"
        assert flat["pipeline_llm_max_tokens"] == 256
        assert flat["pipeline_llm_timeout"] == 12.5

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
        assert config.top_k == 7
        assert config.threshold == 0.30
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

    def test_glob_zero_matches_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
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
            '[hooks]\nevents = ["PreToolUse", "Stop"]\n'
        )
        config = load_config(project_dir=project, home_dir=home)
        assert config.hook_events == ("PreToolUse", "Stop")

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


class TestPipelineConfig:
    def test_default_pipeline_config(self, tmp_path: Path) -> None:
        """No [pipeline] section yields default PipelineConfig."""
        home = tmp_path / "home"
        home.mkdir()
        config = load_config(home_dir=home)
        assert config.pipeline.mode == "embedding"
        assert config.pipeline.local_endpoint == "http://localhost:8081/v1"
        assert config.pipeline.haiku_model == "claude-haiku-4-5"
        assert config.pipeline.thinking is False

    def test_pipeline_mode_from_project(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            '[pipeline]\nmode = "rerank"\n'
        )
        config = load_config(project_dir=project, home_dir=home)
        assert config.pipeline.mode == "rerank"

    def test_pipeline_llm_settings(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            '[pipeline]\n'
            'mode = "rerank-llm-local"\n\n'
            '[pipeline.llm]\n'
            'local_endpoint = "http://localhost:9999/v1"\n'
            'haiku_model = "claude-haiku-4-5-2025"\n'
            'thinking = true\n'
        )
        config = load_config(project_dir=project, home_dir=home)
        assert config.pipeline.mode == "rerank-llm-local"
        assert config.pipeline.local_endpoint == "http://localhost:9999/v1"
        assert config.pipeline.haiku_model == "claude-haiku-4-5-2025"
        assert config.pipeline.thinking is True

    def test_pipeline_invalid_mode_raises(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            '[pipeline]\nmode = "invalid-mode"\n'
        )
        with pytest.raises(ConfigError, match="Invalid pipeline mode"):
            load_config(project_dir=project, home_dir=home)

    def test_haiku_model_invalid_raises(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            '[pipeline]\n'
            'mode = "rerank-llm-haiku"\n\n'
            '[pipeline.llm]\n'
            'haiku_model = "gpt-4o"\n'
        )
        with pytest.raises(ConfigError, match="not in allowlist"):
            load_config(project_dir=project, home_dir=home)

    def test_thinking_non_bool_raises(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            '[pipeline]\nmode = "embedding"\n\n'
            '[pipeline.llm]\nthinking = "yes"\n'
        )
        with pytest.raises(ConfigError, match="must be a boolean"):
            load_config(project_dir=project, home_dir=home)

    def test_pipeline_extract_flat(self) -> None:
        raw = {
            "pipeline": {
                "mode": "rerank",
                "llm": {
                    "local_endpoint": "http://localhost:1234/v1",
                    "haiku_model": "test-model",
                    "thinking": True,
                },
            },
        }
        flat = _extract_flat(raw)
        assert flat["pipeline_mode"] == "rerank"
        assert flat["pipeline_local_endpoint"] == "http://localhost:1234/v1"
        assert flat["pipeline_haiku_model"] == "test-model"
        assert flat["pipeline_thinking"] is True

    def test_pipeline_extract_flat_empty(self) -> None:
        """Empty pipeline section produces no pipeline keys."""
        flat = _extract_flat({})
        assert "pipeline_mode" not in flat

    def test_pipeline_global_fallback(self, tmp_path: Path) -> None:
        """Global pipeline config used when project doesn't set it."""
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        (cuecard_dir / "config.toml").write_text(
            '[pipeline]\nmode = "rerank"\n'
        )
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text("")
        config = load_config(project_dir=project, home_dir=home)
        assert config.pipeline.mode == "rerank"


class TestAllowedModels:
    def test_all_defaults_in_allowlist(self) -> None:
        assert "BAAI/bge-small-en-v1.5" in _ALLOWED_MODELS

    def test_allowlist_is_frozenset(self) -> None:
        assert isinstance(_ALLOWED_MODELS, frozenset)

    def test_valid_hook_events_is_frozenset(self) -> None:
        assert isinstance(_VALID_HOOK_EVENTS, frozenset)


class TestEnrichedRetrievalConfig:
    """Tests for enriched retrieval config fields."""

    def test_defaults(self, tmp_path: Path) -> None:
        """Enriched fields default correctly when no TOML is present."""
        home = tmp_path / "home"
        (home / ".cuecard").mkdir(parents=True)
        config = load_config(home_dir=home)
        assert config.fusion_k == 10
        assert config.llm_candidates == 12
        assert config.sparse_enabled is True
        assert config.dense_weight == 0.7
        assert config.sparse_weight == 0.3
        assert config.expansion_max_per_rule == 5
        assert config.expansion_max_length == 500

    def test_from_global_toml(self, tmp_path: Path) -> None:
        """Enriched fields are read from [retrieval] and [expansion]."""
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        (cuecard_dir / "config.toml").write_text(
            "[retrieval]\n"
            "fusion_k = 100\n"
            "sparse_enabled = false\n"
            "dense_weight = 0.6\n"
            "sparse_weight = 0.4\n"
            "\n"
            "[expansion]\n"
            "max_per_rule = 20\n"
            "max_length = 150\n"
        )
        config = load_config(home_dir=home)
        assert config.fusion_k == 100
        assert config.sparse_enabled is False
        assert config.dense_weight == 0.6
        assert config.sparse_weight == 0.4
        assert config.expansion_max_per_rule == 20
        assert config.expansion_max_length == 150

    def test_project_overrides_global(self, tmp_path: Path) -> None:
        """Project TOML overrides global for enriched fields."""
        home = tmp_path / "home"
        cuecard_dir = home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        (cuecard_dir / "config.toml").write_text(
            "[retrieval]\nfusion_k = 100\n"
        )
        project = tmp_path / "project"
        project.mkdir()
        (project / "cuecard.toml").write_text(
            "[retrieval]\nfusion_k = 42\n"
        )
        config = load_config(project_dir=project, home_dir=home)
        assert config.fusion_k == 42

    def test_extract_flat_retrieval_fields(self) -> None:
        """_extract_flat reads retrieval fields from [retrieval]."""
        raw = {
            "retrieval": {
                "fusion_k": 80,
                "sparse_enabled": False,
                "dense_weight": 0.65,
                "sparse_weight": 0.35,
            },
        }
        flat = _extract_flat(raw)
        assert flat["fusion_k"] == 80
        assert flat["sparse_enabled"] is False
        assert flat["dense_weight"] == 0.65
        assert flat["sparse_weight"] == 0.35

    def test_extract_flat_expansion_fields(self) -> None:
        """_extract_flat reads max_per_rule and max_length from [expansion]."""
        raw = {
            "expansion": {"max_per_rule": 15, "max_length": 300},
        }
        flat = _extract_flat(raw)
        assert flat["expansion_max_per_rule"] == 15
        assert flat["expansion_max_length"] == 300

    def test_fusion_k_validation(self, tmp_path: Path) -> None:
        """fusion_k must be an integer in [1, 1000]."""
        home = tmp_path / "home"
        (home / ".cuecard").mkdir(parents=True)
        (home / ".cuecard" / "config.toml").write_text(
            "[retrieval]\nfusion_k = 0\n"
        )
        with pytest.raises(ConfigError, match="fusion_k"):
            load_config(home_dir=home)

    def test_sparse_enabled_validation(self, tmp_path: Path) -> None:
        """sparse_enabled must be a boolean."""
        home = tmp_path / "home"
        (home / ".cuecard").mkdir(parents=True)
        (home / ".cuecard" / "config.toml").write_text(
            "[retrieval]\nsparse_enabled = 42\n"
        )
        with pytest.raises(ConfigError, match="sparse_enabled"):
            load_config(home_dir=home)

    def test_dense_weight_validation(self, tmp_path: Path) -> None:
        """dense_weight must be within [0, 1]."""
        home = tmp_path / "home"
        (home / ".cuecard").mkdir(parents=True)
        (home / ".cuecard" / "config.toml").write_text(
            "[retrieval]\ndense_weight = 1.2\n"
        )
        with pytest.raises(ConfigError, match="dense_weight"):
            load_config(home_dir=home)

    def test_sparse_weight_validation(self, tmp_path: Path) -> None:
        """sparse_weight must be within [0, 1]."""
        home = tmp_path / "home"
        (home / ".cuecard").mkdir(parents=True)
        (home / ".cuecard" / "config.toml").write_text(
            "[retrieval]\nsparse_weight = -0.1\n"
        )
        with pytest.raises(ConfigError, match="sparse_weight"):
            load_config(home_dir=home)

    def test_affinity_mode_default(self, tmp_path: Path) -> None:
        """affinity_mode defaults to 'infer'."""
        home = tmp_path / "home"
        (home / ".cuecard").mkdir(parents=True)
        config = load_config(home_dir=home)
        assert config.affinity_mode == "infer"

    def test_affinity_mode_strict(self, tmp_path: Path) -> None:
        """affinity_mode can be set to 'strict'."""
        home = tmp_path / "home"
        (home / ".cuecard").mkdir(parents=True)
        (home / ".cuecard" / "config.toml").write_text(
            '[retrieval]\naffinity_mode = "strict"\n',
        )
        config = load_config(home_dir=home)
        assert config.affinity_mode == "strict"

    def test_affinity_mode_invalid_raises(
        self, tmp_path: Path,
    ) -> None:
        """Invalid affinity_mode raises ConfigError."""
        home = tmp_path / "home"
        (home / ".cuecard").mkdir(parents=True)
        (home / ".cuecard" / "config.toml").write_text(
            '[retrieval]\naffinity_mode = "bogus"\n',
        )
        with pytest.raises(ConfigError, match="affinity_mode"):
            load_config(home_dir=home)

    def test_extract_flat_affinity_mode(self) -> None:
        """_extract_flat reads affinity_mode from [retrieval]."""
        raw = {"retrieval": {"affinity_mode": "strict"}}
        flat = _extract_flat(raw)
        assert flat["affinity_mode"] == "strict"
