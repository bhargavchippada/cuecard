"""Load, merge, and validate TOML configuration files."""

from __future__ import annotations

import dataclasses
import glob as glob_module
import logging
import os
import tomllib
from pathlib import Path
from typing import Any

from cuecard.models import (
    DEFAULT_HAIKU_MODEL,
    DEFAULT_LLM_ENDPOINT,
    KNOWN_HOOK_EVENTS,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    PipelineConfig,
    ResolvedConfig,
)
from cuecard.security import ConfigError, validate_source_path

logger = logging.getLogger(__name__)

# --- Model allowlist (fastembed catalog) ---

_ALLOWED_MODELS: frozenset[str] = frozenset({
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "jinaai/jina-embeddings-v2-base-code",
    "nomic-ai/nomic-embed-text-v1.5",
    "snowflake/snowflake-arctic-embed-m",
    "mixedbread-ai/mxbai-embed-large-v1",
    "snowflake/snowflake-arctic-embed-s",
    "BAAI/bge-large-en-v1.5",
})

# --- Valid hook events (canonical set from models.py + SessionStart) ---

_VALID_HOOK_EVENTS: frozenset[str] = (
    KNOWN_HOOK_EVENTS | {"SessionStart"}
)

# --- Defaults and validators derived from ResolvedConfig ---

# TOML-configurable field names (flat keys used after _extract_flat).
# "model" in TOML maps to "model_name" in ResolvedConfig.
_TOML_TO_FIELD: dict[str, str] = {"model": "model_name"}
_FIELD_TO_TOML: dict[str, str] = {"model_name": "model"}

# Fields that participate in TOML merge (project wins → global → default).
# Derived from ResolvedConfig fields that have defaults.
_TOML_DEFAULTS: dict[str, Any] = {}
for _f in dataclasses.fields(ResolvedConfig):
    if _f.default is not dataclasses.MISSING:
        _toml_key = _FIELD_TO_TOML.get(_f.name, _f.name)
        _TOML_DEFAULTS[_toml_key] = _f.default
    elif _f.default_factory is not dataclasses.MISSING:
        _toml_key = _FIELD_TO_TOML.get(_f.name, _f.name)
        _TOML_DEFAULTS[_toml_key] = _f.default_factory()

# Range validators derived from field metadata {"min": ..., "max": ...}.
_VALIDATORS: dict[str, tuple[type, float | int, float | int]] = {}
for _f in dataclasses.fields(ResolvedConfig):
    if "min" in _f.metadata:
        _toml_key = _FIELD_TO_TOML.get(_f.name, _f.name)
        # Resolve type from default value (annotation is str due to __future__).
        # Falls back to int if default_factory — no current field hits this,
        # but if one does, add explicit type resolution here.
        _val_type = type(_f.default) if _f.default is not dataclasses.MISSING else int
        _VALIDATORS[_toml_key] = (_val_type, _f.metadata["min"], _f.metadata["max"])

# Global-only fields: project config cannot override these.
_GLOBAL_ONLY: frozenset[str] = frozenset({"redact"})

# Fields excluded from the generic merge loop (computed or special-cased).
_SKIP_IN_MERGE: frozenset[str] = frozenset({
    "source_paths", "global_source_paths", "project_source_paths",
    "global_cache_dir", "project_cache_dir", "allowed_dirs",
    "pipeline", "source_rules",
})


# --- Validation ---


def _validate_field(name: str, value: object) -> None:
    """Validate a config field against its constraints."""
    if name not in _VALIDATORS:
        return
    expected_type, min_val, max_val = _VALIDATORS[name]
    if not isinstance(value, expected_type):
        # Accept int for float fields
        if expected_type is float and isinstance(value, int):
            pass
        else:
            msg = (
                f"Config field {name!r} must be"
                f" {expected_type.__name__},"
                f" got {type(value).__name__}"
            )
            raise ConfigError(msg)
    if min_val is not None and value < min_val:  # type: ignore[operator]
        msg = f"Config field {name!r} must be >= {min_val}, got {value}"
        raise ConfigError(msg)
    if max_val is not None and value > max_val:  # type: ignore[operator]
        msg = f"Config field {name!r} must be <= {max_val}, got {value}"
        raise ConfigError(msg)


def _validate_bool(name: str, value: object) -> None:
    """Validate that a value is a boolean."""
    if not isinstance(value, bool):
        msg = f"Config field {name!r} must be bool, got {type(value).__name__}"
        raise ConfigError(msg)


def _validate_str(name: str, value: object) -> None:
    """Validate that a value is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        msg = f"Config field {name!r} must be a non-empty string"
        raise ConfigError(msg)


# --- TOML loading ---


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file, returning empty dict if not found."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _extract_flat(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract a flat dict of config values from nested TOML structure."""
    flat: dict[str, Any] = {}

    retrieval = raw.get("retrieval", {})
    for key in (
        "top_k", "threshold", "dedup_threshold",
        "fusion_k", "llm_candidates", "sparse_enabled",
        "dense_weight", "sparse_weight",
        "affinity_mode", "llm_recall_threshold",
        "reranker_model",
    ):
        if key in retrieval:
            flat[key] = retrieval[key]

    hooks = raw.get("hooks", {})
    if "query_max_length" in hooks:
        flat["query_max_length"] = hooks["query_max_length"]
    if "events" in hooks:
        flat["hook_events"] = hooks["events"]

    log = raw.get("logging", {})
    for key in ("max_log_size_mb", "verbose", "redact"):
        if key in log:
            flat[key] = log[key]

    embedding = raw.get("embedding", {})
    if "model" in embedding:
        flat["model"] = embedding["model"]

    sources = raw.get("sources", {})
    if "rules" in sources:
        flat["source_rules"] = sources["rules"]
    if "allowed_dirs" in sources:
        flat["allowed_dirs"] = sources["allowed_dirs"]

    serve = raw.get("serve", {})
    if "port" in serve:
        flat["serve_port"] = serve["port"]

    expansion = raw.get("expansion", {})
    for key in ("max_per_rule", "max_length", "dedup_threshold"):
        if key in expansion:
            flat[f"expansion_{key}"] = expansion[key]

    pipeline_section = raw.get("pipeline", {})
    if pipeline_section:
        if "mode" in pipeline_section:
            flat["pipeline_mode"] = pipeline_section["mode"]
        llm_section = pipeline_section.get("llm", {})
        if llm_section:
            if "local_endpoint" in llm_section:
                flat["pipeline_local_endpoint"] = llm_section["local_endpoint"]
            if "haiku_model" in llm_section:
                flat["pipeline_haiku_model"] = llm_section["haiku_model"]
            if "thinking" in llm_section:
                flat["pipeline_thinking"] = llm_section["thinking"]
            if "max_tokens" in llm_section:
                flat["pipeline_llm_max_tokens"] = llm_section["max_tokens"]
            if "timeout" in llm_section:
                flat["pipeline_llm_timeout"] = llm_section["timeout"]

    return flat


# --- Path resolution with glob expansion ---


def _resolve_source_paths(
    raw_paths: list[str],
    config_dir: Path,
    allowed_dirs: tuple[str, ...],
    *,
    is_global: bool,
) -> tuple[str, ...]:
    """Resolve, validate, and glob-expand source paths."""
    resolved: list[str] = []
    seen: set[str] = set()

    for raw in raw_paths:
        # Check for glob patterns
        if "*" in raw or "?" in raw:
            # Expand tilde first for glob
            expanded_pattern = os.path.expanduser(raw)
            if not Path(expanded_pattern).is_absolute():
                expanded_pattern = str(config_dir / expanded_pattern)

            matches = sorted(glob_module.glob(expanded_pattern, recursive=True))
            if not matches:
                logger.warning("Glob pattern %r matched zero files", raw)
                continue

            for match in matches:
                validated = validate_source_path(
                    match,
                    config_dir,
                    allowed_dirs,
                    is_global=is_global,
                )
                path_str = str(validated)
                if path_str not in seen:
                    seen.add(path_str)
                    resolved.append(path_str)
        else:
            validated = validate_source_path(
                raw,
                config_dir,
                allowed_dirs,
                is_global=is_global,
            )
            path_str = str(validated)
            if path_str not in seen:
                seen.add(path_str)
                resolved.append(path_str)

    return tuple(resolved)


# --- Main entry point ---


def load_config(
    project_dir: Path | None = None,
    *,
    home_dir: Path | None = None,
    allow_custom_model: bool = False,
) -> ResolvedConfig:
    """Load, merge, and validate cuecard configuration.

    Args:
        project_dir: Project directory containing cuecard.toml, or None.
        home_dir: Override home directory (for testing). Defaults to Path.home().
        allow_custom_model: If True, skip model allowlist validation.

    Returns:
        Fully resolved and validated ResolvedConfig.
    """
    home = home_dir if home_dir is not None else Path.home()
    global_dir = home / ".cuecard"
    global_config_path = global_dir / "config.toml"

    # Load raw configs
    global_raw = _load_toml(global_config_path)
    project_raw: dict[str, Any] = {}
    if project_dir is not None:
        project_config_path = project_dir / "cuecard.toml"
        project_raw = _load_toml(project_config_path)

    # Extract flat values
    global_flat = _extract_flat(global_raw)
    project_flat = _extract_flat(project_raw)

    # Merge all TOML-configurable fields: project wins → global → default
    merged: dict[str, Any] = {}
    for key, default in _TOML_DEFAULTS.items():
        if key in _SKIP_IN_MERGE:
            continue
        if key in _GLOBAL_ONLY:
            merged[key] = global_flat.get(key, default)
        elif key in project_flat:
            merged[key] = project_flat[key]
        elif key in global_flat:
            merged[key] = global_flat[key]
        else:
            merged[key] = default

    # Validate all range-checked fields
    for key in _VALIDATORS:
        if key in merged:
            _validate_field(key, merged[key])

    # Validate bool fields
    for key in ("verbose", "redact", "sparse_enabled"):
        if key in merged:
            _validate_bool(key, merged[key])

    # Validate string fields
    _validate_str("model", merged["model"])

    # Validate model against allowlist
    if not allow_custom_model and merged["model"] not in _ALLOWED_MODELS:
        msg = (
            f"Model {merged['model']!r} is not in the allowed model list. "
            f"Allowed: {sorted(_ALLOWED_MODELS)}"
        )
        raise ConfigError(msg)

    # Validate hook_events
    events = merged["hook_events"]
    if not isinstance(events, (list, tuple)):
        msg = (
            f"Config field 'hook_events' must be a list,"
            f" got {type(events).__name__}"
        )
        raise ConfigError(msg)
    for event in events:
        if event not in _VALID_HOOK_EVENTS:
            msg = (
                f"Unknown hook event {event!r}. "
                f"Valid events: {sorted(_VALID_HOOK_EVENTS)}"
            )
            raise ConfigError(msg)

    # Validate affinity_mode
    valid_affinity_modes = ("infer", "strict")
    affinity_mode = merged["affinity_mode"]
    if (
        not isinstance(affinity_mode, str)
        or affinity_mode not in valid_affinity_modes
    ):
        msg = (
            f"Config field 'affinity_mode' must be one of"
            f" {valid_affinity_modes}, got {affinity_mode!r}"
        )
        raise ConfigError(msg)

    # Merge allowed_dirs
    global_allowed = global_flat.get("allowed_dirs", [])
    project_allowed = project_flat.get("allowed_dirs", [])
    all_allowed = tuple(dict.fromkeys(global_allowed + project_allowed))

    # Resolve source paths (union of global + project)
    global_source_rules = global_flat.get("source_rules", [])
    project_source_rules = project_flat.get("source_rules", [])

    global_resolved = _resolve_source_paths(
        global_source_rules,
        global_dir,
        all_allowed,
        is_global=True,
    )
    project_resolved: tuple[str, ...] = ()
    if project_dir is not None:
        project_resolved = _resolve_source_paths(
            project_source_rules,
            project_dir,
            all_allowed,
            is_global=False,
        )

    # Union by resolved path (dedup)
    seen_paths: set[str] = set()
    all_paths: list[str] = []
    for p in (*global_resolved, *project_resolved):
        if p not in seen_paths:
            seen_paths.add(p)
            all_paths.append(p)

    # Fixed cache locations
    global_cache = str(global_dir / "index")
    project_cache = (
        str(project_dir / ".cuecard" / "index")
        if project_dir is not None
        else None
    )

    # Build PipelineConfig (project wins, then global, then default)
    from cuecard.retrieval.pipeline import VALID_MODES

    pipeline_mode = project_flat.get(
        "pipeline_mode",
        global_flat.get("pipeline_mode", "embedding"),
    )
    if pipeline_mode not in VALID_MODES:
        msg = f"Invalid pipeline mode {pipeline_mode!r}"
        raise ConfigError(msg)

    pipeline_local_endpoint = project_flat.get(
        "pipeline_local_endpoint",
        global_flat.get("pipeline_local_endpoint", DEFAULT_LLM_ENDPOINT),
    )
    pipeline_haiku_model = project_flat.get(
        "pipeline_haiku_model",
        global_flat.get("pipeline_haiku_model", DEFAULT_HAIKU_MODEL),
    )
    pipeline_thinking = project_flat.get(
        "pipeline_thinking",
        global_flat.get("pipeline_thinking", False),
    )
    pipeline_llm_max_tokens = project_flat.get(
        "pipeline_llm_max_tokens",
        global_flat.get("pipeline_llm_max_tokens", LLM_MAX_TOKENS),
    )
    pipeline_llm_timeout = project_flat.get(
        "pipeline_llm_timeout",
        global_flat.get("pipeline_llm_timeout", LLM_TIMEOUT),
    )

    # Validate endpoint at config load time (H1 fix)
    if pipeline_mode in ("rerank-llm-local", "llm-local"):
        from cuecard.retrieval.llm_utils import validate_endpoint
        validate_endpoint(pipeline_local_endpoint)

    # Validate haiku model at config load time (M2 fix)
    if pipeline_mode in ("rerank-llm-haiku", "llm-haiku"):
        from cuecard.retrieval.llm_utils import _ALLOWED_HAIKU_MODELS
        if pipeline_haiku_model not in _ALLOWED_HAIKU_MODELS:
            msg = (
                f"Haiku model {pipeline_haiku_model!r} not in allowlist. "
                f"Allowed: {sorted(_ALLOWED_HAIKU_MODELS)}"
            )
            raise ConfigError(msg)

    # Validate thinking is bool (M3 fix)
    if not isinstance(pipeline_thinking, bool):
        got = type(pipeline_thinking).__name__
        msg = f"pipeline.llm.thinking must be a boolean, got {got}"
        raise ConfigError(msg)

    pipeline_config = PipelineConfig(
        mode=pipeline_mode,
        local_endpoint=pipeline_local_endpoint,
        haiku_model=pipeline_haiku_model,
        thinking=pipeline_thinking,
        llm_max_tokens=pipeline_llm_max_tokens,
        llm_timeout=pipeline_llm_timeout,
    )

    return ResolvedConfig(
        source_paths=tuple(all_paths),
        global_source_paths=global_resolved,
        project_source_paths=project_resolved,
        global_cache_dir=global_cache,
        project_cache_dir=project_cache,
        allowed_dirs=all_allowed,
        model_name=merged["model"],
        top_k=merged["top_k"],
        threshold=float(merged["threshold"]),
        dedup_threshold=float(merged["dedup_threshold"]),
        query_max_length=merged["query_max_length"],
        hook_events=tuple(merged["hook_events"]),
        verbose=merged["verbose"],
        redact=merged["redact"],
        max_log_size_mb=merged["max_log_size_mb"],
        pipeline=pipeline_config,
        fusion_k=merged["fusion_k"],
        llm_candidates=merged["llm_candidates"],
        sparse_enabled=merged["sparse_enabled"],
        dense_weight=float(merged["dense_weight"]),
        sparse_weight=float(merged["sparse_weight"]),
        reranker_model=merged["reranker_model"],
        serve_port=merged["serve_port"],
        expansion_max_per_rule=merged["expansion_max_per_rule"],
        expansion_max_length=merged["expansion_max_length"],
        expansion_dedup_threshold=float(merged["expansion_dedup_threshold"]),
        llm_recall_threshold=float(merged["llm_recall_threshold"]),
        affinity_mode=merged["affinity_mode"],
    )
