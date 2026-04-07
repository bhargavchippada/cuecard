"""Load, merge, and validate TOML configuration files."""

from __future__ import annotations

import glob as glob_module
import logging
import os
import tomllib
from pathlib import Path
from typing import Any

from cuecard.models import KNOWN_HOOK_EVENTS, PipelineConfig, ResolvedConfig
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

# --- Defaults ---

_DEFAULTS: dict[str, Any] = {
    "top_k": 5,
    "threshold": 0.30,
    "dedup_threshold": 0.95,
    "query_max_length": 500,
    "max_log_size_mb": 10,
    "verbose": False,
    "redact": True,
    "model": "BAAI/bge-small-en-v1.5",
    "hook_events": ["PreToolUse"],
}


# --- Validation ---

_VALIDATORS: dict[str, tuple[type, float | int | None, float | int | None]] = {
    "top_k": (int, 1, 50),
    "threshold": (float, 0.0, 1.0),
    "dedup_threshold": (float, 0.0, 1.0),
    "query_max_length": (int, 50, 2000),
    "max_log_size_mb": (int, 1, 1000),
    "fusion_k": (int, 1, 1000),
    "expansion_max_per_rule": (int, 1, 100),
    "expansion_max_length": (int, 10, 2000),
}


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
    for key in ("top_k", "threshold", "dedup_threshold"):
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

    # Retrieval: fusion_k, sparse_enabled, affinity_mode
    for key in ("fusion_k", "sparse_enabled", "affinity_mode"):
        if key in retrieval:
            flat[key] = retrieval[key]

    # Expansion section
    expansion = raw.get("expansion", {})
    for key in ("max_per_rule", "max_length"):
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

    # Merge scalars: project wins, then global, then defaults
    merged: dict[str, Any] = {}
    for key, default in _DEFAULTS.items():
        if key == "redact":
            # Global-only field: project cannot override
            merged[key] = global_flat.get(key, default)
        elif key in project_flat:
            merged[key] = project_flat[key]
        elif key in global_flat:
            merged[key] = global_flat[key]
        else:
            merged[key] = default

    # Validate all fields
    for key in (
        "top_k", "threshold", "dedup_threshold",
        "query_max_length", "max_log_size_mb",
    ):
        _validate_field(key, merged[key])
    _validate_bool("verbose", merged["verbose"])
    _validate_bool("redact", merged["redact"])
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
    if not isinstance(events, list):
        msg = f"Config field 'hook_events' must be a list, got {type(events).__name__}"
        raise ConfigError(msg)
    for event in events:
        if event not in _VALID_HOOK_EVENTS:
            msg = (
                f"Unknown hook event {event!r}. "
                f"Valid events: {sorted(_VALID_HOOK_EVENTS)}"
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
    from cuecard.pipeline import VALID_MODES

    pipeline_mode = project_flat.get(
        "pipeline_mode",
        global_flat.get("pipeline_mode", "embedding"),
    )
    if pipeline_mode not in VALID_MODES:
        msg = f"Invalid pipeline mode {pipeline_mode!r}"
        raise ConfigError(msg)

    pipeline_local_endpoint = project_flat.get(
        "pipeline_local_endpoint",
        global_flat.get("pipeline_local_endpoint", "http://localhost:8081/v1"),
    )
    pipeline_haiku_model = project_flat.get(
        "pipeline_haiku_model",
        global_flat.get("pipeline_haiku_model", "claude-haiku-4-5"),
    )
    pipeline_thinking = project_flat.get(
        "pipeline_thinking",
        global_flat.get("pipeline_thinking", False),
    )

    # Validate endpoint at config load time (H1 fix)
    if pipeline_mode in ("rerank-llm-local", "llm-local"):
        from cuecard.llm_utils import validate_endpoint
        validate_endpoint(pipeline_local_endpoint)

    # Validate haiku model at config load time (M2 fix)
    if pipeline_mode in ("rerank-llm-haiku", "llm-haiku"):
        from cuecard.llm_utils import _ALLOWED_HAIKU_MODELS
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
    )

    # Resolve enriched retrieval fields (project wins → global → default)
    _enriched_defaults: dict[str, int | bool | str] = {
        "fusion_k": 60,
        "sparse_enabled": True,
        "expansion_max_per_rule": 10,
        "expansion_max_length": 200,
        "affinity_mode": "infer",
    }
    enriched: dict[str, Any] = {}
    for key, default in _enriched_defaults.items():
        if key in project_flat:
            enriched[key] = project_flat[key]
        elif key in global_flat:
            enriched[key] = global_flat[key]
        else:
            enriched[key] = default

    # Validate enriched fields
    for key in ("fusion_k", "expansion_max_per_rule", "expansion_max_length"):
        _validate_field(key, enriched[key])
    _validate_bool("sparse_enabled", enriched["sparse_enabled"])

    # Validate affinity_mode
    valid_affinity_modes = ("infer", "strict")
    affinity_mode = enriched["affinity_mode"]
    if (
        not isinstance(affinity_mode, str)
        or affinity_mode not in valid_affinity_modes
    ):
        msg = (
            f"Config field 'affinity_mode' must be one of"
            f" {valid_affinity_modes}, got {affinity_mode!r}"
        )
        raise ConfigError(msg)

    return ResolvedConfig(
        source_paths=tuple(all_paths),
        global_source_paths=global_resolved,
        project_source_paths=project_resolved,
        model_name=merged["model"],
        top_k=merged["top_k"],
        threshold=float(merged["threshold"]),
        dedup_threshold=float(merged["dedup_threshold"]),
        query_max_length=merged["query_max_length"],
        hook_events=tuple(merged["hook_events"]),
        verbose=merged["verbose"],
        redact=merged["redact"],
        max_log_size_mb=merged["max_log_size_mb"],
        global_cache_dir=global_cache,
        project_cache_dir=project_cache,
        allowed_dirs=all_allowed,
        pipeline=pipeline_config,
        fusion_k=enriched["fusion_k"],
        sparse_enabled=enriched["sparse_enabled"],
        expansion_max_per_rule=enriched["expansion_max_per_rule"],
        expansion_max_length=enriched["expansion_max_length"],
        affinity_mode=enriched["affinity_mode"],
    )
