"""Unified index loading with freshness checking and scope composition.

Provides ``load_or_build`` — the single entry point for obtaining a
ready-to-query index.  It checks freshness per scope (global / project),
rebuilds stale scopes, and composes them via ``merge_indexes``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cuecard.freshness import check_freshness
from cuecard.indexer import (
    build_index,
    load_index,
    load_rules_json,
    merge_rules_json,
    save_index,
    save_rules_json,
)
from cuecard.parser import parse_rules
from cuecard.retriever import merge_indexes

if TYPE_CHECKING:
    from cuecard.indexer import EmbeddingModel
    from cuecard.models import Index, ResolvedConfig

logger = logging.getLogger(__name__)


def _load_or_rebuild_scope(
    cache_dir: str,
    source_paths: tuple[str, ...],
    model_name: str,
    model: EmbeddingModel | None,
    *,
    reindex: bool,
) -> Index | None:
    """Load a single scope's index, rebuilding if stale.

    Returns None when no sources exist for this scope or the index
    cannot be built (e.g. no embedding model provided).
    """
    if not source_paths:
        return None

    index = load_index(cache_dir)

    if index is not None and not reindex:
        return index

    if index is not None:
        freshness = check_freshness(source_paths, dict(index.sources))
        if not freshness.is_stale:
            return index
        logger.info(
            "Index stale in %s (changed=%s, new=%s, removed=%s) — rebuilding",
            cache_dir,
            freshness.changed_files,
            freshness.new_files,
            freshness.removed_files,
        )
    else:
        logger.info("No index found in %s — building", cache_dir)

    # (Re)build
    rules = parse_rules(source_paths)
    if not rules:
        logger.warning("No rules parsed from %s", source_paths)
        return None

    # Merge with cached rules.json to preserve expansions
    cached_rules = load_rules_json(cache_dir)
    if cached_rules is not None:
        rules = merge_rules_json(rules, cached_rules)

    # Save the canonical JSON intermediate
    save_rules_json(rules, cache_dir)

    if model is None:
        logger.warning("Embedding model required to build index for %s", cache_dir)
        return None

    freshness = check_freshness(source_paths, {})
    new_index = build_index(
        tuple(rules),
        freshness.updated_sources,
        model_name,
        model=model,
    )
    save_index(new_index, cache_dir)
    logger.info("Built index: %d rules in %s", new_index.size, cache_dir)
    return new_index


def load_or_build(
    config: ResolvedConfig,
    model: EmbeddingModel | None = None,
    *,
    reindex: bool = True,
) -> Index | None:
    """Load a ready-to-query index, rebuilding stale scopes as needed.

    This is the primary entry point for both the adapter and CLI
    retrieval paths.  It:

    1. Loads/rebuilds the **global** index from ``global_source_paths``.
    2. Loads/rebuilds the **project** index from ``project_source_paths``
       (if a project scope exists).
    3. Composes them via ``merge_indexes`` so retrieval sees both.

    Args:
        config: Resolved configuration with scoped source paths.
        model: Embedding model for rebuilding stale indexes.
            Required if any scope is stale or unbuilt.
        reindex: If False, skip freshness checks (low-latency mode).

    Returns:
        A composed ``Index``, or ``None`` if no rules exist anywhere.
    """
    indexes: list[Index] = []

    # Global scope
    global_idx = _load_or_rebuild_scope(
        config.global_cache_dir,
        config.global_source_paths,
        config.model_name,
        model,
        reindex=reindex,
    )
    if global_idx is not None:
        indexes.append(global_idx)

    # Project scope
    if config.project_cache_dir and config.project_source_paths:
        project_idx = _load_or_rebuild_scope(
            config.project_cache_dir,
            config.project_source_paths,
            config.model_name,
            model,
            reindex=reindex,
        )
        if project_idx is not None:
            indexes.append(project_idx)

    if not indexes:
        return None

    if len(indexes) == 1:
        return indexes[0]

    # Compose global + project (dedup by text)
    merged = merge_indexes(*indexes)
    if not merged.rules:
        return None

    return merged
