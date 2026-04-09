"""Semantic retrieval: query encoding, scoring, dedup, and index merging."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from cuecard._math import l2_normalize
from cuecard.models import Index, RankedResult, Rule, SourceMeta

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# --- Query normalization ---

_TOOL_PREFIX_RE = re.compile(
    r"^(Bash|Read|Write|Edit|Glob|Grep|NotebookEdit|WebSearch"
    r"|WebFetch|Agent|AskUserQuestion):\s*",
)


def normalize_query(query: str) -> str:
    """Normalize a tool-call query for better semantic matching.

    Strips the tool name prefix (e.g. "Bash: ") so the embedding
    focuses on the action semantics rather than the tool type.
    Preserves the original text after the prefix.
    """
    return _TOOL_PREFIX_RE.sub("", query)


def retrieve(
    index: Index,
    query: str,
    *,
    top_k: int,
    threshold: float,
    dedup_threshold: float,
    max_query_length: int,
    model: TextEmbedding | None = None,
) -> list[RankedResult]:
    """Retrieve the top-k rules most relevant to *query*.

    Args:
        index: Pre-built embedding index over rules.
        query: Natural-language query string.
        top_k: Maximum number of results to return.
        threshold: Minimum cosine similarity to include a result.
        dedup_threshold: Embedding dot-product above which the
            lower-scored candidate is dropped as a near-duplicate.
        max_query_length: Maximum character length for the query
            string.  Longer queries are truncated with a warning.
        model: fastembed ``TextEmbedding`` instance used for
            asymmetric query encoding.  Required when *index* is
            non-empty.

    Returns:
        Ranked list of :class:`RankedResult`, best first.
    """
    if index.size == 0:
        return []

    query = normalize_query(query)

    if len(query) > max_query_length:
        logger.warning(
            "Query truncated from %d to %d characters",
            len(query),
            max_query_length,
        )
        query = query[:max_query_length]

    if model is None:
        msg = "model is required for non-empty index"
        raise ValueError(msg)

    # --- encode query (asymmetric: query_embed, NOT passage_embed) ---
    query_vec = np.array(list(model.query_embed([query])), dtype=np.float32)
    query_vec = l2_normalize(query_vec)  # (1, dim)

    # --- dot product (== cosine because embeddings are L2-normalized) ---
    raw_scores: npt.NDArray[np.floating] = (
        index.embeddings @ query_vec.T
    ).flatten()  # (total_embeddings,)

    # --- parent collapse: max score per parent rule via rule_map ---
    num_rules = len(index.rules)
    parent_scores = np.full(num_rules, -np.inf)
    np.maximum.at(parent_scores, list(index.rule_map), raw_scores)

    # --- threshold filter (on parent scores) ---
    mask = parent_scores >= threshold
    candidate_idxs = np.where(mask)[0]

    if candidate_idxs.size == 0:
        return []

    # --- sort descending ---
    order = np.argsort(-parent_scores[candidate_idxs])
    sorted_idxs = candidate_idxs[order]

    # --- precompute canonical embedding row per parent ---
    canonical_row: dict[int, int] = {}
    for emb_idx, parent_idx in enumerate(index.rule_map):
        if parent_idx not in canonical_row:
            canonical_row[parent_idx] = emb_idx

    # --- semantic dedup on canonical embeddings ---
    accepted_idxs: list[int] = []
    for idx in sorted_idxs:
        idx_int: int = int(idx)
        emb_row = canonical_row[idx_int]
        if _is_near_duplicate(
            emb_row,
            [canonical_row[a] for a in accepted_idxs],
            index.embeddings,
            dedup_threshold,
        ):
            continue
        accepted_idxs.append(idx_int)
        if len(accepted_idxs) == top_k:
            break

    # --- build results (accepted_idxs are parent rule indices) ---
    return [
        RankedResult(rule=index.rules[i], score=float(parent_scores[i]))
        for i in accepted_idxs
    ]


def merge_indexes(*indexes: Index) -> Index:
    """Merge multiple indexes, deduplicating rules with identical text.

    Returns an ``Index`` with all unique rules and their embeddings
    (including expansion embeddings), properly remapped ``rule_map``
    and concatenated ``bm25_corpus``.

    Args:
        indexes: One or more :class:`Index` instances.

    Returns:
        Merged ``Index`` with exact-text duplicates removed
        (first occurrence wins).

    Raises:
        ValueError: If *model_name* or *dim* differs across indexes.
    """
    if not indexes:
        return Index(
            embeddings=np.empty((0, 0), dtype=np.float32),
            rules=(),
            model_name="",
            dim=0,
            sources={},
            rule_map=(),
            bm25_corpus=None,
        )

    # --- validate compatibility ---
    ref = indexes[0]
    for idx in indexes[1:]:
        if idx.model_name != ref.model_name:
            msg = (
                f"Model name mismatch: {ref.model_name!r} vs "
                f"{idx.model_name!r}"
            )
            raise ValueError(msg)
        if idx.dim != ref.dim:
            msg = f"Dimension mismatch: {ref.dim} vs {idx.dim}"
            raise ValueError(msg)

    # --- exact-text dedup with expansion embedding support ---
    # If ANY index lacks bm25_corpus, discard BM25 for the merged result
    # (avoids misalignment between rule_map and bm25_corpus lengths)
    all_have_bm25 = all(idx.bm25_corpus is not None for idx in indexes)
    if not all_have_bm25:
        logger.warning(
            "BM25 corpus missing in some indexes — sparse retrieval"
            " disabled for merged result",
        )

    seen_texts: set[str] = set()
    keep_rules: list[Rule] = []
    keep_embs: list[npt.NDArray[np.floating]] = []
    keep_rule_map: list[int] = []
    keep_bm25: list[str] = []
    merged_sources: dict[str, SourceMeta] = {}

    for idx in indexes:
        merged_sources.update(idx.sources)

        # Precompute parent → embedding rows for O(E) instead of O(R×E)
        rows_by_parent: dict[int, list[int]] = defaultdict(list)
        for emb_idx, parent in enumerate(idx.rule_map):
            rows_by_parent[parent].append(emb_idx)

        for rule_idx, rule in enumerate(idx.rules):
            if rule.text in seen_texts:
                continue
            seen_texts.add(rule.text)
            new_parent = len(keep_rules)
            keep_rules.append(rule)

            # Copy ALL embedding rows for this parent rule
            for emb_idx in rows_by_parent.get(rule_idx, []):
                keep_embs.append(idx.embeddings[emb_idx])
                keep_rule_map.append(new_parent)
                if all_have_bm25 and idx.bm25_corpus is not None:
                    keep_bm25.append(idx.bm25_corpus[emb_idx])

    if not keep_rules:
        return Index(
            embeddings=np.empty((0, ref.dim), dtype=np.float32),
            rules=(),
            model_name=ref.model_name,
            dim=ref.dim,
            sources=merged_sources,
            rule_map=(),
            bm25_corpus=None,
        )

    embeddings = np.stack(keep_embs, axis=0)
    return Index(
        embeddings=embeddings,
        rules=tuple(keep_rules),
        model_name=ref.model_name,
        dim=ref.dim,
        sources=merged_sources,
        rule_map=tuple(keep_rule_map),
        bm25_corpus=tuple(keep_bm25) if all_have_bm25 else None,
    )


# --- private helpers ---



def _is_near_duplicate(
    candidate_idx: int,
    accepted_idxs: list[int],
    embeddings: npt.NDArray[np.floating],
    dedup_threshold: float,
) -> bool:
    """Return True if *candidate_idx* is a near-duplicate of any accepted."""
    if not accepted_idxs:
        return False
    candidate_emb = embeddings[candidate_idx]  # (dim,)
    accepted_embs = embeddings[accepted_idxs]  # (K, dim)
    sims = accepted_embs @ candidate_emb  # (K,)
    return bool(np.any(sims > dedup_threshold))
