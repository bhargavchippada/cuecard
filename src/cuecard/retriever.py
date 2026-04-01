"""Semantic retrieval: query encoding, scoring, dedup, and index merging."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from cuecard._math import l2_normalize
from cuecard.models import Index, RankedResult, Rule

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
    top_k: int = 5,
    threshold: float = 0.30,
    dedup_threshold: float = 0.95,
    max_query_length: int = 500,
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
    scores: npt.NDArray[np.floating] = (
        index.embeddings @ query_vec.T
    ).flatten()  # (N,)

    # --- threshold filter ---
    mask = scores >= threshold
    candidate_idxs = np.where(mask)[0]

    if candidate_idxs.size == 0:
        return []

    # --- sort descending ---
    order = np.argsort(-scores[candidate_idxs])
    sorted_idxs = candidate_idxs[order]

    # --- semantic dedup ---
    accepted_idxs: list[int] = []
    for idx in sorted_idxs:
        idx_int: int = int(idx)
        if _is_near_duplicate(
            idx_int, accepted_idxs, index.embeddings, dedup_threshold,
        ):
            continue
        accepted_idxs.append(idx_int)
        if len(accepted_idxs) == top_k:
            break

    # --- build results ---
    return [
        RankedResult(rule=index.rules[i], score=float(scores[i]))
        for i in accepted_idxs
    ]


def merge_indexes(*indexes: Index) -> tuple[npt.NDArray[np.floating], tuple[Rule, ...]]:
    """Merge multiple indexes, deduplicating rules with identical text.

    Args:
        indexes: One or more :class:`Index` instances.

    Returns:
        ``(embeddings, rules)`` tuple with exact-text duplicates removed
        (first occurrence wins).

    Raises:
        ValueError: If *model_name* or *dim* differs across indexes.
    """
    if not indexes:
        return np.empty((0, 0), dtype=np.float32), ()

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

    # --- exact-text dedup (first occurrence wins) ---
    seen_texts: set[str] = set()
    keep_embs: list[npt.NDArray[np.floating]] = []
    keep_rules: list[Rule] = []

    for idx in indexes:
        for i, rule in enumerate(idx.rules):
            if rule.text in seen_texts:
                continue
            seen_texts.add(rule.text)
            keep_embs.append(idx.embeddings[i])
            keep_rules.append(rule)

    if not keep_rules:
        return np.empty((0, ref.dim), dtype=np.float32), ()

    embeddings = np.stack(keep_embs, axis=0)
    return embeddings, tuple(keep_rules)


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
