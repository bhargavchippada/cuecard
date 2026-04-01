"""Cross-encoder re-ranking (Stage 2)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    from cuecard.models import ResolvedConfig

from cuecard.models import RankedResult

logger = logging.getLogger(__name__)

# Only these models can be loaded (security allowlist)
ALLOWED_RERANKER_MODELS: frozenset[str] = frozenset({
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "jinaai/jina-reranker-v1-tiny-en",
    "jinaai/jina-reranker-v1-turbo-en",
    "BAAI/bge-reranker-base",
})

DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def rerank(
    candidates: list[RankedResult],
    query: str,
    *,
    model_name: str = DEFAULT_RERANKER_MODEL,
    top_k: int = 5,
    model: TextCrossEncoder | None = None,
    config: ResolvedConfig | None = None,
) -> list[RankedResult]:
    """Re-rank candidates using a cross-encoder.

    Args:
        candidates: Results from Stage 1 embedding retrieval.
        query: The tool context query string.
        model_name: fastembed cross-encoder model (must be in allowlist).
        top_k: Number of results to return after re-ranking.
        model: Pre-loaded TextCrossEncoder (for reuse across calls).
        config: Pipeline config (reserved for future per-config overrides).

    Returns:
        Top-k results re-ranked by cross-encoder score.
        Provenance preserved from input candidates.
    """
    if not candidates:
        return []

    if model_name not in ALLOWED_RERANKER_MODELS:
        msg = (
            f"Model {model_name!r} not in allowlist. "
            f"Allowed: {sorted(ALLOWED_RERANKER_MODELS)}"
        )
        raise ValueError(msg)

    encoder = model if model is not None else _load_model(model_name)

    documents = [c.rule.text for c in candidates]

    # fastembed rerank() returns list of (index, score) tuples
    raw_results = list(encoder.rerank(
        query=query,
        documents=documents,
        top_k=len(candidates),
    ))

    # Pair scores with original candidates and sort descending
    scored: list[tuple[int, float]] = []
    for result in raw_results:
        # fastembed returns objects with .index and .score attributes
        idx: int = result.index if hasattr(result, "index") else result[0]  # type: ignore[index]
        score: float = result.score if hasattr(result, "score") else result[1]  # type: ignore[index]
        scored.append((idx, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    effective_top_k = min(top_k, len(scored))

    return [
        RankedResult(rule=candidates[idx].rule, score=float(score))
        for idx, score in scored[:effective_top_k]
    ]


def _load_model(model_name: str) -> TextCrossEncoder:
    """Load a cross-encoder model by name."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    logger.info("Loading cross-encoder model: %s", model_name)
    return TextCrossEncoder(model_name=model_name)
