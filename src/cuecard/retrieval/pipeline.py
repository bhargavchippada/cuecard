"""Multi-stage retrieval pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from cuecard.models import (
    PipelineResult,
    RetrievalStageTrace,
    RetrieverTrace,
    StageTrace,
)
from cuecard.security import scrub_secrets

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from fastembed import TextEmbedding

    from cuecard.models import AffinityIndex, Index, RankedResult, ResolvedConfig
    from cuecard.retrieval.fusion import ScoredCandidate

logger = logging.getLogger(__name__)

# Valid pipeline modes
VALID_MODES = frozenset({
    "embedding",
    "rerank",
    "rerank-llm-local",
    "rerank-llm-haiku",
    "llm-local",
    "llm-haiku",
})


def run_pipeline(
    query: str,
    index: Index,
    config: ResolvedConfig,
    *,
    embedding_model: TextEmbedding | None = None,
    mode: str | None = None,
    event: str = "",
    tool_name: str = "",
    affinity: AffinityIndex | None = None,
) -> PipelineResult:
    """Execute the multi-stage retrieval pipeline.

    Args:
        query: Tool context query string.
        index: Pre-built embedding index.
        config: Resolved configuration.
        embedding_model: fastembed TextEmbedding for Stage 1.
        mode: Override config mode. If None, uses config.
        event: Hook event name (e.g. "PreToolUse"). Used for event mask.
        tool_name: Tool name for event mask filtering.
        affinity: Affinity index for event mask. If None, no mask applied.

    Returns:
        PipelineResult with final results and per-stage traces.

    The pipeline reads mode to determine which stages run:
      - "embedding": Stage 1 only
      - "rerank": Stage 1 + Stage 2 (cross-encoder)
      - "rerank-llm-local": Stage 1 + Stage 2 + Stage 3 (local LLM)
      - "rerank-llm-haiku": Stage 1 + Stage 2 + Stage 3 (Haiku)
      - "llm-local": Stage 1 + Stage 3 (skip cross-encoder, local LLM)
      - "llm-haiku": Stage 1 + Stage 3 (skip cross-encoder, Haiku)

    Each stage failure degrades gracefully to previous stage results.
    """
    effective_mode = _resolve_mode(mode, config)

    # Build event mask if affinity is available and event is specified
    event_mask: npt.NDArray[np.bool_] | None = None
    event_mask_applied = False
    rules_masked = 0
    if affinity is not None and event:
        from cuecard.retrieval.affinity import build_event_mask

        event_mask = build_event_mask(
            index, affinity, event,
            tool_name=tool_name if tool_name else None,
        )
        event_mask_applied = True
        rules_masked = int((~event_mask).sum())

    stages: list[StageTrace] = []

    # Stage 1: always (multi-retriever + fusion)
    results, trace = _run_retrieval_stage(
        query, index, config, effective_mode, embedding_model,
        mask=event_mask,
    )
    stages.append(trace)

    # Stage 2: cross-encoder (only for "rerank" and "rerank-llm-*" modes)
    if effective_mode.startswith("rerank"):
        results, trace = _run_rerank_stage(results, query, config)
        stages.append(trace)

    # Stage 3: LLM (for any mode containing "llm")
    if "llm" in effective_mode:
        backend = "local" if "local" in effective_mode else "haiku"
        results, trace = _run_llm_stage(results, query, config, backend)
        stages.append(trace)

    return PipelineResult(
        results=tuple(results),
        stages=tuple(stages),
        mode=effective_mode,
        event=event,
        event_mask_applied=event_mask_applied,
        rules_masked=rules_masked,
    )


def _resolve_mode(mode: str | None, config: ResolvedConfig) -> str:
    """Determine effective pipeline mode from explicit override or config."""
    effective = mode if mode is not None else config.pipeline.mode

    if effective not in VALID_MODES:
        capped = effective[:50] if isinstance(effective, str) else str(effective)[:50]
        msg = f"Invalid pipeline mode {capped!r}. Valid: {sorted(VALID_MODES)}"
        raise ValueError(msg)

    return effective


def _retrieval_params(
    effective_mode: str, config: ResolvedConfig,
) -> tuple[int, float]:
    """Determine top_k and threshold based on pipeline mode."""
    if effective_mode == "embedding":
        return config.top_k, config.threshold
    # LLM modes use wider recall to give the reranker more candidates
    return config.llm_candidates, config.llm_recall_threshold


def _run_sparse(
    query: str,
    index: Index,
    top_k: int,
    mask: npt.NDArray[np.bool_] | None,
) -> tuple[list[ScoredCandidate], float]:
    """Run sparse retrieval, returning results and latency in ms."""
    from cuecard.retrieval.sparse import SparseRetriever

    try:
        sparse = SparseRetriever()
        t0 = time.monotonic()
        results = sparse.retrieve(
            query, index, top_k=top_k, threshold=0.0, mask=mask,
        )
        return results, (time.monotonic() - t0) * 1000.0
    except Exception as exc:
        logger.warning(
            "Sparse retrieval failed (%s), falling back to dense-only",
            exc,
        )
        return [], 0.0


def _build_retriever_traces(
    dense_results: list[ScoredCandidate],
    sparse_results: list[ScoredCandidate],
    t_dense_ms: float,
    t_sparse_ms: float,
    sparse_ran: bool,
) -> tuple[RetrieverTrace, ...]:
    """Build per-retriever trace objects with unique rule counts."""
    dense_texts = {c.rule.text for c in dense_results}
    sparse_texts = {c.rule.text for c in sparse_results}

    traces: list[RetrieverTrace] = [
        RetrieverTrace(
            name="dense",
            candidate_count=len(dense_results),
            latency_ms=t_dense_ms,
            unique_rules=len(dense_texts - sparse_texts),
        ),
    ]
    if sparse_ran:
        traces.append(RetrieverTrace(
            name="sparse",
            candidate_count=len(sparse_results),
            latency_ms=t_sparse_ms,
            unique_rules=len(sparse_texts - dense_texts),
        ))
    return tuple(traces)


def _run_retrieval_stage(
    query: str,
    index: Index,
    config: ResolvedConfig,
    effective_mode: str,
    embedding_model: TextEmbedding | None,
    *,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[list[RankedResult], StageTrace]:
    """Stage 1: multi-retriever + RRF fusion (always runs)."""
    from cuecard.models import RankedResult
    from cuecard.retrieval.dense import DenseRetriever
    from cuecard.retrieval.fusion import fuse

    top_k, threshold = _retrieval_params(effective_mode, config)
    input_count = index.size
    t0 = time.monotonic()

    # Dense (always)
    dense = DenseRetriever(
        model=embedding_model,
        dedup_threshold=config.dedup_threshold,
        max_query_length=config.query_max_length,
    )
    t_dense_start = time.monotonic()
    dense_results = dense.retrieve(
        query, index, top_k=top_k, threshold=threshold, mask=mask,
    )
    t_dense_ms = (time.monotonic() - t_dense_start) * 1000.0

    # Sparse (when available and enabled)
    all_results: list[list[ScoredCandidate]] = [dense_results]
    sparse_ran = config.sparse_enabled and index.bm25_corpus is not None
    sparse_results: list[ScoredCandidate] = []
    t_sparse_ms = 0.0

    if sparse_ran:
        sparse_results, t_sparse_ms = _run_sparse(
            query, index, top_k, mask,
        )
        if sparse_results:
            all_results.append(sparse_results)

    # Fusion
    t_fusion_start = time.monotonic()
    fused = (
        fuse(all_results, k=config.fusion_k, top_k=top_k)
        if len(all_results) > 1 else dense_results
    )
    t_fusion_ms = (time.monotonic() - t_fusion_start) * 1000.0

    results: list[RankedResult] = [
        RankedResult(rule=sc.rule, score=sc.score) for sc in fused
    ]

    return results, RetrievalStageTrace(
        stage="retrieval",
        input_count=input_count,
        output_count=len(results),
        latency_ms=(time.monotonic() - t0) * 1000.0,
        retrievers=_build_retriever_traces(
            dense_results, sparse_results,
            t_dense_ms, t_sparse_ms, sparse_ran,
        ),
        fusion_latency_ms=t_fusion_ms,
    )


def _run_rerank_stage(
    candidates: list[RankedResult],
    query: str,
    config: ResolvedConfig,
) -> tuple[list[RankedResult], StageTrace]:
    """Stage 2: cross-encoder re-ranking (if available)."""
    input_count = len(candidates)
    t0 = time.monotonic()

    try:
        from cuecard.retrieval import reranker

        results = reranker.rerank(candidates, query, config=config)
        latency_ms = (time.monotonic() - t0) * 1000.0
        return results, StageTrace(
            stage="rerank",
            input_count=input_count,
            output_count=len(results),
            latency_ms=latency_ms,
        )
    except (ImportError, AttributeError):
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.warning("reranker module not available — skipping Stage 2")
        return candidates, StageTrace(
            stage="rerank",
            input_count=input_count,
            output_count=len(candidates),
            latency_ms=latency_ms,
            error="reranker module not available",
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.error("Stage 2 rerank failed: %s", exc)  # noqa: TRY400
        return candidates, StageTrace(
            stage="rerank",
            input_count=input_count,
            output_count=len(candidates),
            latency_ms=latency_ms,
            error=scrub_secrets(str(exc)),
        )


def _run_llm_stage(
    candidates: list[RankedResult],
    query: str,
    config: ResolvedConfig,
    backend: str,
) -> tuple[list[RankedResult], StageTrace]:
    """Stage 3: LLM re-ranking (if available)."""
    input_count = len(candidates)
    t0 = time.monotonic()

    try:
        from cuecard.retrieval import llm_reranker

        results = llm_reranker.rerank_llm(
            candidates,
            query,
            backend=backend,
            endpoint=config.pipeline.local_endpoint,
            haiku_model=config.pipeline.haiku_model,
            thinking=config.pipeline.thinking,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0
        return results, StageTrace(
            stage="llm",
            input_count=input_count,
            output_count=len(results),
            latency_ms=latency_ms,
        )
    except (ImportError, AttributeError):
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.warning("llm_reranker module not available — skipping Stage 3")
        return candidates, StageTrace(
            stage="llm",
            input_count=input_count,
            output_count=len(candidates),
            latency_ms=latency_ms,
            error="llm_reranker module not available",
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.error("Stage 3 LLM rerank failed: %s", exc)  # noqa: TRY400
        return candidates, StageTrace(
            stage="llm",
            input_count=input_count,
            output_count=len(candidates),
            latency_ms=latency_ms,
            error=scrub_secrets(str(exc)),
        )
