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
        from cuecard.affinity import build_event_mask

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
    if mode is not None:
        effective = mode
    elif hasattr(config, "pipeline") and hasattr(config.pipeline, "mode"):
        effective = config.pipeline.mode
    else:
        effective = "embedding"

    if effective not in VALID_MODES:
        capped = effective[:50] if isinstance(effective, str) else str(effective)[:50]
        msg = f"Invalid pipeline mode {capped!r}. Valid: {sorted(VALID_MODES)}"
        raise ValueError(msg)

    return effective


def _run_retrieval_stage(
    query: str,
    index: Index,
    config: ResolvedConfig,
    effective_mode: str,
    embedding_model: TextEmbedding | None,
    *,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[list[RankedResult], StageTrace]:
    """Stage 1: multi-retriever + RRF fusion (always runs).

    Runs dense retrieval (always) and sparse/BM25 retrieval (when
    ``index.bm25_corpus`` is available and ``sparse_enabled`` is True).
    Fuses results via Reciprocal Rank Fusion when both retrievers run.
    Falls back to dense-only otherwise.
    """
    from cuecard.models import RankedResult
    from cuecard.retrievers import ScoredCandidate, fuse
    from cuecard.retrievers.dense import DenseRetriever
    from cuecard.retrievers.sparse import SparseRetriever

    # Use higher recall params when reranking follows
    if effective_mode == "embedding":
        top_k = config.top_k
        threshold = config.threshold
    else:
        # LLM modes use wider recall to give the reranker more candidates
        top_k = 20
        threshold = 0.20

    input_count = index.size
    t0 = time.monotonic()

    # Dense retriever (always)
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

    retriever_traces: list[RetrieverTrace] = []
    all_results: list[list[ScoredCandidate]] = [dense_results]

    # Sparse retriever (when bm25_corpus available and enabled)
    sparse_enabled = config.sparse_enabled
    sparse_results: list[ScoredCandidate] = []
    t_sparse_ms = 0.0

    if sparse_enabled and index.bm25_corpus is not None:
        try:
            sparse = SparseRetriever()
            t_sparse_start = time.monotonic()
            sparse_results = sparse.retrieve(
                query, index, top_k=top_k, threshold=0.0, mask=mask,
            )
            t_sparse_ms = (time.monotonic() - t_sparse_start) * 1000.0
            all_results.append(sparse_results)
        except Exception as exc:
            logger.warning(
                "Sparse retrieval failed (%s), falling back to dense-only",
                exc,
            )
            t_sparse_ms = 0.0

    # Fusion (only if multiple retrievers ran)
    fusion_k = config.fusion_k
    t_fusion_start = time.monotonic()

    if len(all_results) > 1:
        fused = fuse(all_results, k=fusion_k, top_k=top_k)
    else:
        fused = dense_results

    t_fusion_ms = (time.monotonic() - t_fusion_start) * 1000.0

    # Compute unique rules per retriever
    dense_rule_texts = {c.rule.text for c in dense_results}
    sparse_rule_texts = {c.rule.text for c in sparse_results}

    dense_unique = len(dense_rule_texts - sparse_rule_texts)
    sparse_unique = len(sparse_rule_texts - dense_rule_texts)

    retriever_traces.append(RetrieverTrace(
        name="dense",
        candidate_count=len(dense_results),
        latency_ms=t_dense_ms,
        unique_rules=dense_unique,
    ))
    if sparse_enabled and index.bm25_corpus is not None:
        retriever_traces.append(RetrieverTrace(
            name="sparse",
            candidate_count=len(sparse_results),
            latency_ms=t_sparse_ms,
            unique_rules=sparse_unique,
        ))

    latency_ms = (time.monotonic() - t0) * 1000.0

    # Convert ScoredCandidate -> RankedResult
    results: list[RankedResult] = [
        RankedResult(rule=sc.rule, score=sc.score)
        for sc in fused
    ]

    return results, RetrievalStageTrace(
        stage="retrieval",
        input_count=input_count,
        output_count=len(results),
        latency_ms=latency_ms,
        retrievers=tuple(retriever_traces),
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
        from cuecard import reranker

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
        from cuecard import llm_reranker

        llm_kwargs: dict[str, object] = {"backend": backend}
        if hasattr(config, "pipeline"):
            llm_kwargs["endpoint"] = config.pipeline.local_endpoint
            llm_kwargs["haiku_model"] = config.pipeline.haiku_model
            llm_kwargs["thinking"] = config.pipeline.thinking
        results = llm_reranker.rerank_llm(
            candidates, query, **llm_kwargs,  # type: ignore[arg-type]
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
