"""Multi-stage retrieval pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from cuecard.models import PipelineResult, StageTrace
from cuecard.security import scrub_secrets

if TYPE_CHECKING:
    from fastembed import TextEmbedding

    from cuecard.models import Index, RankedResult, ResolvedConfig

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
) -> PipelineResult:
    """Execute the multi-stage retrieval pipeline.

    Args:
        query: Tool context query string.
        index: Pre-built embedding index.
        config: Resolved configuration.
        embedding_model: fastembed TextEmbedding for Stage 1.
        mode: Override config mode. If None, uses config.

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

    stages: list[StageTrace] = []

    # Stage 1: always
    results, trace = _run_embedding_stage(
        query, index, config, effective_mode, embedding_model,
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
        results=tuple(results), stages=tuple(stages), mode=effective_mode,
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


def _run_embedding_stage(
    query: str,
    index: Index,
    config: ResolvedConfig,
    effective_mode: str,
    embedding_model: TextEmbedding | None,
) -> tuple[list[RankedResult], StageTrace]:
    """Stage 1: embedding retrieval (always runs)."""
    from cuecard import retriever

    # Use higher recall params when reranking follows
    if effective_mode == "embedding":
        top_k = config.top_k
        threshold = config.threshold
    else:
        retrieval = getattr(config, "retrieval", None)
        top_k = getattr(retrieval, "recall_top_k", 20) if retrieval else 20
        threshold = getattr(retrieval, "recall_threshold", 0.20) if retrieval else 0.20

    input_count = index.size
    t0 = time.monotonic()

    results = retriever.retrieve(
        index,
        query,
        top_k=top_k,
        threshold=threshold,
        dedup_threshold=config.dedup_threshold,
        max_query_length=config.query_max_length,
        model=embedding_model,
    )

    latency_ms = (time.monotonic() - t0) * 1000.0

    return results, StageTrace(
        stage="embedding",
        input_count=input_count,
        output_count=len(results),
        latency_ms=latency_ms,
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
