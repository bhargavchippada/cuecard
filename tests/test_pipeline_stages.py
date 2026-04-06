"""Tests for cuecard.pipeline — retrieval stages, graceful degradation, and traces."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from cuecard.models import (
    PipelineResult,
    Provenance,
    RetrievalStageTrace,
    Rule,
)
from cuecard.pipeline import run_pipeline
from cuecard.retrievers import ScoredCandidate

if TYPE_CHECKING:
    from cuecard.models import Index, ResolvedConfig


# --- helpers ---


def _make_scored_candidates(count: int) -> list[ScoredCandidate]:
    """Build deterministic ScoredCandidate list for testing."""
    return [
        ScoredCandidate(
            rule=Rule(
                text=f"Rule {i}",
                provenance=Provenance(
                    file="/tmp/rules.txt", line_start=i, line_end=i,
                ),
            ),
            score=1.0 - i * 0.1,
            retriever="dense",
        )
        for i in range(count)
    ]


# --- fixtures ---


@pytest.fixture
def config(sample_index: Index) -> ResolvedConfig:
    """Minimal ResolvedConfig for pipeline tests."""
    from cuecard.models import ResolvedConfig

    return ResolvedConfig(
        source_paths=(),
        global_source_paths=(),
        project_source_paths=(),
        model_name="test",
        top_k=5,
        threshold=0.30,
        dedup_threshold=0.95,
        query_max_length=500,
        hook_events=(),
        verbose=False,
        redact=False,
        max_log_size_mb=10,
        global_cache_dir="/tmp",
        project_cache_dir=None,
        allowed_dirs=(),
    )


@pytest.fixture
def fake_candidates() -> list[ScoredCandidate]:
    return _make_scored_candidates(3)


# --- test cases ---


class TestRetrievalStageMode:
    """Stage 1 (retrieval) runs with multi-retriever + fusion."""

    def test_embedding_only_mode(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        with patch(
            "cuecard.retrievers.dense.DenseRetriever.retrieve",
            return_value=fake_candidates,
        ):
            result = run_pipeline(
                "test query", sample_index, config, mode="embedding",
            )

        assert isinstance(result, PipelineResult)
        assert result.mode == "embedding"
        assert len(result.results) == len(fake_candidates)
        assert len(result.stages) == 1
        assert result.stages[0].stage == "retrieval"
        assert result.stages[0].error is None
        assert result.stages[0].input_count == sample_index.size
        assert result.stages[0].output_count == len(fake_candidates)

    def test_retrieval_stage_is_retrieval_stage_trace(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        with patch(
            "cuecard.retrievers.dense.DenseRetriever.retrieve",
            return_value=fake_candidates,
        ):
            result = run_pipeline(
                "test query", sample_index, config, mode="embedding",
            )

        trace = result.stages[0]
        assert isinstance(trace, RetrievalStageTrace)
        assert len(trace.retrievers) >= 1
        assert trace.retrievers[0].name == "dense"
        assert trace.fusion_latency_ms >= 0.0


class TestRerankModeGraceful:
    """Stage 2 degrades when reranker is absent."""

    def test_rerank_mode_with_no_reranker(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        import cuecard

        real_mod = getattr(cuecard, "reranker", None)
        with (
            patch(
                "cuecard.retrievers.dense.DenseRetriever.retrieve",
                return_value=fake_candidates,
            ),
            patch.dict(sys.modules, {"cuecard.reranker": None}),
        ):
            if hasattr(cuecard, "reranker"):
                delattr(cuecard, "reranker")
            try:
                result = run_pipeline(
                    "test query", sample_index, config, mode="rerank",
                )
            finally:
                if real_mod is not None:
                    cuecard.reranker = real_mod  # type: ignore[attr-defined]

        assert result.mode == "rerank"
        assert len(result.results) == len(fake_candidates)
        assert len(result.stages) == 2
        assert result.stages[0].stage == "retrieval"
        assert result.stages[0].error is None
        assert result.stages[1].stage == "rerank"
        assert result.stages[1].error is not None
        assert "not available" in result.stages[1].error


class TestLLMModeGraceful:
    """Stage 2 + Stage 3 degrade when modules are absent."""

    def test_rerank_llm_local_mode_graceful(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        import cuecard

        saved = {
            "reranker": getattr(cuecard, "reranker", None),
            "llm_reranker": getattr(cuecard, "llm_reranker", None),
        }
        with (
            patch(
                "cuecard.retrievers.dense.DenseRetriever.retrieve",
                return_value=fake_candidates,
            ),
            patch.dict(
                sys.modules,
                {"cuecard.reranker": None, "cuecard.llm_reranker": None},
            ),
        ):
            for attr in ("reranker", "llm_reranker"):
                if hasattr(cuecard, attr):
                    delattr(cuecard, attr)
            try:
                result = run_pipeline(
                    "test query", sample_index, config,
                    mode="rerank-llm-local",
                )
            finally:
                for attr, mod in saved.items():
                    if mod is not None:
                        setattr(cuecard, attr, mod)

        assert result.mode == "rerank-llm-local"
        assert len(result.results) == len(fake_candidates)
        assert len(result.stages) == 3
        assert result.stages[1].error is not None
        assert result.stages[2].stage == "llm"
        assert result.stages[2].error is not None

    def test_rerank_llm_haiku_mode_graceful(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        import cuecard

        saved = {
            "reranker": getattr(cuecard, "reranker", None),
            "llm_reranker": getattr(cuecard, "llm_reranker", None),
        }
        with (
            patch(
                "cuecard.retrievers.dense.DenseRetriever.retrieve",
                return_value=fake_candidates,
            ),
            patch.dict(
                sys.modules,
                {"cuecard.reranker": None, "cuecard.llm_reranker": None},
            ),
        ):
            for attr in ("reranker", "llm_reranker"):
                if hasattr(cuecard, attr):
                    delattr(cuecard, attr)
            try:
                result = run_pipeline(
                    "test query", sample_index, config,
                    mode="rerank-llm-haiku",
                )
            finally:
                for attr, mod in saved.items():
                    if mod is not None:
                        setattr(cuecard, attr, mod)

        assert result.mode == "rerank-llm-haiku"
        assert len(result.stages) == 3
        assert result.stages[2].stage == "llm"
        assert result.stages[2].error is not None


class TestInvalidMode:
    """Bad mode raises ValueError."""

    def test_invalid_mode_raises(
        self,
        sample_index: Index,
        config: ResolvedConfig,
    ) -> None:
        with pytest.raises(ValueError, match="Invalid pipeline mode"):
            run_pipeline("test query", sample_index, config, mode="bogus")


class TestEmptyIndex:
    """Empty index returns empty results."""

    def test_empty_index(self, config: ResolvedConfig) -> None:
        from cuecard.models import Index

        empty_index = Index(
            embeddings=np.empty((0, 384), dtype=np.float32),
            rules=(),
            model_name="test",
            dim=384,
            sources={},
        )
        result = run_pipeline(
            "test query", empty_index, config, mode="embedding",
        )

        assert result.results == ()
        assert result.stages[0].input_count == 0
        assert result.stages[0].output_count == 0


class TestStageTraces:
    """StageTrace fields are populated correctly."""

    def test_stage_traces_recorded(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        with patch(
            "cuecard.retrievers.dense.DenseRetriever.retrieve",
            return_value=fake_candidates,
        ):
            result = run_pipeline(
                "test query", sample_index, config, mode="embedding",
            )

        trace = result.stages[0]
        assert trace.stage == "retrieval"
        assert trace.input_count == sample_index.size
        assert trace.output_count == len(fake_candidates)
        assert trace.error is None

    def test_embedding_stage_timing(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        with patch(
            "cuecard.retrievers.dense.DenseRetriever.retrieve",
            return_value=fake_candidates,
        ):
            result = run_pipeline(
                "test query", sample_index, config, mode="embedding",
            )
        assert result.stages[0].latency_ms >= 0.0
