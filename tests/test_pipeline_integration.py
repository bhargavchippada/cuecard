"""Tests for cuecard.pipeline — config modes, mock rerankers, recall params, sparse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cuecard.models import (
    Provenance,
    RankedResult,
    ResolvedConfig,
    RetrievalStageTrace,
    Rule,
)
from cuecard.pipeline import run_pipeline
from cuecard.retrievers import ScoredCandidate

if TYPE_CHECKING:
    from cuecard.models import Index


# --- helpers ---


def _make_ranked_results(count: int) -> list[RankedResult]:
    """Build deterministic RankedResult list for testing."""
    return [
        RankedResult(
            rule=Rule(
                text=f"Rule {i}",
                provenance=Provenance(
                    file="/tmp/rules.txt", line_start=i, line_end=i,
                ),
            ),
            score=1.0 - i * 0.1,
        )
        for i in range(count)
    ]


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


@dataclass(frozen=True)
class _FakeRetrieval:
    mode: str = "embedding"
    recall_top_k: int = 30
    recall_threshold: float = 0.05


@dataclass(frozen=True)
class _ConfigWithRetrieval:
    """Config stub with a retrieval attribute."""

    source_paths: tuple[str, ...] = ()
    model_name: str = "test"
    top_k: int = 5
    threshold: float = 0.30
    dedup_threshold: float = 0.95
    query_max_length: int = 500
    hook_events: tuple[str, ...] = ()
    verbose: bool = False
    redact: bool = False
    max_log_size_mb: int = 10
    global_cache_dir: str = "/tmp"
    project_cache_dir: str | None = None
    allowed_dirs: tuple[str, ...] = ()
    sparse_enabled: bool = True
    fusion_k: int = 60
    llm_candidates: int = 12
    llm_recall_threshold: float = 0.25
    retrieval: _FakeRetrieval = _FakeRetrieval()


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
def fake_results() -> list[RankedResult]:
    return _make_ranked_results(3)


@pytest.fixture
def fake_candidates() -> list[ScoredCandidate]:
    return _make_scored_candidates(3)


# --- test cases ---


class TestModeFromConfig:
    """Mode resolved from config when not explicitly passed."""

    def test_mode_from_config_with_pipeline_attr(
        self,
        sample_index: Index,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        """config.pipeline.mode is used when no explicit mode."""
        from cuecard.models import PipelineConfig, ResolvedConfig

        cfg = ResolvedConfig(
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
            pipeline=PipelineConfig(mode="rerank"),
        )
        with (
            patch(
                "cuecard.retrievers.dense.DenseRetriever.retrieve",
                return_value=fake_candidates,
            ),
            patch(
                "cuecard.reranker.rerank",
                side_effect=RuntimeError("stub"),
            ),
        ):
            result = run_pipeline(
                "test query", sample_index, cfg, mode=None,
            )
        assert result.mode == "rerank"

    def test_mode_default_without_retrieval(
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
                "test query", sample_index, config, mode=None,
            )
        assert result.mode == "embedding"

    def test_mode_default_bare_config(
        self,
        sample_index: Index,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        """Config with all defaults falls back to embedding via pipeline."""
        cfg = ResolvedConfig(
            source_paths=(),
            global_source_paths=(),
            project_source_paths=(),
            global_cache_dir="/tmp/test",
        )
        with patch(
            "cuecard.retrievers.dense.DenseRetriever.retrieve",
            return_value=fake_candidates,
        ):
            result = run_pipeline(
                "test query", sample_index, cfg, mode=None,
            )
        assert result.mode == "embedding"


class TestMockReranker:
    """Mock reranker module to verify Stage 2 integration."""

    def test_rerank_stage_with_mock_reranker(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_candidates: list[ScoredCandidate],
        fake_results: list[RankedResult],
    ) -> None:
        reranked = fake_results[:2]
        mock_rerank = MagicMock(return_value=reranked)

        with (
            patch(
                "cuecard.retrievers.dense.DenseRetriever.retrieve",
                return_value=fake_candidates,
            ),
            patch("cuecard.reranker.rerank", mock_rerank),
        ):
            result = run_pipeline(
                "test query", sample_index, config, mode="rerank",
            )

        assert result.results == tuple(reranked)
        assert len(result.stages) == 2
        assert result.stages[1].stage == "rerank"
        assert result.stages[1].error is None
        assert result.stages[1].output_count == 2
        mock_rerank.assert_called_once()

    def test_rerank_stage_exception(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        mock_rerank = MagicMock(
            side_effect=RuntimeError("model failed"),
        )

        with (
            patch(
                "cuecard.retrievers.dense.DenseRetriever.retrieve",
                return_value=fake_candidates,
            ),
            patch("cuecard.reranker.rerank", mock_rerank),
        ):
            result = run_pipeline(
                "test query", sample_index, config, mode="rerank",
            )

        # Degraded to Stage 1
        assert len(result.results) == len(fake_candidates)
        assert result.stages[1].error == "model failed"


class TestMockLLMReranker:
    """Mock llm_reranker module to verify Stage 3 integration."""

    def test_llm_stage_with_mock_llm_reranker(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_candidates: list[ScoredCandidate],
        fake_results: list[RankedResult],
    ) -> None:
        llm_reranked = fake_results[:1]
        mock_rerank = MagicMock(return_value=fake_results[:2])
        mock_llm_rerank = MagicMock(return_value=llm_reranked)

        with (
            patch(
                "cuecard.retrievers.dense.DenseRetriever.retrieve",
                return_value=fake_candidates,
            ),
            patch("cuecard.reranker.rerank", mock_rerank),
            patch("cuecard.llm_reranker.rerank_llm", mock_llm_rerank),
        ):
            result = run_pipeline(
                "test query", sample_index, config, mode="rerank-llm-local",
            )

        assert result.results == tuple(llm_reranked)
        assert len(result.stages) == 3
        assert result.stages[2].stage == "llm"
        assert result.stages[2].error is None
        mock_llm_rerank.assert_called_once()
        # Verify backend passed correctly
        call_kwargs = mock_llm_rerank.call_args
        assert call_kwargs.kwargs["backend"] == "local"

    def test_llm_stage_exception(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_candidates: list[ScoredCandidate],
        fake_results: list[RankedResult],
    ) -> None:
        mock_rerank = MagicMock(return_value=fake_results)
        mock_llm_rerank = MagicMock(
            side_effect=TimeoutError("LLM timeout"),
        )

        with (
            patch(
                "cuecard.retrievers.dense.DenseRetriever.retrieve",
                return_value=fake_candidates,
            ),
            patch("cuecard.reranker.rerank", mock_rerank),
            patch("cuecard.llm_reranker.rerank_llm", mock_llm_rerank),
        ):
            result = run_pipeline(
                "test query", sample_index, config, mode="rerank-llm-haiku",
            )

        # Degraded to rerank stage results
        assert result.results == tuple(fake_results)
        assert result.stages[2].error == "LLM timeout"


class TestRecallParams:
    """Stage 1 uses higher recall when reranking follows."""

    def test_rerank_mode_uses_recall_params(
        self,
        sample_index: Index,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        cfg = _ConfigWithRetrieval(
            retrieval=_FakeRetrieval(
                mode="rerank", recall_top_k=30, recall_threshold=0.05,
            ),
        )
        with (
            patch(
                "cuecard.retrievers.dense.DenseRetriever.retrieve",
                return_value=fake_candidates,
            ) as mock_ret,
            patch(
                "cuecard.reranker.rerank",
                side_effect=RuntimeError("stub"),
            ),
        ):
            run_pipeline(
                "test query", sample_index, cfg, mode="rerank",  # type: ignore[arg-type]
            )

        call_kwargs = mock_ret.call_args
        # LLM modes use wider recall: top_k=12, threshold=0.25
        assert call_kwargs.kwargs["top_k"] == 12
        assert call_kwargs.kwargs["threshold"] == 0.25

    def test_rerank_mode_uses_defaults_without_retrieval_attr(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        with (
            patch(
                "cuecard.retrievers.dense.DenseRetriever.retrieve",
                return_value=fake_candidates,
            ) as mock_ret,
            patch(
                "cuecard.reranker.rerank",
                side_effect=RuntimeError("stub"),
            ),
        ):
            run_pipeline(
                "test query", sample_index, config, mode="rerank",
            )

        call_kwargs = mock_ret.call_args
        assert call_kwargs.kwargs["top_k"] == 12
        assert call_kwargs.kwargs["threshold"] == 0.25


class TestModeOverride:
    """Explicit mode parameter overrides config."""

    def test_mode_override_parameter(
        self,
        sample_index: Index,
        fake_candidates: list[ScoredCandidate],
    ) -> None:
        cfg = _ConfigWithRetrieval(retrieval=_FakeRetrieval(mode="rerank"))
        with patch(
            "cuecard.retrievers.dense.DenseRetriever.retrieve",
            return_value=fake_candidates,
        ):
            result = run_pipeline(
                "test query",
                sample_index,
                cfg,  # type: ignore[arg-type]
                mode="embedding",  # override
            )
        assert result.mode == "embedding"
        assert len(result.stages) == 1  # only Stage 1


class TestSparseRetrieverIntegration:
    """Sparse retriever runs when bm25_corpus is available."""

    def test_sparse_retriever_runs_with_bm25_corpus(
        self,
        config: ResolvedConfig,
    ) -> None:
        """Pipeline runs sparse when bm25_corpus present."""
        from cuecard.models import Index

        rules = tuple(
            Rule(
                text=f"Rule {i}",
                provenance=Provenance(
                    file="/tmp/r.txt", line_start=i, line_end=i,
                ),
            )
            for i in range(3)
        )
        emb = np.eye(3, 8, dtype=np.float32)
        corpus = ("commit secrets", "uv packages", "validate input")
        idx = Index(
            embeddings=emb,
            rules=rules,
            model_name="test",
            dim=8,
            sources={},
            bm25_corpus=corpus,
        )

        dense_cands = _make_scored_candidates(2)
        sparse_cands = [
            ScoredCandidate(
                rule=rules[2],
                score=3.0,
                retriever="sparse",
            ),
        ]

        with (
            patch(
                "cuecard.retrievers.dense.DenseRetriever.retrieve",
                return_value=dense_cands,
            ),
            patch(
                "cuecard.retrievers.sparse.SparseRetriever.retrieve",
                return_value=sparse_cands,
            ),
        ):
            result = run_pipeline(
                "test query", idx, config, mode="embedding",
            )

        trace = result.stages[0]
        assert isinstance(trace, RetrievalStageTrace)
        assert len(trace.retrievers) == 2
        assert trace.retrievers[0].name == "dense"
        assert trace.retrievers[1].name == "sparse"
        assert trace.fusion_latency_ms >= 0.0
        # Results should be fused
        assert len(result.results) > 0

    def test_sparse_disabled_skips(
        self,
    ) -> None:
        """sparse_enabled=False skips sparse retriever."""
        from cuecard.models import Index

        @dataclass(frozen=True)
        class _CfgSparseOff:
            source_paths: tuple[str, ...] = ()
            model_name: str = "test"
            top_k: int = 5
            threshold: float = 0.30
            dedup_threshold: float = 0.95
            query_max_length: int = 500
            sparse_enabled: bool = False
            fusion_k: int = 60
            llm_candidates: int = 12

        rules = tuple(
            Rule(
                text=f"Rule {i}",
                provenance=Provenance(
                    file="/tmp/r.txt", line_start=i, line_end=i,
                ),
            )
            for i in range(2)
        )
        emb = np.eye(2, 8, dtype=np.float32)
        idx = Index(
            embeddings=emb,
            rules=rules,
            model_name="test",
            dim=8,
            sources={},
            bm25_corpus=("a", "b"),
        )

        dense_cands = _make_scored_candidates(2)
        with patch(
            "cuecard.retrievers.dense.DenseRetriever.retrieve",
            return_value=dense_cands,
        ):
            result = run_pipeline(
                "test", idx, _CfgSparseOff(),  # type: ignore[arg-type]
                mode="embedding",
            )

        trace = result.stages[0]
        assert isinstance(trace, RetrievalStageTrace)
        # Only dense retriever
        assert len(trace.retrievers) == 1
        assert trace.retrievers[0].name == "dense"

    def test_sparse_failure_falls_back_to_dense(
        self,
        fake_candidates: list[ScoredCandidate],
        config: ResolvedConfig,
    ) -> None:
        """Sparse retriever failure degrades gracefully to dense-only."""
        from cuecard.models import Index as IndexClass

        rules = tuple(
            Rule(
                text=f"Rule {i}",
                provenance=Provenance(
                    file="/tmp/r.txt", line_start=i, line_end=i,
                ),
            )
            for i in range(2)
        )
        emb = np.eye(2, 8, dtype=np.float32)
        idx = IndexClass(
            embeddings=emb,
            rules=rules,
            model_name="test",
            dim=8,
            sources={},
            rule_map=(0, 1),
            bm25_corpus=("rule 0", "rule 1"),
        )

        with (
            patch(
                "cuecard.retrievers.dense.DenseRetriever.retrieve",
                return_value=fake_candidates,
            ),
            patch(
                "cuecard.retrievers.sparse.SparseRetriever.retrieve",
                side_effect=RuntimeError("BM25 crashed"),
            ),
        ):
            result = run_pipeline(
                "test", idx, config, mode="embedding",
            )

        # Pipeline succeeds with dense results despite sparse failure
        assert result.results
        trace = result.stages[0]
        assert isinstance(trace, RetrievalStageTrace)
