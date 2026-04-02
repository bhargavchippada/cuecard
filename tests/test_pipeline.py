"""Tests for the multi-stage retrieval pipeline orchestrator."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from cuecard.models import PipelineResult, Provenance, RankedResult, Rule
from cuecard.pipeline import run_pipeline

if TYPE_CHECKING:
    from cuecard.models import Index, ResolvedConfig


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
    retrieval: _FakeRetrieval = _FakeRetrieval()


# --- fixtures ---


@pytest.fixture
def config(sample_index: Index) -> ResolvedConfig:
    """Minimal ResolvedConfig for pipeline tests."""
    from cuecard.models import ResolvedConfig

    return ResolvedConfig(
        source_paths=(),
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


# --- test cases ---


class TestEmbeddingOnlyMode:
    """Stage 1 only."""

    def test_embedding_only_mode(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_results: list[RankedResult],
    ) -> None:
        with patch("cuecard.retriever.retrieve", return_value=fake_results) as mock_ret:
            result = run_pipeline(
                "test query", sample_index, config, mode="embedding",
            )

        assert isinstance(result, PipelineResult)
        assert result.mode == "embedding"
        assert result.results == tuple(fake_results)
        assert len(result.stages) == 1
        assert result.stages[0].stage == "embedding"
        assert result.stages[0].error is None
        assert result.stages[0].input_count == sample_index.size
        assert result.stages[0].output_count == len(fake_results)
        mock_ret.assert_called_once()


class TestRerankModeGraceful:
    """Stage 2 degrades when reranker is absent."""

    def test_rerank_mode_with_no_reranker(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_results: list[RankedResult],
    ) -> None:
        import cuecard

        real_mod = getattr(cuecard, "reranker", None)
        with (
            patch("cuecard.retriever.retrieve", return_value=fake_results),
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
        assert result.results == tuple(fake_results)
        assert len(result.stages) == 2
        assert result.stages[0].stage == "embedding"
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
        fake_results: list[RankedResult],
    ) -> None:
        import cuecard

        saved = {
            "reranker": getattr(cuecard, "reranker", None),
            "llm_reranker": getattr(cuecard, "llm_reranker", None),
        }
        with (
            patch("cuecard.retriever.retrieve", return_value=fake_results),
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
        assert result.results == tuple(fake_results)
        assert len(result.stages) == 3
        assert result.stages[1].error is not None
        assert result.stages[2].stage == "llm"
        assert result.stages[2].error is not None

    def test_rerank_llm_haiku_mode_graceful(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_results: list[RankedResult],
    ) -> None:
        import cuecard

        saved = {
            "reranker": getattr(cuecard, "reranker", None),
            "llm_reranker": getattr(cuecard, "llm_reranker", None),
        }
        with (
            patch("cuecard.retriever.retrieve", return_value=fake_results),
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
        import numpy as np

        from cuecard.models import Index

        empty_index = Index(
            embeddings=np.empty((0, 384), dtype=np.float32),
            rules=(),
            model_name="test",
            dim=384,
            sources={},
        )
        with patch("cuecard.retriever.retrieve", return_value=[]):
            result = run_pipeline(
                "test query", empty_index, config, mode="embedding",
            )

        assert result.results == ()
        assert result.stages[0].input_count == 0
        assert result.stages[0].output_count == 0


class TestModeFromConfig:
    """Mode resolved from config when not explicitly passed."""

    def test_mode_from_config_with_pipeline_attr(
        self,
        sample_index: Index,
        fake_results: list[RankedResult],
    ) -> None:
        """config.pipeline.mode is used when no explicit mode."""
        from cuecard.models import PipelineConfig, ResolvedConfig

        cfg = ResolvedConfig(
            source_paths=(),
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
            patch("cuecard.retriever.retrieve", return_value=fake_results),
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
        fake_results: list[RankedResult],
    ) -> None:
        with patch("cuecard.retriever.retrieve", return_value=fake_results):
            result = run_pipeline(
                "test query", sample_index, config, mode=None,
            )
        assert result.mode == "embedding"

    def test_mode_default_bare_config(
        self,
        sample_index: Index,
        fake_results: list[RankedResult],
    ) -> None:
        """Config with neither pipeline nor retrieval falls back to embedding."""

        @dataclass(frozen=True)
        class _BareConfig:
            source_paths: tuple[str, ...] = ()
            model_name: str = "test"
            top_k: int = 5
            threshold: float = 0.30
            dedup_threshold: float = 0.95
            query_max_length: int = 500

        cfg = _BareConfig()
        with patch("cuecard.retriever.retrieve", return_value=fake_results):
            result = run_pipeline(
                "test query", sample_index, cfg, mode=None,  # type: ignore[arg-type]
            )
        assert result.mode == "embedding"


class TestStageTraces:
    """StageTrace fields are populated correctly."""

    def test_stage_traces_recorded(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_results: list[RankedResult],
    ) -> None:
        with patch("cuecard.retriever.retrieve", return_value=fake_results):
            result = run_pipeline(
                "test query", sample_index, config, mode="embedding",
            )

        trace = result.stages[0]
        assert trace.stage == "embedding"
        assert trace.input_count == sample_index.size
        assert trace.output_count == len(fake_results)
        assert trace.error is None

    def test_embedding_stage_timing(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_results: list[RankedResult],
    ) -> None:
        with patch("cuecard.retriever.retrieve", return_value=fake_results):
            result = run_pipeline(
                "test query", sample_index, config, mode="embedding",
            )
        assert result.stages[0].latency_ms >= 0.0


class TestMockReranker:
    """Mock reranker module to verify Stage 2 integration."""

    def test_rerank_stage_with_mock_reranker(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_results: list[RankedResult],
    ) -> None:
        reranked = fake_results[:2]
        mock_rerank = MagicMock(return_value=reranked)

        with (
            patch("cuecard.retriever.retrieve", return_value=fake_results),
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
        fake_results: list[RankedResult],
    ) -> None:
        mock_rerank = MagicMock(
            side_effect=RuntimeError("model failed"),
        )

        with (
            patch("cuecard.retriever.retrieve", return_value=fake_results),
            patch("cuecard.reranker.rerank", mock_rerank),
        ):
            result = run_pipeline(
                "test query", sample_index, config, mode="rerank",
            )

        assert result.results == tuple(fake_results)  # degraded to Stage 1
        assert result.stages[1].error == "model failed"


class TestMockLLMReranker:
    """Mock llm_reranker module to verify Stage 3 integration."""

    def test_llm_stage_with_mock_llm_reranker(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_results: list[RankedResult],
    ) -> None:
        llm_reranked = fake_results[:1]
        mock_rerank = MagicMock(return_value=fake_results[:2])
        mock_llm_rerank = MagicMock(return_value=llm_reranked)

        with (
            patch("cuecard.retriever.retrieve", return_value=fake_results),
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
        fake_results: list[RankedResult],
    ) -> None:
        mock_rerank = MagicMock(return_value=fake_results)
        mock_llm_rerank = MagicMock(
            side_effect=TimeoutError("LLM timeout"),
        )

        with (
            patch("cuecard.retriever.retrieve", return_value=fake_results),
            patch("cuecard.reranker.rerank", mock_rerank),
            patch("cuecard.llm_reranker.rerank_llm", mock_llm_rerank),
        ):
            result = run_pipeline(
                "test query", sample_index, config, mode="rerank-llm-haiku",
            )

        assert result.results == tuple(fake_results)  # degraded
        assert result.stages[2].error == "LLM timeout"


class TestRecallParams:
    """Stage 1 uses higher recall when reranking follows."""

    def test_rerank_mode_uses_recall_params(
        self,
        sample_index: Index,
        fake_results: list[RankedResult],
    ) -> None:
        cfg = _ConfigWithRetrieval(
            retrieval=_FakeRetrieval(
                mode="rerank", recall_top_k=30, recall_threshold=0.05,
            ),
        )
        with (
            patch("cuecard.retriever.retrieve", return_value=fake_results) as mock_ret,
            patch(
                "cuecard.reranker.rerank",
                side_effect=RuntimeError("stub"),
            ),
        ):
            run_pipeline(
                "test query", sample_index, cfg, mode="rerank",  # type: ignore[arg-type]
            )

        call_kwargs = mock_ret.call_args
        assert call_kwargs.kwargs["top_k"] == 30
        assert call_kwargs.kwargs["threshold"] == 0.05

    def test_rerank_mode_uses_defaults_without_retrieval_attr(
        self,
        sample_index: Index,
        config: ResolvedConfig,
        fake_results: list[RankedResult],
    ) -> None:
        with (
            patch("cuecard.retriever.retrieve", return_value=fake_results) as mock_ret,
            patch(
                "cuecard.reranker.rerank",
                side_effect=RuntimeError("stub"),
            ),
        ):
            run_pipeline(
                "test query", sample_index, config, mode="rerank",
            )

        call_kwargs = mock_ret.call_args
        assert call_kwargs.kwargs["top_k"] == 20
        assert call_kwargs.kwargs["threshold"] == 0.20


class TestModeOverride:
    """Explicit mode parameter overrides config."""

    def test_mode_override_parameter(
        self,
        sample_index: Index,
        fake_results: list[RankedResult],
    ) -> None:
        cfg = _ConfigWithRetrieval(retrieval=_FakeRetrieval(mode="rerank"))
        with patch("cuecard.retriever.retrieve", return_value=fake_results):
            result = run_pipeline(
                "test query",
                sample_index,
                cfg,  # type: ignore[arg-type]
                mode="embedding",  # override
            )
        assert result.mode == "embedding"
        assert len(result.stages) == 1  # only Stage 1
