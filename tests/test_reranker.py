"""Tests for cross-encoder re-ranking (Stage 2)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from cuecard.models import Provenance, RankedResult, Rule
from cuecard.reranker import (
    ALLOWED_RERANKER_MODELS,
    DEFAULT_RERANKER_MODEL,
    _load_model,
    rerank,
)


@dataclass(frozen=True)
class MockRerankResult:
    """Mimics fastembed's RerankResult with .score and .index."""

    score: float
    index: int


def _make_mock_model(scores: list[float]) -> MagicMock:
    """Create a mock TextCrossEncoder that returns given scores."""
    model = MagicMock()

    def mock_rerank(
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[MockRerankResult]:
        return [
            MockRerankResult(score=s, index=i)
            for i, s in enumerate(scores)
        ]

    model.rerank = mock_rerank
    return model


def _make_candidates(n: int) -> list[RankedResult]:
    """Create n dummy RankedResult candidates with distinct provenance."""
    results: list[RankedResult] = []
    for i in range(n):
        prov = Provenance(
            file=f"/tmp/rules_{i}.txt",
            line_start=i + 1,
            line_end=i + 1,
            section_path=(f"section-{i}",),
        )
        rule = Rule(text=f"Rule number {i}", provenance=prov)
        results.append(RankedResult(rule=rule, score=0.5 + i * 0.1))
    return results


class TestRerankBasic:
    """Core re-ranking behavior."""

    def test_rerank_basic(self) -> None:
        """5 candidates, top_k=3, returns top 3 by cross-encoder score."""
        candidates = _make_candidates(5)
        scores = [0.1, 0.9, 0.3, 0.7, 0.5]
        mock_model = _make_mock_model(scores)

        result = rerank(candidates, "test query", top_k=3, model=mock_model)

        assert len(result) == 3
        assert result[0].score == 0.9
        assert result[1].score == 0.7
        assert result[2].score == 0.5

    def test_rerank_empty_candidates(self) -> None:
        """Empty list returns empty list without touching the model."""
        result = rerank([], "test query")
        assert result == []

    def test_rerank_single_candidate(self) -> None:
        """Single candidate returns it with the cross-encoder score."""
        candidates = _make_candidates(1)
        mock_model = _make_mock_model([0.85])

        result = rerank(candidates, "test query", top_k=5, model=mock_model)

        assert len(result) == 1
        assert result[0].score == 0.85
        assert result[0].rule.text == "Rule number 0"


class TestRerankProvenance:
    """Provenance and rule preservation."""

    def test_rerank_preserves_provenance(self) -> None:
        """Rule text and provenance are unchanged after re-ranking."""
        candidates = _make_candidates(3)
        scores = [0.3, 0.9, 0.1]
        mock_model = _make_mock_model(scores)

        result = rerank(candidates, "test query", top_k=3, model=mock_model)

        # Best score was index 1
        assert result[0].rule.text == "Rule number 1"
        assert result[0].rule.provenance.file == "/tmp/rules_1.txt"
        assert result[0].rule.provenance.section_path == ("section-1",)

    def test_rerank_replaces_score(self) -> None:
        """Original embedding scores are replaced with cross-encoder scores."""
        candidates = _make_candidates(2)
        original_scores = [c.score for c in candidates]
        ce_scores = [0.99, 0.01]
        mock_model = _make_mock_model(ce_scores)

        result = rerank(candidates, "test query", top_k=2, model=mock_model)

        result_scores = [r.score for r in result]
        assert result_scores != original_scores
        assert result_scores == [0.99, 0.01]


class TestRerankSorting:
    """Sort order verification."""

    def test_rerank_sorts_by_score(self) -> None:
        """Results are sorted descending by cross-encoder score."""
        candidates = _make_candidates(4)
        scores = [0.2, 0.8, 0.4, 0.6]
        mock_model = _make_mock_model(scores)

        result = rerank(candidates, "test query", top_k=4, model=mock_model)

        result_scores = [r.score for r in result]
        assert result_scores == sorted(result_scores, reverse=True)
        assert result_scores == [0.8, 0.6, 0.4, 0.2]

    def test_rerank_top_k_larger_than_candidates(self) -> None:
        """top_k=10 with 3 candidates returns all 3."""
        candidates = _make_candidates(3)
        scores = [0.5, 0.3, 0.8]
        mock_model = _make_mock_model(scores)

        result = rerank(candidates, "test query", top_k=10, model=mock_model)

        assert len(result) == 3
        assert result[0].score == 0.8


class TestRerankModelAllowlist:
    """Model allowlist enforcement."""

    def test_rerank_model_allowlist_rejects_invalid(self) -> None:
        """Invalid model_name raises ValueError."""
        candidates = _make_candidates(1)

        with pytest.raises(ValueError, match="not in allowlist"):
            rerank(
                candidates,
                "test query",
                model_name="evil/malicious-model",
                model=_make_mock_model([0.5]),
            )

    def test_rerank_all_allowed_models_accepted(self) -> None:
        """Each model in ALLOWED_RERANKER_MODELS passes validation."""
        candidates = _make_candidates(1)
        mock_model = _make_mock_model([0.5])

        for model_name in ALLOWED_RERANKER_MODELS:
            result = rerank(
                candidates,
                "test query",
                model_name=model_name,
                model=mock_model,
            )
            assert len(result) == 1

    def test_default_model_is_in_allowlist(self) -> None:
        """DEFAULT_RERANKER_MODEL is a member of ALLOWED_RERANKER_MODELS."""
        assert DEFAULT_RERANKER_MODEL in ALLOWED_RERANKER_MODELS


class TestRerankModelLoading:
    """Model instantiation and reuse."""

    def test_rerank_uses_passed_model(self) -> None:
        """When model= is passed, no new model is instantiated."""
        candidates = _make_candidates(2)
        mock_model = _make_mock_model([0.5, 0.3])

        with patch("cuecard.reranker._load_model") as load_mock:
            rerank(candidates, "test query", model=mock_model)
            load_mock.assert_not_called()

    def test_rerank_creates_model_when_none(self) -> None:
        """When model=None, _load_model is called."""
        candidates = _make_candidates(2)
        mock_model = _make_mock_model([0.5, 0.3])

        with patch(
            "cuecard.reranker._load_model", return_value=mock_model,
        ) as load_mock:
            result = rerank(candidates, "test query")
            load_mock.assert_called_once_with(DEFAULT_RERANKER_MODEL)
            assert len(result) == 2


class TestRerankConfig:
    """Config parameter handling."""

    def test_rerank_accepts_config_kwarg(self) -> None:
        """config= parameter is accepted without error."""
        candidates = _make_candidates(2)
        mock_model = _make_mock_model([0.5, 0.3])
        mock_config = MagicMock()

        result = rerank(
            candidates, "test query", model=mock_model, config=mock_config,
        )
        assert len(result) == 2


class TestLoadModel:
    """Model loading helper."""

    def test_load_model_calls_text_cross_encoder(self) -> None:
        """_load_model imports and instantiates TextCrossEncoder."""
        mock_cls = MagicMock()
        with patch(
            "cuecard.reranker.TextCrossEncoder", mock_cls, create=True,
        ):
            # Patch the import inside _load_model
            import types as _types

            mock_module = _types.ModuleType("fastembed.rerank.cross_encoder")
            mock_module.TextCrossEncoder = mock_cls  # type: ignore[attr-defined]
            with patch.dict(
                "sys.modules",
                {"fastembed.rerank.cross_encoder": mock_module},
            ):
                result = _load_model("Xenova/ms-marco-MiniLM-L-6-v2")

        mock_cls.assert_called_once_with(
            model_name="Xenova/ms-marco-MiniLM-L-6-v2",
        )
        assert result == mock_cls.return_value


class TestRerankRawFloatScores:
    """Test reranker with raw float scores (as returned by real fastembed)."""

    def test_raw_float_scores(self) -> None:
        """fastembed TextCrossEncoder returns raw floats, not objects."""
        candidates = _make_candidates(3)
        model = MagicMock()
        # Real fastembed returns list of floats in document order
        model.rerank.return_value = [-5.0, -2.0, -8.0]

        result = rerank(candidates, "test query", model=model, top_k=2)

        assert len(result) == 2
        # Sorted by score desc: index 1 (-2.0) > index 0 (-5.0)
        assert result[0].rule.text == "Rule number 1"
        assert result[1].rule.text == "Rule number 0"

    def test_generic_fallback_scores(self) -> None:
        """Non-float, non-object values are cast to float."""
        candidates = _make_candidates(2)
        model = MagicMock()
        # Simulate unexpected type that's still float-castable
        import numpy as np

        model.rerank.return_value = [np.float32(-1.0), np.float32(-3.0)]

        result = rerank(candidates, "test query", model=model, top_k=2)

        assert len(result) == 2
        assert result[0].rule.text == "Rule number 0"
