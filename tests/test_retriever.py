"""Tests for cuecard.retriever — retrieve() and merge_indexes()."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from cuecard._math import l2_normalize
from cuecard.models import Index, Provenance, Rule
from cuecard.retriever import merge_indexes, retrieve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(query_vec: np.ndarray) -> MagicMock:
    """Create a mock TextEmbedding that returns *query_vec* from query_embed."""
    model = MagicMock()
    # Use side_effect to return a fresh iterator on every call
    model.query_embed.side_effect = lambda _texts: iter([query_vec.flatten()])
    return model


def _make_index(
    embeddings: np.ndarray,
    rules: tuple[Rule, ...],
    *,
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> Index:
    dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
    return Index(
        embeddings=embeddings,
        rules=rules,
        model_name=model_name,
        dim=dim,
        sources={},
    )


def _l2(vec: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norms, 1e-12)


def _rule(text: str, line: int = 1) -> Rule:
    return Rule(
        text=text,
        provenance=Provenance(file="/tmp/rules.txt", line_start=line, line_end=line),
    )


# ---------------------------------------------------------------------------
# TestRetrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    """Tests for the retrieve() function."""

    def test_basic_retrieval_sorted_by_score(
        self,
        sample_index: Index,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Results come back sorted best-first."""
        # Use the first embedding as query (perfect match for rule 0)
        query_vec = sample_embeddings[0].copy()
        model = _make_model(query_vec)

        results = retrieve(sample_index, "test query", model=model, threshold=0.0)

        assert len(results) > 0
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        # Best match should be rule 0 (dot with itself == 1.0)
        assert results[0].rule == sample_index.rules[0]
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_threshold_filters_low_scores(
        self,
        sample_index: Index,
        sample_embeddings: np.ndarray,
    ) -> None:
        """High threshold excludes distant results."""
        query_vec = sample_embeddings[0].copy()
        model = _make_model(query_vec)

        all_results = retrieve(
            sample_index, "q", model=model, threshold=0.0,
        )
        strict_results = retrieve(
            sample_index, "q", model=model, threshold=0.99,
        )

        assert len(strict_results) < len(all_results)
        for r in strict_results:
            assert r.score >= 0.99

    def test_top_k_limits_results(
        self,
        sample_index: Index,
        sample_embeddings: np.ndarray,
    ) -> None:
        """top_k caps the number of returned results."""
        query_vec = sample_embeddings[0].copy()
        model = _make_model(query_vec)

        results = retrieve(
            sample_index, "q", model=model, top_k=2, threshold=0.0,
        )
        assert len(results) <= 2

    def test_empty_index_returns_empty(self) -> None:
        """An empty index yields no results without touching the model."""
        empty_idx = Index(
            embeddings=np.empty((0, 384), dtype=np.float32),
            rules=(),
            model_name="BAAI/bge-small-en-v1.5",
            dim=384,
            sources={},
        )
        results = retrieve(empty_idx, "anything")
        assert results == []

    def test_semantic_dedup_removes_near_duplicates(self) -> None:
        """Near-duplicate embeddings are collapsed to the higher-scored one."""
        base = _l2(np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32))
        # second embedding almost identical to first (cosine > 0.99)
        near_dup = _l2(base + np.array([[0, 0.01, 0, 0, 0, 0, 0, 0]], dtype=np.float32))
        distant = _l2(np.array([[0, 0, 0, 0, 0, 0, 0, 1.0]], dtype=np.float32))

        embs = np.vstack([base, near_dup, distant])
        rules = (_rule("rule A"), _rule("rule B", 2), _rule("rule C", 3))
        idx = _make_index(embs, rules)

        query_vec = base.flatten()
        model = _make_model(query_vec)

        results = retrieve(
            idx, "q", model=model, threshold=0.0, dedup_threshold=0.95,
        )

        result_texts = [r.rule.text for r in results]
        # Both A and B match the query well, but B is near-dup of A → dropped
        assert "rule A" in result_texts
        assert "rule B" not in result_texts

    def test_query_vector_isl2_normalized(self) -> None:
        """Even an unnormalized query vector produces correct cosine scores."""
        emb = _l2(np.array([[1, 0, 0, 0]], dtype=np.float32))
        rules = (_rule("only rule"),)
        idx = _make_index(emb, rules)

        # Unnormalized query pointing in the same direction
        raw_query = np.array([5.0, 0, 0, 0], dtype=np.float32)
        model = _make_model(raw_query)

        results = retrieve(idx, "q", model=model, threshold=0.0)
        assert len(results) == 1
        # After normalization, dot product should be 1.0
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_uses_query_embed_not_passage_embed(
        self,
        sample_index: Index,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Asymmetric encoding: model.query_embed is called, not passage_embed."""
        query_vec = sample_embeddings[0].copy()
        model = _make_model(query_vec)
        model.passage_embed = MagicMock()

        retrieve(sample_index, "my query", model=model, threshold=0.0)

        model.query_embed.assert_called_once_with(["my query"])
        model.passage_embed.assert_not_called()

    def test_no_results_above_threshold(
        self,
        sample_index: Index,
    ) -> None:
        """When no scores meet threshold, return empty list."""
        # Orthogonal query vector
        query_vec = np.zeros(384, dtype=np.float32)
        query_vec[0] = 1.0
        model = _make_model(query_vec)

        results = retrieve(
            sample_index, "q", model=model, threshold=2.0,
        )
        assert results == []

    def test_model_required_for_nonempty_index(
        self,
        sample_index: Index,
    ) -> None:
        """ValueError raised when model is None but index is non-empty."""
        with pytest.raises(ValueError, match="model is required"):
            retrieve(sample_index, "query")

    def test_long_query_truncated(
        self,
        sample_index: Index,
        sample_embeddings: np.ndarray,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Query exceeding max_query_length is truncated with warning."""
        import logging

        query_vec = sample_embeddings[0].copy()
        model = _make_model(query_vec)

        long_query = "x" * 600
        with caplog.at_level(logging.WARNING, logger="cuecard.retriever"):
            retrieve(
                sample_index, long_query, model=model,
                threshold=0.0, max_query_length=100,
            )

        assert "truncated" in caplog.text.lower()
        # Verify the model received a truncated query
        called_texts = model.query_embed.call_args[0][0]
        assert len(called_texts[0]) == 100

    def test_query_within_limit_not_truncated(
        self,
        sample_index: Index,
        sample_embeddings: np.ndarray,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Query within max_query_length passes through unchanged."""
        import logging

        query_vec = sample_embeddings[0].copy()
        model = _make_model(query_vec)

        with caplog.at_level(logging.WARNING, logger="cuecard.retriever"):
            retrieve(
                sample_index, "short query", model=model,
                threshold=0.0, max_query_length=500,
            )

        assert "truncated" not in caplog.text.lower()


# ---------------------------------------------------------------------------
# TestMergeIndexes
# ---------------------------------------------------------------------------


class TestMergeIndexes:
    """Tests for the merge_indexes() function."""

    def test_two_indexes_merged(self) -> None:
        """Embeddings and rules from two indexes are concatenated."""
        dim = 4
        emb1 = _l2(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
        emb2 = _l2(np.array([[0, 0, 1, 0]], dtype=np.float32))
        rules1 = (_rule("rule 1"), _rule("rule 2", 2))
        rules2 = (_rule("rule 3", 3),)
        idx1 = _make_index(emb1, rules1)
        idx2 = _make_index(emb2, rules2)

        merged_embs, merged_rules = merge_indexes(idx1, idx2)

        assert merged_embs.shape == (3, dim)
        assert len(merged_rules) == 3
        assert [r.text for r in merged_rules] == ["rule 1", "rule 2", "rule 3"]

    def test_exact_text_dedup(self) -> None:
        """Rules with identical text are deduplicated (first wins)."""
        dim = 4
        emb1 = _l2(np.array([[1, 0, 0, 0]], dtype=np.float32))
        emb2 = _l2(np.array([[0, 1, 0, 0]], dtype=np.float32))
        rules1 = (_rule("shared rule"),)
        rules2 = (_rule("shared rule"),)
        idx1 = _make_index(emb1, rules1)
        idx2 = _make_index(emb2, rules2)

        merged_embs, merged_rules = merge_indexes(idx1, idx2)

        assert len(merged_rules) == 1
        assert merged_embs.shape == (1, dim)
        # First occurrence's embedding is kept
        np.testing.assert_array_almost_equal(merged_embs[0], emb1[0])

    def test_model_name_mismatch_raises(self) -> None:
        """ValueError on mismatched model names."""
        emb = _l2(np.array([[1, 0, 0, 0]], dtype=np.float32))
        rules = (_rule("r"),)
        idx1 = _make_index(emb, rules, model_name="model-a")
        idx2 = _make_index(emb.copy(), rules, model_name="model-b")

        with pytest.raises(ValueError, match="Model name mismatch"):
            merge_indexes(idx1, idx2)

    def test_dim_mismatch_raises(self) -> None:
        """ValueError on mismatched dimensions."""
        emb1 = _l2(np.array([[1, 0, 0, 0]], dtype=np.float32))
        emb2 = _l2(np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32))
        idx1 = _make_index(emb1, (_rule("r1"),))
        idx2 = _make_index(emb2, (_rule("r2"),), model_name="BAAI/bge-small-en-v1.5")
        # Manually set dim to differ
        object.__setattr__(idx2, "dim", 8)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            merge_indexes(idx1, idx2)

    def test_single_index_passthrough(self) -> None:
        """A single index passes through unchanged."""
        emb = _l2(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
        rules = (_rule("r1"), _rule("r2", 2))
        idx = _make_index(emb, rules)

        merged_embs, merged_rules = merge_indexes(idx)

        assert len(merged_rules) == 2
        np.testing.assert_array_almost_equal(merged_embs, emb)

    def test_empty_indexes(self) -> None:
        """Merging empty indexes returns empty results."""
        emb = np.empty((0, 4), dtype=np.float32)
        idx = _make_index(emb, ())

        merged_embs, merged_rules = merge_indexes(idx)

        assert merged_rules == ()
        assert merged_embs.shape[0] == 0

    def test_no_indexes(self) -> None:
        """Calling merge_indexes with no arguments returns empty."""
        merged_embs, merged_rules = merge_indexes()

        assert merged_rules == ()
        assert merged_embs.shape[0] == 0


class TestL2Normalize:
    """Tests for the l2_normalize helper (now in _math.py)."""

    def test_1d_vector_normalized(self) -> None:
        """A 1D vector is L2-normalized correctly."""
        vec = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        result = l2_normalize(vec)

        assert result.ndim == 1
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6
        # Direction preserved: [3,4,0] -> [0.6, 0.8, 0.0]
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0], atol=1e-6)

    def test_1d_zero_vector(self) -> None:
        """A 1D zero vector doesn't produce NaN."""
        vec = np.zeros(4, dtype=np.float32)
        result = l2_normalize(vec)

        assert result.ndim == 1
        assert not np.any(np.isnan(result))
        np.testing.assert_allclose(result, 0.0)
