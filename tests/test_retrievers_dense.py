"""Tests for the DenseRetriever adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from cuecard.models import Index, Provenance, Rule, SourceMeta
from cuecard.retrievers import ScoredCandidate
from cuecard.retrievers.dense import DenseRetriever


def _make_rules(count: int) -> tuple[Rule, ...]:
    """Create count distinct rules."""
    return tuple(
        Rule(
            text=f"Rule {i}",
            provenance=Provenance(file="/tmp/r.txt", line_start=i, line_end=i),
        )
        for i in range(count)
    )


def _make_embeddings(n: int, dim: int = 384) -> np.ndarray:
    """Create n L2-normalized random embedding vectors."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


def _make_model(query_vec: np.ndarray) -> MagicMock:
    """Create a mock TextEmbedding that returns query_vec from query_embed."""
    model = MagicMock()
    model.query_embed.return_value = iter([query_vec.flatten()])
    return model


class TestEmptyIndex:
    """DenseRetriever returns empty list for empty index."""

    def test_empty_index(self) -> None:
        idx = Index(
            embeddings=np.empty((0, 384), dtype=np.float32),
            rules=(),
            model_name="test",
            dim=384,
            sources={},
        )
        retriever = DenseRetriever()
        result = retriever.retrieve("test", idx, top_k=5, threshold=0.3)
        assert result == []


class TestBasicRetrieval:
    """Dense retrieval with identity rule_map (one embedding per rule)."""

    def test_returns_scored_candidates(self) -> None:
        rules = _make_rules(3)
        emb = _make_embeddings(3)
        idx = Index(
            embeddings=emb,
            rules=rules,
            model_name="test",
            dim=384,
            sources={
                "/tmp/r.txt": SourceMeta(
                    mtime=1.0, content_hash="h", rule_count=3,
                ),
            },
        )
        # Use first rule's embedding as query (should match itself perfectly)
        model = _make_model(emb[0])
        retriever = DenseRetriever(model=model, dedup_threshold=0.95)
        result = retriever.retrieve("test", idx, top_k=5, threshold=0.0)

        assert len(result) > 0
        assert all(isinstance(c, ScoredCandidate) for c in result)
        assert all(c.retriever == "dense" for c in result)
        # First result should be Rule 0 (exact match)
        assert result[0].rule.text == "Rule 0"
        assert result[0].score > 0.99  # near-perfect match

    def test_threshold_filters(self) -> None:
        rules = _make_rules(3)
        emb = _make_embeddings(3)
        idx = Index(
            embeddings=emb, rules=rules, model_name="test", dim=384, sources={},
        )
        model = _make_model(emb[0])
        retriever = DenseRetriever(model=model)
        # Very high threshold should filter most results
        result = retriever.retrieve("test", idx, top_k=5, threshold=0.999)
        # Only exact match (self) should survive
        assert len(result) == 1
        assert result[0].rule.text == "Rule 0"

    def test_top_k_limits(self) -> None:
        rules = _make_rules(10)
        emb = _make_embeddings(10)
        idx = Index(
            embeddings=emb, rules=rules, model_name="test", dim=384, sources={},
        )
        model = _make_model(emb[0])
        retriever = DenseRetriever(model=model)
        result = retriever.retrieve("test", idx, top_k=3, threshold=0.0)
        assert len(result) <= 3


class TestParentCollapse:
    """Parent collapse via rule_map (expansion embeddings folded to parent)."""

    def test_max_score_per_parent(self) -> None:
        """With 2 rules and 4 embeddings (2 per rule), parent collapse picks max."""
        rules = _make_rules(2)
        dim = 8
        rng = np.random.default_rng(99)

        # Rule 0 has 2 embeddings, Rule 1 has 2 embeddings
        emb = rng.standard_normal((4, dim)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms

        rule_map = (0, 0, 1, 1)
        idx = Index(
            embeddings=emb, rules=rules, model_name="test", dim=dim,
            sources={}, rule_map=rule_map,
        )

        # Query matches second embedding of rule 0 best
        model = _make_model(emb[1])
        retriever = DenseRetriever(model=model, dedup_threshold=0.99)
        result = retriever.retrieve("test", idx, top_k=5, threshold=0.0)

        # Both rules should appear (2 parents), Rule 0 first (exact match on emb[1])
        assert len(result) == 2
        assert result[0].rule.text == "Rule 0"
        assert result[0].score > 0.99

    def test_returns_one_result_per_parent(self) -> None:
        """Even with many expansion embeddings, each parent appears once."""
        rules = _make_rules(2)
        dim = 8
        rng = np.random.default_rng(42)

        # 5 embeddings for rule 0, 3 for rule 1
        emb = rng.standard_normal((8, dim)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms

        rule_map = (0, 0, 0, 0, 0, 1, 1, 1)
        idx = Index(
            embeddings=emb, rules=rules, model_name="test", dim=dim,
            sources={}, rule_map=rule_map,
        )
        model = _make_model(emb[0])
        retriever = DenseRetriever(model=model, dedup_threshold=0.99)
        result = retriever.retrieve("test", idx, top_k=10, threshold=0.0)

        assert len(result) == 2  # only 2 unique parents


class TestAllBelowThreshold:
    """All parent scores below threshold returns empty."""

    def test_high_threshold_returns_empty(self) -> None:
        rules = _make_rules(3)
        dim = 8
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((3, dim)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms

        # Create query orthogonal to all embeddings (very low scores)
        query_vec = np.zeros(dim, dtype=np.float32)
        query_vec[0] = 1.0
        # Make embeddings point away from query
        for i in range(3):
            emb[i] = np.zeros(dim, dtype=np.float32)
            emb[i][i + 1 if i + 1 < dim else 0] = 1.0

        idx = Index(
            embeddings=emb, rules=rules, model_name="test",
            dim=dim, sources={},
        )
        model = _make_model(query_vec)
        retriever = DenseRetriever(model=model)
        result = retriever.retrieve(
            "test", idx, top_k=5, threshold=0.99,
        )
        assert result == []


class TestQueryNormalization:
    """Query is normalized (tool prefix stripped)."""

    def test_tool_prefix_stripped(self) -> None:
        rules = _make_rules(1)
        emb = _make_embeddings(1)
        idx = Index(
            embeddings=emb, rules=rules, model_name="test", dim=384, sources={},
        )
        model = _make_model(emb[0])
        retriever = DenseRetriever(model=model)
        retriever.retrieve("Bash: test query", idx, top_k=5, threshold=0.0)
        # Verify query_embed was called with normalized query
        call_args = model.query_embed.call_args
        assert call_args[0][0] == ["test query"]


class TestQueryTruncation:
    """Long queries are truncated."""

    def test_long_query_truncated(self) -> None:
        rules = _make_rules(1)
        emb = _make_embeddings(1)
        idx = Index(
            embeddings=emb, rules=rules, model_name="test", dim=384, sources={},
        )
        model = _make_model(emb[0])
        retriever = DenseRetriever(model=model, max_query_length=10)
        retriever.retrieve("a" * 100, idx, top_k=5, threshold=0.0)
        call_args = model.query_embed.call_args
        query_text = call_args[0][0][0]
        assert len(query_text) == 10


class TestModelRequired:
    """Model must be provided for non-empty index."""

    def test_no_model_raises(self) -> None:
        rules = _make_rules(1)
        emb = _make_embeddings(1)
        idx = Index(
            embeddings=emb, rules=rules, model_name="test", dim=384, sources={},
        )
        retriever = DenseRetriever(model=None)
        with pytest.raises(ValueError, match="model is required"):
            retriever.retrieve("test", idx, top_k=5, threshold=0.3)


class TestDenseParentCollapseValues:
    """Parent collapse score values for DenseRetriever."""

    def test_expansion_score_propagates_to_parent(self) -> None:
        """Expansion row with higher score wins via np.maximum.at."""
        rules = _make_rules(1)
        dim = 8
        # Canonical: dim 0
        row0 = np.zeros((1, dim), dtype=np.float32)
        row0[0, 0] = 1.0
        # Expansion: dim 7 (will match query)
        row1 = np.zeros((1, dim), dtype=np.float32)
        row1[0, 7] = 1.0
        emb = np.vstack([row0, row1])

        idx = Index(
            embeddings=emb, rules=rules, model_name="test", dim=dim,
            sources={}, rule_map=(0, 0),
        )
        # Query aligned with dim 7 (matches expansion, not canonical)
        query = np.zeros(dim, dtype=np.float32)
        query[7] = 1.0
        model = _make_model(query)
        retriever = DenseRetriever(model=model, dedup_threshold=0.99)
        result = retriever.retrieve("test", idx, top_k=5, threshold=0.0)

        assert len(result) == 1
        # Score should be ~1.0 (from expansion), not ~0.0 (from canonical)
        assert result[0].score > 0.95


class TestDedup:
    """Near-duplicate suppression via canonical embeddings."""

    def test_near_duplicates_filtered(self) -> None:
        """Two rules with identical embeddings: only first is kept."""
        rules = _make_rules(2)
        dim = 8
        # Both rules get the same embedding
        emb_base = np.ones((1, dim), dtype=np.float32)
        emb_base /= np.linalg.norm(emb_base)
        emb = np.vstack([emb_base, emb_base])

        idx = Index(
            embeddings=emb, rules=rules, model_name="test", dim=dim, sources={},
        )
        model = _make_model(emb_base.flatten())
        retriever = DenseRetriever(model=model, dedup_threshold=0.95)
        result = retriever.retrieve("test", idx, top_k=5, threshold=0.0)

        # Only one should survive dedup
        assert len(result) == 1

    def test_dedup_continue_preserves_later_rules(self) -> None:
        """Dedup loop uses 'continue' (not 'break') so later non-dup rules survive.

        3 rules: A and B are near-duplicates, C is distant.
        After dedup: A and C should be returned.
        If 'continue' were 'break', only A would be returned.
        """
        dim = 8
        # Rule A: unit vector in dim 0
        emb_a = np.zeros((1, dim), dtype=np.float32)
        emb_a[0, 0] = 1.0
        # Rule B: near-duplicate of A (cosine > 0.99)
        emb_b = np.zeros((1, dim), dtype=np.float32)
        emb_b[0, 0] = 1.0
        emb_b[0, 1] = 0.02
        emb_b /= np.linalg.norm(emb_b)
        # Rule C: orthogonal to A (dim 7)
        emb_c = np.zeros((1, dim), dtype=np.float32)
        emb_c[0, 7] = 1.0

        emb = np.vstack([emb_a, emb_b, emb_c])
        rules = _make_rules(3)
        idx = Index(
            embeddings=emb, rules=rules, model_name="test", dim=dim, sources={},
        )

        # Query: mix of dim 0 and dim 7 so all rules score above threshold
        query = np.zeros(dim, dtype=np.float32)
        query[0] = 0.9
        query[7] = 0.4
        query /= np.linalg.norm(query)
        model = _make_model(query)

        retriever = DenseRetriever(model=model, dedup_threshold=0.95)
        result = retriever.retrieve("test", idx, top_k=10, threshold=0.0)

        result_texts = [r.rule.text for r in result]
        assert "Rule 0" in result_texts  # A: kept
        assert "Rule 1" not in result_texts  # B: deduped as near-dup of A
        assert "Rule 2" in result_texts  # C: distant, must survive
        assert len(result) == 2
