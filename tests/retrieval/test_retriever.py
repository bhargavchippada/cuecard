"""Tests for cuecard.retriever — retrieve() and merge_indexes()."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from cuecard.models import Index, Provenance, Rule
from cuecard.retrieval.retriever import merge_indexes, normalize_query, retrieve

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

        results = retrieve(
            sample_index, "test query", model=model, threshold=0.0,
            top_k=5, dedup_threshold=0.95, max_query_length=500,
        )

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
            top_k=5, dedup_threshold=0.95, max_query_length=500,
        )
        strict_results = retrieve(
            sample_index, "q", model=model, threshold=0.99,
            top_k=5, dedup_threshold=0.95, max_query_length=500,
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
            dedup_threshold=0.95, max_query_length=500,
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
        results = retrieve(
            empty_idx, "anything",
            top_k=5, threshold=0.30, dedup_threshold=0.95, max_query_length=500,
        )
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
            top_k=5, max_query_length=500,
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

        results = retrieve(
            idx, "q", model=model, threshold=0.0,
            top_k=5, dedup_threshold=0.95, max_query_length=500,
        )
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

        retrieve(
            sample_index, "my query", model=model, threshold=0.0,
            top_k=5, dedup_threshold=0.95, max_query_length=500,
        )

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
            top_k=5, dedup_threshold=0.95, max_query_length=500,
        )
        assert results == []

    def test_model_required_for_nonempty_index(
        self,
        sample_index: Index,
    ) -> None:
        """ValueError raised when model is None but index is non-empty."""
        with pytest.raises(ValueError, match="model is required"):
            retrieve(
                sample_index, "query",
                top_k=5, threshold=0.30, dedup_threshold=0.95, max_query_length=500,
            )

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
                top_k=5, dedup_threshold=0.95,
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
                top_k=5, dedup_threshold=0.95,
            )

        assert "truncated" not in caplog.text.lower()

    def test_parent_collapse_expansion_score_propagates(self) -> None:
        """Expansion embedding score propagates to parent via np.maximum.at.

        1 rule with 3 embedding rows (canonical + 2 expansions).
        Query matches expansion 2 best. The returned score must equal
        the MAX across all 3 rows, not just the canonical row.
        """
        dim = 8
        # Canonical: points mostly in dim 0
        canonical = _l2(np.array([[1.0, 0.1, 0, 0, 0, 0, 0, 0]], dtype=np.float32))
        # Expansion 1: points mostly in dim 1
        exp1 = _l2(np.array([[0.1, 1.0, 0, 0, 0, 0, 0, 0]], dtype=np.float32))
        # Expansion 2: points mostly in dim 2
        exp2 = _l2(np.array([[0, 0, 1.0, 0, 0, 0, 0, 0]], dtype=np.float32))

        embs = np.vstack([canonical, exp1, exp2])
        rules = (_rule("parent rule"),)
        idx = Index(
            embeddings=embs,
            rules=rules,
            model_name="BAAI/bge-small-en-v1.5",
            dim=dim,
            sources={},
            rule_map=(0, 0, 0),  # all 3 rows map to rule 0
        )

        # Query aligned with expansion 2 (dim 2)
        query_vec = np.zeros(dim, dtype=np.float32)
        query_vec[2] = 1.0
        model = _make_model(query_vec)

        results = retrieve(
            idx, "q", model=model, threshold=0.0,
            top_k=5, dedup_threshold=0.95, max_query_length=500,
        )

        assert len(results) == 1
        assert results[0].rule.text == "parent rule"
        # Score should be dot(exp2, query) which is ~1.0 (both unit vectors in dim 2)
        # NOT dot(canonical, query) which is ~0.0
        assert results[0].score > 0.95

    def test_parent_collapse_max_not_mean(self) -> None:
        """Parent collapse uses MAX, not MEAN across expansions.

        If it used mean, the score would be dragged down by low-scoring rows.
        """
        dim = 4
        # Rule 0: canonical (orthogonal to query) + expansion (aligned with query)
        row0 = _l2(np.array([[0, 1, 0, 0]], dtype=np.float32))  # score ≈ 0
        row1 = _l2(np.array([[1, 0, 0, 0]], dtype=np.float32))  # score ≈ 1
        embs = np.vstack([row0, row1])
        rules = (_rule("collapsed rule"),)
        idx = Index(
            embeddings=embs,
            rules=rules,
            model_name="BAAI/bge-small-en-v1.5",
            dim=dim,
            sources={},
            rule_map=(0, 0),
        )

        query_vec = np.array([1, 0, 0, 0], dtype=np.float32)
        model = _make_model(query_vec)

        results = retrieve(
            idx, "q", model=model, threshold=0.0,
            top_k=5, dedup_threshold=0.95, max_query_length=500,
        )

        assert len(results) == 1
        # MAX should give ≈ 1.0, MEAN would give ≈ 0.5
        assert results[0].score > 0.9

    def test_dedup_continue_not_break_preserves_later_rules(self) -> None:
        """Dedup 'continue' skips the dup but keeps scanning.

        If 'continue' were replaced by 'break', the loop would stop
        after the first near-duplicate and miss rule 3.

        Setup: rule A and rule B are near-duplicates (cosine > 0.95).
        rule C is distant. After dedup: A and C should be returned.
        If 'continue' were 'break': only A would be returned.
        """
        base = _l2(np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32))
        near_dup = _l2(np.array(
            [[1.0, 0.02, 0, 0, 0, 0, 0, 0]], dtype=np.float32,
        ))
        distant = _l2(np.array(
            [[0, 0, 0, 0, 0, 0, 0, 1.0]], dtype=np.float32,
        ))

        embs = np.vstack([base, near_dup, distant])
        rules = (_rule("rule A"), _rule("rule B", 2), _rule("rule C", 3))
        idx = _make_index(embs, rules)

        # Mix query: mostly dim 0 + bit of dim 7 so all 3 pass threshold
        query_vec = _l2(np.array(
            [0.9, 0, 0, 0, 0, 0, 0, 0.4], dtype=np.float32,
        )).flatten()
        model = _make_model(query_vec)

        results = retrieve(
            idx, "q", model=model, threshold=0.0,
            dedup_threshold=0.95, top_k=10, max_query_length=500,
        )

        result_texts = [r.rule.text for r in results]
        assert "rule A" in result_texts
        assert "rule B" not in result_texts  # deduped
        assert "rule C" in result_texts  # must survive — 'continue' kept scanning
        assert len(result_texts) == 2


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

        merged = merge_indexes(idx1, idx2)

        assert merged.embeddings.shape == (3, dim)
        assert len(merged.rules) == 3
        assert [r.text for r in merged.rules] == ["rule 1", "rule 2", "rule 3"]

    def test_exact_text_dedup(self) -> None:
        """Rules with identical text are deduplicated (first wins)."""
        dim = 4
        emb1 = _l2(np.array([[1, 0, 0, 0]], dtype=np.float32))
        emb2 = _l2(np.array([[0, 1, 0, 0]], dtype=np.float32))
        rules1 = (_rule("shared rule"),)
        rules2 = (_rule("shared rule"),)
        idx1 = _make_index(emb1, rules1)
        idx2 = _make_index(emb2, rules2)

        merged = merge_indexes(idx1, idx2)

        assert len(merged.rules) == 1
        assert merged.embeddings.shape == (1, dim)
        # First occurrence's embedding is kept
        np.testing.assert_array_almost_equal(merged.embeddings[0], emb1[0])

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

        merged = merge_indexes(idx)

        assert len(merged.rules) == 2
        np.testing.assert_array_almost_equal(merged.embeddings, emb)

    def test_empty_indexes(self) -> None:
        """Merging empty indexes returns empty results."""
        emb = np.empty((0, 4), dtype=np.float32)
        idx = _make_index(emb, ())

        merged = merge_indexes(idx)

        assert merged.rules == ()
        assert merged.embeddings.shape[0] == 0

    def test_no_indexes(self) -> None:
        """Calling merge_indexes with no arguments returns empty."""
        merged = merge_indexes()

        assert merged.rules == ()
        assert merged.embeddings.shape[0] == 0

    def test_merge_with_expansion_embeddings(self) -> None:
        """Expansion embeddings are correctly remapped during merge."""
        dim = 4
        # Index 1: 1 rule with 1 expansion (2 embedding rows)
        emb1 = _l2(np.array([[1, 0, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32))
        rules1 = (_rule("rule 1"),)
        idx1 = Index(
            embeddings=emb1,
            rules=rules1,
            model_name="BAAI/bge-small-en-v1.5",
            dim=dim,
            sources={},
            rule_map=(0, 0),
            bm25_corpus=("rule 1", "expansion 1"),
        )
        # Index 2: 1 different rule (1 embedding row)
        emb2 = _l2(np.array([[0, 0, 1, 0]], dtype=np.float32))
        rules2 = (_rule("rule 2", 2),)
        idx2 = _make_index(emb2, rules2)

        merged = merge_indexes(idx1, idx2)

        assert len(merged.rules) == 2
        assert merged.embeddings.shape[0] == 3  # 2 from idx1 + 1 from idx2
        assert merged.rule_map == (0, 0, 1)  # remapped
        # bm25_corpus is None when any input index lacks it (prevents misalignment)
        assert merged.bm25_corpus is None

    def test_merge_dedup_keeps_all_expansions(self) -> None:
        """When deduplicating, all expansion rows of the kept rule are preserved."""
        dim = 4
        emb1 = _l2(np.array(
            [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0.8, 0.2, 0, 0]],
            dtype=np.float32,
        ))
        rules1 = (_rule("same rule"),)
        idx1 = Index(
            embeddings=emb1,
            rules=rules1,
            model_name="BAAI/bge-small-en-v1.5",
            dim=dim,
            sources={},
            rule_map=(0, 0, 0),
            bm25_corpus=("same rule", "exp1", "exp2"),
        )

        emb2 = _l2(np.array([[0, 1, 0, 0]], dtype=np.float32))
        rules2 = (_rule("same rule"),)
        idx2 = _make_index(emb2, rules2)

        merged = merge_indexes(idx1, idx2)

        assert len(merged.rules) == 1
        # All 3 expansion rows from first index are kept
        assert merged.embeddings.shape[0] == 3
        assert merged.rule_map == (0, 0, 0)

    def test_merge_sources_combined(self) -> None:
        """Sources from all indexes are merged."""
        from cuecard.models import SourceMeta

        emb1 = _l2(np.array([[1, 0, 0, 0]], dtype=np.float32))
        idx1 = Index(
            embeddings=emb1,
            rules=(_rule("r1"),),
            model_name="BAAI/bge-small-en-v1.5",
            dim=4,
            sources={
                "/a.txt": SourceMeta(mtime=1.0, content_hash="sha256:a", rule_count=1),
            },
        )
        emb2 = _l2(np.array([[0, 1, 0, 0]], dtype=np.float32))
        idx2 = Index(
            embeddings=emb2,
            rules=(_rule("r2", 2),),
            model_name="BAAI/bge-small-en-v1.5",
            dim=4,
            sources={
                "/b.txt": SourceMeta(mtime=2.0, content_hash="sha256:b", rule_count=1),
            },
        )

        merged = merge_indexes(idx1, idx2)
        assert "/a.txt" in merged.sources
        assert "/b.txt" in merged.sources

    def test_merge_empty_returns_proper_index(self) -> None:
        """Merging empty indexes returns Index with correct fields."""
        emb = np.empty((0, 4), dtype=np.float32)
        idx = _make_index(emb, ())

        merged = merge_indexes(idx)

        assert isinstance(merged, Index)
        assert merged.rule_map == ()
        assert merged.bm25_corpus is None

    def test_merge_both_have_bm25(self) -> None:
        """When both indexes have bm25_corpus, merged result preserves it."""
        dim = 4
        emb1 = _l2(np.array([[1, 0, 0, 0]], dtype=np.float32))
        emb2 = _l2(np.array([[0, 1, 0, 0]], dtype=np.float32))
        rules1 = (_rule("rule 1"),)
        rules2 = (_rule("rule 2"),)
        idx1 = Index(
            embeddings=emb1, rules=rules1, model_name="BAAI/bge-small-en-v1.5",
            dim=dim, sources={}, rule_map=(0,), bm25_corpus=("rule 1 text",),
        )
        idx2 = Index(
            embeddings=emb2, rules=rules2, model_name="BAAI/bge-small-en-v1.5",
            dim=dim, sources={}, rule_map=(0,), bm25_corpus=("rule 2 text",),
        )

        merged = merge_indexes(idx1, idx2)

        assert merged.bm25_corpus == ("rule 1 text", "rule 2 text")
        assert merged.rule_map == (0, 1)
        assert len(merged.rules) == 2


# ---------------------------------------------------------------------------
# TestNormalizeQuery
# ---------------------------------------------------------------------------


class TestNormalizeQuery:
    def test_strips_bash_prefix(self) -> None:
        assert normalize_query("Bash: git commit -m 'fix'") == "git commit -m 'fix'"

    def test_strips_read_prefix(self) -> None:
        query = 'Read: {"file_path": "src/main.py"}'
        assert normalize_query(query) == '{"file_path": "src/main.py"}'

    def test_strips_edit_prefix(self) -> None:
        assert normalize_query("Edit: src/auth.py") == "src/auth.py"

    def test_strips_write_prefix(self) -> None:
        assert normalize_query("Write: /tmp/out.txt") == "/tmp/out.txt"

    def test_preserves_non_tool_query(self) -> None:
        assert normalize_query("how to commit safely") == "how to commit safely"

    def test_preserves_bash_in_middle(self) -> None:
        assert normalize_query("run Bash: ls") == "run Bash: ls"

    def test_empty_string(self) -> None:
        assert normalize_query("") == ""

    def test_tool_prefix_only(self) -> None:
        assert normalize_query("Bash: ") == ""

    def test_case_sensitive(self) -> None:
        assert normalize_query("bash: ls") == "bash: ls"

    def test_all_tool_types(self) -> None:
        for tool in [
            "Bash", "Read", "Write", "Edit", "Glob", "Grep",
            "NotebookEdit", "WebSearch", "WebFetch", "Agent",
            "AskUserQuestion",
        ]:
            result = normalize_query(f"{tool}: some action")
            assert result == "some action", f"Failed for {tool}"
