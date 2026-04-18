"""Tests for RRF fusion and ScoredCandidate."""

from __future__ import annotations

import pytest

from cuecard.models import Provenance, Rule
from cuecard.retrieval.fusion import Retriever, ScoredCandidate, fuse


def _rule(text: str) -> Rule:
    """Create a minimal Rule for testing."""
    prov = Provenance(file="/tmp/r.txt", line_start=1, line_end=1)
    return Rule(text=text, provenance=prov)


class TestScoredCandidate:
    """ScoredCandidate is frozen and has expected fields."""

    def test_frozen(self) -> None:
        sc = ScoredCandidate(rule=_rule("r1"), score=0.9, retriever="dense")
        assert sc.rule.text == "r1"
        assert sc.score == 0.9
        assert sc.retriever == "dense"

    def test_immutable(self) -> None:
        sc = ScoredCandidate(rule=_rule("r1"), score=0.9, retriever="dense")
        try:
            sc.score = 0.5  # type: ignore[misc]
        except AttributeError:
            pass
        else:
            raise AssertionError("ScoredCandidate should be frozen")  # noqa: EM101, TRY003


class TestRetrieverProtocol:
    """Retriever Protocol is runtime-checkable."""

    def test_protocol_check(self) -> None:
        class _FakeRetriever:
            def retrieve(
                self,
                query: str,
                index: object,
                *,
                top_k: int,
                threshold: float,
            ) -> list[ScoredCandidate]:
                return []

        assert isinstance(_FakeRetriever(), Retriever)


class TestFuseEmpty:
    """Edge cases for fuse()."""

    def test_empty_input(self) -> None:
        assert fuse([], k=10) == []

    def test_single_retriever(self) -> None:
        r1 = _rule("r1")
        candidates = [ScoredCandidate(rule=r1, score=0.9, retriever="dense")]
        result = fuse([candidates], k=60)
        assert len(result) == 1
        assert result[0].rule.text == "r1"
        assert result[0].retriever == "fused"
        # Score = 1 / (60 + 1) for rank 0 (1-based rank 1)
        assert abs(result[0].score - 1.0 / 61) < 1e-9

    def test_single_empty_list(self) -> None:
        result = fuse([[]], k=10)
        assert result == []


class TestFuseBasic:
    """Basic RRF fusion behavior."""

    def test_two_retrievers_same_rule(self) -> None:
        """Same rule from two retrievers gets combined RRF score."""
        r1 = _rule("r1")
        dense = [ScoredCandidate(rule=r1, score=0.9, retriever="dense")]
        sparse = [ScoredCandidate(rule=r1, score=5.0, retriever="sparse")]

        result = fuse([dense, sparse], k=60)
        assert len(result) == 1
        expected_score = 1.0 / 61 + 1.0 / 61  # rank 1 in both
        assert abs(result[0].score - expected_score) < 1e-9

    def test_two_retrievers_different_rules(self) -> None:
        """Different rules from each retriever appear in output."""
        r1 = _rule("r1")
        r2 = _rule("r2")
        dense = [ScoredCandidate(rule=r1, score=0.9, retriever="dense")]
        sparse = [ScoredCandidate(rule=r2, score=5.0, retriever="sparse")]

        result = fuse([dense, sparse], k=60)
        assert len(result) == 2
        texts = {c.rule.text for c in result}
        assert texts == {"r1", "r2"}

    def test_ranking_order(self) -> None:
        """Rule appearing in both retrievers ranks higher than one-retriever rules."""
        r1 = _rule("shared")
        r2 = _rule("dense_only")
        r3 = _rule("sparse_only")

        dense = [
            ScoredCandidate(rule=r1, score=0.9, retriever="dense"),
            ScoredCandidate(rule=r2, score=0.8, retriever="dense"),
        ]
        sparse = [
            ScoredCandidate(rule=r1, score=5.0, retriever="sparse"),
            ScoredCandidate(rule=r3, score=4.0, retriever="sparse"),
        ]

        result = fuse([dense, sparse], k=60)
        assert result[0].rule.text == "shared"  # highest combined score

    def test_top_k_truncation(self) -> None:
        """top_k limits output length."""
        rules = [_rule(f"r{i}") for i in range(10)]
        candidates = [
            ScoredCandidate(rule=r, score=1.0 - i * 0.1, retriever="dense")
            for i, r in enumerate(rules)
        ]
        result = fuse([candidates], k=60, top_k=3)
        assert len(result) == 3

    def test_rrf_score_formula(self) -> None:
        """Verify exact RRF score computation."""
        r1 = _rule("r1")
        r2 = _rule("r2")
        # r1 is rank 1 (0-based 0) in dense, rank 2 (0-based 1) in sparse
        # r2 is rank 2 (0-based 1) in dense, rank 1 (0-based 0) in sparse
        dense = [
            ScoredCandidate(rule=r1, score=0.9, retriever="dense"),
            ScoredCandidate(rule=r2, score=0.8, retriever="dense"),
        ]
        sparse = [
            ScoredCandidate(rule=r2, score=5.0, retriever="sparse"),
            ScoredCandidate(rule=r1, score=4.0, retriever="sparse"),
        ]

        result = fuse([dense, sparse], k=60)
        # r1: 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
        # r2: 1/62 + 1/61 = 0.01613 + 0.01639 = 0.03252
        # Scores are equal — either ordering is fine
        assert len(result) == 2
        assert abs(result[0].score - result[1].score) < 1e-9

    def test_first_rule_reference_wins(self) -> None:
        """First Rule ref is kept when same text from multiple retrievers."""
        prov1 = Provenance(file="/a.txt", line_start=1, line_end=1)
        prov2 = Provenance(file="/b.txt", line_start=2, line_end=2)
        r1a = Rule(text="shared", provenance=prov1)
        r1b = Rule(text="shared", provenance=prov2)

        dense = [ScoredCandidate(rule=r1a, score=0.9, retriever="dense")]
        sparse = [ScoredCandidate(rule=r1b, score=5.0, retriever="sparse")]

        result = fuse([dense, sparse], k=10)
        assert result[0].rule.provenance.file == "/a.txt"

    def test_custom_k_parameter(self) -> None:
        """Different k values change relative scores."""
        r1 = _rule("r1")
        candidates = [ScoredCandidate(rule=r1, score=0.9, retriever="dense")]

        result_k10 = fuse([candidates], k=10)
        result_k100 = fuse([candidates], k=100)

        # Smaller k gives higher score: 1/11 > 1/101
        assert result_k10[0].score > result_k100[0].score

    def test_weighted_retrievers_apply_custom_multipliers(self) -> None:
        """Retriever weights scale each retriever's RRF contribution."""
        r1 = _rule("shared")
        dense = [ScoredCandidate(rule=r1, score=0.9, retriever="dense")]
        sparse = [ScoredCandidate(rule=r1, score=5.0, retriever="sparse")]

        result = fuse(
            [dense, sparse],
            k=60,
            retriever_weights={"dense": 0.7, "sparse": 0.3},
        )

        expected_score = 0.7 * (1.0 / 61) + 0.3 * (1.0 / 61)
        assert len(result) == 1
        assert result[0].score == pytest.approx(expected_score, rel=1e-9)


class TestFuseRRFSignAndRank:
    """Kill RRF sign-flip and rank-offset mutants.

    The formula is 1/(k + rank_zero + 1). Mutating to k - rank_zero + 1
    or k + rank_zero - 1 would produce different scores and orderings.
    """

    def test_rrf_exact_scores_two_retrievers_different_orders(self) -> None:
        """Two retrievers, overlapping candidates in different orders.

        Dense: [r1(rank0), r2(rank1), r3(rank2)]
        Sparse: [r3(rank0), r2(rank1), r1(rank2)]

        RRF(r1) = 1/(60+0+1) + 1/(60+2+1) = 1/61 + 1/63
        RRF(r2) = 1/(60+1+1) + 1/(60+1+1) = 2/62
        RRF(r3) = 1/(60+2+1) + 1/(60+0+1) = 1/63 + 1/61

        r1 and r3 should tie; r2 has a different score.
        """
        r1, r2, r3 = _rule("r1"), _rule("r2"), _rule("r3")
        dense = [
            ScoredCandidate(rule=r1, score=0.9, retriever="dense"),
            ScoredCandidate(rule=r2, score=0.8, retriever="dense"),
            ScoredCandidate(rule=r3, score=0.7, retriever="dense"),
        ]
        sparse = [
            ScoredCandidate(rule=r3, score=5.0, retriever="sparse"),
            ScoredCandidate(rule=r2, score=4.0, retriever="sparse"),
            ScoredCandidate(rule=r1, score=3.0, retriever="sparse"),
        ]

        result = fuse([dense, sparse], k=60)

        scores_by_text = {c.rule.text: c.score for c in result}

        expected_r1 = 1.0 / 61 + 1.0 / 63
        expected_r2 = 1.0 / 62 + 1.0 / 62
        expected_r3 = 1.0 / 63 + 1.0 / 61

        assert scores_by_text["r1"] == pytest.approx(expected_r1, rel=1e-9)
        assert scores_by_text["r2"] == pytest.approx(expected_r2, rel=1e-9)
        assert scores_by_text["r3"] == pytest.approx(expected_r3, rel=1e-9)

        # r1 and r3 should be equal (symmetric ranks)
        assert scores_by_text["r1"] == pytest.approx(scores_by_text["r3"], rel=1e-9)

        # r2 at rank 1 in both should differ from r1/r3
        # 2/62 = 1/31, 1/61+1/63 = 124/3843 ≈ 0.03227
        # 2/62 ≈ 0.03226 — very close but NOT equal
        # If sign were flipped (k - rank + 1), scores diverge
        assert expected_r2 != pytest.approx(expected_r1, rel=1e-6)

    def test_rrf_rank_matters_not_original_score(self) -> None:
        """RRF uses rank position, not the original retriever scores.

        If the formula used k - rank + 1 instead of k + rank + 1,
        higher ranks would get HIGHER scores (wrong direction).
        """
        r1, r2 = _rule("r1"), _rule("r2")
        # r1 at rank 0 (best), r2 at rank 1
        candidates = [
            ScoredCandidate(rule=r1, score=0.9, retriever="dense"),
            ScoredCandidate(rule=r2, score=0.1, retriever="dense"),
        ]

        result = fuse([candidates], k=60)

        # With correct formula: r1 gets 1/61, r2 gets 1/62
        # With sign flip (k-rank+1): r1 gets 1/61, r2 gets 1/60 (WRONG: r2 > r1)
        assert result[0].rule.text == "r1"
        assert result[1].rule.text == "r2"
        assert result[0].score == pytest.approx(1.0 / 61, rel=1e-9)
        assert result[1].score == pytest.approx(1.0 / 62, rel=1e-9)
        assert result[0].score > result[1].score

    def test_rrf_k_parameter_in_denominator(self) -> None:
        """Verify k is added (not subtracted) in denominator.

        With k=60: score for rank 0 = 1/(60+0+1) = 1/61
        With k=2:  score for rank 0 = 1/(2+0+1) = 1/3

        If formula were k-rank+1: k=2, rank=0 => 1/3 (same)
        but k=2, rank=2 => 1/1 (bigger than rank 0, WRONG).
        """
        r1, r2, r3 = _rule("r1"), _rule("r2"), _rule("r3")
        candidates = [
            ScoredCandidate(rule=r1, score=0.9, retriever="d"),
            ScoredCandidate(rule=r2, score=0.5, retriever="d"),
            ScoredCandidate(rule=r3, score=0.1, retriever="d"),
        ]

        result = fuse([candidates], k=2)

        # rank 0: 1/(2+0+1)=1/3, rank 1: 1/(2+1+1)=1/4, rank 2: 1/(2+2+1)=1/5
        assert result[0].score == pytest.approx(1.0 / 3, rel=1e-9)
        assert result[1].score == pytest.approx(1.0 / 4, rel=1e-9)
        assert result[2].score == pytest.approx(1.0 / 5, rel=1e-9)
        # Strictly decreasing
        assert result[0].score > result[1].score > result[2].score


class TestFuseBranchesNormalization:
    """Two-level fusion: _fuse_branches groups by family and balances weights.

    Guards against the class of bug where query expansion amplifies one
    retriever family's effective weight with the number of branches.
    """

    def test_single_branch_per_family_matches_flat_fuse(self) -> None:
        """No expansions → exactly the same result as one-level fuse."""
        from cuecard.retrieval.pipeline import _fuse_branches

        r1, r2 = _rule("shared"), _rule("dense_only")
        dense = [
            ScoredCandidate(rule=r1, score=0.9, retriever="dense"),
            ScoredCandidate(rule=r2, score=0.8, retriever="dense"),
        ]
        sparse = [
            ScoredCandidate(rule=r1, score=5.0, retriever="sparse"),
        ]
        weights = {"dense": 0.7, "sparse": 0.3}

        branched = _fuse_branches(
            [dense, sparse], k=60, top_k=None, retriever_weights=weights,
        )
        flat = fuse([dense, sparse], k=60, retriever_weights=weights)

        # Same order, same scores
        assert [c.rule.text for c in branched] == [c.rule.text for c in flat]
        for b, f in zip(branched, flat, strict=True):
            assert b.score == pytest.approx(f.score, rel=1e-9)

    def test_expansion_branches_do_not_dominate_family_weight(self) -> None:
        """N dense branches must not multiply 'dense_weight' by N.

        Scenario: 3 dense branches (raw + 2 expansions) all rank the same
        rule at position 0; 1 sparse branch ranks a DIFFERENT rule at
        position 0.  With flat fuse + retriever_weights, the dense rule
        gets 3 * 0.7 * (1/61) and the sparse rule gets 0.3 * (1/61) —
        3x amplification from branches alone.

        With _fuse_branches, each family contributes exactly once to the
        outer weighting.
        """
        from cuecard.retrieval.pipeline import _fuse_branches

        r_dense = _rule("dense_rule")
        r_sparse = _rule("sparse_rule")

        dense_raw = [ScoredCandidate(rule=r_dense, score=0.9, retriever="dense")]
        dense_exp1 = [ScoredCandidate(rule=r_dense, score=0.8, retriever="dense")]
        dense_exp2 = [ScoredCandidate(rule=r_dense, score=0.7, retriever="dense")]
        sparse_raw = [
            ScoredCandidate(rule=r_sparse, score=5.0, retriever="sparse"),
        ]

        weights = {"dense": 0.7, "sparse": 0.3}
        out = _fuse_branches(
            [dense_raw, dense_exp1, dense_exp2, sparse_raw],
            k=60, top_k=None, retriever_weights=weights,
        )

        scores = {c.rule.text: c.score for c in out}
        # After inner fuse: each family has one ranking with its best rule
        # at position 0. Outer fuse applies weight * 1/(k+1) per family.
        expected_dense = 0.7 * (1.0 / 61)
        expected_sparse = 0.3 * (1.0 / 61)
        assert scores["dense_rule"] == pytest.approx(expected_dense, rel=1e-9)
        assert scores["sparse_rule"] == pytest.approx(expected_sparse, rel=1e-9)
        # Ratio is exactly the configured weight ratio
        assert (
            scores["dense_rule"] / scores["sparse_rule"]
            == pytest.approx(0.7 / 0.3, rel=1e-9)
        )

    def test_intra_family_branches_equal_weighted(self) -> None:
        """Within a family, each branch contributes equally (no weighting)."""
        from cuecard.retrieval.pipeline import _fuse_branches

        r_raw = _rule("only_in_raw")
        r_exp = _rule("only_in_expansion")

        dense_raw = [
            ScoredCandidate(rule=r_raw, score=0.9, retriever="dense"),
        ]
        dense_exp = [
            ScoredCandidate(rule=r_exp, score=0.9, retriever="dense"),
        ]

        out = _fuse_branches(
            [dense_raw, dense_exp], k=60, top_k=None, retriever_weights=None,
        )
        scores = {c.rule.text: c.score for c in out}
        # Both rules sit at rank 0 inside their own branch, equal RRF
        assert scores["only_in_raw"] == pytest.approx(
            scores["only_in_expansion"], rel=1e-9,
        )

    def test_empty_branches(self) -> None:
        from cuecard.retrieval.pipeline import _fuse_branches

        assert _fuse_branches(
            [], k=60, top_k=None, retriever_weights=None,
        ) == []
        assert _fuse_branches(
            [[], []], k=60, top_k=None, retriever_weights=None,
        ) == []

    def test_top_k_truncation_outer(self) -> None:
        from cuecard.retrieval.pipeline import _fuse_branches

        rules = [_rule(f"r{i}") for i in range(5)]
        dense = [
            ScoredCandidate(rule=r, score=1.0 - i * 0.1, retriever="dense")
            for i, r in enumerate(rules)
        ]
        sparse = [
            ScoredCandidate(rule=r, score=0.5, retriever="sparse")
            for r in rules[2:]
        ]
        out = _fuse_branches(
            [dense, sparse], k=60, top_k=3,
            retriever_weights={"dense": 0.7, "sparse": 0.3},
        )
        assert len(out) == 3

    def test_single_family_single_branch_truncates(self) -> None:
        """Single branch, single family: still honors top_k."""
        from cuecard.retrieval.pipeline import _fuse_branches

        rules = [_rule(f"r{i}") for i in range(5)]
        dense = [
            ScoredCandidate(rule=r, score=1.0 - i * 0.1, retriever="dense")
            for i, r in enumerate(rules)
        ]
        out = _fuse_branches(
            [dense], k=60, top_k=2, retriever_weights={"dense": 0.7},
        )
        assert len(out) == 2
        assert [c.rule.text for c in out] == ["r0", "r1"]
