"""Tests for cuecard.eval — metric functions and quality score."""

from __future__ import annotations

import math

import pytest

from cuecard.eval.harness import (
    anti_precision,
    context_waste_ratio,
    mrr,
    ndcg_at_k,
    noise_ratio,
    precision_at_k,
    quality_score,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b"], {"a", "b"}) == 1.0

    def test_none_relevant(self) -> None:
        assert precision_at_k(["x", "y"], {"a", "b"}) == 0.0

    def test_partial(self) -> None:
        assert precision_at_k(["a", "x", "b"], {"a", "b"}) == pytest.approx(2 / 3)

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], {"a"}) == 0.0


class TestRecallAtK:
    def test_all_found(self) -> None:
        assert recall_at_k(["a", "b"], {"a", "b"}) == 1.0

    def test_none_found(self) -> None:
        assert recall_at_k(["x", "y"], {"a", "b"}) == 0.0

    def test_partial(self) -> None:
        assert recall_at_k(["a", "x"], {"a", "b", "c"}) == pytest.approx(1 / 3)

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a", "b"], set()) == 0.0


class TestMRR:
    def test_first_result_relevant(self) -> None:
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_result_relevant(self) -> None:
        assert mrr(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_none_relevant(self) -> None:
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_retrieved(self) -> None:
        assert mrr([], {"a"}) == 0.0


class TestNDCG:
    def test_perfect_ranking(self) -> None:
        # Two relevant items at positions 1 and 2 — ideal ordering
        result = ndcg_at_k(["a", "b", "x"], {"a", "b"})
        assert result == pytest.approx(1.0)

    def test_imperfect_ranking(self) -> None:
        # Two relevant items, but one is at position 3 instead of 2
        # DCG  = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0.0 + 0.5 = 1.5
        # iDCG = 1/log2(2) + 1/log2(3)              = 1.0 + 0.631 = 1.631
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = dcg / idcg
        result = ndcg_at_k(["a", "x", "b"], {"a", "b"})
        assert result == pytest.approx(expected)

    def test_no_relevant_results(self) -> None:
        assert ndcg_at_k(["x", "y"], {"a"}) == 0.0

    def test_empty_retrieved(self) -> None:
        assert ndcg_at_k([], {"a"}) == 0.0

    def test_empty_relevant(self) -> None:
        assert ndcg_at_k(["a", "b"], set()) == 0.0

    def test_single_relevant_at_first(self) -> None:
        assert ndcg_at_k(["a"], {"a"}) == pytest.approx(1.0)


class TestAntiPrecision:
    def test_no_anti_relevant(self) -> None:
        assert anti_precision(["a", "b"], {"x"}) == 0.0

    def test_some_anti_relevant(self) -> None:
        assert anti_precision(["a", "x"], {"x"}) == pytest.approx(0.5)

    def test_all_anti_relevant(self) -> None:
        assert anti_precision(["x", "y"], {"x", "y"}) == 1.0

    def test_empty_retrieved(self) -> None:
        assert anti_precision([], {"x"}) == 0.0


class TestNoiseRatio:
    def test_all_relevant(self) -> None:
        assert noise_ratio(["a", "b"], {"a", "b"}) == 0.0

    def test_all_irrelevant(self) -> None:
        assert noise_ratio(["x", "y"], {"a"}) == 1.0

    def test_mixed(self) -> None:
        assert noise_ratio(["a", "x", "b"], {"a", "b"}) == pytest.approx(1 / 3)

    def test_empty_retrieved(self) -> None:
        assert noise_ratio([], {"a"}) == 0.0

    def test_empty_relevant(self) -> None:
        assert noise_ratio(["x", "y"], set()) == 1.0


class TestContextWasteRatio:
    def test_all_relevant(self) -> None:
        assert context_waste_ratio(["abc", "de"], {"abc", "de"}) == 0.0

    def test_all_irrelevant(self) -> None:
        assert context_waste_ratio(["abc", "de"], set()) == 1.0

    def test_mixed_char_weighted(self) -> None:
        # "short" (5 chars) relevant, "a very long irrelevant rule" (27 chars) not
        retrieved = ["short", "a very long irrelevant rule"]
        relevant = {"short"}
        ratio = context_waste_ratio(retrieved, relevant)
        assert ratio == pytest.approx(27 / 32)

    def test_empty_retrieved(self) -> None:
        assert context_waste_ratio([], {"a"}) == 0.0

    def test_empty_strings(self) -> None:
        assert context_waste_ratio(["", ""], {"a"}) == 0.0


class TestQualityScore:
    def test_positive_perfect(self) -> None:
        # Found all, no noise
        score = quality_score(["a", "b"], {"a", "b"}, is_negative=False)
        assert score == pytest.approx(1.0)

    def test_positive_half_recall_no_noise(self) -> None:
        # Found 1 of 2, no noise → R=0.5, P=1.0, F2=5*1*0.5/(4*1+0.5)=0.556
        score = quality_score(["a"], {"a", "b"}, is_negative=False)
        assert score == pytest.approx(5.0 * 1.0 * 0.5 / (4.0 + 0.5))

    def test_positive_full_recall_with_noise(self) -> None:
        # Found all + 1 noise → R=1.0, P=0.5, F2=5*0.5*1.0/(4*0.5+1.0)=0.833
        score = quality_score(["a", "noise"], {"a"}, is_negative=False)
        assert score == pytest.approx(5.0 * 0.5 * 1.0 / (2.0 + 1.0))

    def test_positive_nothing_retrieved(self) -> None:
        score = quality_score([], {"a", "b"}, is_negative=False)
        assert score == 0.0

    def test_negative_silent(self) -> None:
        score = quality_score([], set(), is_negative=True)
        assert score == 1.0

    def test_negative_not_silent(self) -> None:
        score = quality_score(["noise"], set(), is_negative=True)
        assert score == 0.0

    def test_positive_empty_relevant_silent(self) -> None:
        # Degenerate: positive fixture with no should_match, stays silent
        score = quality_score([], set(), is_negative=False)
        assert score == 1.0

    def test_positive_empty_relevant_noisy(self) -> None:
        # Degenerate: positive fixture with no should_match, returns something
        score = quality_score(["x"], set(), is_negative=False)
        assert score == 0.0
