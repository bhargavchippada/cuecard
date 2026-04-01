"""Tests for cuecard.eval — fixtures, metrics, pipeline, and reporting."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cuecard.eval import (
    EvalSummary,
    Fixture,
    FixtureResult,
    anti_precision,
    format_eval_report,
    load_fixtures,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    run_eval,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fixtures(path: str, data: list[dict[str, object]]) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def _valid_fixture_data() -> list[dict[str, object]]:
    return [
        {
            "id": "test-1",
            "query": "how to commit safely",
            "corpus": "rules.txt",
            "should_match": ["Never commit secrets"],
            "should_not_match": ["Send Enter after tmux"],
        },
    ]


# ---------------------------------------------------------------------------
# TestLoadFixtures
# ---------------------------------------------------------------------------


class TestLoadFixtures:
    def test_valid_json(self, tmp_path: object) -> None:
        p = str(tmp_path / "fixtures.json")  # type: ignore[operator]
        _write_fixtures(p, _valid_fixture_data())

        result = load_fixtures(p)

        assert len(result) == 1
        assert isinstance(result[0], Fixture)
        assert result[0].id == "test-1"
        assert result[0].query == "how to commit safely"
        assert result[0].corpus == "rules.txt"
        assert result[0].should_match == ("Never commit secrets",)
        assert result[0].should_not_match == ("Send Enter after tmux",)

    def test_missing_fields_raises(self, tmp_path: object) -> None:
        p = str(tmp_path / "bad.json")  # type: ignore[operator]
        _write_fixtures(p, [{"id": "x", "query": "q"}])

        with pytest.raises(ValueError, match="missing fields"):
            load_fixtures(p)

    def test_empty_list(self, tmp_path: object) -> None:
        p = str(tmp_path / "empty.json")  # type: ignore[operator]
        _write_fixtures(p, [])

        result = load_fixtures(p)
        assert result == []

    def test_not_a_list_raises(self, tmp_path: object) -> None:
        p = str(tmp_path / "obj.json")  # type: ignore[operator]
        with open(p, "w") as f:
            json.dump({"not": "a list"}, f)

        with pytest.raises(ValueError, match="JSON array"):
            load_fixtures(p)

    def test_entry_not_dict_raises(self, tmp_path: object) -> None:
        p = str(tmp_path / "str.json")  # type: ignore[operator]
        with open(p, "w") as f:
            json.dump(["not a dict"], f)

        with pytest.raises(ValueError, match="must be an object"):
            load_fixtures(p)

    def test_should_match_not_list_raises(self, tmp_path: object) -> None:
        p = str(tmp_path / "bad_match.json")  # type: ignore[operator]
        data = _valid_fixture_data()
        data[0]["should_match"] = "not a list"
        _write_fixtures(p, data)

        with pytest.raises(ValueError, match="should_match must be a list"):
            load_fixtures(p)

    def test_should_not_match_not_list_raises(self, tmp_path: object) -> None:
        p = str(tmp_path / "bad_anti.json")  # type: ignore[operator]
        data = _valid_fixture_data()
        data[0]["should_not_match"] = "not a list"
        _write_fixtures(p, data)

        with pytest.raises(ValueError, match="should_not_match must be a list"):
            load_fixtures(p)

    def test_multiple_fixtures(self, tmp_path: object) -> None:
        p = str(tmp_path / "multi.json")  # type: ignore[operator]
        data = _valid_fixture_data()
        data.append({
            "id": "test-2",
            "query": "formatting code",
            "corpus": "rules.txt",
            "should_match": ["Run formatter"],
            "should_not_match": [],
        })
        _write_fixtures(p, data)

        result = load_fixtures(p)
        assert len(result) == 2
        assert result[1].id == "test-2"
        assert result[1].should_not_match == ()


# ---------------------------------------------------------------------------
# TestPrecisionAtK
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b"], {"a", "b"}) == 1.0

    def test_none_relevant(self) -> None:
        assert precision_at_k(["x", "y"], {"a", "b"}) == 0.0

    def test_partial(self) -> None:
        assert precision_at_k(["a", "x", "b"], {"a", "b"}) == pytest.approx(2 / 3)

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# TestRecallAtK
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_all_found(self) -> None:
        assert recall_at_k(["a", "b"], {"a", "b"}) == 1.0

    def test_none_found(self) -> None:
        assert recall_at_k(["x", "y"], {"a", "b"}) == 0.0

    def test_partial(self) -> None:
        assert recall_at_k(["a", "x"], {"a", "b", "c"}) == pytest.approx(1 / 3)

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a", "b"], set()) == 0.0


# ---------------------------------------------------------------------------
# TestMRR
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_result_relevant(self) -> None:
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_result_relevant(self) -> None:
        assert mrr(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_none_relevant(self) -> None:
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_retrieved(self) -> None:
        assert mrr([], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# TestNDCG
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_perfect_ranking(self) -> None:
        # Two relevant items at positions 1 and 2 — ideal ordering
        result = ndcg_at_k(["a", "b", "x"], {"a", "b"})
        assert result == pytest.approx(1.0)

    def test_imperfect_ranking(self) -> None:
        # Two relevant items, but one is at position 3 instead of 2
        # DCG  = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0.0 + 0.5 = 1.5
        # iDCG = 1/log2(2) + 1/log2(3)              = 1.0 + 0.631 = 1.631
        import math

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


# ---------------------------------------------------------------------------
# TestAntiPrecision
# ---------------------------------------------------------------------------


class TestAntiPrecision:
    def test_no_anti_relevant(self) -> None:
        assert anti_precision(["a", "b"], {"x"}) == 0.0

    def test_some_anti_relevant(self) -> None:
        assert anti_precision(["a", "x"], {"x"}) == pytest.approx(0.5)

    def test_all_anti_relevant(self) -> None:
        assert anti_precision(["x", "y"], {"x", "y"}) == 1.0

    def test_empty_retrieved(self) -> None:
        assert anti_precision([], {"x"}) == 0.0


# ---------------------------------------------------------------------------
# Mock model for run_eval
# ---------------------------------------------------------------------------

_DIM = 384


class MockModel:
    """Deterministic embedding model for testing."""

    def passage_embed(
        self, texts: list[str], **kwargs: object,
    ) -> list[np.ndarray]:
        vecs = []
        for i, _ in enumerate(texts):
            vec = np.zeros(_DIM, dtype=np.float32)
            vec[i % _DIM] = 1.0
            vecs.append(vec)
        return vecs

    def query_embed(
        self, texts: list[str], **kwargs: object,
    ) -> list[np.ndarray]:
        vecs = []
        for _ in texts:
            vec = np.zeros(_DIM, dtype=np.float32)
            vec[0] = 1.0  # Matches first passage
            vecs.append(vec)
        return vecs


# ---------------------------------------------------------------------------
# TestRunEval
# ---------------------------------------------------------------------------


class TestRunEval:
    def _setup_corpus(self, tmp_path: object) -> tuple[str, list[Fixture]]:
        """Create a corpus file and matching fixtures."""
        corpus_dir = str(tmp_path)  # type: ignore[arg-type]
        corpus_file = str(tmp_path / "rules.txt")  # type: ignore[operator]
        with open(corpus_file, "w") as f:
            f.write("Never commit secrets\n")
            f.write("Run formatter before push\n")
            f.write("Send Enter after tmux\n")

        fixtures = [
            Fixture(
                id="test-1",
                query="how to commit safely",
                corpus="rules.txt",
                should_match=("Never commit secrets",),
                should_not_match=("Send Enter after tmux",),
            ),
        ]
        return corpus_dir, fixtures

    def test_full_pipeline(self, tmp_path: object) -> None:
        corpus_dir, fixtures = self._setup_corpus(tmp_path)
        model = MockModel()

        summary = run_eval(
            fixtures,
            corpus_dir,
            "test-model",
            model=model,
            top_k=5,
            threshold=0.0,
        )

        assert summary.fixture_count == 1
        assert len(summary.per_fixture) == 1

        fr = summary.per_fixture[0]
        assert fr.fixture_id == "test-1"
        assert fr.query == "how to commit safely"
        assert len(fr.retrieved) > 0
        assert fr.latency_ms >= 0.0
        # Metrics are computable (not NaN)
        assert fr.precision_at_k >= 0.0
        assert fr.recall_at_k >= 0.0
        assert fr.mrr >= 0.0
        assert fr.ndcg_at_k >= 0.0
        assert fr.anti_precision >= 0.0

    def test_all_metrics_computed(self, tmp_path: object) -> None:
        corpus_dir, fixtures = self._setup_corpus(tmp_path)
        model = MockModel()

        summary = run_eval(
            fixtures, corpus_dir, "test-model", model=model, threshold=0.0,
        )

        assert summary.mean_precision >= 0.0
        assert summary.mean_recall >= 0.0
        assert summary.mean_mrr >= 0.0
        assert summary.mean_ndcg >= 0.0
        assert summary.mean_anti_precision >= 0.0
        assert summary.latency_p50_ms >= 0.0
        assert summary.latency_p95_ms >= 0.0
        assert summary.latency_p99_ms >= 0.0

    def test_latency_measured(self, tmp_path: object) -> None:
        corpus_dir, fixtures = self._setup_corpus(tmp_path)
        model = MockModel()

        summary = run_eval(
            fixtures, corpus_dir, "test-model", model=model, threshold=0.0,
        )

        fr = summary.per_fixture[0]
        # Latency must be a positive number (retrieval takes nonzero time)
        assert fr.latency_ms > 0.0

    def test_empty_fixtures(self) -> None:
        summary = run_eval([], "/nonexistent", "test-model")

        assert summary.fixture_count == 0
        assert summary.per_fixture == ()
        assert summary.mean_precision == 0.0
        assert summary.latency_p50_ms == 0.0

    def test_first_rule_matched(self, tmp_path: object) -> None:
        """MockModel query vector matches first passage — verify it's retrieved."""
        corpus_dir, fixtures = self._setup_corpus(tmp_path)
        model = MockModel()

        summary = run_eval(
            fixtures, corpus_dir, "test-model", model=model, threshold=0.0,
        )

        fr = summary.per_fixture[0]
        # First rule ("Never commit secrets") should be retrieved
        # because query_embed returns vec[0]=1.0 matching passage 0's vec[0]=1.0
        assert "Never commit secrets" in fr.retrieved

    def test_multiple_fixtures(self, tmp_path: object) -> None:
        corpus_dir = str(tmp_path)  # type: ignore[arg-type]
        corpus_file = str(tmp_path / "rules.txt")  # type: ignore[operator]
        with open(corpus_file, "w") as f:
            f.write("Rule A\n")
            f.write("Rule B\n")

        fixtures = [
            Fixture(
                id="f1", query="q1", corpus="rules.txt",
                should_match=("Rule A",), should_not_match=(),
            ),
            Fixture(
                id="f2", query="q2", corpus="rules.txt",
                should_match=("Rule B",), should_not_match=(),
            ),
        ]
        model = MockModel()
        summary = run_eval(
            fixtures, corpus_dir, "test-model", model=model, threshold=0.0,
        )

        assert summary.fixture_count == 2
        assert len(summary.per_fixture) == 2


# ---------------------------------------------------------------------------
# TestFormatEvalReport
# ---------------------------------------------------------------------------


class TestFormatEvalReport:
    def _make_summary(
        self, fixture_count: int = 1,
    ) -> EvalSummary:
        per_fixture = tuple(
            FixtureResult(
                fixture_id=f"fixture-{i}",
                query=f"query {i}",
                retrieved=("rule-a",),
                precision_at_k=0.8,
                recall_at_k=0.6,
                mrr=1.0,
                ndcg_at_k=0.9,
                anti_precision=0.0,
                latency_ms=1.5,
            )
            for i in range(fixture_count)
        )
        return EvalSummary(
            fixture_count=fixture_count,
            mean_precision=0.8,
            mean_recall=0.6,
            mean_mrr=1.0,
            mean_ndcg=0.9,
            mean_anti_precision=0.0,
            latency_p50_ms=1.5,
            latency_p95_ms=2.0,
            latency_p99_ms=2.5,
            per_fixture=per_fixture,
        )

    def test_produces_readable_output(self) -> None:
        summary = self._make_summary(fixture_count=2)
        report = format_eval_report(summary)

        assert "Evaluation Report" in report
        assert "fixture-0" in report
        assert "fixture-1" in report
        assert "Mean Precision@k" in report
        assert "Mean MRR" in report
        assert "Latency p50" in report
        assert "0.800" in report
        assert "1.000" in report

    def test_handles_empty_results(self) -> None:
        summary = EvalSummary(
            fixture_count=0,
            mean_precision=0.0,
            mean_recall=0.0,
            mean_mrr=0.0,
            mean_ndcg=0.0,
            mean_anti_precision=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            per_fixture=(),
        )
        report = format_eval_report(summary)

        assert "Evaluation Report" in report
        assert "Fixtures:          0" in report

    def test_contains_all_metric_labels(self) -> None:
        summary = self._make_summary()
        report = format_eval_report(summary)

        for label in [
            "P@k", "R@k", "MRR", "nDCG@k", "AntiP", "Latency(ms)",
            "Mean Precision@k", "Mean Recall@k", "Mean MRR",
            "Mean nDCG@k", "Mean Anti-P",
            "Latency p50", "Latency p95", "Latency p99",
        ]:
            assert label in report, f"Missing label: {label}"
