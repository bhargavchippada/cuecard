"""Tests for cuecard.eval — fixtures, metrics, pipeline, and reporting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from cuecard.eval import (
    EvalSummary,
    Fixture,
    FixtureResult,
    TierSummary,
    anti_precision,
    context_waste_ratio,
    format_eval_report,
    load_fixtures,
    mrr,
    ndcg_at_k,
    noise_ratio,
    precision_at_k,
    quality_score,
    recall_at_k,
    run_eval,
)

if TYPE_CHECKING:
    from pathlib import Path

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
            "difficulty": "medium",
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
        assert result[0].difficulty == "medium"

    def test_difficulty_defaults_to_unknown(
        self, tmp_path: object,
    ) -> None:
        data = _valid_fixture_data()
        del data[0]["difficulty"]
        p = str(tmp_path / "no_diff.json")  # type: ignore[operator]
        _write_fixtures(p, data)
        result = load_fixtures(p)
        assert result[0].difficulty == "unknown"

    def test_difficulty_invalid_type_raises(
        self, tmp_path: object,
    ) -> None:
        data = _valid_fixture_data()
        data[0]["difficulty"] = 42
        p = str(tmp_path / "bad_diff.json")  # type: ignore[operator]
        _write_fixtures(p, data)
        with pytest.raises(ValueError, match="difficulty must be a string"):
            load_fixtures(p)

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
# TestNoiseRatio
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestContextWasteRatio
# ---------------------------------------------------------------------------


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
                difficulty="medium",
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
        assert fr.difficulty == "medium"
        assert len(fr.retrieved) > 0
        assert fr.latency_ms >= 0.0
        assert fr.retrieved_count > 0
        # Metrics are computable (not NaN)
        assert fr.precision_at_k >= 0.0
        assert fr.recall_at_k >= 0.0
        assert fr.mrr >= 0.0
        assert fr.ndcg_at_k >= 0.0
        assert fr.anti_precision >= 0.0
        assert 0.0 <= fr.noise_ratio <= 1.0
        assert 0.0 <= fr.context_waste_ratio <= 1.0

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
        assert 0.0 <= summary.mean_noise_ratio <= 1.0
        assert 0.0 <= summary.mean_context_waste_ratio <= 1.0
        assert summary.mean_retrieved_count >= 0.0
        assert 0.0 <= summary.mean_quality <= 1.0
        assert summary.positive_recall >= 0.0
        assert summary.positive_quality >= 0.0
        assert len(summary.per_tier) > 0
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
                difficulty="easy",
            ),
            Fixture(
                id="f2", query="q2", corpus="rules.txt",
                should_match=("Rule B",), should_not_match=(),
                difficulty="hard",
            ),
            Fixture(
                id="f3", query="q3", corpus="rules.txt",
                should_match=(), should_not_match=(),
                difficulty="custom_tier",
            ),
        ]
        model = MockModel()
        summary = run_eval(
            fixtures, corpus_dir, "test-model", model=model, threshold=0.0,
        )

        assert summary.fixture_count == 3
        assert len(summary.per_fixture) == 3
        # Per-tier: easy, hard, and custom_tier (unknown)
        tiers = {t.tier: t for t in summary.per_tier}
        assert "easy" in tiers
        assert "hard" in tiers
        assert "custom_tier" in tiers
        assert tiers["easy"].count == 1
        assert tiers["hard"].count == 1
        assert tiers["custom_tier"].count == 1


class TestRunEvalSampling:
    """Test run_eval with sample_ratio."""

    def test_sample_ratio_reduces_fixtures(self, tmp_path: object) -> None:
        corpus_dir = str(tmp_path)  # type: ignore[arg-type]
        corpus_file = str(tmp_path / "rules.txt")  # type: ignore[operator]
        with open(corpus_file, "w") as f:
            f.write("Rule A\nRule B\nRule C\n")

        fixtures = [
            Fixture(id=f"fix-{i}", query=f"q {i}", corpus="rules.txt",
                    should_match=("Rule A",), should_not_match=(),
                    difficulty=d)
            for i, d in enumerate(
                ["easy"] * 10 + ["medium"] * 10 + ["hard"] * 10 + ["negative"] * 10
            )
        ]
        model = MockModel()
        summary = run_eval(
            fixtures, corpus_dir, "test-model",
            model=model, top_k=5, threshold=0.0,
            sample_ratio=0.3,
        )
        # Stratified: 3 per tier (30% of 10), 4 tiers = 12
        assert summary.fixture_count == 12

    def test_sample_ratio_one_uses_all(self, tmp_path: object) -> None:
        corpus_dir = str(tmp_path)  # type: ignore[arg-type]
        corpus_file = str(tmp_path / "rules.txt")  # type: ignore[operator]
        with open(corpus_file, "w") as f:
            f.write("Rule A\n")

        fixtures = [
            Fixture(id=f"fix-{i}", query=f"q {i}", corpus="rules.txt",
                    should_match=("Rule A",), should_not_match=(),
                    difficulty="easy")
            for i in range(5)
        ]
        model = MockModel()
        summary = run_eval(
            fixtures, corpus_dir, "test-model",
            model=model, top_k=5, threshold=0.0,
            sample_ratio=1.0,
        )
        assert summary.fixture_count == 5

    def test_tqdm_import_failure_graceful(self, tmp_path: object) -> None:
        """Verify eval works when tqdm is not available."""
        corpus_dir = str(tmp_path)  # type: ignore[arg-type]
        corpus_file = str(tmp_path / "rules.txt")  # type: ignore[operator]
        with open(corpus_file, "w") as f:
            f.write("Rule A\n")

        fixtures = [
            Fixture(id="fix-1", query="q", corpus="rules.txt",
                    should_match=("Rule A",), should_not_match=(),
                    difficulty="easy"),
        ]
        model = MockModel()

        import builtins
        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "tqdm":
                raise ImportError("no tqdm")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            summary = run_eval(
                fixtures, corpus_dir, "test-model",
                model=model, top_k=5, threshold=0.0,
            )
        assert summary.fixture_count == 1


class TestRunEvalWithMode:
    """Test run_eval with pipeline mode parameter."""

    def test_mode_rerank_calls_pipeline(self, tmp_path: object) -> None:
        corpus_dir = str(tmp_path)  # type: ignore[arg-type]
        corpus_file = str(tmp_path / "rules.txt")  # type: ignore[operator]
        with open(corpus_file, "w") as f:
            f.write("Rule A\n")

        fixtures = [
            Fixture(
                id="mode-test",
                query="test",
                corpus="rules.txt",
                should_match=("Rule A",),
                should_not_match=(),
                difficulty="easy",
            ),
        ]
        model = MockModel()

        from cuecard.models import (
            PipelineResult,
            Provenance,
            RankedResult,
            Rule,
            StageTrace,
        )

        fake_results = [
            RankedResult(
                rule=Rule(
                    text="Rule A",
                    provenance=Provenance(file=corpus_file, line_start=1, line_end=1),
                ),
                score=0.9,
            ),
        ]
        fake_pipeline = PipelineResult(
            results=tuple(fake_results),
            stages=(StageTrace(
                stage="embedding", input_count=1,
                output_count=1, latency_ms=0.5,
            ),),
            mode="rerank",
        )

        with patch(
            "cuecard.pipeline.run_pipeline",
            return_value=fake_pipeline,
        ) as mock_pipe:
            summary = run_eval(
                fixtures, corpus_dir, "test-model", model=model, mode="rerank",
            )

        assert summary.fixture_count == 1
        mock_pipe.assert_called_once()
        assert summary.per_fixture[0].retrieved == ("Rule A",)


# ---------------------------------------------------------------------------
# TestCorpusOverride
# ---------------------------------------------------------------------------


class TestCorpusOverride:
    def test_corpus_override_builds_unified_index(
        self, tmp_path: Path,
    ) -> None:
        """corpus_override forces all fixtures to use a combined corpus."""
        # Create two corpus files
        corpus_a = tmp_path / "corpus_a.txt"
        corpus_a.write_text("Rule from corpus A\n")
        corpus_b = tmp_path / "corpus_b.txt"
        corpus_b.write_text("Rule from corpus B\n")

        fixtures = [
            Fixture(
                id="fix-a",
                query="test query A",
                corpus="corpus_a.txt",
                should_match=("Rule from corpus A",),
                should_not_match=(),
                difficulty="easy",
            ),
            Fixture(
                id="fix-b",
                query="test query B",
                corpus="corpus_b.txt",
                should_match=("Rule from corpus B",),
                should_not_match=(),
                difficulty="easy",
            ),
        ]

        model = MockModel()
        summary = run_eval(
            fixtures,
            str(tmp_path),
            "test-model",
            model=model,
            corpus_override=(str(corpus_a), str(corpus_b)),
        )

        # Both fixtures ran against the unified index (2 rules)
        assert summary.fixture_count == 2


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
                difficulty="medium",
                retrieved=("rule-a",),
                precision_at_k=0.8,
                recall_at_k=0.6,
                mrr=1.0,
                ndcg_at_k=0.9,
                anti_precision=0.0,
                noise_ratio=0.2,
                context_waste_ratio=0.15,
                retrieved_count=1,
                latency_ms=1.5,
                quality_score=0.48,
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
            mean_noise_ratio=0.2,
            mean_context_waste_ratio=0.15,
            negative_silence_rate=1.0,
            mean_retrieved_count=1.0,
            mean_quality=0.48,
            positive_recall=0.6,
            positive_quality=0.48,
            latency_p50_ms=1.5,
            latency_p95_ms=2.0,
            latency_p99_ms=2.5,
            per_fixture=per_fixture,
            per_tier=(
                TierSummary(
                    tier="medium",
                    count=fixture_count,
                    mean_precision=0.8,
                    mean_recall=0.6,
                    mean_mrr=1.0,
                    mean_noise_ratio=0.2,
                    mean_context_waste_ratio=0.15,
                    silence_rate=0.0,
                    mean_retrieved_count=1.0,
                    mean_quality=0.48,
                ),
            ),
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
            mean_noise_ratio=0.0,
            mean_context_waste_ratio=0.0,
            negative_silence_rate=0.0,
            mean_retrieved_count=0.0,
            mean_quality=0.0,
            positive_recall=0.0,
            positive_quality=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            per_fixture=(),
            per_tier=(),
        )
        report = format_eval_report(summary)

        assert "Evaluation Report" in report
        assert "Fixtures:" in report

    def test_contains_all_metric_labels(self) -> None:
        summary = self._make_summary()
        report = format_eval_report(summary)

        for label in [
            "P@k", "R@k", "MRR", "nDCG", "Noise", "Waste",
            "AntiP", "#Ret", "Lat(ms)", "F2",
            "Quality (F2)", "Positive Quality", "Positive Recall@k",
            "Mean Precision@k", "Mean Recall@k", "Mean MRR",
            "Mean nDCG@k", "Mean Anti-P",
            "Mean Noise Ratio", "Mean Context Waste",
            "Neg Silence Rate", "Mean Retrieved Count",
            "Latency p50", "Latency p95", "Latency p99",
            "Per-Tier Breakdown",
        ]:
            assert label in report, f"Missing label: {label}"
