"""Tests for per-event metrics in cuecard.eval."""

from __future__ import annotations

import json

import pytest

from cuecard.eval.harness import (
    _DEFAULT_EVENT,
    Fixture,
    FixtureResult,
    PerEventMetrics,
    evaluate_per_event,
    format_per_event_report,
    load_fixtures,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    fixture_id: str,
    difficulty: str,
    retrieved: tuple[str, ...] = (),
    recall: float = 0.0,
    noise: float = 0.0,
    quality: float = 0.0,
) -> FixtureResult:
    return FixtureResult(
        fixture_id=fixture_id,
        query=f"query for {fixture_id}",
        difficulty=difficulty,
        retrieved=retrieved,
        precision_at_k=0.0,
        recall_at_k=recall,
        mrr=0.0,
        ndcg_at_k=0.0,
        anti_precision=0.0,
        noise_ratio=noise,
        context_waste_ratio=0.0,
        retrieved_count=len(retrieved),
        latency_ms=1.0,
        quality_score=quality,
    )


def _make_fixture(
    fixture_id: str,
    difficulty: str,
    event: str = "PreToolUse",
) -> Fixture:
    return Fixture(
        id=fixture_id,
        query=f"query for {fixture_id}",
        corpus="rules.txt",
        should_match=("rule-a",) if difficulty != "negative" else (),
        should_not_match=(),
        difficulty=difficulty,
        event=event,
    )


# ---------------------------------------------------------------------------
# TestPerEventMetrics
# ---------------------------------------------------------------------------


class TestPerEventMetrics:
    def test_frozen(self) -> None:
        m = PerEventMetrics(
            event="PreToolUse", fixture_count=1,
            quality=0.8, positive_recall=0.7,
            noise_ratio=0.2, negative_silence=1.0,
        )
        with pytest.raises(AttributeError):
            m.event = "Stop"  # type: ignore[misc]

    def test_fields(self) -> None:
        m = PerEventMetrics(
            event="Stop", fixture_count=5,
            quality=0.6, positive_recall=0.5,
            noise_ratio=0.3, negative_silence=0.8,
        )
        assert m.event == "Stop"
        assert m.fixture_count == 5
        assert m.quality == 0.6
        assert m.positive_recall == 0.5
        assert m.noise_ratio == 0.3
        assert m.negative_silence == 0.8


# ---------------------------------------------------------------------------
# TestEvaluatePerEvent
# ---------------------------------------------------------------------------


class TestEvaluatePerEvent:
    def test_single_event(self) -> None:
        fixtures = [
            _make_fixture("f1", "easy", "PreToolUse"),
            _make_fixture("f2", "medium", "PreToolUse"),
        ]
        results = [
            _make_result("f1", "easy", quality=0.9, recall=0.8, noise=0.1),
            _make_result("f2", "medium", quality=0.7, recall=0.6, noise=0.3),
        ]
        per_event = evaluate_per_event(results, fixtures)
        assert len(per_event) == 1
        assert per_event[0].event == "PreToolUse"
        assert per_event[0].fixture_count == 2
        assert per_event[0].quality == pytest.approx(0.8)
        assert per_event[0].positive_recall == pytest.approx(0.7)
        assert per_event[0].noise_ratio == pytest.approx(0.2)

    def test_multiple_events(self) -> None:
        fixtures = [
            _make_fixture("f1", "easy", "PreToolUse"),
            _make_fixture("f2", "easy", "UserPromptSubmit"),
            _make_fixture("f3", "easy", "Stop"),
        ]
        results = [
            _make_result("f1", "easy", quality=0.9, recall=0.8, noise=0.1),
            _make_result("f2", "easy", quality=0.7, recall=0.6, noise=0.2),
            _make_result("f3", "easy", quality=0.5, recall=0.4, noise=0.3),
        ]
        per_event = evaluate_per_event(results, fixtures)
        assert len(per_event) == 3
        # Verify canonical order
        assert per_event[0].event == "PreToolUse"
        assert per_event[1].event == "UserPromptSubmit"
        assert per_event[2].event == "Stop"

    def test_negative_fixtures_silence(self) -> None:
        fixtures = [
            _make_fixture("f1", "negative", "Stop"),
            _make_fixture("f2", "negative", "Stop"),
        ]
        results = [
            _make_result("f1", "negative", retrieved=(), quality=1.0),
            _make_result("f2", "negative", retrieved=("noise",), quality=0.0),
        ]
        per_event = evaluate_per_event(results, fixtures)
        assert len(per_event) == 1
        assert per_event[0].negative_silence == pytest.approx(0.5)
        assert per_event[0].positive_recall == 0.0

    def test_no_negatives_defaults_silence_to_one(self) -> None:
        fixtures = [_make_fixture("f1", "easy", "Stop")]
        results = [_make_result("f1", "easy", quality=0.8, recall=0.7)]
        per_event = evaluate_per_event(results, fixtures)
        assert per_event[0].negative_silence == 1.0

    def test_empty_results(self) -> None:
        per_event = evaluate_per_event([], [])
        assert per_event == []

    def test_canonical_order(self) -> None:
        """Events appear in _EVENT_ORDER, then alphabetically for unknowns."""
        fixtures = [
            _make_fixture("f1", "easy", "Stop"),
            _make_fixture("f2", "easy", "PreToolUse"),
            _make_fixture("f3", "easy", "UserPromptSubmit"),
            _make_fixture("f4", "easy", "CustomEvent"),
        ]
        results = [
            _make_result("f1", "easy", quality=0.5),
            _make_result("f2", "easy", quality=0.5),
            _make_result("f3", "easy", quality=0.5),
            _make_result("f4", "easy", quality=0.5),
        ]
        per_event = evaluate_per_event(results, fixtures)
        events = [m.event for m in per_event]
        assert events == [
            "PreToolUse", "UserPromptSubmit", "Stop", "CustomEvent",
        ]

    def test_missing_fixture_defaults_to_pretooluse(self) -> None:
        """Result with no matching fixture gets PreToolUse event."""
        fixtures: list[Fixture] = []
        results = [_make_result("orphan", "easy", quality=0.5)]
        per_event = evaluate_per_event(results, fixtures)
        assert len(per_event) == 1
        assert per_event[0].event == "PreToolUse"


# ---------------------------------------------------------------------------
# TestLoadFixturesEvent
# ---------------------------------------------------------------------------


class TestLoadFixturesEvent:
    def test_event_field_parsed(self, tmp_path: object) -> None:
        data = [{
            "id": "test-1",
            "query": "Bash: git commit",
            "corpus": "rules.txt",
            "should_match": ["rule"],
            "should_not_match": [],
            "difficulty": "easy",
            "event": "Stop",
        }]
        p = str(tmp_path / "fx.json")  # type: ignore[operator]
        with open(p, "w") as f:
            json.dump(data, f)
        result = load_fixtures(p)
        assert result[0].event == "Stop"

    def test_event_defaults_to_pretooluse(self, tmp_path: object) -> None:
        data = [{
            "id": "test-2",
            "query": "Bash: ls",
            "corpus": "rules.txt",
            "should_match": [],
            "should_not_match": [],
            "difficulty": "negative",
        }]
        p = str(tmp_path / "fx.json")  # type: ignore[operator]
        with open(p, "w") as f:
            json.dump(data, f)
        result = load_fixtures(p)
        assert result[0].event == _DEFAULT_EVENT

    def test_event_invalid_type_raises(self, tmp_path: object) -> None:
        data = [{
            "id": "test-3",
            "query": "q",
            "corpus": "rules.txt",
            "should_match": [],
            "should_not_match": [],
            "difficulty": "easy",
            "event": 42,
        }]
        p = str(tmp_path / "fx.json")  # type: ignore[operator]
        with open(p, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="event must be a string"):
            load_fixtures(p)


# ---------------------------------------------------------------------------
# TestFormatPerEventReport
# ---------------------------------------------------------------------------


class TestFormatPerEventReport:
    def test_produces_readable_output(self) -> None:
        metrics = [
            PerEventMetrics(
                event="PreToolUse", fixture_count=100,
                quality=0.78, positive_recall=0.80,
                noise_ratio=0.25, negative_silence=0.85,
            ),
            PerEventMetrics(
                event="Stop", fixture_count=30,
                quality=0.65, positive_recall=0.60,
                noise_ratio=0.30, negative_silence=0.90,
            ),
        ]
        report = format_per_event_report(metrics)
        assert "Per-Event Breakdown" in report
        assert "PreToolUse" in report
        assert "Stop" in report
        assert "0.780" in report
        assert "0.650" in report

    def test_empty_metrics(self) -> None:
        report = format_per_event_report([])
        assert "Per-Event Breakdown" in report

    def test_contains_all_column_headers(self) -> None:
        metrics = [
            PerEventMetrics(
                event="Stop", fixture_count=5,
                quality=0.5, positive_recall=0.4,
                noise_ratio=0.3, negative_silence=1.0,
            ),
        ]
        report = format_per_event_report(metrics)
        for label in ["Event", "N", "F2", "PosRecal", "Noise", "NegSil"]:
            assert label in report, f"Missing column header: {label}"
