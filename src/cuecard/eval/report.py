"""Evaluation report formatting and per-event/per-tier aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from cuecard.eval.metrics import _mean
from cuecard.models import DEFAULT_HOOK_EVENT as _DEFAULT_EVENT

if TYPE_CHECKING:
    from cuecard.eval.harness import (
        EvalSummary,
        Fixture,
        FixtureResult,
        PerEventMetrics,
        TierSummary,
    )


# ---------------------------------------------------------------------------
# Per-tier aggregation
# ---------------------------------------------------------------------------

_TIER_ORDER = ("easy", "medium", "hard", "negative")


def _compute_tier_summaries(
    results: list[FixtureResult],
) -> list[TierSummary]:
    """Group results by difficulty tier and compute per-tier metrics."""
    by_tier: dict[str, list[FixtureResult]] = defaultdict(list)
    for r in results:
        by_tier[r.difficulty].append(r)

    summaries: list[TierSummary] = []
    for tier in _TIER_ORDER:
        tier_results = by_tier.get(tier, [])
        if not tier_results:
            continue
        summaries.append(
            _make_tier_summary(tier, tier_results),
        )

    # Include any tiers not in _TIER_ORDER (e.g., "unknown")
    for tier in sorted(by_tier.keys()):
        if tier not in _TIER_ORDER:
            summaries.append(
                _make_tier_summary(tier, by_tier[tier]),
            )

    return summaries


def _make_tier_summary(
    tier: str, tier_results: list[FixtureResult],
) -> TierSummary:
    """Build a TierSummary for a single tier."""
    from cuecard.eval.harness import TierSummary

    n = len(tier_results)
    neg_silent = sum(
        1 for r in tier_results if r.retrieved_count == 0
    )
    return TierSummary(
        tier=tier,
        count=n,
        mean_precision=_mean(
            r.precision_at_k for r in tier_results
        ),
        mean_recall=_mean(
            r.recall_at_k for r in tier_results
        ),
        mean_mrr=_mean(r.mrr for r in tier_results),
        mean_noise_ratio=_mean(
            r.noise_ratio for r in tier_results
        ),
        mean_context_waste_ratio=_mean(
            r.context_waste_ratio for r in tier_results
        ),
        silence_rate=neg_silent / n,
        mean_retrieved_count=_mean(
            float(r.retrieved_count) for r in tier_results
        ),
        mean_quality=_mean(
            r.quality_score for r in tier_results
        ),
    )


# ---------------------------------------------------------------------------
# Per-event metrics
# ---------------------------------------------------------------------------

_EVENT_ORDER = (
    "PreToolUse", "UserPromptSubmit",
    "SubagentStart", "Stop",
)


def evaluate_per_event(
    results: list[FixtureResult],
    fixtures: list[Fixture],
) -> list[PerEventMetrics]:
    """Group fixtures by event and compute quality metrics per group.

    Fixtures without an event field are treated as PreToolUse (backwards compat).
    Returns metrics sorted by _EVENT_ORDER, then alphabetically for unknowns.
    """
    from cuecard.eval.harness import Fixture as FixtureCls
    from cuecard.eval.harness import PerEventMetrics as PerEventMetricsType

    # Build fixture lookup: id -> Fixture
    fixture_by_id: dict[str, Fixture] = {f.id: f for f in fixtures}

    # Group results by event
    by_event: dict[str, list[tuple[FixtureResult, Fixture]]] = defaultdict(list)
    for r in results:
        fx = fixture_by_id.get(r.fixture_id)
        event = fx.event if fx is not None else _DEFAULT_EVENT
        by_event[event].append((r, fx or FixtureCls(
            id=r.fixture_id, query=r.query, corpus="",
            should_match=(), should_not_match=(),
            difficulty=r.difficulty, event=event,
        )))

    metrics: list[PerEventMetricsType] = []
    # Process in canonical order first
    for event in _EVENT_ORDER:
        if event in by_event:
            metrics.append(_make_event_metrics(event, by_event[event]))
    # Then any unknown events alphabetically
    for event in sorted(by_event.keys()):
        if event not in _EVENT_ORDER:
            metrics.append(_make_event_metrics(event, by_event[event]))

    return metrics


def _make_event_metrics(
    event: str,
    pairs: list[tuple[FixtureResult, Fixture]],
) -> PerEventMetrics:
    """Build PerEventMetrics for a single event type."""
    from cuecard.eval.harness import PerEventMetrics

    results_only = [r for r, _ in pairs]
    positives = [r for r, fx in pairs if fx.difficulty != "negative"]
    negatives = [r for r, fx in pairs if fx.difficulty == "negative"]

    neg_silent = sum(1 for r in negatives if r.retrieved_count == 0)
    neg_silence = neg_silent / len(negatives) if negatives else 1.0

    pos_recall = (
        _mean(r.recall_at_k for r in positives) if positives else 0.0
    )

    return PerEventMetrics(
        event=event,
        fixture_count=len(results_only),
        quality=_mean(r.quality_score for r in results_only),
        positive_recall=pos_recall,
        noise_ratio=_mean(r.noise_ratio for r in results_only),
        negative_silence=neg_silence,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_HEADER_FMT = (
    "{:<25s} {:>6s} {:>6s} {:>6s} {:>6s}"
    " {:>6s} {:>6s} {:>6s} {:>4s} {:>10s}"
)
_ROW_FMT = (
    "{:<25s} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f}"
    " {:>6.3f} {:>6.3f} {:>6.3f} {:>4d} {:>10.1f}"
)
_TIER_HEADER_FMT = (
    "{:<10s} {:>5s} {:>6s} {:>6s} {:>6s}"
    " {:>6s} {:>6s} {:>6s} {:>5s} {:>6s}"
)
_TIER_ROW_FMT = (
    "{:<10s} {:>5d} {:>6.3f} {:>6.3f} {:>6.3f}"
    " {:>6.3f} {:>6.3f} {:>6.3f} {:>5.1f} {:>6.3f}"
)


def format_eval_report(summary: EvalSummary) -> str:
    """Format an EvalSummary into a human-readable text report.

    Includes per-fixture table, per-tier breakdown, and aggregates.
    """
    lines: list[str] = []
    lines.append("=" * 95)
    lines.append("Evaluation Report")
    lines.append("=" * 95)
    lines.append("")

    # Per-fixture table
    header = _HEADER_FMT.format(
        "Fixture", "P@k", "R@k", "MRR", "nDCG",
        "Noise", "Waste", "AntiP", "#Ret", "Lat(ms)",
    )
    lines.append(header)
    lines.append("-" * 95)

    for fr in summary.per_fixture:
        fixture_id = fr.fixture_id[:25]
        row = _ROW_FMT.format(
            fixture_id,
            fr.precision_at_k,
            fr.recall_at_k,
            fr.mrr,
            fr.ndcg_at_k,
            fr.noise_ratio,
            fr.context_waste_ratio,
            fr.anti_precision,
            fr.retrieved_count,
            fr.latency_ms,
        )
        lines.append(row)

    lines.append("-" * 95)
    lines.append("")

    # Per-tier breakdown
    if summary.per_tier:
        lines.append("Per-Tier Breakdown")
        tier_header = _TIER_HEADER_FMT.format(
            "Tier", "N", "P@k", "R@k", "MRR",
            "Noise", "Waste", "Silen", "AvgRt", "F2",
        )
        lines.append(tier_header)
        lines.append("-" * 78)
        for ts in summary.per_tier:
            row = _TIER_ROW_FMT.format(
                ts.tier,
                ts.count,
                ts.mean_precision,
                ts.mean_recall,
                ts.mean_mrr,
                ts.mean_noise_ratio,
                ts.mean_context_waste_ratio,
                ts.silence_rate,
                ts.mean_retrieved_count,
                ts.mean_quality,
            )
            lines.append(row)
        lines.append("")

    # Aggregates
    lines.append("Aggregate Metrics")
    lines.append(f"  Fixtures:             {summary.fixture_count}")
    lines.append(f"  Quality (F2):         {summary.mean_quality:.3f}")
    lines.append(f"  Positive Quality:     {summary.positive_quality:.3f}")
    lines.append(f"  Positive Recall@k:    {summary.positive_recall:.3f}")
    lines.append(f"  Mean Precision@k:     {summary.mean_precision:.3f}")
    lines.append(f"  Mean Recall@k:        {summary.mean_recall:.3f}")
    lines.append(f"  Mean MRR:             {summary.mean_mrr:.3f}")
    lines.append(f"  Mean nDCG@k:          {summary.mean_ndcg:.3f}")
    lines.append(f"  Mean Anti-P:          {summary.mean_anti_precision:.3f}")
    lines.append(f"  Mean Noise Ratio:     {summary.mean_noise_ratio:.3f}")
    lines.append(f"  Mean Context Waste:   {summary.mean_context_waste_ratio:.3f}")
    lines.append(f"  Neg Silence Rate:     {summary.negative_silence_rate:.3f}")
    lines.append(f"  Mean Retrieved Count: {summary.mean_retrieved_count:.1f}")
    lines.append(f"  Latency p50:          {summary.latency_p50_ms:.1f} ms")
    lines.append(f"  Latency p95:          {summary.latency_p95_ms:.1f} ms")
    lines.append(f"  Latency p99:          {summary.latency_p99_ms:.1f} ms")
    lines.append("")

    return "\n".join(lines)


_EVENT_HEADER_FMT = (
    "{:<20s} {:>5s} {:>6s} {:>8s} {:>6s} {:>6s}"
)
_EVENT_ROW_FMT = (
    "{:<20s} {:>5d} {:>6.3f} {:>8.3f} {:>6.3f} {:>6.3f}"
)


def format_per_event_report(
    per_event: list[PerEventMetrics],
) -> str:
    """Format per-event metrics into a human-readable text table."""
    lines: list[str] = []
    lines.append("Per-Event Breakdown")
    header = _EVENT_HEADER_FMT.format(
        "Event", "N", "F2", "PosRecal", "Noise", "NegSil",
    )
    lines.append(header)
    lines.append("-" * 58)
    for em in per_event:
        row = _EVENT_ROW_FMT.format(
            em.event[:20],
            em.fixture_count,
            em.quality,
            em.positive_recall,
            em.noise_ratio,
            em.negative_silence,
        )
        lines.append(row)
    lines.append("")
    return "\n".join(lines)
