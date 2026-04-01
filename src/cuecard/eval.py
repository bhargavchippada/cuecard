"""Evaluation harness: load golden fixtures, run retrieval, compute IR metrics."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

import numpy as np

from cuecard.indexer import build_index
from cuecard.parser import parse_rules
from cuecard.retriever import retrieve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fixture:
    """A single golden test case for retrieval evaluation."""

    id: str
    query: str
    corpus: str
    should_match: tuple[str, ...]
    should_not_match: tuple[str, ...]
    difficulty: str


@dataclass(frozen=True)
class FixtureResult:
    """Per-fixture evaluation result."""

    fixture_id: str
    query: str
    difficulty: str
    retrieved: tuple[str, ...]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    anti_precision: float
    noise_ratio: float
    context_waste_ratio: float
    retrieved_count: int
    latency_ms: float


@dataclass(frozen=True)
class TierSummary:
    """Aggregate metrics for a single difficulty tier."""

    tier: str
    count: int
    mean_precision: float
    mean_recall: float
    mean_mrr: float
    mean_noise_ratio: float
    mean_context_waste_ratio: float
    silence_rate: float
    mean_retrieved_count: float


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate evaluation results across all fixtures."""

    fixture_count: int
    mean_precision: float
    mean_recall: float
    mean_mrr: float
    mean_ndcg: float
    mean_anti_precision: float
    mean_noise_ratio: float
    mean_context_waste_ratio: float
    negative_silence_rate: float
    mean_retrieved_count: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    per_fixture: tuple[FixtureResult, ...]
    per_tier: tuple[TierSummary, ...]


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Fraction of retrieved items that are relevant.

    Returns 0.0 if retrieved is empty.
    """
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(retrieved)


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Fraction of relevant items that appear in retrieved.

    Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant result.

    Returns 0.0 if no relevant result is found.
    """
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Normalized Discounted Cumulative Gain with binary relevance.

    Uses DCG formula: sum(rel_i / log2(i+1)) for i=1..k.
    Returns 0.0 if no relevant items exist.
    """
    if not relevant or not retrieved:
        return 0.0

    # DCG of the actual ranking
    dcg = 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: all relevant items ranked first
    n_relevant_in_k = min(len(relevant), len(retrieved))
    idcg = 0.0
    for i in range(1, n_relevant_in_k + 1):
        idcg += 1.0 / math.log2(i + 1)

    # idcg > 0 guaranteed: we early-return when relevant or retrieved is empty
    return dcg / idcg


def anti_precision(retrieved: list[str], anti_relevant: set[str]) -> float:
    """Fraction of retrieved items that are in the anti-relevant set.

    Desired value is always 0.0. Returns 0.0 if retrieved is empty.
    """
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in anti_relevant)
    return hits / len(retrieved)


def noise_ratio(retrieved: list[str], relevant: set[str]) -> float:
    """Fraction of retrieved items that are NOT relevant.

    Returns 0.0 if retrieved is empty.
    """
    if not retrieved:
        return 0.0
    irrelevant = sum(1 for r in retrieved if r not in relevant)
    return irrelevant / len(retrieved)


def context_waste_ratio(retrieved: list[str], relevant: set[str]) -> float:
    """Char-weighted waste: fraction of injected chars that are irrelevant.

    Returns 0.0 if retrieved is empty or total chars is 0.
    """
    if not retrieved:
        return 0.0
    total_chars = sum(len(r) for r in retrieved)
    if total_chars == 0:
        return 0.0
    waste_chars = sum(len(r) for r in retrieved if r not in relevant)
    return waste_chars / total_chars


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = frozenset(
    {"id", "query", "corpus", "should_match", "should_not_match"},
)


def load_fixtures(path: str) -> list[Fixture]:
    """Load golden test fixtures from a JSON file.

    Args:
        path: Absolute path to the fixtures JSON file.

    Returns:
        List of Fixture objects.

    Raises:
        ValueError: On missing or invalid fields.
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        msg = f"Fixtures file must contain a JSON array, got {type(data).__name__}"
        raise ValueError(msg)

    fixtures: list[Fixture] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            msg = f"Fixture at index {i} must be an object, got {type(entry).__name__}"
            raise ValueError(msg)

        missing = _REQUIRED_FIELDS - entry.keys()
        if missing:
            msg = f"Fixture at index {i} missing fields: {sorted(missing)}"
            raise ValueError(msg)

        should_match = entry["should_match"]
        should_not_match = entry["should_not_match"]

        if not isinstance(should_match, list):
            msg = f"Fixture {entry['id']!r}: should_match must be a list"
            raise ValueError(msg)

        if not isinstance(should_not_match, list):
            msg = f"Fixture {entry['id']!r}: should_not_match must be a list"
            raise ValueError(msg)

        difficulty = entry.get("difficulty", "unknown")
        if not isinstance(difficulty, str):
            msg = f"Fixture {entry['id']!r}: difficulty must be a string"
            raise ValueError(msg)

        fixtures.append(
            Fixture(
                id=entry["id"],
                query=entry["query"],
                corpus=entry["corpus"],
                should_match=tuple(should_match),
                should_not_match=tuple(should_not_match),
                difficulty=difficulty,
            ),
        )

    return fixtures


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile from a non-empty list of values."""
    arr = np.array(sorted(values), dtype=np.float64)
    return float(np.percentile(arr, pct))


@dataclass(frozen=True)
class _EvalConfig:
    """Minimal config stub for pipeline calls from eval harness."""

    top_k: int = 5
    threshold: float = 0.30
    dedup_threshold: float = 0.95
    query_max_length: int = 500


def run_eval(
    fixtures: list[Fixture],
    corpus_dir: str,
    model_name: str,
    *,
    model: object | None = None,
    top_k: int = 5,
    threshold: float = 0.30,
    dedup_threshold: float = 0.95,
    mode: str | None = None,
) -> EvalSummary:
    """Run evaluation across all fixtures and aggregate metrics.

    For each fixture:
      1. Parse the corpus file (corpus_dir / fixture.corpus)
      2. Build an index from the corpus rules
      3. Run retrieve() with the fixture's query
      4. Compute all metrics
      5. Measure latency (time the retrieve call)

    Args:
        fixtures: List of golden test fixtures.
        corpus_dir: Directory containing corpus files.
        model_name: Embedding model identifier.
        model: Embedding model instance (fastembed-compatible).
        top_k: Number of results to retrieve per query.
        threshold: Minimum cosine similarity threshold.
        dedup_threshold: Near-duplicate dedup threshold.

    Returns:
        EvalSummary with per-fixture and aggregate metrics.
    """
    results: list[FixtureResult] = []

    for fixture in fixtures:
        corpus_path = str(Path(corpus_dir) / fixture.corpus)
        rules = tuple(parse_rules((corpus_path,)))

        sources: dict[str, object] = {}
        index = build_index(
            rules,
            sources,  # type: ignore[arg-type]
            model_name,
            model=model,  # type: ignore[arg-type]
        )

        start = time.perf_counter()
        if mode is not None and mode != "embedding":
            from cuecard.pipeline import run_pipeline

            pipeline_result = run_pipeline(
                fixture.query,
                index,
                _EvalConfig(
                    top_k=top_k,
                    threshold=threshold,
                    dedup_threshold=dedup_threshold,
                    query_max_length=500,
                ),  # type: ignore[arg-type]
                embedding_model=model,  # type: ignore[arg-type]
                mode=mode,
            )
            ranked = pipeline_result.results
        else:
            ranked = retrieve(
                index,
                fixture.query,
                top_k=top_k,
                threshold=threshold,
                dedup_threshold=dedup_threshold,
                model=model,  # type: ignore[arg-type]
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        retrieved_texts = [r.rule.text for r in ranked]
        relevant = set(fixture.should_match)
        anti_rel = set(fixture.should_not_match)

        result = FixtureResult(
            fixture_id=fixture.id,
            query=fixture.query,
            difficulty=fixture.difficulty,
            retrieved=tuple(retrieved_texts),
            precision_at_k=precision_at_k(retrieved_texts, relevant),
            recall_at_k=recall_at_k(retrieved_texts, relevant),
            mrr=mrr(retrieved_texts, relevant),
            ndcg_at_k=ndcg_at_k(retrieved_texts, relevant),
            anti_precision=anti_precision(retrieved_texts, anti_rel),
            noise_ratio=noise_ratio(retrieved_texts, relevant),
            context_waste_ratio=context_waste_ratio(
                retrieved_texts, relevant,
            ),
            retrieved_count=len(retrieved_texts),
            latency_ms=elapsed_ms,
        )
        results.append(result)

    latencies = [r.latency_ms for r in results]
    fixture_count = len(results)

    if fixture_count == 0:
        return EvalSummary(
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
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            per_fixture=(),
            per_tier=(),
        )

    negatives = [r for r in results if r.difficulty == "negative"]
    neg_silent = sum(1 for r in negatives if r.retrieved_count == 0)
    neg_silence_rate = (
        neg_silent / len(negatives) if negatives else 1.0
    )

    tier_summaries = _compute_tier_summaries(results)

    return EvalSummary(
        fixture_count=fixture_count,
        mean_precision=_mean(r.precision_at_k for r in results),
        mean_recall=_mean(r.recall_at_k for r in results),
        mean_mrr=_mean(r.mrr for r in results),
        mean_ndcg=_mean(r.ndcg_at_k for r in results),
        mean_anti_precision=_mean(r.anti_precision for r in results),
        mean_noise_ratio=_mean(r.noise_ratio for r in results),
        mean_context_waste_ratio=_mean(
            r.context_waste_ratio for r in results
        ),
        negative_silence_rate=neg_silence_rate,
        mean_retrieved_count=_mean(
            float(r.retrieved_count) for r in results
        ),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        latency_p99_ms=_percentile(latencies, 99),
        per_fixture=tuple(results),
        per_tier=tuple(tier_summaries),
    )


def _mean(values: Iterable[float]) -> float:
    """Compute mean from an iterable. Returns 0.0 for empty input."""
    items = list(values)
    return sum(items) / len(items) if items else 0.0


_TIER_ORDER = ("easy", "medium", "hard", "negative")


def _compute_tier_summaries(
    results: list[FixtureResult],
) -> list[TierSummary]:
    """Group results by difficulty tier and compute per-tier metrics."""
    from collections import defaultdict

    by_tier: dict[str, list[FixtureResult]] = defaultdict(list)
    for r in results:
        by_tier[r.difficulty].append(r)

    summaries: list[TierSummary] = []
    for tier in _TIER_ORDER:
        tier_results = by_tier.get(tier, [])
        if not tier_results:
            continue
        n = len(tier_results)
        neg_silent = sum(
            1 for r in tier_results if r.retrieved_count == 0
        )
        summaries.append(
            TierSummary(
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
            ),
        )

    # Include any tiers not in _TIER_ORDER (e.g., "unknown")
    for tier in sorted(by_tier.keys()):
        if tier not in _TIER_ORDER:
            tier_results = by_tier[tier]
            n = len(tier_results)
            neg_silent = sum(
                1 for r in tier_results if r.retrieved_count == 0
            )
            summaries.append(
                TierSummary(
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
                ),
            )

    return summaries


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
    " {:>6s} {:>6s} {:>6s} {:>5s}"
)
_TIER_ROW_FMT = (
    "{:<10s} {:>5d} {:>6.3f} {:>6.3f} {:>6.3f}"
    " {:>6.3f} {:>6.3f} {:>6.3f} {:>5.1f}"
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
            "Noise", "Waste", "Silen", "AvgRt",
        )
        lines.append(tier_header)
        lines.append("-" * 70)
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
            )
            lines.append(row)
        lines.append("")

    # Aggregates
    lines.append("Aggregate Metrics")
    lines.append(f"  Fixtures:             {summary.fixture_count}")
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
