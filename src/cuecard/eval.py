"""Evaluation harness: load golden fixtures, run retrieval, compute IR metrics."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class FixtureResult:
    """Per-fixture evaluation result."""

    fixture_id: str
    query: str
    retrieved: tuple[str, ...]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    anti_precision: float
    latency_ms: float


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate evaluation results across all fixtures."""

    fixture_count: int
    mean_precision: float
    mean_recall: float
    mean_mrr: float
    mean_ndcg: float
    mean_anti_precision: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    per_fixture: tuple[FixtureResult, ...]


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

        fixtures.append(
            Fixture(
                id=entry["id"],
                query=entry["query"],
                corpus=entry["corpus"],
                should_match=tuple(should_match),
                should_not_match=tuple(should_not_match),
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


def run_eval(
    fixtures: list[Fixture],
    corpus_dir: str,
    model_name: str,
    *,
    model: object | None = None,
    top_k: int = 5,
    threshold: float = 0.35,
    dedup_threshold: float = 0.95,
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
            retrieved=tuple(retrieved_texts),
            precision_at_k=precision_at_k(retrieved_texts, relevant),
            recall_at_k=recall_at_k(retrieved_texts, relevant),
            mrr=mrr(retrieved_texts, relevant),
            ndcg_at_k=ndcg_at_k(retrieved_texts, relevant),
            anti_precision=anti_precision(retrieved_texts, anti_rel),
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
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            per_fixture=(),
        )

    return EvalSummary(
        fixture_count=fixture_count,
        mean_precision=sum(r.precision_at_k for r in results) / fixture_count,
        mean_recall=sum(r.recall_at_k for r in results) / fixture_count,
        mean_mrr=sum(r.mrr for r in results) / fixture_count,
        mean_ndcg=sum(r.ndcg_at_k for r in results) / fixture_count,
        mean_anti_precision=sum(r.anti_precision for r in results) / fixture_count,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        latency_p99_ms=_percentile(latencies, 99),
        per_fixture=tuple(results),
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_HEADER_FMT = "{:<25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>12s}"
_ROW_FMT = "{:<25s} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>12.1f}"


def format_eval_report(summary: EvalSummary) -> str:
    """Format an EvalSummary into a human-readable text report.

    Includes a per-fixture table and aggregate metrics.
    """
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("Evaluation Report")
    lines.append("=" * 90)
    lines.append("")

    # Per-fixture table
    header = _HEADER_FMT.format(
        "Fixture", "P@k", "R@k", "MRR", "nDCG@k", "AntiP", "Latency(ms)",
    )
    lines.append(header)
    lines.append("-" * 90)

    for fr in summary.per_fixture:
        fixture_id = fr.fixture_id[:25]
        row = _ROW_FMT.format(
            fixture_id,
            fr.precision_at_k,
            fr.recall_at_k,
            fr.mrr,
            fr.ndcg_at_k,
            fr.anti_precision,
            fr.latency_ms,
        )
        lines.append(row)

    lines.append("-" * 90)
    lines.append("")

    # Aggregates
    lines.append("Aggregate Metrics")
    lines.append(f"  Fixtures:          {summary.fixture_count}")
    lines.append(f"  Mean Precision@k:  {summary.mean_precision:.3f}")
    lines.append(f"  Mean Recall@k:     {summary.mean_recall:.3f}")
    lines.append(f"  Mean MRR:          {summary.mean_mrr:.3f}")
    lines.append(f"  Mean nDCG@k:       {summary.mean_ndcg:.3f}")
    lines.append(f"  Mean Anti-P:       {summary.mean_anti_precision:.3f}")
    lines.append(f"  Latency p50:       {summary.latency_p50_ms:.1f} ms")
    lines.append(f"  Latency p95:       {summary.latency_p95_ms:.1f} ms")
    lines.append(f"  Latency p99:       {summary.latency_p99_ms:.1f} ms")
    lines.append("")

    return "\n".join(lines)
