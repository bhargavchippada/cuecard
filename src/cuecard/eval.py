"""Evaluation harness: load golden fixtures, run retrieval, compute IR metrics."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from cuecard.models import AffinityIndex, Index, RankedResult, Rule

import numpy as np

from cuecard.indexer import build_index
from cuecard.parser import parse_rules

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


_DEFAULT_EVENT = "PreToolUse"


@dataclass(frozen=True)
class Fixture:
    """A single golden test case for retrieval evaluation."""

    id: str
    query: str
    corpus: str
    should_match: tuple[str, ...]
    should_not_match: tuple[str, ...]
    difficulty: str
    event: str = _DEFAULT_EVENT


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
    quality_score: float


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
    mean_quality: float


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
    mean_quality: float
    positive_recall: float
    positive_quality: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    per_fixture: tuple[FixtureResult, ...]
    per_tier: tuple[TierSummary, ...]


@dataclass(frozen=True)
class PerEventMetrics:
    """Quality metrics broken down by event type."""

    event: str
    fixture_count: int
    quality: float
    positive_recall: float
    noise_ratio: float
    negative_silence: float


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


def quality_score(
    retrieved: list[str],
    relevant: set[str],
    is_negative: bool,
) -> float:
    """Single quality score per fixture: F2 with correct-abstention convention.

    Uses F-beta with beta=2 (recall-weighted), extended with the
    empty-set convention for retrieval systems with negative queries.

    For positive fixtures (has should_match):
        F2 = 5 * P * R / (4P + R), where:
          P = precision (hits / retrieved)
          R = recall (hits / relevant)
        Returns 0.0 if both P and R are 0.

    For negative fixtures (no should_match):
        1.0 if silent (correct abstention), 0.0 otherwise.
        This is the standard empty-set convention — correct silence
        is a perfect retrieval outcome.

    The F2 formulation naturally penalizes noise (low precision) while
    weighting recall 4x higher than precision. It handles partial
    matches and noisy retrieval in a single principled number.
    """
    if is_negative:
        return 1.0 if not retrieved else 0.0

    if not relevant:
        return 1.0 if not retrieved else 0.0

    p = precision_at_k(retrieved, relevant)
    r = recall_at_k(retrieved, relevant)

    if p == 0.0 and r == 0.0:
        return 0.0

    # F2: beta=2 → (1 + 4) * P * R / (4 * P + R)
    return 5.0 * p * r / (4.0 * p + r)


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

        event = entry.get("event", _DEFAULT_EVENT)
        if not isinstance(event, str):
            msg = f"Fixture {entry['id']!r}: event must be a string"
            raise ValueError(msg)

        fixtures.append(
            Fixture(
                id=entry["id"],
                query=entry["query"],
                corpus=entry["corpus"],
                should_match=tuple(should_match),
                should_not_match=tuple(should_not_match),
                difficulty=difficulty,
                event=event,
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

    top_k: int = 7
    threshold: float = 0.30
    dedup_threshold: float = 0.95
    query_max_length: int = 500
    sparse_enabled: bool = True
    fusion_k: int = 10
    llm_candidates: int = 12


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
    query_max_length: int = 500,
    corpus_override: tuple[str, ...] | None = None,
    sample_ratio: float = 1.0,
    seed: int = 42,
    affinity: AffinityIndex | None = None,
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
        top_k: Number of results to retrieve per query (embedding mode only).
        threshold: Minimum cosine similarity threshold (embedding mode only).
        dedup_threshold: Near-duplicate dedup threshold (embedding mode only).
        mode: Pipeline mode. When not None or "embedding", uses run_pipeline()
            which applies its own recall-widening parameters (top_k=20,
            threshold=0.10) for the embedding stage, ignoring the top_k and
            threshold args above. This is intentional: re-ranking stages
            need a wider candidate pool to be effective.
        query_max_length: Maximum character length for queries before truncation.
        corpus_override: If set, use these corpus file paths for ALL fixtures
            instead of each fixture's corpus field. Builds a single unified
            index. Useful for testing cross-domain noise.
        sample_ratio: Fraction of fixtures to evaluate (0.0-1.0). Uses
            stratified sampling to preserve tier distribution. Default 1.0.
        seed: Random seed for reproducible sampling.

    Returns:
        EvalSummary with per-fixture and aggregate metrics.
    """
    # Stratified sampling: preserve tier distribution
    if sample_ratio < 1.0:
        import random

        rng = random.Random(seed)
        by_tier: dict[str, list[Fixture]] = {}
        for fx in fixtures:
            by_tier.setdefault(fx.difficulty, []).append(fx)
        sampled: list[Fixture] = []
        for tier_fixtures in by_tier.values():
            n = max(1, int(len(tier_fixtures) * sample_ratio))
            sampled.extend(rng.sample(tier_fixtures, min(n, len(tier_fixtures))))
        rng.shuffle(sampled)
        fixtures = sampled
        total = sum(len(v) for v in by_tier.values())
        logger.info(
            "Sampled %d/%d fixtures (ratio=%.2f)",
            len(sampled), total, sample_ratio,
        )

    results: list[FixtureResult] = []

    # Index cache: corpus key -> (rules, index)
    _index_cache: dict[tuple[str, ...], tuple[tuple[Rule, ...], Index]] = {}

    effective_mode = mode if mode is not None else "embedding"
    eval_config = _EvalConfig(
        top_k=top_k,
        threshold=threshold,
        dedup_threshold=dedup_threshold,
        query_max_length=query_max_length,
    )

    # Pre-build all indexes (not parallelizable — depends on corpus_key)
    for fixture in fixtures:
        if corpus_override is not None:
            corpus_key = corpus_override
        else:
            corpus_key = (str(Path(corpus_dir) / fixture.corpus),)

        if corpus_key not in _index_cache:
            rules = tuple(parse_rules(corpus_key))
            sources: dict[str, object] = {}
            idx = build_index(
                rules,
                sources,  # type: ignore[arg-type]
                model_name,
                model=model,  # type: ignore[arg-type]
            )
            _index_cache[corpus_key] = (rules, idx)

    from cuecard.pipeline import run_pipeline

    def _eval_one(fixture: Fixture) -> FixtureResult:
        if corpus_override is not None:
            ckey = corpus_override
        else:
            ckey = (str(Path(corpus_dir) / fixture.corpus),)
        _cached_rules, index = _index_cache[ckey]
        event = fixture.event if fixture.event else ""

        start = time.perf_counter()
        pipeline_result = run_pipeline(
            fixture.query,
            index,
            eval_config,  # type: ignore[arg-type]
            embedding_model=model,  # type: ignore[arg-type]
            mode=effective_mode,
            event=event,
            affinity=affinity,
        )
        ranked: Sequence[RankedResult] = pipeline_result.results
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        retrieved_texts = [r.rule.text for r in ranked]
        relevant = set(fixture.should_match)
        anti_rel = set(fixture.should_not_match)
        is_negative = fixture.difficulty == "negative"

        return FixtureResult(
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
            quality_score=quality_score(
                retrieved_texts, relevant, is_negative,
            ),
        )

    # Parallel execution for LLM modes (bottleneck is LLM call ~1-2s each)
    use_parallel = effective_mode != "embedding" and len(fixtures) > 1

    if use_parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = 5  # match llama-server -np 5

        try:
            from tqdm import tqdm
            pbar: Any = tqdm(
                total=len(fixtures), desc=effective_mode, unit="fix",
            )
        except ImportError:
            pbar = None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_eval_one, fx): fx for fx in fixtures
            }
            for future in as_completed(futures):
                results.append(future.result())
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()
    else:
        try:
            from tqdm import tqdm
            fixture_iter: Iterable[Fixture] = tqdm(
                fixtures, desc=effective_mode, unit="fix",
            )
        except ImportError:
            fixture_iter = fixtures

        for fixture in fixture_iter:
            results.append(_eval_one(fixture))

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
            mean_quality=0.0,
            positive_recall=0.0,
            positive_quality=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            per_fixture=(),
            per_tier=(),
        )

    negatives = [r for r in results if r.difficulty == "negative"]
    positives = [r for r in results if r.difficulty != "negative"]
    neg_silent = sum(1 for r in negatives if r.retrieved_count == 0)
    neg_silence_rate = (
        neg_silent / len(negatives) if negatives else 1.0
    )

    # positive_recall / positive_quality: averaged only over positive
    # fixtures (negatives have their own metric — negative_silence_rate)
    pos_recall = (
        _mean(r.recall_at_k for r in positives)
        if positives
        else 0.0
    )
    pos_quality = (
        _mean(r.quality_score for r in positives)
        if positives
        else 0.0
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
        mean_quality=_mean(r.quality_score for r in results),
        positive_recall=pos_recall,
        positive_quality=pos_quality,
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
    "PreToolUse", "PostToolUse", "UserPromptSubmit",
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
    from collections import defaultdict

    # Build fixture lookup: id -> Fixture
    fixture_by_id: dict[str, Fixture] = {f.id: f for f in fixtures}

    # Group results by event
    by_event: dict[str, list[tuple[FixtureResult, Fixture]]] = defaultdict(list)
    for r in results:
        fx = fixture_by_id.get(r.fixture_id)
        event = fx.event if fx is not None else _DEFAULT_EVENT
        by_event[event].append((r, fx or Fixture(
            id=r.fixture_id, query=r.query, corpus="",
            should_match=(), should_not_match=(),
            difficulty=r.difficulty, event=event,
        )))

    metrics: list[PerEventMetrics] = []
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
