"""Evaluation harness: load golden fixtures, run retrieval, compute IR metrics."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from cuecard.models import AffinityIndex, Index, RankedResult, Rule

# Re-export metric functions so existing callers (tests, CLI, bench) keep working
from cuecard.eval.metrics import (  # noqa: F401
    _mean,
    _percentile,
    anti_precision,
    context_waste_ratio,
    mrr,
    ndcg_at_k,
    noise_ratio,
    precision_at_k,
    quality_score,
    recall_at_k,
)

# Re-export report functions so existing callers keep working
from cuecard.eval.report import (  # noqa: F401
    _compute_tier_summaries,
    evaluate_per_event,
    format_eval_report,
    format_per_event_report,
)
from cuecard.indexing.indexer import build_index
from cuecard.indexing.parser import parse_rules
from cuecard.models import DEFAULT_HOOK_EVENT as _DEFAULT_EVENT
from cuecard.models import ResolvedConfig

logger = logging.getLogger(__name__)

__all__ = [
    "EvalSummary",
    "Fixture",
    "FixtureResult",
    "PerEventMetrics",
    "TierSummary",
    "anti_precision",
    "context_waste_ratio",
    "evaluate_per_event",
    "format_eval_report",
    "format_per_event_report",
    "load_fixtures",
    "mrr",
    "ndcg_at_k",
    "noise_ratio",
    "precision_at_k",
    "quality_score",
    "recall_at_k",
    "run_eval",
]


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


def _make_eval_config(
    *,
    top_k: int = 7,
    threshold: float = 0.30,
    dedup_threshold: float = 0.95,
    query_max_length: int = 500,
    llm_candidates: int | None = None,
    sparse_enabled: bool = False,
) -> ResolvedConfig:
    """Build a ResolvedConfig for eval with dummy computed fields."""
    kwargs: dict[str, object] = {
        "source_paths": (),
        "global_source_paths": (),
        "project_source_paths": (),
        "global_cache_dir": "",
        "top_k": top_k,
        "threshold": threshold,
        "dedup_threshold": dedup_threshold,
        "query_max_length": query_max_length,
        "sparse_enabled": sparse_enabled,
    }
    if llm_candidates is not None:
        kwargs["llm_candidates"] = llm_candidates
    return ResolvedConfig(**kwargs)  # type: ignore[arg-type]


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
    llm_candidates: int | None = None,
    query_expansion_enabled: bool = False,
    query_expansion_endpoint: str = "http://localhost:8081/v1",
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
        fixtures = _sample_fixtures(fixtures, sample_ratio, seed)

    results: list[FixtureResult] = []

    # Index cache: corpus key -> (rules, index)
    _index_cache: dict[tuple[str, ...], tuple[tuple[Rule, ...], Index]] = {}

    effective_mode = mode if mode is not None else "embedding"
    eval_config = _make_eval_config(
        top_k=top_k,
        threshold=threshold,
        dedup_threshold=dedup_threshold,
        query_max_length=query_max_length,
        llm_candidates=llm_candidates,
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

    from cuecard.retrieval.pipeline import run_pipeline

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
            eval_config,
            embedding_model=model,  # type: ignore[arg-type]
            mode=effective_mode,
            event=event,
            affinity=affinity,
            query_expansion_enabled=query_expansion_enabled,
            query_expansion_endpoint=query_expansion_endpoint,
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
        _run_parallel(fixtures, _eval_one, results, effective_mode)
    else:
        _run_sequential(fixtures, _eval_one, results, effective_mode)

    return _build_summary(results)


def _sample_fixtures(
    fixtures: list[Fixture], sample_ratio: float, seed: int,
) -> list[Fixture]:
    """Stratified sampling: preserve tier distribution."""
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
    total = sum(len(v) for v in by_tier.values())
    logger.info(
        "Sampled %d/%d fixtures (ratio=%.2f)",
        len(sampled), total, sample_ratio,
    )
    return sampled


def _run_parallel(
    fixtures: list[Fixture],
    eval_one: Any,
    results: list[FixtureResult],
    mode_label: str,
) -> None:
    """Run eval in parallel using ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    max_workers = 5  # match llama-server -np 5

    try:
        from tqdm import tqdm
        pbar: Any = tqdm(
            total=len(fixtures), desc=mode_label, unit="fix",
        )
    except ImportError:
        pbar = None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(eval_one, fx): fx for fx in fixtures
        }
        for future in as_completed(futures):
            results.append(future.result())
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()


def _run_sequential(
    fixtures: list[Fixture],
    eval_one: Any,
    results: list[FixtureResult],
    mode_label: str,
) -> None:
    """Run eval sequentially with optional progress bar."""
    try:
        from tqdm import tqdm
        fixture_iter: Iterable[Fixture] = tqdm(
            fixtures, desc=mode_label, unit="fix",
        )
    except ImportError:
        fixture_iter = fixtures

    for fixture in fixture_iter:
        results.append(eval_one(fixture))


def _build_summary(results: list[FixtureResult]) -> EvalSummary:
    """Aggregate per-fixture results into an EvalSummary."""
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

    latencies = [r.latency_ms for r in results]
    negatives = [r for r in results if r.difficulty == "negative"]
    positives = [r for r in results if r.difficulty != "negative"]
    neg_silent = sum(1 for r in negatives if r.retrieved_count == 0)
    neg_silence_rate = (
        neg_silent / len(negatives) if negatives else 1.0
    )

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
