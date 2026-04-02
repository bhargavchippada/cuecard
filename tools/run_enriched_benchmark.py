#!/usr/bin/env python3
"""Run enriched retrieval benchmarks (expansions + BM25 + RRF).

Loads pre-built enriched rules.json corpora, builds indexes with
expansion embeddings + bm25_corpus, and evaluates against all fixture sets.

Usage:
    uv run python tools/run_enriched_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure cuecard is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuecard.eval import (  # noqa: E402
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
    recall_at_k,
)
from cuecard.indexer import build_index, load_rules_json  # noqa: E402
from cuecard.pipeline import run_pipeline  # noqa: E402
from cuecard.retriever import merge_indexes  # noqa: E402

MODEL = "BAAI/bge-small-en-v1.5"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
FIXTURES_DIR = EVAL_DIR / "fixtures"
CORPORA_DIR = EVAL_DIR / "corpora"
RESULTS_DIR = EVAL_DIR / "results"

_TIER_ORDER = ("easy", "medium", "hard", "negative")


@dataclass(frozen=True)
class BenchConfig:
    """Minimal config stub for pipeline calls."""

    top_k: int = 5
    threshold: float = 0.30
    dedup_threshold: float = 0.95
    query_max_length: int = 500
    sparse_enabled: bool = True
    fusion_k: int = 60


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    arr = np.array(sorted(values), dtype=np.float64)
    return float(np.percentile(arr, pct))


def _compute_tier_summaries(results: list[FixtureResult]) -> list[TierSummary]:
    by_tier: dict[str, list[FixtureResult]] = defaultdict(list)
    for r in results:
        by_tier[r.difficulty].append(r)

    summaries: list[TierSummary] = []
    for tier in _TIER_ORDER:
        tier_results = by_tier.get(tier, [])
        if not tier_results:
            continue
        n = len(tier_results)
        neg_silent = sum(1 for r in tier_results if r.retrieved_count == 0)
        summaries.append(
            TierSummary(
                tier=tier,
                count=n,
                mean_precision=_mean([r.precision_at_k for r in tier_results]),
                mean_recall=_mean([r.recall_at_k for r in tier_results]),
                mean_mrr=_mean([r.mrr for r in tier_results]),
                mean_noise_ratio=_mean([r.noise_ratio for r in tier_results]),
                mean_context_waste_ratio=_mean(
                    [r.context_waste_ratio for r in tier_results]
                ),
                silence_rate=neg_silent / n,
                mean_retrieved_count=_mean(
                    [float(r.retrieved_count) for r in tier_results]
                ),
            ),
        )
    return summaries


def evaluate_fixtures(
    fixtures: list[Fixture],
    index_map: dict[str, object],
    config: BenchConfig,
    model: object,
) -> EvalSummary:
    """Run pipeline on each fixture using its corpus's enriched index."""
    results: list[FixtureResult] = []

    for fixture in fixtures:
        idx = index_map.get(fixture.corpus)
        if idx is None:
            print(f"  SKIP {fixture.id}: no index for corpus {fixture.corpus}")
            continue

        start = time.perf_counter()
        pipeline_result = run_pipeline(
            fixture.query,
            idx,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            embedding_model=model,  # type: ignore[arg-type]
            mode="embedding",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        retrieved_texts = [r.rule.text for r in pipeline_result.results]
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
            context_waste_ratio=context_waste_ratio(retrieved_texts, relevant),
            retrieved_count=len(retrieved_texts),
            latency_ms=elapsed_ms,
        )
        results.append(result)

    if not results:
        return EvalSummary(
            fixture_count=0, mean_precision=0.0, mean_recall=0.0,
            mean_mrr=0.0, mean_ndcg=0.0, mean_anti_precision=0.0,
            mean_noise_ratio=0.0, mean_context_waste_ratio=0.0,
            negative_silence_rate=0.0, mean_retrieved_count=0.0,
            latency_p50_ms=0.0, latency_p95_ms=0.0, latency_p99_ms=0.0,
            per_fixture=(), per_tier=(),
        )

    latencies = [r.latency_ms for r in results]
    negatives = [r for r in results if r.difficulty == "negative"]
    neg_silent = sum(1 for r in negatives if r.retrieved_count == 0)
    neg_silence_rate = neg_silent / len(negatives) if negatives else 1.0

    return EvalSummary(
        fixture_count=len(results),
        mean_precision=_mean([r.precision_at_k for r in results]),
        mean_recall=_mean([r.recall_at_k for r in results]),
        mean_mrr=_mean([r.mrr for r in results]),
        mean_ndcg=_mean([r.ndcg_at_k for r in results]),
        mean_anti_precision=_mean([r.anti_precision for r in results]),
        mean_noise_ratio=_mean([r.noise_ratio for r in results]),
        mean_context_waste_ratio=_mean([r.context_waste_ratio for r in results]),
        negative_silence_rate=neg_silence_rate,
        mean_retrieved_count=_mean([float(r.retrieved_count) for r in results]),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        latency_p99_ms=_percentile(latencies, 99),
        per_fixture=tuple(results),
        per_tier=tuple(_compute_tier_summaries(results)),
    )


def summary_to_dict(s: EvalSummary) -> dict:
    """Convert EvalSummary to JSON-serializable dict."""
    return {
        "fixture_count": s.fixture_count,
        "mean_precision": round(s.mean_precision, 4),
        "mean_recall": round(s.mean_recall, 4),
        "mean_mrr": round(s.mean_mrr, 4),
        "mean_ndcg": round(s.mean_ndcg, 4),
        "mean_anti_precision": round(s.mean_anti_precision, 4),
        "mean_noise_ratio": round(s.mean_noise_ratio, 4),
        "mean_context_waste_ratio": round(s.mean_context_waste_ratio, 4),
        "negative_silence_rate": round(s.negative_silence_rate, 4),
        "mean_retrieved_count": round(s.mean_retrieved_count, 2),
        "latency_p50_ms": round(s.latency_p50_ms, 1),
        "latency_p95_ms": round(s.latency_p95_ms, 1),
        "latency_p99_ms": round(s.latency_p99_ms, 1),
        "per_tier": [
            {
                "tier": t.tier,
                "count": t.count,
                "mean_precision": round(t.mean_precision, 4),
                "mean_recall": round(t.mean_recall, 4),
                "mean_mrr": round(t.mean_mrr, 4),
                "mean_noise_ratio": round(t.mean_noise_ratio, 4),
                "mean_context_waste_ratio": round(t.mean_context_waste_ratio, 4),
                "silence_rate": round(t.silence_rate, 4),
                "mean_retrieved_count": round(t.mean_retrieved_count, 2),
            }
            for t in s.per_tier
        ],
    }


def main() -> None:
    print("=" * 80)
    print("Enriched Retrieval Benchmark (expansions + BM25 + RRF)")
    print(f"Model: {MODEL}")
    print(f"Mode: embedding (dense + sparse, no LLM reranker)")
    print("=" * 80)

    # Load embedding model
    print("\nLoading embedding model...")
    from fastembed import TextEmbedding

    t0 = time.time()
    model = TextEmbedding(model_name=MODEL)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Load enriched rules from JSON
    print("\nLoading enriched rules...")
    basic_rules = load_rules_json(str(CORPORA_DIR / "enriched_basic"))
    workflow_rules = load_rules_json(str(CORPORA_DIR / "enriched_workflow"))

    if basic_rules is None:
        print("ERROR: Failed to load enriched_basic/rules.json")
        sys.exit(1)
    if workflow_rules is None:
        print("ERROR: Failed to load enriched_workflow/rules.json")
        sys.exit(1)

    basic_expansions = sum(len(r.expansions) for r in basic_rules)
    workflow_expansions = sum(len(r.expansions) for r in workflow_rules)
    print(f"  Basic: {len(basic_rules)} rules, {basic_expansions} expansions")
    print(f"  Workflow: {len(workflow_rules)} rules, {workflow_expansions} expansions")

    # Build enriched indexes
    print("\nBuilding enriched indexes...")
    t0 = time.time()
    basic_idx = build_index(tuple(basic_rules), {}, MODEL, model=model)
    t_basic = time.time() - t0
    print(
        f"  Basic: {basic_idx.size} rules, "
        f"{basic_idx.embeddings.shape[0]} embedding rows, "
        f"bm25={basic_idx.bm25_corpus is not None} "
        f"({t_basic:.1f}s)"
    )

    t0 = time.time()
    workflow_idx = build_index(tuple(workflow_rules), {}, MODEL, model=model)
    t_workflow = time.time() - t0
    print(
        f"  Workflow: {workflow_idx.size} rules, "
        f"{workflow_idx.embeddings.shape[0]} embedding rows, "
        f"bm25={workflow_idx.bm25_corpus is not None} "
        f"({t_workflow:.1f}s)"
    )

    t0 = time.time()
    merged_idx = merge_indexes(basic_idx, workflow_idx)
    t_merge = time.time() - t0
    print(
        f"  Merged: {merged_idx.size} rules, "
        f"{merged_idx.embeddings.shape[0]} embedding rows "
        f"({t_merge:.1f}s)"
    )

    config = BenchConfig()

    # Index maps per fixture set
    index_map_basic = {"rules_basic.txt": basic_idx}
    index_map_workflow = {"rules_workflow.txt": workflow_idx}
    index_map_merged = {
        "rules_basic.txt": merged_idx,
        "rules_workflow.txt": merged_idx,
    }

    runs = [
        ("basic", "basic.json", index_map_basic),
        ("workflow", "workflow.json", index_map_workflow),
        ("mined-sessions", "mined-sessions.json", index_map_basic),
        ("mined-sessions-v2", "mined-sessions-v2.json", index_map_merged),
        ("combined", "combined.json", index_map_merged),
    ]

    all_results: dict[str, dict] = {}

    for name, fixture_file, idx_map in runs:
        fixture_path = FIXTURES_DIR / fixture_file
        if not fixture_path.exists():
            print(f"\n  SKIP {name}: {fixture_path} not found")
            continue

        fixtures = load_fixtures(str(fixture_path))
        print(f"\n{'='*60}")
        print(f"  {name}: {len(fixtures)} fixtures")
        print(f"{'='*60}")

        summary = evaluate_fixtures(fixtures, idx_map, config, model)
        all_results[name] = summary_to_dict(summary)

        print(format_eval_report(summary))

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "enriched-embedding-2026-04-02.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Compare with raw baseline if available
    baseline_path = RESULTS_DIR / "raw-baseline-2026-04-02.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print("\n" + "=" * 80)
        print("COMPARISON: Enriched vs Raw Baseline")
        print("=" * 80)
        print(
            f"{'Dataset':<20s} {'Metric':<25s} "
            f"{'Raw':>8s} {'Enriched':>8s} {'Delta':>8s}"
        )
        print("-" * 72)
        for dset in all_results:
            if dset not in baseline:
                continue
            b = baseline[dset]
            e = all_results[dset]
            for metric in (
                "mean_recall", "mean_precision", "mean_noise_ratio",
                "negative_silence_rate", "mean_retrieved_count",
            ):
                bv = b.get(metric, 0.0)
                ev = e.get(metric, 0.0)
                delta = ev - bv
                sign = "+" if delta >= 0 else ""
                print(
                    f"{dset:<20s} {metric:<25s} "
                    f"{bv:>8.3f} {ev:>8.3f} {sign}{delta:>7.3f}"
                )
            print()
    else:
        print(f"\nNo raw baseline at {baseline_path} — skipping comparison")

    # Compact summary table
    print("\n" + "=" * 80)
    print("ENRICHED EMBEDDING SUMMARY (bge-small + expansions + BM25 + RRF)")
    print("=" * 80)
    print(
        f"{'Dataset':<20s} {'N':>4s} {'Recall':>7s} {'Prec':>7s} "
        f"{'Noise':>7s} {'NegSil':>7s} {'p50ms':>7s}"
    )
    print("-" * 60)
    for name, data in all_results.items():
        print(
            f"{name:<20s} {data['fixture_count']:>4d} "
            f"{data['mean_recall']:>7.3f} {data['mean_precision']:>7.3f} "
            f"{data['mean_noise_ratio']:>7.3f} "
            f"{data['negative_silence_rate']:>7.3f} "
            f"{data['latency_p50_ms']:>7.1f}"
        )


if __name__ == "__main__":
    main()
