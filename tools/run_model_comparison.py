#!/usr/bin/env python3
"""Compare embedding models on enriched indexes.

Evaluates multiple fastembed models against basic (354) and workflow (84)
fixture sets using the enriched retrieval pipeline (dense + BM25 + RRF).

Usage:
    uv run python tools/run_model_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuecard.eval import (  # noqa: E402
    EvalSummary,
    Fixture,
    FixtureResult,
    TierSummary,
    format_eval_report,
    load_fixtures,
    mrr,
    ndcg_at_k,
    noise_ratio,
    precision_at_k,
    recall_at_k,
    anti_precision,
    context_waste_ratio,
)
from cuecard.indexer import build_index, load_rules_json  # noqa: E402
from cuecard.pipeline import run_pipeline  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
FIXTURES_DIR = EVAL_DIR / "fixtures"
CORPORA_DIR = EVAL_DIR / "corpora"
RESULTS_DIR = EVAL_DIR / "results"

_TIER_ORDER = ("easy", "medium", "hard", "negative")

MODELS = [
    "jinaai/jina-embeddings-v2-base-code",
    "mixedbread-ai/mxbai-embed-large-v1",
    "nomic-ai/nomic-embed-text-v1.5",
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-small-en-v1.5",
    "jinaai/jina-embeddings-v3",
]

# Models checked but NOT in fastembed:
# - BAAI/bge-m3 (not supported)
# - Xenova/CodeRankEmbed (not supported)
# - Qwen3-Embedding-0.6B (not supported)
# - embeddinggemma-300m (not supported)


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
    index: object,
    config: BenchConfig,
    model: object,
) -> EvalSummary:
    """Run pipeline on each fixture against a single index."""
    results: list[FixtureResult] = []

    for fixture in fixtures:
        start = time.perf_counter()
        pipeline_result = run_pipeline(
            fixture.query,
            index,  # type: ignore[arg-type]
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
    return {
        "fixture_count": s.fixture_count,
        "mean_precision": round(s.mean_precision, 4),
        "mean_recall": round(s.mean_recall, 4),
        "mean_mrr": round(s.mean_mrr, 4),
        "mean_ndcg": round(s.mean_ndcg, 4),
        "mean_noise_ratio": round(s.mean_noise_ratio, 4),
        "negative_silence_rate": round(s.negative_silence_rate, 4),
        "mean_retrieved_count": round(s.mean_retrieved_count, 2),
        "latency_p50_ms": round(s.latency_p50_ms, 1),
        "latency_p95_ms": round(s.latency_p95_ms, 1),
        "per_tier": [
            {
                "tier": t.tier,
                "count": t.count,
                "mean_recall": round(t.mean_recall, 4),
                "mean_noise_ratio": round(t.mean_noise_ratio, 4),
                "silence_rate": round(t.silence_rate, 4),
            }
            for t in s.per_tier
        ],
    }


def main() -> None:
    print("=" * 80)
    print("Embedding Model Comparison (enriched index: dense + BM25 + RRF)")
    print("=" * 80)

    from fastembed import TextEmbedding

    # Load enriched rules (shared across models — only embeddings differ)
    print("\nLoading enriched rules...")
    basic_rules = load_rules_json(str(CORPORA_DIR / "enriched_basic"))
    workflow_rules = load_rules_json(str(CORPORA_DIR / "enriched_workflow"))

    if basic_rules is None or workflow_rules is None:
        print("ERROR: Failed to load enriched corpora")
        sys.exit(1)

    basic_exp = sum(len(r.expansions) for r in basic_rules)
    workflow_exp = sum(len(r.expansions) for r in workflow_rules)
    print(f"  Basic: {len(basic_rules)} rules, {basic_exp} expansions")
    print(f"  Workflow: {len(workflow_rules)} rules, {workflow_exp} expansions")

    # Load fixtures
    basic_fixtures = load_fixtures(str(FIXTURES_DIR / "basic.json"))
    workflow_fixtures = load_fixtures(str(FIXTURES_DIR / "workflow.json"))
    print(f"  Basic fixtures: {len(basic_fixtures)}")
    print(f"  Workflow fixtures: {len(workflow_fixtures)}")

    config = BenchConfig()
    all_results: dict[str, dict] = {}

    for model_name in MODELS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")

        # Load model
        try:
            t0 = time.time()
            model = TextEmbedding(model_name=model_name)
            print(f"  Loaded in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"  FAILED to load: {exc}")
            all_results[model_name] = {"error": str(exc)}
            continue

        # Build indexes
        try:
            t0 = time.time()
            basic_idx = build_index(
                tuple(basic_rules), {}, model_name, model=model,
            )
            t_basic = time.time() - t0
            print(
                f"  Basic index: {basic_idx.embeddings.shape[0]} rows, "
                f"dim={basic_idx.dim} ({t_basic:.1f}s)"
            )

            t0 = time.time()
            workflow_idx = build_index(
                tuple(workflow_rules), {}, model_name, model=model,
            )
            t_workflow = time.time() - t0
            print(
                f"  Workflow index: {workflow_idx.embeddings.shape[0]} rows, "
                f"dim={workflow_idx.dim} ({t_workflow:.1f}s)"
            )
        except Exception as exc:
            print(f"  FAILED to build index: {exc}")
            all_results[model_name] = {"error": str(exc)}
            continue

        # Evaluate basic fixtures
        print(f"\n  Evaluating basic ({len(basic_fixtures)} fixtures)...")
        basic_summary = evaluate_fixtures(
            basic_fixtures, basic_idx, config, model,
        )
        print(format_eval_report(basic_summary))

        # Evaluate workflow fixtures
        print(f"\n  Evaluating workflow ({len(workflow_fixtures)} fixtures)...")
        workflow_summary = evaluate_fixtures(
            workflow_fixtures, workflow_idx, config, model,
        )
        print(format_eval_report(workflow_summary))

        all_results[model_name] = {
            "dim": basic_idx.dim,
            "basic": summary_to_dict(basic_summary),
            "workflow": summary_to_dict(workflow_summary),
        }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "model-comparison-2026-04-02.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print comparison table
    print("\n" + "=" * 100)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 100)
    header = (
        f"{'Model':<45s} {'Dim':>4s} "
        f"{'B-Recall':>8s} {'B-Noise':>8s} {'B-NegSil':>8s} "
        f"{'W-Recall':>8s} {'W-Noise':>8s} {'W-NegSil':>8s} "
        f"{'p50ms':>6s}"
    )
    print(header)
    print("-" * 100)

    for model_name, data in all_results.items():
        if "error" in data:
            print(f"{model_name:<45s}  ERROR: {data['error'][:40]}")
            continue

        b = data["basic"]
        w = data["workflow"]
        print(
            f"{model_name:<45s} {data['dim']:>4d} "
            f"{b['mean_recall']:>8.3f} {b['mean_noise_ratio']:>8.3f} "
            f"{b['negative_silence_rate']:>8.3f} "
            f"{w['mean_recall']:>8.3f} {w['mean_noise_ratio']:>8.3f} "
            f"{w['negative_silence_rate']:>8.3f} "
            f"{b['latency_p50_ms']:>6.1f}"
        )

    # Per-tier breakdown for each model
    print("\n" + "=" * 100)
    print("PER-TIER BREAKDOWN (Basic fixtures)")
    print("=" * 100)
    for model_name, data in all_results.items():
        if "error" in data:
            continue
        print(f"\n  {model_name}:")
        for t in data["basic"]["per_tier"]:
            print(
                f"    {t['tier']:<10s} n={t['count']:>3d}  "
                f"recall={t['mean_recall']:.3f}  "
                f"noise={t['mean_noise_ratio']:.3f}  "
                f"silence={t['silence_rate']:.3f}"
            )

    print("\n" + "=" * 100)
    print("PER-TIER BREAKDOWN (Workflow fixtures)")
    print("=" * 100)
    for model_name, data in all_results.items():
        if "error" in data:
            continue
        print(f"\n  {model_name}:")
        for t in data["workflow"]["per_tier"]:
            print(
                f"    {t['tier']:<10s} n={t['count']:>3d}  "
                f"recall={t['mean_recall']:.3f}  "
                f"noise={t['mean_noise_ratio']:.3f}  "
                f"silence={t['silence_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
