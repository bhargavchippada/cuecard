#!/usr/bin/env python3
"""Run enriched retrieval benchmarks with v3 expansions.

Evaluates v3 expansion prompt against v1, and compares bge-small vs jina-code
on v3 expansions. Also compares against raw baseline.

Usage:
    uv run python tools/run_enriched_v3_benchmark.py
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
FIXTURES_DIR = EVAL_DIR / "fixtures"
CORPORA_DIR = EVAL_DIR / "corpora"
RESULTS_DIR = EVAL_DIR / "results"

_TIER_ORDER = ("easy", "medium", "hard", "negative")

BGE_SMALL = "BAAI/bge-small-en-v1.5"
JINA_CODE = "jinaai/jina-embeddings-v2-base-code"


@dataclass(frozen=True)
class BenchConfig:
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


def print_comparison(
    title: str,
    label_a: str,
    label_b: str,
    results_a: dict[str, dict],
    results_b: dict[str, dict],
    datasets: list[str],
) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)
    print(
        f"{'Dataset':<20s} {'Metric':<25s} "
        f"{label_a:>10s} {label_b:>10s} {'Delta':>8s}"
    )
    print("-" * 76)
    for dset in datasets:
        a = results_a.get(dset)
        b = results_b.get(dset)
        if a is None or b is None:
            continue
        for metric in (
            "mean_recall", "mean_precision", "mean_noise_ratio",
            "negative_silence_rate", "mean_retrieved_count",
        ):
            av = a.get(metric, 0.0)
            bv = b.get(metric, 0.0)
            delta = bv - av
            sign = "+" if delta >= 0 else ""
            print(
                f"{dset:<20s} {metric:<25s} "
                f"{av:>10.4f} {bv:>10.4f} {sign}{delta:>7.4f}"
            )
        print()

    # Per-tier comparison
    print(f"\nPer-tier comparison ({label_a} → {label_b}):")
    print(
        f"{'Dataset':<16s} {'Tier':<10s} "
        f"{'Recall-A':>9s} {'Recall-B':>9s} {'Delta':>7s}  "
        f"{'Noise-A':>8s} {'Noise-B':>8s} {'Delta':>7s}"
    )
    print("-" * 85)
    for dset in datasets:
        a = results_a.get(dset)
        b = results_b.get(dset)
        if a is None or b is None:
            continue
        tiers_a = {t["tier"]: t for t in a.get("per_tier", [])}
        tiers_b = {t["tier"]: t for t in b.get("per_tier", [])}
        for tier in _TIER_ORDER:
            ta = tiers_a.get(tier)
            tb = tiers_b.get(tier)
            if ta is None or tb is None:
                continue
            r_delta = tb["mean_recall"] - ta["mean_recall"]
            n_delta = tb["mean_noise_ratio"] - ta["mean_noise_ratio"]
            r_sign = "+" if r_delta >= 0 else ""
            n_sign = "+" if n_delta >= 0 else ""
            print(
                f"{dset:<16s} {tier:<10s} "
                f"{ta['mean_recall']:>9.4f} {tb['mean_recall']:>9.4f} {r_sign}{r_delta:>6.4f}  "
                f"{ta['mean_noise_ratio']:>8.4f} {tb['mean_noise_ratio']:>8.4f} {n_sign}{n_delta:>6.4f}"
            )
        print()


def run_model_suite(
    model_name: str,
    basic_rules: tuple,
    workflow_rules: tuple,
    fixture_runs: list[tuple[str, str, str]],
    all_fixtures: dict[str, list[Fixture]],
) -> dict[str, dict]:
    """Build indexes and evaluate all fixture sets for a given model."""
    from fastembed import TextEmbedding

    print(f"\nLoading model: {model_name}")
    t0 = time.time()
    model = TextEmbedding(model_name=model_name)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print(f"\nBuilding indexes with {model_name}...")
    t0 = time.time()
    basic_idx = build_index(basic_rules, {}, model_name, model=model)
    print(
        f"  Basic: {basic_idx.size} rules, "
        f"{basic_idx.embeddings.shape[0]} rows, dim={basic_idx.dim} "
        f"({time.time() - t0:.1f}s)"
    )

    t0 = time.time()
    workflow_idx = build_index(workflow_rules, {}, model_name, model=model)
    print(
        f"  Workflow: {workflow_idx.size} rules, "
        f"{workflow_idx.embeddings.shape[0]} rows, dim={workflow_idx.dim} "
        f"({time.time() - t0:.1f}s)"
    )

    t0 = time.time()
    merged_idx = merge_indexes(basic_idx, workflow_idx)
    print(
        f"  Merged: {merged_idx.size} rules, "
        f"{merged_idx.embeddings.shape[0]} rows "
        f"({time.time() - t0:.1f}s)"
    )

    index_map_basic = {"rules_basic.txt": basic_idx}
    index_map_workflow = {"rules_workflow.txt": workflow_idx}
    index_map_merged = {
        "rules_basic.txt": merged_idx,
        "rules_workflow.txt": merged_idx,
    }

    idx_maps = {
        "basic": index_map_basic,
        "workflow": index_map_workflow,
        "merged": index_map_merged,
    }

    config = BenchConfig()
    results: dict[str, dict] = {}

    for name, fixture_file, idx_type in fixture_runs:
        fixtures = all_fixtures.get(fixture_file)
        if fixtures is None:
            print(f"\n  SKIP {name}: fixture file not loaded")
            continue

        idx_map = idx_maps[idx_type]
        print(f"\n{'='*60}")
        print(f"  {name}: {len(fixtures)} fixtures ({model_name})")
        print(f"{'='*60}")

        summary = evaluate_fixtures(fixtures, idx_map, config, model)
        results[name] = summary_to_dict(summary)
        print(format_eval_report(summary))

    return results


def main() -> None:
    print("=" * 80)
    print("Enriched v3 Benchmark (v3 expansions + BM25 + RRF)")
    print("=" * 80)

    # Load v3 enriched rules
    print("\nLoading v3 enriched rules...")
    basic_rules = load_rules_json(str(CORPORA_DIR / "enriched_basic_v3"))
    workflow_rules = load_rules_json(str(CORPORA_DIR / "enriched_workflow_v3"))

    if basic_rules is None:
        print("ERROR: Failed to load enriched_basic_v3/rules.json")
        sys.exit(1)
    if workflow_rules is None:
        print("ERROR: Failed to load enriched_workflow_v3/rules.json")
        sys.exit(1)

    basic_exp = sum(len(r.expansions) for r in basic_rules)
    workflow_exp = sum(len(r.expansions) for r in workflow_rules)
    print(f"  Basic v3: {len(basic_rules)} rules, {basic_exp} expansions")
    print(f"  Workflow v3: {len(workflow_rules)} rules, {workflow_exp} expansions")

    basic_rules_t = tuple(basic_rules)
    workflow_rules_t = tuple(workflow_rules)

    # Load all fixture files
    print("\nLoading fixtures...")
    all_fixtures: dict[str, list[Fixture]] = {}
    for fname in ("basic.json", "workflow.json", "mined-sessions.json",
                  "mined-sessions-v2.json", "combined.json"):
        fpath = FIXTURES_DIR / fname
        if fpath.exists():
            all_fixtures[fname] = load_fixtures(str(fpath))
            print(f"  {fname}: {len(all_fixtures[fname])} fixtures")
        else:
            print(f"  {fname}: NOT FOUND")

    # Define runs: (name, fixture_file, index_type)
    fixture_runs: list[tuple[str, str, str]] = [
        ("basic", "basic.json", "basic"),
        ("workflow", "workflow.json", "workflow"),
        ("mined-sessions", "mined-sessions.json", "merged"),
        ("mined-sessions-v2", "mined-sessions-v2.json", "merged"),
        ("combined", "combined.json", "merged"),
    ]

    # ── Run 1: bge-small ──
    print("\n" + "=" * 80)
    print("RUN 1: bge-small + v3 expansions")
    print("=" * 80)
    bge_results = run_model_suite(
        BGE_SMALL, basic_rules_t, workflow_rules_t,
        fixture_runs, all_fixtures,
    )

    # ── Run 2: jina-code (basic + workflow only) ──
    print("\n" + "=" * 80)
    print("RUN 2: jina-code + v3 expansions")
    print("=" * 80)
    jina_runs = [
        ("basic", "basic.json", "basic"),
        ("workflow", "workflow.json", "workflow"),
    ]
    jina_results = run_model_suite(
        JINA_CODE, basic_rules_t, workflow_rules_t,
        jina_runs, all_fixtures,
    )

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "meta": {
            "date": "2026-04-02",
            "expansion_version": "v3",
            "basic_rules": len(basic_rules),
            "basic_expansions": basic_exp,
            "workflow_rules": len(workflow_rules),
            "workflow_expansions": workflow_exp,
        },
        "bge_small": bge_results,
        "jina_code": jina_results,
    }
    output_path = RESULTS_DIR / "enriched-v3-embedding-2026-04-02.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ── Comparison 1: v1 vs v3 (bge-small) ──
    v1_path = RESULTS_DIR / "enriched-embedding-2026-04-02.json"
    if v1_path.exists():
        with open(v1_path) as f:
            v1_results = json.load(f)
        print_comparison(
            "COMPARISON 1: v1 vs v3 expansions (bge-small, embedding mode)",
            "v1", "v3",
            v1_results, bge_results,
            ["basic", "workflow", "mined-sessions", "mined-sessions-v2", "combined"],
        )
    else:
        print(f"\nNo v1 results at {v1_path} — skipping v1 vs v3 comparison")

    # ── Comparison 2: v3 bge-small vs v3 jina-code ──
    print_comparison(
        "COMPARISON 2: v3 bge-small vs v3 jina-code",
        "bge-small", "jina-code",
        bge_results, jina_results,
        ["basic", "workflow"],
    )

    # ── Comparison 3: vs raw baseline ──
    baseline_path = RESULTS_DIR / "raw-baseline-2026-04-02.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print_comparison(
            "COMPARISON 3: raw baseline vs v3 enriched (bge-small)",
            "raw", "v3-enriched",
            baseline, bge_results,
            ["basic", "workflow", "mined-sessions", "mined-sessions-v2", "combined"],
        )
    else:
        print(f"\nNo raw baseline at {baseline_path} — skipping comparison")

    # ── Compact summary ──
    print("\n" + "=" * 80)
    print("V3 ENRICHED EMBEDDING SUMMARY")
    print("=" * 80)
    print(
        f"{'Model':<20s} {'Dataset':<20s} {'N':>4s} {'Recall':>7s} "
        f"{'Prec':>7s} {'Noise':>7s} {'NegSil':>7s} {'p50ms':>7s}"
    )
    print("-" * 72)
    for model_label, model_results in [("bge-small", bge_results), ("jina-code", jina_results)]:
        for name, data in model_results.items():
            print(
                f"{model_label:<20s} {name:<20s} {data['fixture_count']:>4d} "
                f"{data['mean_recall']:>7.3f} {data['mean_precision']:>7.3f} "
                f"{data['mean_noise_ratio']:>7.3f} "
                f"{data['negative_silence_rate']:>7.3f} "
                f"{data['latency_p50_ms']:>7.1f}"
            )


if __name__ == "__main__":
    main()
