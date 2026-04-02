#!/usr/bin/env python3
"""Benchmark Qwen3-0.6B as LLM reranker on basic + workflow fixtures.

Compares against Qwen3.5-35B baseline from enriched-llm-local-2026-04-02.json.

Usage:
    uv run python tools/run_small_model_benchmark.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuecard.eval import (
    EvalSummary,
    Fixture,
    FixtureResult,
    TierSummary,
    load_fixtures,
    mrr,
    ndcg_at_k,
    noise_ratio,
    precision_at_k,
    recall_at_k,
    anti_precision,
    context_waste_ratio,
    format_eval_report,
)
from cuecard.indexer import build_index, load_rules_json
from cuecard.models import PipelineConfig
from cuecard.pipeline import run_pipeline
from cuecard.retriever import merge_indexes

# Monkey-patch: small models repeat JSON output, causing parse failures.
# Extract just the first JSON object before parsing.
import cuecard.llm_reranker as _llm_mod

_original_parse = _llm_mod._parse_llm_response
_decoder = json.JSONDecoder()
_parse_fix_count = 0


def _patched_parse(response: str, max_rule_id: int) -> _llm_mod.LLMParseResult:
    """Try original parse first; on failure, extract first JSON object."""
    global _parse_fix_count
    result = _original_parse(response, max_rule_id)
    if result.indices is not None:
        return result
    # Small models repeat JSON — extract just the first object
    cleaned = _llm_mod._strip_thinking_tags(response)
    brace_pos = cleaned.find("{")
    if brace_pos >= 0:
        try:
            obj, _ = _decoder.raw_decode(cleaned, brace_pos)
            first_json = json.dumps(obj)
            result2 = _original_parse(first_json, max_rule_id)
            if result2.indices is not None:
                _parse_fix_count += 1
                return result2
        except json.JSONDecodeError:
            pass
    return result


_llm_mod._parse_llm_response = _patched_parse

MODEL = "BAAI/bge-small-en-v1.5"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
FIXTURES_DIR = EVAL_DIR / "fixtures"
CORPORA_DIR = EVAL_DIR / "corpora"
RESULTS_DIR = EVAL_DIR / "results"

_TIER_ORDER = ("easy", "medium", "hard", "negative")

LLM_ENDPOINT = "http://localhost:8082/v1"
BASELINE_PATH = RESULTS_DIR / "enriched-llm-local-2026-04-02.json"


@dataclass(frozen=True)
class BenchConfig:
    """Config stub for Qwen3-0.6B benchmark."""

    top_k: int = 5
    threshold: float = 0.30
    dedup_threshold: float = 0.95
    query_max_length: int = 500
    sparse_enabled: bool = True
    fusion_k: int = 60
    pipeline: PipelineConfig = field(
        default_factory=lambda: PipelineConfig(
            mode="llm-local",
            local_endpoint=LLM_ENDPOINT,
            thinking=False,
        ),
    )


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
) -> tuple[EvalSummary, int, int, list[str]]:
    """Run LLM pipeline on each fixture. Returns (summary, successes, failures, sample_errors)."""
    results: list[FixtureResult] = []
    errors = 0
    sample_errors: list[str] = []

    for i, fixture in enumerate(fixtures):
        idx = index_map.get(fixture.corpus)
        if idx is None:
            print(f"  SKIP {fixture.id}: no index for corpus {fixture.corpus}")
            continue

        try:
            start = time.perf_counter()
            pipeline_result = run_pipeline(
                fixture.query,
                idx,
                config,
                embedding_model=model,
                mode="llm-local",
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

            if (i + 1) % 50 == 0:
                print(f"    [{i + 1}/{len(fixtures)}] last latency: {elapsed_ms:.0f}ms")

        except Exception as exc:
            errors += 1
            if len(sample_errors) < 5:
                sample_errors.append(f"{fixture.id}: {exc}")
            if errors <= 3:
                print(f"  ERROR {fixture.id}: {exc}")
            elif errors == 4:
                print("  (suppressing further error messages...)")
            continue

    if errors:
        print(f"  ({errors} fixtures errored out of {len(fixtures)})")

    if not results:
        empty_summary = EvalSummary(
            fixture_count=0, mean_precision=0.0, mean_recall=0.0,
            mean_mrr=0.0, mean_ndcg=0.0, mean_anti_precision=0.0,
            mean_noise_ratio=0.0, mean_context_waste_ratio=0.0,
            negative_silence_rate=0.0, mean_retrieved_count=0.0,
            latency_p50_ms=0.0, latency_p95_ms=0.0, latency_p99_ms=0.0,
            per_fixture=(), per_tier=(),
        )
        return empty_summary, 0, errors, sample_errors

    latencies = [r.latency_ms for r in results]
    negatives = [r for r in results if r.difficulty == "negative"]
    neg_silent = sum(1 for r in negatives if r.retrieved_count == 0)
    neg_silence_rate = neg_silent / len(negatives) if negatives else 1.0

    summary = EvalSummary(
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
    return summary, len(results), errors, sample_errors


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


def print_comparison(all_results: dict[str, dict], baseline: dict) -> None:
    """Print side-by-side comparison table."""
    print("\n" + "=" * 90)
    print("COMPARISON: Qwen3.5-35B (baseline) vs Qwen3-0.6B")
    print("=" * 90)

    for dset in ("basic", "workflow"):
        if dset not in all_results or dset not in baseline:
            continue

        b = baseline[dset]
        e = all_results[dset]

        print(f"\n  {dset.upper()}:")
        print(f"    {'Metric':<30s} {'35B':>8s} {'0.6B':>8s} {'Delta':>8s}")
        print(f"    {'-'*58}")

        for metric in (
            "mean_recall", "mean_precision", "mean_noise_ratio",
            "negative_silence_rate", "mean_retrieved_count",
            "latency_p50_ms", "latency_p95_ms",
        ):
            bv = b.get(metric, 0.0)
            ev = e.get(metric, 0.0)
            delta = ev - bv
            sign = "+" if delta >= 0 else ""
            print(
                f"    {metric:<30s} {bv:>8.3f} {ev:>8.3f} {sign}{delta:>7.3f}"
            )

        # Per-tier comparison
        b_tiers = {t["tier"]: t for t in b.get("per_tier", [])}
        e_tiers = {t["tier"]: t for t in e.get("per_tier", [])}

        print(f"\n    Per-tier recall:")
        print(f"    {'Tier':<12s} {'35B':>8s} {'0.6B':>8s} {'Delta':>8s}")
        print(f"    {'-'*40}")
        for tier in _TIER_ORDER:
            bt = b_tiers.get(tier, {})
            et = e_tiers.get(tier, {})
            if tier == "negative":
                bv = bt.get("silence_rate", 0.0)
                ev = et.get("silence_rate", 0.0)
                label = f"{tier} (silence)"
            else:
                bv = bt.get("mean_recall", 0.0)
                ev = et.get("mean_recall", 0.0)
                label = tier
            delta = ev - bv
            sign = "+" if delta >= 0 else ""
            print(f"    {label:<12s} {bv:>8.3f} {ev:>8.3f} {sign}{delta:>7.3f}")


def main() -> None:
    print("=" * 80)
    print("Small Model Benchmark: Qwen3-0.6B as LLM Reranker")
    print(f"Embedding: {MODEL}")
    print(f"Mode: llm-local (dense + sparse + RRF + Qwen3-0.6B)")
    print(f"LLM endpoint: {LLM_ENDPOINT}")
    print("=" * 80)

    # Verify LLM server
    import urllib.request
    try:
        req = urllib.request.Request(f"{LLM_ENDPOINT}/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            model_name = data.get("data", [{}])[0].get("id", "unknown")
            print(f"LLM server: OK ({model_name})")
    except Exception as exc:
        print(f"ERROR: LLM server not reachable at {LLM_ENDPOINT}: {exc}")
        sys.exit(1)

    # Load baseline
    baseline = {}
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        print(f"Baseline loaded: {BASELINE_PATH.name}")
    else:
        print("WARNING: No baseline file found")

    # Load embedding model
    print("\nLoading embedding model...")
    from fastembed import TextEmbedding
    t0 = time.time()
    emb_model = TextEmbedding(model_name=MODEL)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Load enriched rules
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

    # Build indexes
    print("\nBuilding enriched indexes...")
    t0 = time.time()
    basic_idx = build_index(tuple(basic_rules), {}, MODEL, model=emb_model)
    workflow_idx = build_index(tuple(workflow_rules), {}, MODEL, model=emb_model)
    print(f"  Indexes built in {time.time() - t0:.1f}s")

    config = BenchConfig()

    all_results: dict[str, dict] = {}
    all_meta: dict[str, dict] = {}
    total_start = time.time()

    runs = [
        ("basic", "basic.json", {"rules_basic.txt": basic_idx}),
        ("workflow", "workflow.json", {"rules_workflow.txt": workflow_idx}),
    ]

    for name, fixture_file, idx_map in runs:
        fixture_path = FIXTURES_DIR / fixture_file
        if not fixture_path.exists():
            print(f"\n  SKIP {name}: {fixture_path} not found")
            continue

        fixtures = load_fixtures(str(fixture_path))
        print(f"\n{'='*60}")
        print(f"  {name}: {len(fixtures)} fixtures (Qwen3-0.6B reranker)")
        print(f"{'='*60}")

        run_start = time.time()
        summary, successes, failures, sample_errors = evaluate_fixtures(
            fixtures, idx_map, config, emb_model,
        )
        run_elapsed = time.time() - run_start
        print(f"  Completed in {run_elapsed:.1f}s ({successes} ok, {failures} errors)")

        failure_rate = failures / len(fixtures) if fixtures else 0
        if failure_rate > 0.5:
            print(f"\n  WARNING: >50% failure rate ({failure_rate:.0%})")
            print("  Likely cause: prompt too long for 0.6B context window")
            if sample_errors:
                print("  Sample errors:")
                for err in sample_errors:
                    print(f"    - {err}")

        all_results[name] = summary_to_dict(summary)
        all_meta[name] = {
            "successes": successes,
            "failures": failures,
            "failure_rate": round(failure_rate, 4),
            "sample_errors": sample_errors,
        }
        print(format_eval_report(summary))

    total_elapsed = time.time() - total_start
    print(f"\nTotal benchmark time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print(f"JSON repetition fixes applied: {_parse_fix_count}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": "Qwen3-0.6B-Q4_K_M",
        "endpoint": LLM_ENDPOINT,
        "embedding_model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": all_results,
        "meta": all_meta,
    }
    output_path = RESULTS_DIR / "small-model-qwen3-0.6b-2026-04-02.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Comparison
    if baseline:
        print_comparison(all_results, baseline)

    # Summary table
    print("\n" + "=" * 80)
    print("QWEN3-0.6B SUMMARY (bge-small + expansions + BM25 + RRF + 0.6B)")
    print("=" * 80)
    print(
        f"{'Dataset':<12s} {'N':>4s} {'Recall':>7s} {'Prec':>7s} "
        f"{'Noise':>7s} {'NegSil':>7s} {'p50ms':>7s} {'Errors':>7s}"
    )
    print("-" * 65)
    for name, data in all_results.items():
        meta = all_meta.get(name, {})
        print(
            f"{name:<12s} {data['fixture_count']:>4d} "
            f"{data['mean_recall']:>7.3f} {data['mean_precision']:>7.3f} "
            f"{data['mean_noise_ratio']:>7.3f} "
            f"{data['negative_silence_rate']:>7.3f} "
            f"{data['latency_p50_ms']:>7.1f} "
            f"{meta.get('failures', 0):>7d}"
        )


if __name__ == "__main__":
    main()
