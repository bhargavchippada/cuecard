#!/usr/bin/env python3
"""Benchmark all pipeline modes on the golden fixture set."""

import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cuecard.eval import load_fixtures, run_eval, format_eval_report

FIXTURES_PATH = str(Path(__file__).parent.parent / "eval" / "fixtures" / "basic.json")
CORPUS_DIR = str(Path(__file__).parent.parent / "eval" / "corpora")
RESULTS_DIR = Path(__file__).parent.parent / "eval" / "results"

EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"


def run_mode(mode_name, mode, model, fixtures, **kwargs):
    """Run eval for a specific mode and save results."""
    print(f"\n{'='*60}")
    print(f"Running: {mode_name}")
    print(f"{'='*60}")

    start = time.time()
    summary = run_eval(
        fixtures,
        CORPUS_DIR,
        EMBEDDING_MODEL,
        model=model,
        top_k=5,
        threshold=0.30,
        mode=mode,
        **kwargs,
    )
    elapsed = time.time() - start

    print(format_eval_report(summary))
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    result = {
        "mode": mode_name,
        "n_fixtures": summary.fixture_count,
        "mean_recall": summary.mean_recall,
        "mean_precision": summary.mean_precision,
        "mean_mrr": summary.mean_mrr,
        "mean_noise_ratio": summary.mean_noise_ratio,
        "mean_context_waste": summary.mean_context_waste_ratio,
        "negative_silence_rate": summary.negative_silence_rate,
        "mean_retrieved_count": summary.mean_retrieved_count,
        "latency_p50": summary.latency_p50_ms,
        "latency_p95": summary.latency_p95_ms,
        "per_tier": [
            {
                "tier": t.tier,
                "count": t.count,
                "recall": t.mean_recall,
                "precision": t.mean_precision,
                "noise": t.mean_noise_ratio,
                "waste": t.mean_context_waste_ratio,
                "silence_rate": t.silence_rate,
                "avg_retrieved": t.mean_retrieved_count,
            }
            for t in summary.per_tier
        ],
    }

    out_path = RESULTS_DIR / f"pipeline_{mode_name.replace(' ', '_').lower()}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved: {out_path}")

    return summary


def main():
    fixtures = load_fixtures(FIXTURES_PATH)
    print(f"Loaded {len(fixtures)} fixtures")

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=EMBEDDING_MODEL)

    summaries = {}

    # Mode 1: Embedding only (baseline)
    summaries["embedding"] = run_mode("embedding", None, model, fixtures)

    # Mode 2: Cross-encoder rerank
    summaries["rerank"] = run_mode("rerank", "rerank", model, fixtures)

    # Mode 3: LLM rerank (local) - check if server is available
    import httpx

    try:
        resp = httpx.get("http://localhost:8081/v1/models", timeout=5)
        if resp.status_code == 200:
            summaries["rerank-llm-local"] = run_mode(
                "rerank-llm-local", "rerank-llm-local", model, fixtures,
            )
        else:
            print(f"\n[SKIP] llama-server not ready (status {resp.status_code})")
    except Exception as e:
        print(f"\n[SKIP] llama-server not available: {e}")

    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    header = (
        f"{'Mode':<25s} {'Recall':>8s} {'Noise':>8s} {'Waste':>8s}"
        f" {'Silence':>8s} {'AvgRet':>8s} {'p50ms':>8s}"
    )
    print(header)
    print("-" * 75)
    for name, s in summaries.items():
        print(
            f"{name:<25s} {s.mean_recall:>8.3f} {s.mean_noise_ratio:>8.3f}"
            f" {s.mean_context_waste_ratio:>8.3f} {s.negative_silence_rate:>8.3f}"
            f" {s.mean_retrieved_count:>8.1f} {s.latency_p50_ms:>8.1f}"
        )

    # Per-tier comparison
    if len(summaries) >= 2:
        print(f"\n{'='*60}")
        print("PER-TIER COMPARISON")
        print(f"{'='*60}")
        for tier_name in ("easy", "medium", "hard", "negative"):
            print(f"\n  {tier_name.upper()}")
            print(f"  {'Mode':<25s} {'Recall':>8s} {'Noise':>8s} {'Waste':>8s} {'N':>4s}")
            for mode_name, s in summaries.items():
                for t in s.per_tier:
                    if t.tier == tier_name:
                        print(
                            f"  {mode_name:<25s} {t.mean_recall:>8.3f}"
                            f" {t.mean_noise_ratio:>8.3f}"
                            f" {t.mean_context_waste_ratio:>8.3f}"
                            f" {t.count:>4d}"
                        )


if __name__ == "__main__":
    main()
