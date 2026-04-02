"""Run raw (no enrichment) baseline benchmarks across all fixture sets.

Usage:
    uv run python tools/run_raw_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastembed import TextEmbedding

from cuecard.eval import format_eval_report, load_fixtures, run_eval

MODEL = "BAAI/bge-small-en-v1.5"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = str(PROJECT_ROOT / "eval" / "corpora")

configs: list[tuple[str, str, tuple[str, ...] | None]] = [
    ("basic", str(PROJECT_ROOT / "eval/fixtures/basic.json"), None),
    ("workflow", str(PROJECT_ROOT / "eval/fixtures/workflow.json"), None),
    ("mined-v1", str(PROJECT_ROOT / "eval/fixtures/mined-sessions.json"), None),
    ("mined-v2", str(PROJECT_ROOT / "eval/fixtures/mined-sessions-v2.json"), None),
    (
        "combined-unified",
        str(PROJECT_ROOT / "eval/fixtures/combined.json"),
        (
            str(PROJECT_ROOT / "eval/corpora/rules_basic.txt"),
            str(PROJECT_ROOT / "eval/corpora/rules_workflow.txt"),
        ),
    ),
]


def main() -> None:
    print(f"Loading model: {MODEL}")
    model = TextEmbedding(model_name=MODEL)

    results: dict[str, dict[str, object]] = {}
    for label, fixture_path, corpus_override in configs:
        fixtures = load_fixtures(fixture_path)
        t0 = time.monotonic()
        summary = run_eval(
            fixtures,
            corpus_dir=CORPUS_DIR,
            model_name=MODEL,
            model=model,
            top_k=5,
            threshold=0.30,
            mode="embedding",
            corpus_override=corpus_override,
        )
        elapsed = time.monotonic() - t0
        results[label] = {
            "fixtures": len(fixtures),
            "recall": round(summary.mean_recall, 4),
            "precision": round(summary.mean_precision, 4),
            "noise": round(summary.mean_noise_ratio, 4),
            "neg_silence": round(summary.negative_silence_rate, 4),
            "mrr": round(summary.mean_mrr, 4),
            "ndcg": round(summary.mean_ndcg, 4),
            "context_waste": round(summary.mean_context_waste_ratio, 4),
            "avg_retrieved": round(summary.mean_retrieved_count, 2),
            "p50_ms": round(summary.latency_p50_ms, 1),
            "elapsed_s": round(elapsed, 1),
        }
        print(f"\n=== {label} ({len(fixtures)} fixtures) ===")
        print(format_eval_report(summary))

    # Save results
    out_path = PROJECT_ROOT / "eval" / "results" / "raw-baseline-2026-04-02.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 110)
    print(
        f"{'Set':<20} {'N':>5} {'Recall':>8} {'Prec':>8} {'Noise':>8} "
        f"{'NegSil':>8} {'MRR':>8} {'nDCG':>8} {'Waste':>8} {'AvgK':>6} {'p50ms':>7}"
    )
    print("-" * 110)
    for label, m in results.items():
        print(
            f"{label:<20} {m['fixtures']:>5} {m['recall']:>8.4f} {m['precision']:>8.4f} "
            f"{m['noise']:>8.4f} {m['neg_silence']:>8.4f} {m['mrr']:>8.4f} {m['ndcg']:>8.4f} "
            f"{m['context_waste']:>8.4f} {m['avg_retrieved']:>6.2f} {m['p50_ms']:>7.1f}"
        )
    print("=" * 110)


if __name__ == "__main__":
    main()
