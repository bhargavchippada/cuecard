#!/usr/bin/env python3
"""Final LLM reranker benchmark: new prompt + _MAX_TOKENS=2048.

Runs on all 357 golden fixtures, reports per-tier results.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cuecard.eval import load_fixtures, run_eval, format_eval_report

FIXTURES_PATH = str(Path(__file__).parent.parent / "eval" / "fixtures" / "basic.json")
CORPUS_DIR = str(Path(__file__).parent.parent / "eval" / "corpora")
RESULTS_DIR = Path(__file__).parent.parent / "eval" / "results"


def main():
    EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"

    fixtures = load_fixtures(FIXTURES_PATH)
    print(f"Loaded {len(fixtures)} fixtures", flush=True)

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=EMBEDDING_MODEL)

    print(f"\n{'='*60}", flush=True)
    print("Running: new prompt + _MAX_TOKENS=2048", flush=True)
    print(f"{'='*60}", flush=True)

    start = time.time()
    summary = run_eval(
        fixtures,
        CORPUS_DIR,
        EMBEDDING_MODEL,
        model=model,
        top_k=5,
        threshold=0.30,
        mode="rerank-llm-local",
        query_max_length=1000,
    )
    elapsed = time.time() - start

    print(format_eval_report(summary), flush=True)
    print(f"\nTotal time: {elapsed:.1f}s", flush=True)

    # Save results
    result = {
        "config": "new_prompt_mt2048",
        "max_tokens": 2048,
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
        "latency_p99": summary.latency_p99_ms,
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
        "total_time_s": elapsed,
    }

    out_path = RESULTS_DIR / "llm_prompt_final_mt2048.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}", flush=True)

    # Print compact summary for tmux reporting
    print(f"\n{'='*60}", flush=True)
    print("COMPACT SUMMARY FOR REPORTING:", flush=True)
    for t in summary.per_tier:
        print(
            f"  {t.tier:<10s}: R={t.mean_recall:.1%} N={t.mean_noise_ratio:.1%}"
            f" S={t.silence_rate:.1%} avg={t.mean_retrieved_count:.1f}",
            flush=True,
        )
    print(
        f"  Overall:  R={summary.mean_recall:.1%} N={summary.mean_noise_ratio:.1%}"
        f" NegSilence={summary.negative_silence_rate:.1%}"
        f" p50={summary.latency_p50_ms:.0f}ms p95={summary.latency_p95_ms:.0f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
