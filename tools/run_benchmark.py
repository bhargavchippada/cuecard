#!/usr/bin/env python3
"""Benchmark runner for enriched retrieval evaluation.

Usage:
    python tools/run_benchmark.py --config <name> --fixtures <path> [--corpus <paths>] [--output <path>]

Configs:
    raw-embedding       Dense-only, no expansions, threshold=0.30
    enriched-embedding  Dense+BM25 with expansions, threshold=0.30
    enriched-llm-local  Dense+BM25 + LLM reranker (Qwen3.5-35B)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure cuecard is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuecard.eval import (  # noqa: E402
    EvalSummary,
    format_eval_report,
    load_fixtures,
    run_eval,
)

CONFIGS: dict[str, dict[str, object]] = {
    "raw-embedding": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "top_k": 5,
        "threshold": 0.30,
        "mode": "embedding",
    },
    "enriched-embedding": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "top_k": 5,
        "threshold": 0.30,
        "mode": "embedding",
    },
    "enriched-llm-local": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "top_k": 5,
        "threshold": 0.30,
        "mode": "llm-local",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = str(PROJECT_ROOT / "eval" / "corpora")


def _run_one(
    config_name: str,
    fixtures_path: str,
    corpus_override: tuple[str, ...] | None = None,
) -> EvalSummary:
    cfg = CONFIGS[config_name]
    fixtures = load_fixtures(fixtures_path)

    from fastembed import TextEmbedding

    model_name = str(cfg["model_name"])
    model = TextEmbedding(model_name=model_name)

    t0 = time.monotonic()
    summary = run_eval(
        fixtures,
        corpus_dir=CORPUS_DIR,
        model_name=model_name,
        model=model,
        top_k=int(cfg["top_k"]),
        threshold=float(cfg["threshold"]),
        mode=str(cfg["mode"]) if cfg.get("mode") else None,
        corpus_override=corpus_override,
    )
    elapsed = time.monotonic() - t0

    return summary, elapsed  # type: ignore[return-value]


def _summary_to_dict(summary: EvalSummary) -> dict[str, object]:
    return {
        "fixture_count": summary.fixture_count,
        "mean_precision": round(summary.mean_precision, 4),
        "mean_recall": round(summary.mean_recall, 4),
        "mean_mrr": round(summary.mean_mrr, 4),
        "mean_ndcg": round(summary.mean_ndcg, 4),
        "mean_noise_ratio": round(summary.mean_noise_ratio, 4),
        "mean_context_waste_ratio": round(summary.mean_context_waste_ratio, 4),
        "negative_silence_rate": round(summary.negative_silence_rate, 4),
        "mean_retrieved_count": round(summary.mean_retrieved_count, 2),
        "latency_p50_ms": round(summary.latency_p50_ms, 1),
        "latency_p95_ms": round(summary.latency_p95_ms, 1),
        "latency_p99_ms": round(summary.latency_p99_ms, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cuecard benchmarks")
    parser.add_argument("--config", required=True, choices=list(CONFIGS))
    parser.add_argument("--fixtures", required=True, help="Path to fixtures JSON")
    parser.add_argument(
        "--corpus", nargs="*", default=None,
        help="Override corpus paths (for unified index tests)",
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--label", default=None, help="Label for this run")
    args = parser.parse_args()

    corpus_override = tuple(args.corpus) if args.corpus else None
    label = args.label or f"{args.config}:{Path(args.fixtures).stem}"

    print(f"=== {label} ===")
    print(f"Config: {args.config}")
    print(f"Fixtures: {args.fixtures}")
    if corpus_override:
        print(f"Corpus override: {corpus_override}")

    summary, elapsed = _run_one(args.config, args.fixtures, corpus_override)

    print(f"\nCompleted in {elapsed:.1f}s")
    print(format_eval_report(summary))

    result = {
        "label": label,
        "config": args.config,
        "fixtures": args.fixtures,
        "elapsed_s": round(elapsed, 1),
        "metrics": _summary_to_dict(summary),
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")
    else:
        print(f"\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
