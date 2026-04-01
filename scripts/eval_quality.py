#!/usr/bin/env python3
"""Evaluate retrieval quality against golden fixtures.

Usage:
    uv run python scripts/eval_quality.py [--model MODEL] [--top-k K] [--threshold T]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from cuecard.indexer import build_index
from cuecard.models import Provenance, Rule, SourceMeta
from cuecard.parser import parse_rules
from cuecard.retriever import retrieve


def load_fixtures(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate(
    model_name: str,
    corpus_path: str,
    fixtures_path: str,
    top_k: int = 5,
    threshold: float = 0.35,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Corpus: {corpus_path} | Fixtures: {fixtures_path}")
    print(f"top_k={top_k}, threshold={threshold}")
    print(f"{'='*60}\n")

    # Parse rules
    rules = parse_rules((corpus_path,))
    print(f"Parsed {len(rules)} rules from corpus")

    # Build index
    print(f"Loading model {model_name}...")
    t0 = time.time()
    model = TextEmbedding(model_name=model_name)
    model_load_time = time.time() - t0
    print(f"Model loaded in {model_load_time:.2f}s")

    t0 = time.time()
    sources = {
        corpus_path: SourceMeta(mtime=0.0, content_hash="eval", rule_count=len(rules)),
    }
    index = build_index(
        tuple(rules), sources, model_name,
        model=model,  # type: ignore[arg-type]
    )
    embed_time = time.time() - t0
    print(f"Indexed {index.size} rules in {embed_time:.2f}s (dim={index.dim})")

    # Load fixtures
    fixtures = load_fixtures(fixtures_path)
    print(f"\nEvaluating {len(fixtures)} test cases...\n")

    # Evaluate each fixture
    total_precision = 0.0
    total_recall = 0.0
    total_anti = 0.0
    total_mrr = 0.0
    latencies = []
    details = []

    for fx in fixtures:
        query = fx["query"]
        should_match = set(fx["should_match"])
        should_not_match = set(fx.get("should_not_match", []))

        t0 = time.time()
        results = retrieve(
            index, query, top_k=top_k, threshold=threshold,
            model=model,  # type: ignore[arg-type]
        )
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        retrieved_texts = {r.rule.text for r in results}

        # Precision: of retrieved, how many should match?
        true_pos = retrieved_texts & should_match
        precision = len(true_pos) / len(results) if results else 0.0

        # Recall: of should_match, how many were retrieved?
        recall = len(true_pos) / len(should_match) if should_match else 1.0

        # Anti-precision: of retrieved, how many should NOT match?
        anti = len(retrieved_texts & should_not_match)

        # MRR: reciprocal rank of first should_match hit
        mrr = 0.0
        for rank, r in enumerate(results, 1):
            if r.rule.text in should_match:
                mrr = 1.0 / rank
                break

        total_precision += precision
        total_recall += recall
        total_anti += anti
        total_mrr += mrr

        status = "PASS" if recall > 0 else "MISS"
        detail = {
            "id": fx["id"],
            "status": status,
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "anti": anti,
            "latency_ms": latency,
            "retrieved": [(r.rule.text, round(r.score, 3)) for r in results],
        }
        details.append(detail)

        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {fx['id']:<35} P={precision:.2f} R={recall:.2f} MRR={mrr:.2f} ({latency:.1f}ms)")
        if status == "MISS":
            print(f"    Expected: {should_match}")
            print(f"    Got: {[r.rule.text[:60] for r in results]}")

    n = len(fixtures)
    avg_precision = total_precision / n
    avg_recall = total_recall / n
    avg_mrr = total_mrr / n
    avg_anti = total_anti / n
    lat_arr = np.array(latencies)

    summary = {
        "model": model_name,
        "corpus": corpus_path,
        "top_k": top_k,
        "threshold": threshold,
        "n_rules": len(rules),
        "n_fixtures": n,
        "avg_precision": round(avg_precision, 4),
        "avg_recall": round(avg_recall, 4),
        "avg_mrr": round(avg_mrr, 4),
        "avg_anti_precision": round(avg_anti, 4),
        "latency_p50_ms": round(float(np.percentile(lat_arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(lat_arr, 99)), 2),
        "model_load_time_s": round(model_load_time, 2),
        "embed_time_s": round(embed_time, 2),
        "details": details,
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"  Avg Precision@{top_k}: {avg_precision:.4f}")
    print(f"  Avg Recall@{top_k}:    {avg_recall:.4f}")
    print(f"  Avg MRR:              {avg_mrr:.4f}")
    print(f"  Avg Anti-Precision:   {avg_anti:.4f}")
    print(f"  Latency p50/p95/p99:  {summary['latency_p50_ms']:.1f}/{summary['latency_p95_ms']:.1f}/{summary['latency_p99_ms']:.1f} ms")
    print(f"  Model load:           {model_load_time:.2f}s")
    print(f"  Embed time:           {embed_time:.2f}s")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cuecard retrieval quality")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--corpus", default="eval/corpora/rules_basic.txt")
    parser.add_argument("--fixtures", default="eval/fixtures/basic.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--output", default=None, help="Save results JSON to file")
    args = parser.parse_args()

    summary = evaluate(
        model_name=args.model,
        corpus_path=args.corpus,
        fixtures_path=args.fixtures,
        top_k=args.top_k,
        threshold=args.threshold,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
