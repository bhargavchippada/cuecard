#!/usr/bin/env python3
"""Evaluate cross-encoder re-ranker quality on golden fixtures.

Compares: embedding-only vs embedding + cross-encoder re-ranking.

Usage:
    uv run python scripts/eval_reranker.py --reranker-model MODEL
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from cuecard.indexer import build_index
from cuecard.models import RankedResult, Rule, SourceMeta
from cuecard.parser import parse_rules
from cuecard.retriever import retrieve


def load_fixtures(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate_reranker(
    embedding_model_name: str,
    reranker_model_name: str,
    corpus_path: str,
    fixtures_path: str,
    recall_top_k: int = 20,
    recall_threshold: float = 0.15,
    final_top_k: int = 5,
) -> dict:
    print(f"\n{'='*70}")
    print(f"Embedding: {embedding_model_name}")
    print(f"Re-ranker: {reranker_model_name}")
    print(f"recall_top_k={recall_top_k}, recall_threshold={recall_threshold}, final_top_k={final_top_k}")
    print(f"{'='*70}\n")

    # Parse rules
    rules = parse_rules((corpus_path,))
    print(f"Parsed {len(rules)} rules")

    # Load embedding model + build index
    print(f"Loading embedding model...")
    emb_model = TextEmbedding(model_name=embedding_model_name)
    sources = {corpus_path: SourceMeta(mtime=0.0, content_hash="eval", rule_count=len(rules))}
    index = build_index(tuple(rules), sources, embedding_model_name, model=emb_model)  # type: ignore[arg-type]
    print(f"Indexed {index.size} rules")

    # Load re-ranker
    print(f"Loading re-ranker: {reranker_model_name}...")
    t0 = time.time()
    reranker = TextCrossEncoder(model_name=reranker_model_name)
    reranker_load = time.time() - t0
    print(f"Re-ranker loaded in {reranker_load:.2f}s")

    fixtures = load_fixtures(fixtures_path)
    print(f"\nEvaluating {len(fixtures)} test cases...\n")

    # Metrics accumulators
    emb_recalls = []
    rr_recalls = []
    emb_precisions = []
    rr_precisions = []
    emb_mrrs = []
    rr_mrrs = []
    rr_latencies = []

    for fx in fixtures:
        query = fx["query"]
        should_match = set(fx["should_match"])

        # Stage 1: embedding retrieval (loose threshold, more candidates)
        stage1 = retrieve(
            index, query, top_k=recall_top_k, threshold=recall_threshold,
            model=emb_model,  # type: ignore[arg-type]
        )

        # Also get embedding-only result at final top_k for comparison
        emb_only = retrieve(
            index, query, top_k=final_top_k, threshold=0.35,
            model=emb_model,  # type: ignore[arg-type]
        )

        # Stage 2: cross-encoder re-ranking
        if stage1:
            candidate_texts = [r.rule.text for r in stage1]
            t0 = time.time()
            scores = list(reranker.rerank(query, candidate_texts))
            rr_latency = (time.time() - t0) * 1000
            rr_latencies.append(rr_latency)

            # Pair scores with candidates, sort by score descending, take top_k
            scored = sorted(
                zip(scores, stage1),
                key=lambda x: x[0],
                reverse=True,
            )[:final_top_k]

            reranked = [
                RankedResult(rule=candidate.rule, score=score)
                for score, candidate in scored
            ]
        else:
            reranked = []
            rr_latencies.append(0.0)

        # Compute metrics for embedding-only
        emb_texts = {r.rule.text for r in emb_only}
        emb_tp = emb_texts & should_match
        emb_precision = len(emb_tp) / len(emb_only) if emb_only else 0.0
        emb_recall = len(emb_tp) / len(should_match) if should_match else 1.0
        emb_mrr = 0.0
        for rank, r in enumerate(emb_only, 1):
            if r.rule.text in should_match:
                emb_mrr = 1.0 / rank
                break

        # Compute metrics for reranked
        rr_texts = {r.rule.text for r in reranked}
        rr_tp = rr_texts & should_match
        rr_precision = len(rr_tp) / len(reranked) if reranked else 0.0
        rr_recall = len(rr_tp) / len(should_match) if should_match else 1.0
        rr_mrr = 0.0
        for rank, r in enumerate(reranked, 1):
            if r.rule.text in should_match:
                rr_mrr = 1.0 / rank
                break

        emb_recalls.append(emb_recall)
        rr_recalls.append(rr_recall)
        emb_precisions.append(emb_precision)
        rr_precisions.append(rr_precision)
        emb_mrrs.append(emb_mrr)
        rr_mrrs.append(rr_mrr)

        # Show changes
        status = ""
        if rr_recall > emb_recall:
            status = " [green]+IMPROVED[/green]"
        elif rr_recall < emb_recall:
            status = " [red]-REGRESSED[/red]"
        elif rr_recall == 0 and emb_recall == 0:
            status = " [yellow]BOTH MISS[/yellow]"

        delta_r = rr_recall - emb_recall
        delta_p = rr_precision - emb_precision
        print(f"  {fx['id']:<35} R: {emb_recall:.2f}→{rr_recall:.2f} ({delta_r:+.2f})  P: {emb_precision:.2f}→{rr_precision:.2f} ({delta_p:+.2f}){status}")

    # Summary
    lat_arr = np.array(rr_latencies)
    print(f"\n{'='*70}")
    print(f"COMPARISON: {embedding_model_name} + {reranker_model_name}")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Embedding Only':>15} {'+ Re-Ranker':>15} {'Delta':>10}")
    print(f"{'-'*65}")
    print(f"{'Avg Recall@5':<25} {np.mean(emb_recalls):>15.4f} {np.mean(rr_recalls):>15.4f} {np.mean(rr_recalls)-np.mean(emb_recalls):>+10.4f}")
    print(f"{'Avg Precision@5':<25} {np.mean(emb_precisions):>15.4f} {np.mean(rr_precisions):>15.4f} {np.mean(rr_precisions)-np.mean(emb_precisions):>+10.4f}")
    print(f"{'Avg MRR':<25} {np.mean(emb_mrrs):>15.4f} {np.mean(rr_mrrs):>15.4f} {np.mean(rr_mrrs)-np.mean(emb_mrrs):>+10.4f}")
    print(f"{'Re-rank latency p50':<25} {'':>15} {np.percentile(lat_arr, 50):>12.1f} ms")
    print(f"{'Re-rank latency p95':<25} {'':>15} {np.percentile(lat_arr, 95):>12.1f} ms")

    return {
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "n_fixtures": len(fixtures),
        "embedding_recall": round(float(np.mean(emb_recalls)), 4),
        "reranked_recall": round(float(np.mean(rr_recalls)), 4),
        "embedding_precision": round(float(np.mean(emb_precisions)), 4),
        "reranked_precision": round(float(np.mean(rr_precisions)), 4),
        "embedding_mrr": round(float(np.mean(emb_mrrs)), 4),
        "reranked_mrr": round(float(np.mean(rr_mrrs)), 4),
        "rerank_latency_p50": round(float(np.percentile(lat_arr, 50)), 2),
        "rerank_latency_p95": round(float(np.percentile(lat_arr, 95)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default="jinaai/jina-embeddings-v2-base-code")
    parser.add_argument("--reranker-model", default="Xenova/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--corpus", default="eval/corpora/rules_basic.txt")
    parser.add_argument("--fixtures", default="eval/fixtures/basic.json")
    parser.add_argument("--recall-top-k", type=int, default=20)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = evaluate_reranker(
        embedding_model_name=args.embedding_model,
        reranker_model_name=args.reranker_model,
        corpus_path=args.corpus,
        fixtures_path=args.fixtures,
        recall_top_k=args.recall_top_k,
        final_top_k=args.final_top_k,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
