#!/usr/bin/env python3
"""Benchmark Qwen3.5 dense models as rerankers.

Usage:
    uv run python tools/bench_models.py --model-path ~/models/Qwen3.5-9B-Q4_K_M.gguf --label qwen35-9b
    uv run python tools/bench_models.py --model-path ~/models/Qwen3.5-4B-Q4_K_M.gguf --label qwen35-4b
    uv run python tools/bench_models.py --model-path ~/models/Qwen3.5-2B-Q4_K_M.gguf --label qwen35-2b

Assumes llama-server is NOT running — this script starts it, waits for health,
runs evals, saves results, and kills the server.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import requests

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuecard.eval import EvalSummary, load_fixtures, run_eval

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
RESULTS_DIR = EVAL_DIR / "results"

# Fixture sets and their corpora
FIXTURE_SETS = {
    "basic": {
        "fixtures": EVAL_DIR / "fixtures" / "basic.json",
        "corpus_override": (
            str(EVAL_DIR / "corpora" / "enriched_basic_v3" / "rules.json"),
        ),
    },
    "workflow": {
        "fixtures": EVAL_DIR / "fixtures" / "workflow.json",
        "corpus_override": (
            str(EVAL_DIR / "corpora" / "enriched_workflow_v3" / "rules.json"),
        ),
    },
    "mined": {
        "fixtures": EVAL_DIR / "fixtures" / "mined-sessions-v2.json",
        "corpus_override": (
            str(EVAL_DIR / "corpora" / "enriched_basic_v3" / "rules.json"),
            str(EVAL_DIR / "corpora" / "enriched_workflow_v3" / "rules.json"),
        ),
    },
}

EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"
PORT = 8081
ENDPOINT = f"http://localhost:{PORT}/v1"


def start_server(model_path: str, *, ngl: int = 99) -> subprocess.Popen[bytes]:
    """Start llama-server and wait for health."""
    cmd = [
        "llama-server",
        "-m", model_path,
        "--port", str(PORT),
        "-ngl", str(ngl),
        "-c", "16384",
        "--jinja",
    ]
    print(f"Starting llama-server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for health
    for i in range(60):
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                print(f"Server ready after {i + 1}s")
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(1)

    proc.kill()
    msg = "Server failed to start within 60s"
    raise RuntimeError(msg)


def kill_server(proc: subprocess.Popen[bytes]) -> None:
    """Kill llama-server gracefully."""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Server stopped")


def summary_to_dict(summary: EvalSummary) -> dict:
    """Convert EvalSummary to serializable dict (drop per_fixture for brevity)."""
    d = asdict(summary)
    del d["per_fixture"]
    # Round floats
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = round(v, 4)
    for tier in d.get("per_tier", []):
        for k, v in tier.items():
            if isinstance(v, float):
                tier[k] = round(v, 4)
    return d


def run_benchmark(label: str, *, sample_ratio: float = 1.0) -> dict:
    """Run all fixture sets and return combined results."""
    from fastembed import TextEmbedding

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

    results = {}
    for name, cfg in FIXTURE_SETS.items():
        fixture_path = cfg["fixtures"]
        if not fixture_path.exists():
            print(f"  Skipping {name}: fixture file not found")
            continue

        print(f"\n--- {name} ({fixture_path.name}) ---")
        fixtures = load_fixtures(str(fixture_path))
        print(f"  Loaded {len(fixtures)} fixtures")

        summary = run_eval(
            fixtures,
            str(fixture_path.parent),
            EMBEDDING_MODEL,
            model=embedding_model,
            mode="llm-local",
            corpus_override=cfg.get("corpus_override"),
            sample_ratio=sample_ratio,
        )

        results[name] = summary_to_dict(summary)
        print(f"  recall={summary.mean_recall:.3f}  noise={summary.mean_noise_ratio:.3f}  "
              f"neg_silence={summary.negative_silence_rate:.3f}  "
              f"p50={summary.latency_p50_ms:.0f}ms")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.5 models as rerankers")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model file")
    parser.add_argument("--label", required=True, help="Label for results file")
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers (0 for CPU-only)")
    parser.add_argument("--no-server", action="store_true", help="Don't start server (use existing)")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Fraction of fixtures (0.0-1.0)")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Model not found: {args.model_path}")
        sys.exit(1)

    # Set env for pipeline
    os.environ.setdefault("CUECARD_LLM_ENDPOINT", ENDPOINT)

    proc = None
    if not args.no_server:
        proc = start_server(args.model_path, ngl=args.ngl)

    try:
        results = run_benchmark(args.label, sample_ratio=args.sample_ratio)

        # Save
        out_path = RESULTS_DIR / f"{args.label}-llm-local-2026-04-02.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    finally:
        if proc is not None:
            kill_server(proc)


if __name__ == "__main__":
    main()
