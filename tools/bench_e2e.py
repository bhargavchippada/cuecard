#!/usr/bin/env python3
"""End-to-end benchmark: each model generates its own expansions AND reranks.

This is the DEFAULT benchmarking mode to avoid unfair comparisons where
one model's expansions are used to evaluate another model's reranker.

Usage:
    uv run python tools/bench_e2e.py \\
        --model-path ~/models/Qwen3.5-9B-Q4_K_M.gguf --label qwen35-9b
    uv run python tools/bench_e2e.py \\
        --model-path ~/models/gemma-4-E4B-it-Q8_0.gguf --label gemma-e4b

Per-model corpora are cached at eval/corpora/enriched_{basic,workflow}_{label}/
so repeated benchmarks reuse expansions without regeneration.
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

from cuecard.eval.harness import (
    EvalSummary,
    evaluate_per_event,
    format_per_event_report,
    load_fixtures,
    run_eval,
)
from cuecard.indexing.expander import expand_rules
from cuecard.indexing.indexer import save_rules_json
from cuecard.indexing.parser import parse_rules

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
CORPORA_DIR = EVAL_DIR / "corpora"
RESULTS_DIR = EVAL_DIR / "results"

SOURCE_FILES = {
    "basic": {
        "rules_txt": CORPORA_DIR / "rules_global.txt",
        "event_type": "PreToolUse",
        "fixtures": EVAL_DIR / "fixtures" / "basic.json",
    },
    "workflow": {
        "rules_txt": CORPORA_DIR / "rules_global.txt",
        "event_type": "UserPromptSubmit",
        "fixtures": EVAL_DIR / "fixtures" / "workflow.json",
    },
    "post_tool_use": {
        "rules_txt": CORPORA_DIR / "rules_global.txt",
        "event_type": "PostToolUse",
        "fixtures": EVAL_DIR / "fixtures" / "post_tool_use.json",
    },
    "stop": {
        "rules_txt": CORPORA_DIR / "rules_global.txt",
        "event_type": "Stop",
        "fixtures": EVAL_DIR / "fixtures" / "stop.json",
    },
    "subagent_start": {
        "rules_txt": CORPORA_DIR / "rules_global.txt",
        "event_type": "SubagentStart",
        "fixtures": EVAL_DIR / "fixtures" / "subagent_start.json",
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
        "-c", "98304",
        "--jinja",
        "-np", "5",
        "--reasoning", "off",
    ]
    print(f"Starting llama-server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for i in range(180):
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                print(f"Server ready after {i + 1}s")
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(1)

    proc.kill()
    msg = "Server failed to start within 180s"
    raise RuntimeError(msg)


def kill_server(proc: subprocess.Popen[bytes]) -> None:
    """Kill llama-server gracefully."""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Server stopped")


def ping_server() -> float:
    """PING test — trivial completion. Returns round-trip in seconds."""
    t0 = time.monotonic()
    r = requests.post(
        f"{ENDPOINT}/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 5,
            "temperature": 0.0,
        },
        timeout=30,
    )
    r.raise_for_status()
    elapsed = time.monotonic() - t0
    content = r.json()["choices"][0]["message"]["content"][:30]
    print(f"PING test: {elapsed:.2f}s — response: {content}")
    return elapsed


def generate_expansions_for_label(label: str, *, force: bool = False) -> None:
    """Generate expansions using the currently running model.

    Creates eval/corpora/enriched_{tier}_{label}/rules.json for each tier.
    Includes inline affinity from eval/corpora/affinity.json if available.
    Skips generation if corpus already exists and --force is not set.
    """
    from cuecard.retrieval.affinity import load_affinity

    # Load affinity once to embed inline in each rules.json
    affinity = load_affinity(str(CORPORA_DIR))

    for tier, cfg in SOURCE_FILES.items():
        out_dir = CORPORA_DIR / f"enriched_{tier}_{label}"
        rules_json = out_dir / "rules.json"

        if rules_json.exists() and not force:
            data = json.loads(rules_json.read_text())
            n_rules = len(data.get("rules", []))
            print(f"  {tier}: reusing existing corpus at {out_dir} ({n_rules} rules)")
            continue

        event_type = cfg["event_type"]
        print(f"  {tier}: generating expansions ({event_type})...")
        t0 = time.monotonic()
        rules = parse_rules((str(cfg["rules_txt"]),))
        expanded = expand_rules(
            rules,
            backend="local",
            endpoint=ENDPOINT,
            event_type=cfg["event_type"],
        )
        elapsed = time.monotonic() - t0
        total = sum(len(r.expansions) for r in expanded)
        avg = total / len(expanded) if expanded else 0
        print(
            f"  {tier}: {len(expanded)} rules, {total} expansions "
            f"(avg {avg:.1f}) in {elapsed:.1f}s"
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        save_rules_json(expanded, str(out_dir), affinity=affinity)


def summary_to_dict(summary: EvalSummary) -> dict:
    """Convert EvalSummary to serializable dict (drop per_fixture for brevity)."""
    d = asdict(summary)
    del d["per_fixture"]
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = round(v, 4)
    for tier in d.get("per_tier", []):
        for k, v in tier.items():
            if isinstance(v, float):
                tier[k] = round(v, 4)
    return d


def run_benchmark(label: str, *, sample_ratio: float = 0.2, seed: int = 42) -> dict:
    """Run basic + workflow eval against model-specific corpora."""
    from fastembed import TextEmbedding

    from cuecard.retrieval.affinity import load_affinity
    from cuecard.indexing.indexer import load_rules_json

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

    # Load affinity: prefer inline from rules.json, fall back to sidecar
    affinity = None
    first_corpus = CORPORA_DIR / f"enriched_basic_{label}" / "rules.json"
    if first_corpus.exists():
        result = load_rules_json(str(first_corpus.parent))
        if result is not None:
            _rules, affinity = result
    if affinity is None:
        affinity = load_affinity(str(CORPORA_DIR))
    if affinity:
        print(f"Loaded affinity index: {affinity.mode}, {len(affinity.items)} entries")
    else:
        print("No affinity index found — event mask disabled")

    results = {}
    for tier, cfg in SOURCE_FILES.items():
        fixture_path = cfg["fixtures"]
        if not fixture_path.exists():
            print(f"  Skipping {tier}: fixture file not found")
            continue

        corpus_path = str(
            CORPORA_DIR / f"enriched_{tier}_{label}" / "rules.json"
        )
        print(f"\n--- {tier} (corpus: enriched_{tier}_{label}) ---")
        fixtures = load_fixtures(str(fixture_path))
        print(f"  Loaded {len(fixtures)} fixtures")

        summary = run_eval(
            fixtures,
            str(fixture_path.parent),
            EMBEDDING_MODEL,
            model=embedding_model,
            mode="llm-local",
            corpus_override=(corpus_path,),
            sample_ratio=sample_ratio,
            seed=seed,
            affinity=affinity,
        )

        results[tier] = summary_to_dict(summary)
        print(
            f"  F2={summary.mean_quality:.3f}  "
            f"recall={summary.mean_recall:.3f}  "
            f"noise={summary.mean_noise_ratio:.3f}  "
            f"neg_sil={summary.negative_silence_rate:.3f}  "
            f"p50={summary.latency_p50_ms:.0f}ms"
        )

        # Per-event breakdown
        per_event = evaluate_per_event(summary.per_fixture, fixtures)
        if per_event:
            print(format_per_event_report(per_event))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2E benchmark: model generates its own expansions AND reranks",
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to GGUF model file",
    )
    parser.add_argument(
        "--label", required=True,
        help="Label for corpus+results (e.g. qwen35-9b)",
    )
    parser.add_argument(
        "--ngl", type=int, default=99, help="GPU layers (0 for CPU-only)",
    )
    parser.add_argument(
        "--no-server", action="store_true",
        help="Don't start server (use existing)",
    )
    parser.add_argument(
        "--sample-ratio", type=float, default=0.2,
        help="Fraction of fixtures",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified sampling",
    )
    parser.add_argument(
        "--force-expand", action="store_true",
        help="Regenerate expansions even if cached",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path for results JSON",
    )
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Model not found: {args.model_path}")
        sys.exit(1)

    import re
    if not re.match(r"^[a-zA-Z0-9_\-]+$", args.label):
        print(f"Invalid label: {args.label!r} — only alphanumeric, dash, underscore")
        sys.exit(1)

    os.environ.setdefault("CUECARD_LLM_ENDPOINT", ENDPOINT)

    proc = None
    if not args.no_server:
        proc = start_server(args.model_path, ngl=args.ngl)

    try:
        # PING test first — rule: benchmark single call before pipeline
        ping_seconds = ping_server()
        if ping_seconds > 10:
            print(
                f"WARNING: PING took {ping_seconds:.1f}s — endpoint may be slow. "
                "Proceeding anyway.",
            )

        print(f"\n=== Phase 1: Generate expansions for {args.label} ===")
        generate_expansions_for_label(args.label, force=args.force_expand)

        print(f"\n=== Phase 2: Benchmark {args.label} ===")
        results = run_benchmark(
            args.label,
            sample_ratio=args.sample_ratio,
            seed=args.seed,
        )

        if args.out:
            out_path = Path(args.out)
        else:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = RESULTS_DIR / f"{args.label}-e2e-seed{args.seed}.json"
        payload = {
            "label": args.label,
            "seed": args.seed,
            "sample_ratio": args.sample_ratio,
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {out_path}")

    finally:
        if proc is not None:
            kill_server(proc)


if __name__ == "__main__":
    main()
