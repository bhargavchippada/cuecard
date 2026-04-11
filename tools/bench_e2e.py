#!/usr/bin/env python3
"""End-to-end benchmark with full per-stage tracing.

Each model generates its own expansions AND reranks. The traced eval loop
captures every fixture's pipeline stages, LLM prompts, raw responses,
reasoning, and selected rules, writing them to disk so a run can be
debugged without re-invoking the LLM.

Usage:
    uv run python tools/bench_e2e.py \\
        --model-path ~/models/gemma-4-E4B-it-Q8_0.gguf --label gemma-e4b-s32 \\
        --no-server --sample-ratio 0.20 --seed 42

Per-model corpora are cached at eval/corpora/enriched_{label}/ so repeat
runs reuse expansions without regeneration. Each rule's expansion style
is chosen by its affinity (tool_use → tool-style, workflow → workflow-style,
both → both styles merged). One unified corpus is shared across all tiers,
matching production behaviour.

Artifacts (written to eval/results/{label}-traced-seed{seed}/ by default):
    report.md                   Human-readable summary with gap analysis
    summary.json                Machine-readable metrics per tier
    config.json                 Run configuration (label, seed, ratio, tiers)
    traces/{tier}/{id}.json     Full per-fixture trace
    llm_calls.jsonl             Flat log of every LLM interaction
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuecard.eval.harness import (
    EvalSummary,
    Fixture,
    FixtureResult,
    _build_summary,  # noqa: PLC2701
    evaluate_per_event,
    format_per_event_report,
    load_fixtures,
)
from cuecard.eval.metrics import (
    anti_precision,
    context_waste_ratio,
    mrr,
    ndcg_at_k,
    noise_ratio,
    precision_at_k,
    quality_score,
    recall_at_k,
)
from cuecard.indexing.expander import expand_rules
from cuecard.indexing.indexer import build_index, save_rules_json
from cuecard.indexing.parser import parse_rules
from cuecard.models import MAX_TOOL_NAME_LENGTH, PipelineConfig, ResolvedConfig
from cuecard.retrieval import llm_reranker as _llm_rr
from cuecard.retrieval import pipeline as _pipeline_mod
from cuecard.retrieval.pipeline import run_pipeline

if TYPE_CHECKING:
    from cuecard.models import PipelineResult, RankedResult

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
CORPORA_DIR = EVAL_DIR / "corpora"
RESULTS_DIR = EVAL_DIR / "results"

SOURCE_FILES: dict[str, dict[str, Any]] = {
    "pre_tool_use": {
        "rules_txt": CORPORA_DIR / "rules_global.txt",
        "event_type": "PreToolUse",
        "fixtures": EVAL_DIR / "fixtures" / "pre_tool_use.json",
    },
    "user_prompt_submit": {
        "rules_txt": CORPORA_DIR / "rules_global.txt",
        "event_type": "UserPromptSubmit",
        "fixtures": EVAL_DIR / "fixtures" / "user_prompt_submit.json",
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

MAX_WORKERS = 5  # match llama-server -np 5

# Filename sanitizer for fixture IDs
_UNSAFE_FILENAME_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_filename(text: str, max_len: int = 80) -> str:
    """Sanitize arbitrary text for use as a filename."""
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", text).strip("_")
    return cleaned[:max_len] or "fixture"


def _extract_tool_name(event: str, query: str) -> str:
    """Recover tool_name from benchmark fixtures for PreToolUse traces."""
    if event != "PreToolUse":
        return ""
    prefix, sep, _rest = query.partition(":")
    if not sep:
        return ""
    return prefix.strip()[:MAX_TOOL_NAME_LENGTH]


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Affinity + expansion generation (unchanged)
# ---------------------------------------------------------------------------


def _infer_affinity_once() -> tuple[list[Any], object]:
    """Infer affinity once for all tiers. Returns (rules, AffinityIndex)."""
    from cuecard.retrieval.affinity import infer_affinities

    print("  Inferring affinity from rules via LLM (parallel)...")
    first_rules_txt = next(iter(SOURCE_FILES.values()))["rules_txt"]
    rules = parse_rules((str(first_rules_txt),))
    aff_config = ResolvedConfig(
        source_paths=(), global_source_paths=(),
        project_source_paths=(), global_cache_dir="",
        pipeline=PipelineConfig(mode="llm-local"),
    )
    t0 = time.monotonic()
    affinity = infer_affinities(rules, aff_config, max_workers=MAX_WORKERS)
    elapsed = time.monotonic() - t0
    n = len(affinity.items)
    print(f"  Inferred affinity: {affinity.mode}, {n} entries in {elapsed:.1f}s")
    return rules, affinity


def generate_expansions_for_label(
    label: str, affinity: object, *, force: bool = False,
) -> None:
    """Generate (or reuse) per-model corpus with affinity-aware expansions."""
    out_dir = CORPORA_DIR / f"enriched_{label}"
    rules_json = out_dir / "rules.json"

    if rules_json.exists() and not force:
        data = json.loads(rules_json.read_text())
        n_rules = len(data.get("rules", []))
        print(f"  Reusing existing corpus at {out_dir} ({n_rules} rules)")
        return

    print("  Generating affinity-aware expansions (parallel)...")
    t0 = time.monotonic()
    first_rules_txt = next(iter(SOURCE_FILES.values()))["rules_txt"]
    rules = parse_rules((str(first_rules_txt),))
    expanded = expand_rules(
        rules,
        backend="local",
        endpoint=ENDPOINT,
        haiku_model="claude-haiku-4-5",
        event_type="PreToolUse",
        dedup_threshold=0.80,
        affinity=affinity,
        max_workers=MAX_WORKERS,
        max_per_rule=8,
    )
    elapsed = time.monotonic() - t0
    total = sum(len(r.expansions) for r in expanded)
    avg = total / len(expanded) if expanded else 0
    print(
        f"  {len(expanded)} rules, {total} expansions "
        f"(avg {avg:.1f}) in {elapsed:.1f}s",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_rules_json(expanded, str(out_dir), affinity=affinity)


def _apply_suffix(label: str, suffix: str | None) -> str:
    """Append a validated suffix to a label for corpus/result names."""
    if not suffix:
        return label
    return f"{label}-{suffix}"


# ---------------------------------------------------------------------------
# Tracing infrastructure
# ---------------------------------------------------------------------------


class TraceCapture:
    """Per-fixture LLM-call trace, correlated via thread-local storage.

    The monkey-patched ``rerank_llm`` records the raw prompts/response/
    reasoning under the ``fixture_id`` stamped by the calling thread before
    ``run_pipeline``. Safe for parallel eval workers.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._traces: dict[str, dict[str, Any]] = {}

    def set_fixture(self, fixture_id: str) -> None:
        self._local.fixture_id = fixture_id

    def clear_fixture(self) -> None:
        self._local.fixture_id = None

    def current(self) -> str | None:
        return getattr(self._local, "fixture_id", None)

    def record(self, **data: Any) -> None:
        fid = self.current()
        if fid is None:
            return
        with self._lock:
            existing = self._traces.get(fid, {})
            existing.update(data)
            self._traces[fid] = existing

    def pop(self, fixture_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._traces.pop(fixture_id, None)


_TRACE = TraceCapture()


def _traced_rerank_llm(
    candidates: list[RankedResult],
    query: str,
    *,
    backend: str,
    endpoint: str,
    haiku_model: str,
    thinking: bool = False,
    top_k: int,
) -> list[RankedResult]:
    """Instrumented replacement for ``llm_reranker.rerank_llm``.

    Behaviourally identical to the original (same fallback semantics, same
    single retry) but records stage-1 candidates, user prompt, raw response,
    parsed reasoning and selected indices on every call.
    """
    from cuecard.retrieval.llm_utils import (
        call_haiku,
        call_local,
        validate_endpoint,
    )
    from cuecard.security import ConfigError

    if not candidates:
        _TRACE.record(
            stage1_candidates=[],
            raw_response=None,
            reasoning=None,
            selected_indices=None,
            prompt_user=None,
            attempts=0,
            error="no_candidates",
        )
        return []

    if backend not in ("local", "haiku"):
        msg = f"Invalid backend: {backend!r}, expected 'local' or 'haiku'"
        raise ValueError(msg)

    stage1_dump = [
        {
            "idx": i + 1,
            "text": c.rule.text,
            "stage1_score": round(float(c.score), 6),
        }
        for i, c in enumerate(candidates)
    ]
    fallback = candidates[:top_k]

    try:
        if backend == "local":
            validate_endpoint(endpoint)

        nonce = secrets.token_hex(6)
        system_prompt, user_prompt = _llm_rr._build_prompt(
            candidates, query, nonce,
        )

        raw: str | None = None
        parsed = None
        for attempt in range(2):
            if backend == "local":
                raw = call_local(
                    system_prompt, user_prompt, endpoint, thinking,
                    stop=None,
                )
            else:
                raw = call_haiku(system_prompt, user_prompt, haiku_model)
            parsed = _llm_rr._parse_llm_response(raw, len(candidates))
            if parsed.indices is not None:
                break
            if attempt == 0:
                logger.info("LLM response unparseable, retrying once")

        if parsed is None or parsed.indices is None:
            _TRACE.record(
                stage1_candidates=stage1_dump,
                raw_response=raw,
                reasoning=parsed.reasoning if parsed else None,
                selected_indices=None,
                prompt_user=user_prompt,
                attempts=2,
                error="unparseable_after_retry",
            )
            return fallback

        _TRACE.record(
            stage1_candidates=stage1_dump,
            raw_response=raw,
            reasoning=parsed.reasoning,
            selected_indices=list(parsed.indices),
            prompt_user=user_prompt,
            attempts=1 if parsed.indices is not None else 2,
            error=None,
        )
        return _llm_rr._compute_ordinal_scores(
            parsed.indices, candidates,
        )[:top_k]

    except (ConfigError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001
        _TRACE.record(
            stage1_candidates=stage1_dump,
            raw_response=None,
            reasoning=None,
            selected_indices=None,
            prompt_user=None,
            attempts=0,
            error=f"{type(exc).__name__}: {exc}",
        )
        logger.warning("Traced LLM re-rank failed; returning fallback")
        return fallback


def _traced_run_retrieval_stage(
    query: str,
    index: Any,
    config: ResolvedConfig,
    effective_mode: str,
    embedding_model: Any,
    *,
    mask: Any = None,
    query_expansions: tuple[str, ...] = (),
) -> tuple[list[Any], Any]:
    """Instrumented retrieval stage capturing dense/sparse/fused sets."""
    from cuecard.models import RankedResult
    from cuecard.retrieval.dense import DenseRetriever
    from cuecard.retrieval.fusion import fuse

    top_k, threshold = _pipeline_mod._retrieval_params(
        effective_mode, config,
    )
    input_count = index.size
    t0 = time.monotonic()

    queries = (query, *query_expansions)

    dense = DenseRetriever(
        model=embedding_model,
        dedup_threshold=config.dedup_threshold,
        max_query_length=config.query_max_length,
    )
    t_dense_start = time.monotonic()
    dense_sets: list[list[Any]] = []
    for q in queries:
        dense_sets.append(
            dense.retrieve(
                q, index, top_k=top_k, threshold=threshold, mask=mask,
            ),
        )
    dense_results = dense_sets[0] if dense_sets else []
    t_dense_ms = (time.monotonic() - t_dense_start) * 1000.0

    sparse_ran = config.sparse_enabled and index.bm25_corpus is not None
    sparse_sets: list[list[Any]] = []
    sparse_results: list[Any] = []
    t_sparse_ms = 0.0
    if sparse_ran:
        t_sparse_start = time.monotonic()
        for q in queries:
            s, _ = _pipeline_mod._run_sparse(q, index, top_k, mask)
            if s:
                sparse_sets.append(s)
        t_sparse_ms = (time.monotonic() - t_sparse_start) * 1000.0
        if sparse_sets:
            sparse_results = sparse_sets[0]

    raw_only_results: list[list[Any]] = []
    if dense_results:
        raw_only_results.append(dense_results)
    if sparse_results:
        raw_only_results.append(sparse_results)
    fused_raw_query = (
        fuse(raw_only_results, k=config.fusion_k, top_k=top_k)
        if len(raw_only_results) > 1
        else dense_results
    )

    all_results: list[list[Any]] = [s for s in dense_sets if s] + sparse_sets
    t_fusion_start = time.monotonic()
    fused = (
        fuse(all_results, k=config.fusion_k, top_k=top_k)
        if len(all_results) > 1
        else (all_results[0] if all_results else [])
    )
    t_fusion_ms = (time.monotonic() - t_fusion_start) * 1000.0

    results: list[RankedResult] = [
        RankedResult(rule=sc.rule, score=sc.score) for sc in fused
    ]

    def _dump(candidates: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "text": candidate.rule.text,
                "score": round(float(candidate.score), 6),
                "retriever": getattr(candidate, "retriever", ""),
            }
            for candidate in candidates
        ]

    _TRACE.record(
        retrieval_sets={
            "query_expansion_count": len(query_expansions),
            "dense_raw": _dump(dense_results),
            "sparse_raw": _dump(sparse_results),
            "fused_raw_query": _dump(fused_raw_query),
            "fused_raw": _dump(fused),
        },
    )

    return results, _pipeline_mod.RetrievalStageTrace(
        stage="retrieval",
        input_count=input_count,
        output_count=len(results),
        latency_ms=(time.monotonic() - t0) * 1000.0,
        retrievers=_pipeline_mod._build_retriever_traces(
            dense_results, sparse_results,
            t_dense_ms, t_sparse_ms, sparse_ran,
        ),
        fusion_latency_ms=t_fusion_ms,
    )


def _install_tracing() -> tuple[Any, Any]:
    """Monkey-patch LLM and retrieval tracing hooks; return originals."""
    orig_rerank = _llm_rr.rerank_llm
    orig_retrieval = _pipeline_mod._run_retrieval_stage
    _llm_rr.rerank_llm = _traced_rerank_llm
    _pipeline_mod._run_retrieval_stage = _traced_run_retrieval_stage
    return orig_rerank, orig_retrieval


def _uninstall_tracing(orig: tuple[Any, Any]) -> None:
    orig_rerank, orig_retrieval = orig
    _llm_rr.rerank_llm = orig_rerank
    _pipeline_mod._run_retrieval_stage = orig_retrieval


# ---------------------------------------------------------------------------
# Fixture-level trace helpers
# ---------------------------------------------------------------------------


def _stage_trace_to_dict(stage: Any) -> dict[str, Any]:
    """Serialize a StageTrace or RetrievalStageTrace to JSON-safe dict."""
    base = {
        "stage": stage.stage,
        "input_count": stage.input_count,
        "output_count": stage.output_count,
        "latency_ms": round(float(stage.latency_ms), 3),
        "error": stage.error,
    }
    retrievers = getattr(stage, "retrievers", None)
    if retrievers is not None:
        base["retrievers"] = [
            {
                "name": r.name,
                "candidate_count": r.candidate_count,
                "latency_ms": round(float(r.latency_ms), 3),
                "unique_rules": r.unique_rules,
            }
            for r in retrievers
        ]
        base["fusion_latency_ms"] = round(
            float(getattr(stage, "fusion_latency_ms", 0.0)), 3,
        )
    return base


def _count_rules_masked(
    index: Any,
    affinity: Any,
    event: str,
    tool_name: str | None = None,
) -> tuple[int, int, int]:
    """Count rules (not embedding rows) excluded by the event mask.

    Returns (total_rules, rules_included, rules_masked). The pipeline's
    own ``rules_masked`` field counts masked embedding rows, which
    over-reports after expansion fan-out (code-review finding #3 —
    2026-04-10). This helper uses the parent-rule list for a correct
    rule-level count.
    """
    if affinity is None or not event:
        total = len(index.rules)
        return total, total, 0

    total = 0
    included = 0
    for rule in index.rules:
        total += 1
        aff = affinity.get(rule) if hasattr(affinity, "get") else None
        if aff is None:
            included += 1  # default: show rule when no affinity known
            continue
        if (
            event in aff.events
            and (
                not tool_name
                or not aff.tools
                or tool_name in aff.tools
            )
        ):
            included += 1
    return total, included, total - included


def _count_expected_rules_visible(
    fx: Fixture,
    index: Any,
    affinity: Any,
    event: str,
    tool_name: str | None = None,
) -> tuple[int, int]:
    """Count expected rules that survive the event mask for a fixture."""
    expected = set(fx.should_match)
    if not expected:
        return 0, 0

    visible = 0
    for rule in index.rules:
        if rule.text not in expected:
            continue
        aff = affinity.get(rule) if hasattr(affinity, "get") else None
        if aff is None or (
            event in aff.events
            and (
                not tool_name
                or not aff.tools
                or tool_name in aff.tools
            )
        ):
            visible += 1
    return len(expected), visible


def _compute_fixture_metrics(
    retrieved_texts: list[str],
    *,
    relevant: set[str],
    anti_relevant: set[str],
    is_negative: bool,
) -> dict[str, float]:
    """Compute the benchmark metric bundle for one retrieved set."""
    return {
        "precision_at_k": precision_at_k(retrieved_texts, relevant),
        "recall_at_k": recall_at_k(retrieved_texts, relevant),
        "mrr": mrr(retrieved_texts, relevant),
        "ndcg_at_k": ndcg_at_k(retrieved_texts, relevant),
        "anti_precision": anti_precision(retrieved_texts, anti_relevant),
        "noise_ratio": noise_ratio(retrieved_texts, relevant),
        "context_waste_ratio": context_waste_ratio(retrieved_texts, relevant),
        "quality_score_f2": quality_score(
            retrieved_texts, relevant, is_negative,
        ),
    }


def _round_metric_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively round floats for JSON/report friendliness."""
    rounded: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, float):
            rounded[key] = round(value, 4)
        elif isinstance(value, dict):
            rounded[key] = _round_metric_dict(value)
        else:
            rounded[key] = value
    return rounded


def _build_stage_metrics(
    traces: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate stage quality and transition diagnostics for one tier.

    The benchmark already stores final metrics. This helper adds:
    - Stage 0 mask coverage / recall ceiling
    - Stage 1 retrieval quality from candidates shown to the LLM
    - Stage 3 final quality from the reranked output
    - Transition diagnostics that attribute failures between stages
    """
    n = len(traces)
    if n == 0:
        return {
            "stage0_mask": {},
            "stage1_dense": {},
            "stage1_sparse": {},
            "stage1_fused": {},
            "stage1_retrieval": {},
            "stage3_final": {},
            "transitions": {},
        }

    total_rules = [
        t["pipeline"]["total_rules"]
        for t in traces
        if t["pipeline"]["total_rules"] is not None
    ]
    rules_included = [
        t["pipeline"]["rules_included"]
        for t in traces
        if t["pipeline"]["rules_included"] is not None
    ]
    rules_masked = [
        t["pipeline"]["rules_masked"]
        for t in traces
        if t["pipeline"]["rules_masked"] is not None
    ]

    positives = [t for t in traces if t["difficulty"] != "negative"]
    negatives = [t for t in traces if t["difficulty"] == "negative"]

    stage1_dense_rows: list[FixtureResult] = []
    stage1_sparse_rows: list[FixtureResult] = []
    stage1_raw_query_rows: list[FixtureResult] = []
    stage1_fused_rows: list[FixtureResult] = []
    stage1_rows: list[FixtureResult] = []
    stage3_rows: list[FixtureResult] = []

    stage0_full = 0
    stage0_partial = 0
    stage0_zero = 0
    stage0_expected_rules = 0
    stage0_masked_expected_rules = 0

    stage1_full_recall = 0
    stage1_any_hit = 0
    stage1_neg_silent = 0

    llm_relevant_seen = 0
    llm_relevant_kept = 0
    llm_irrelevant_seen = 0
    llm_irrelevant_pruned = 0
    llm_empty_from_nonempty = 0
    llm_fallback_errors = 0
    sparse_ran_count = 0
    query_expansion_fixture_count = 0
    query_expansion_helped = 0
    query_expansion_hurt = 0
    query_expansion_net_new_hits = 0
    query_expansion_net_new_noise = 0

    for t in traces:
        relevant = set(t["should_match"])
        anti_relevant = set(t["should_not_match"])
        final_retrieved = list(t["final_retrieved"])
        retrieval = t.get("retrieval") or {}
        llm = t.get("llm") or {}
        dense_candidates = [
            candidate["text"]
            for candidate in (retrieval.get("dense_raw") or [])
        ]
        sparse_candidates = [
            candidate["text"]
            for candidate in (retrieval.get("sparse_raw") or [])
        ]
        fused_candidates = [
            candidate["text"]
            for candidate in (retrieval.get("fused_raw") or [])
        ]
        raw_query_candidates = [
            candidate["text"]
            for candidate in (retrieval.get("fused_raw_query") or [])
        ]
        stage1_candidates = [
            candidate["text"]
            for candidate in (llm.get("stage1_candidates") or [])
        ]
        is_negative = t["difficulty"] == "negative"
        if llm.get("error") is not None:
            llm_fallback_errors += 1
        if any(
            retr["name"] == "sparse"
            for s in t["pipeline"]["stages"]
            if s["stage"] == "retrieval"
            for retr in s.get("retrievers", [])
        ):
            sparse_ran_count += 1
        if retrieval.get("query_expansion_count", 0) > 0:
            query_expansion_fixture_count += 1
            raw_hits = len(set(raw_query_candidates) & relevant)
            fused_hits = len(set(fused_candidates) & relevant)
            raw_noise = sum(
                1 for text in raw_query_candidates if text not in relevant
            )
            fused_noise = sum(
                1 for text in fused_candidates if text not in relevant
            )
            if fused_hits > raw_hits:
                query_expansion_helped += 1
            elif fused_hits < raw_hits or (
                fused_hits == raw_hits and fused_noise > raw_noise
            ):
                query_expansion_hurt += 1
            query_expansion_net_new_hits += max(0, fused_hits - raw_hits)
            query_expansion_net_new_noise += max(0, fused_noise - raw_noise)

        stage1_dense_metric_values = _compute_fixture_metrics(
            dense_candidates,
            relevant=relevant,
            anti_relevant=anti_relevant,
            is_negative=is_negative,
        )
        stage1_sparse_metric_values = _compute_fixture_metrics(
            sparse_candidates,
            relevant=relevant,
            anti_relevant=anti_relevant,
            is_negative=is_negative,
        )
        stage1_fused_metric_values = _compute_fixture_metrics(
            fused_candidates,
            relevant=relevant,
            anti_relevant=anti_relevant,
            is_negative=is_negative,
        )
        stage1_raw_query_metric_values = _compute_fixture_metrics(
            raw_query_candidates,
            relevant=relevant,
            anti_relevant=anti_relevant,
            is_negative=is_negative,
        )
        stage1_metric_values = _compute_fixture_metrics(
            stage1_candidates,
            relevant=relevant,
            anti_relevant=anti_relevant,
            is_negative=is_negative,
        )
        stage3_metric_values = _compute_fixture_metrics(
            final_retrieved,
            relevant=relevant,
            anti_relevant=anti_relevant,
            is_negative=is_negative,
        )

        stage1_dense_rows.append(FixtureResult(
            fixture_id=t["fixture_id"],
            query=t["query"],
            difficulty=t["difficulty"],
            retrieved=tuple(dense_candidates),
            precision_at_k=stage1_dense_metric_values["precision_at_k"],
            recall_at_k=stage1_dense_metric_values["recall_at_k"],
            mrr=stage1_dense_metric_values["mrr"],
            ndcg_at_k=stage1_dense_metric_values["ndcg_at_k"],
            anti_precision=stage1_dense_metric_values["anti_precision"],
            noise_ratio=stage1_dense_metric_values["noise_ratio"],
            context_waste_ratio=stage1_dense_metric_values["context_waste_ratio"],
            retrieved_count=len(dense_candidates),
            latency_ms=next(
                (
                    retr["latency_ms"]
                    for s in t["pipeline"]["stages"]
                    if s["stage"] == "retrieval"
                    for retr in s.get("retrievers", [])
                    if retr["name"] == "dense"
                ),
                0.0,
            ),
            quality_score=stage1_dense_metric_values["quality_score_f2"],
        ))
        stage1_sparse_rows.append(FixtureResult(
            fixture_id=t["fixture_id"],
            query=t["query"],
            difficulty=t["difficulty"],
            retrieved=tuple(sparse_candidates),
            precision_at_k=stage1_sparse_metric_values["precision_at_k"],
            recall_at_k=stage1_sparse_metric_values["recall_at_k"],
            mrr=stage1_sparse_metric_values["mrr"],
            ndcg_at_k=stage1_sparse_metric_values["ndcg_at_k"],
            anti_precision=stage1_sparse_metric_values["anti_precision"],
            noise_ratio=stage1_sparse_metric_values["noise_ratio"],
            context_waste_ratio=stage1_sparse_metric_values["context_waste_ratio"],
            retrieved_count=len(sparse_candidates),
            latency_ms=next(
                (
                    retr["latency_ms"]
                    for s in t["pipeline"]["stages"]
                    if s["stage"] == "retrieval"
                    for retr in s.get("retrievers", [])
                    if retr["name"] == "sparse"
                ),
                0.0,
            ),
            quality_score=stage1_sparse_metric_values["quality_score_f2"],
        ))
        stage1_raw_query_rows.append(FixtureResult(
            fixture_id=t["fixture_id"],
            query=t["query"],
            difficulty=t["difficulty"],
            retrieved=tuple(raw_query_candidates),
            precision_at_k=stage1_raw_query_metric_values["precision_at_k"],
            recall_at_k=stage1_raw_query_metric_values["recall_at_k"],
            mrr=stage1_raw_query_metric_values["mrr"],
            ndcg_at_k=stage1_raw_query_metric_values["ndcg_at_k"],
            anti_precision=stage1_raw_query_metric_values["anti_precision"],
            noise_ratio=stage1_raw_query_metric_values["noise_ratio"],
            context_waste_ratio=stage1_raw_query_metric_values["context_waste_ratio"],
            retrieved_count=len(raw_query_candidates),
            latency_ms=next(
                (
                    s["latency_ms"]
                    for s in t["pipeline"]["stages"]
                    if s["stage"] == "retrieval"
                ),
                0.0,
            ),
            quality_score=stage1_raw_query_metric_values["quality_score_f2"],
        ))
        stage1_fused_rows.append(FixtureResult(
            fixture_id=t["fixture_id"],
            query=t["query"],
            difficulty=t["difficulty"],
            retrieved=tuple(fused_candidates),
            precision_at_k=stage1_fused_metric_values["precision_at_k"],
            recall_at_k=stage1_fused_metric_values["recall_at_k"],
            mrr=stage1_fused_metric_values["mrr"],
            ndcg_at_k=stage1_fused_metric_values["ndcg_at_k"],
            anti_precision=stage1_fused_metric_values["anti_precision"],
            noise_ratio=stage1_fused_metric_values["noise_ratio"],
            context_waste_ratio=stage1_fused_metric_values["context_waste_ratio"],
            retrieved_count=len(fused_candidates),
            latency_ms=next(
                (
                    s["latency_ms"]
                    for s in t["pipeline"]["stages"]
                    if s["stage"] == "retrieval"
                ),
                0.0,
            ),
            quality_score=stage1_fused_metric_values["quality_score_f2"],
        ))
        stage1_rows.append(FixtureResult(
            fixture_id=t["fixture_id"],
            query=t["query"],
            difficulty=t["difficulty"],
            retrieved=tuple(stage1_candidates),
            precision_at_k=stage1_metric_values["precision_at_k"],
            recall_at_k=stage1_metric_values["recall_at_k"],
            mrr=stage1_metric_values["mrr"],
            ndcg_at_k=stage1_metric_values["ndcg_at_k"],
            anti_precision=stage1_metric_values["anti_precision"],
            noise_ratio=stage1_metric_values["noise_ratio"],
            context_waste_ratio=stage1_metric_values["context_waste_ratio"],
            retrieved_count=len(stage1_candidates),
            latency_ms=next(
                (
                    s["latency_ms"]
                    for s in t["pipeline"]["stages"]
                    if s["stage"] == "retrieval"
                ),
                0.0,
            ),
            quality_score=stage1_metric_values["quality_score_f2"],
        ))
        stage3_rows.append(FixtureResult(
            fixture_id=t["fixture_id"],
            query=t["query"],
            difficulty=t["difficulty"],
            retrieved=tuple(final_retrieved),
            precision_at_k=stage3_metric_values["precision_at_k"],
            recall_at_k=stage3_metric_values["recall_at_k"],
            mrr=stage3_metric_values["mrr"],
            ndcg_at_k=stage3_metric_values["ndcg_at_k"],
            anti_precision=stage3_metric_values["anti_precision"],
            noise_ratio=stage3_metric_values["noise_ratio"],
            context_waste_ratio=stage3_metric_values["context_waste_ratio"],
            retrieved_count=len(final_retrieved),
            latency_ms=t["metrics"]["latency_ms"],
            quality_score=stage3_metric_values["quality_score_f2"],
        ))

        if is_negative:
            if not stage1_candidates:
                stage1_neg_silent += 1
            continue

        expected_rule_count = t["pipeline"].get("expected_rules")
        visible_expected = t["pipeline"].get("expected_rules_visible")
        if expected_rule_count is None:
            expected_rule_count = len(relevant)
        if visible_expected is None:
            visible_expected = expected_rule_count

        stage0_expected_rules += expected_rule_count
        stage0_masked_here = expected_rule_count - visible_expected
        stage0_masked_expected_rules += stage0_masked_here
        if visible_expected == expected_rule_count:
            stage0_full += 1
        elif visible_expected == 0:
            stage0_zero += 1
        else:
            stage0_partial += 1

        stage1_hits = set(stage1_candidates) & relevant
        final_hits = set(final_retrieved) & relevant
        if stage1_hits:
            stage1_any_hit += 1
        if len(stage1_hits) == len(relevant):
            stage1_full_recall += 1

        llm_relevant_seen += len(stage1_hits)
        llm_relevant_kept += len(final_hits & stage1_hits)

        stage1_irrelevant = [text for text in stage1_candidates if text not in relevant]
        llm_irrelevant_seen += len(stage1_irrelevant)
        llm_irrelevant_pruned += sum(
            1 for text in stage1_irrelevant if text not in final_retrieved
        )

        if stage1_candidates and not final_retrieved:
            llm_empty_from_nonempty += 1

    stage0_mask = {
        "mean_total_rules": (
            sum(total_rules) / len(total_rules) if total_rules else 0.0
        ),
        "mean_rules_included": (
            sum(rules_included) / len(rules_included) if rules_included else 0.0
        ),
        "mean_rules_masked": (
            sum(rules_masked) / len(rules_masked) if rules_masked else 0.0
        ),
        "rule_visibility_rate": (
            (sum(rules_included) / sum(total_rules)) if total_rules else 0.0
        ),
        "positive_full_recall_ceiling_rate": (
            stage0_full / len(positives) if positives else 0.0
        ),
        "positive_partial_recall_ceiling_rate": (
            stage0_partial / len(positives) if positives else 0.0
        ),
        "positive_zero_recall_ceiling_rate": (
            stage0_zero / len(positives) if positives else 0.0
        ),
        "expected_rule_mask_rate": (
            stage0_masked_expected_rules / stage0_expected_rules
            if stage0_expected_rules
            else 0.0
        ),
    }

    transitions = {
        "stage1_positive_any_hit_rate": (
            stage1_any_hit / len(positives) if positives else 0.0
        ),
        "stage1_positive_full_recall_rate": (
            stage1_full_recall / len(positives) if positives else 0.0
        ),
        "stage1_negative_silence_rate": (
            stage1_neg_silent / len(negatives) if negatives else 1.0
        ),
        "sparse_ran_rate": sparse_ran_count / n,
        "query_expansion_fixture_rate": (
            query_expansion_fixture_count / n
        ),
        "query_expansion_help_rate": (
            query_expansion_helped / query_expansion_fixture_count
            if query_expansion_fixture_count
            else 0.0
        ),
        "query_expansion_hurt_rate": (
            query_expansion_hurt / query_expansion_fixture_count
            if query_expansion_fixture_count
            else 0.0
        ),
        "query_expansion_avg_new_hits": (
            query_expansion_net_new_hits / query_expansion_fixture_count
            if query_expansion_fixture_count
            else 0.0
        ),
        "query_expansion_avg_new_noise": (
            query_expansion_net_new_noise / query_expansion_fixture_count
            if query_expansion_fixture_count
            else 0.0
        ),
        "llm_relevant_keep_rate": (
            llm_relevant_kept / llm_relevant_seen if llm_relevant_seen else 1.0
        ),
        "llm_irrelevant_prune_rate": (
            llm_irrelevant_pruned / llm_irrelevant_seen
            if llm_irrelevant_seen
            else 1.0
        ),
        "llm_abstain_from_nonempty_rate": (
            llm_empty_from_nonempty / len(positives) if positives else 0.0
        ),
        "llm_error_rate": llm_fallback_errors / n,
    }

    return {
        "stage0_mask": _round_metric_dict(stage0_mask),
        "stage1_dense": _summary_to_dict(_build_summary(stage1_dense_rows)),
        "stage1_sparse": _summary_to_dict(_build_summary(stage1_sparse_rows)),
        "stage1_raw_query": _summary_to_dict(_build_summary(stage1_raw_query_rows)),
        "stage1_fused": _summary_to_dict(_build_summary(stage1_fused_rows)),
        "stage1_retrieval": _summary_to_dict(_build_summary(stage1_rows)),
        "stage3_final": _summary_to_dict(_build_summary(stage3_rows)),
        "transitions": _round_metric_dict(transitions),
    }


def _build_fixture_trace(
    tier: str,
    tool_name: str,
    fx: Fixture,
    pr: PipelineResult,
    fr: FixtureResult,
    llm_capture: dict[str, Any] | None = None,
    *,
    mask_stats: tuple[int, int, int] | None = None,
    expected_mask_stats: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Assemble the per-fixture trace record."""
    relevant = set(fx.should_match)
    retrieved_texts = list(fr.retrieved)
    retrieved_set = set(retrieved_texts)

    hits = sorted(relevant & retrieved_set)
    misses = sorted(relevant - retrieved_set)
    extras = [r for r in retrieved_texts if r not in relevant]

    classification: str
    if fx.difficulty == "negative":
        classification = "tn" if not retrieved_texts else "fp"
    elif not relevant:
        classification = "unknown"
    elif not misses and not extras:
        classification = "tp_exact"
    elif not misses:
        classification = "tp_noisy"
    elif hits:
        classification = "partial"
    else:
        classification = "fn"

    llm_block: dict[str, Any] | None = None
    if llm_capture is not None:
        llm_block = {
            "stage1_candidates": llm_capture.get("stage1_candidates"),
            "prompt_user": llm_capture.get("prompt_user"),
            "raw_response": llm_capture.get("raw_response"),
            "reasoning": llm_capture.get("reasoning"),
            "selected_indices": llm_capture.get("selected_indices"),
            "attempts": llm_capture.get("attempts"),
            "error": llm_capture.get("error"),
        }

    retrieval_block = None
    if llm_capture is not None:
        retrieval_block = llm_capture.get("retrieval_sets")

    return {
        "tier": tier,
        "fixture_id": fx.id,
        "event": fx.event,
        "tool_name": tool_name,
        "difficulty": fx.difficulty,
        "query": fx.query,
        "should_match": list(fx.should_match),
        "should_not_match": list(fx.should_not_match),
        "classification": classification,
        "pipeline": {
            "mode": pr.mode,
            "event": pr.event,
            "event_mask_applied": pr.event_mask_applied,
            # NOTE: PipelineResult.rules_masked counts MASKED EMBEDDING ROWS,
            # not masked rules (see code-review-2026-04-10.md finding #3).
            # Kept under its original name for back-compat; real rule counts
            # exposed below.
            "embeddings_masked": pr.rules_masked,
            "total_rules": mask_stats[0] if mask_stats else None,
            "rules_included": mask_stats[1] if mask_stats else None,
            "rules_masked": mask_stats[2] if mask_stats else None,
            "expected_rules": (
                expected_mask_stats[0] if expected_mask_stats else None
            ),
            "expected_rules_visible": (
                expected_mask_stats[1] if expected_mask_stats else None
            ),
            "expected_rules_masked": (
                expected_mask_stats[0] - expected_mask_stats[1]
                if expected_mask_stats
                else None
            ),
            "stages": [_stage_trace_to_dict(s) for s in pr.stages],
        },
        "retrieval": retrieval_block,
        "llm": llm_block,
        "final_retrieved": retrieved_texts,
        "hits": hits,
        "misses": misses,
        "extras": extras,
        "metrics": {
            **_round_metric_dict(_compute_fixture_metrics(
                retrieved_texts,
                relevant=relevant,
                anti_relevant=set(fx.should_not_match),
                is_negative=fx.difficulty == "negative",
            )),
            "latency_ms": round(fr.latency_ms, 2),
        },
    }


# ---------------------------------------------------------------------------
# Sampling + eval loop
# ---------------------------------------------------------------------------


def _stratified_sample(
    fixtures: list[Fixture], sample_ratio: float, seed: int,
) -> list[Fixture]:
    """Stratified sampling preserving tier distribution."""
    rng = random.Random(seed)
    by_tier: dict[str, list[Fixture]] = {}
    for fx in fixtures:
        by_tier.setdefault(fx.difficulty, []).append(fx)
    sampled: list[Fixture] = []
    for tier_fixtures in by_tier.values():
        n = max(1, int(len(tier_fixtures) * sample_ratio))
        sampled.extend(rng.sample(tier_fixtures, min(n, len(tier_fixtures))))
    rng.shuffle(sampled)
    return sampled


def _make_eval_config(llm_candidates: int | None) -> ResolvedConfig:
    """ResolvedConfig tailored for benchmark eval."""
    kwargs: dict[str, Any] = {
        "source_paths": (),
        "global_source_paths": (),
        "project_source_paths": (),
        "global_cache_dir": "",
        "top_k": 7,
        "threshold": 0.30,
        "dedup_threshold": 0.95,
        "query_max_length": 500,
        "sparse_enabled": False,
    }
    if llm_candidates is not None:
        kwargs["llm_candidates"] = llm_candidates
    return ResolvedConfig(**kwargs)


def _summary_to_dict(summary: EvalSummary) -> dict[str, Any]:
    """Convert EvalSummary to serialisable dict (omit per_fixture for brevity)."""
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


# ---------------------------------------------------------------------------
# Traced benchmark
# ---------------------------------------------------------------------------


def run_benchmark_traced(
    label: str,
    affinity: object,
    *,
    sample_ratio: float,
    seed: int,
    tiers: tuple[str, ...] | None,
    llm_candidates: int | None,
    query_expansion_enabled: bool,
    artifacts_dir: Path,
) -> dict[str, Any]:
    """Run traced benchmark: replaces legacy run_benchmark.

    Writes per-fixture traces, llm_calls.jsonl, summary.json, config.json,
    and report.md into artifacts_dir. Returns the per-tier results dict.
    """
    from fastembed import TextEmbedding

    n_affinity = len(affinity.items)  # type: ignore[attr-defined]
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print(
        f"Using pre-inferred affinity: "
        f"{affinity.mode}, {n_affinity} entries",  # type: ignore[attr-defined]
    )
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

    corpus_path = str(CORPORA_DIR / f"enriched_{label}" / "rules.json")
    if not Path(corpus_path).exists():
        msg = f"Corpus not found: {corpus_path}"
        raise FileNotFoundError(msg)

    eval_config = _make_eval_config(llm_candidates)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    traces_root = artifacts_dir / "traces"
    traces_root.mkdir(exist_ok=True)
    llm_log_path = artifacts_dir / "llm_calls.jsonl"
    llm_log_path.unlink(missing_ok=True)

    # Shared corpus: build index once
    rules = tuple(parse_rules((corpus_path,)))
    index = build_index(rules, {}, EMBEDDING_MODEL, model=embedding_model)
    print(
        f"  Built shared index: {len(rules)} rules, "
        f"{index.embeddings.shape[0]} embeddings "
        f"(rules + expansions)",
    )

    orig_rerank = _install_tracing()
    results_by_tier: dict[str, Any] = {}
    all_fixture_traces: list[dict[str, Any]] = []

    try:
        for tier, cfg in SOURCE_FILES.items():
            if tiers is not None and tier not in tiers:
                continue
            fixture_path = cfg["fixtures"]
            if not fixture_path.exists():
                print(f"  Skipping {tier}: fixture file not found")
                continue

            print(f"\n--- {tier} ---")
            fixtures = load_fixtures(str(fixture_path))
            if sample_ratio < 1.0:
                fixtures = _stratified_sample(fixtures, sample_ratio, seed)
            print(f"  Evaluating {len(fixtures)} fixtures")

            tier_trace_dir = traces_root / tier
            tier_trace_dir.mkdir(exist_ok=True)

            tier_event = cfg["event_type"]
            mask_stats = _count_rules_masked(index, affinity, tier_event)
            print(
                f"  Event mask: {mask_stats[1]}/{mask_stats[0]} rules "
                f"visible for {tier_event} "
                f"({mask_stats[2]} masked)",
            )

            tier_results, tier_traces, tier_llm_log = _eval_tier(
                fixtures=fixtures,
                index=index,
                embedding_model=embedding_model,
                eval_config=eval_config,
                affinity=affinity,
                query_expansion_enabled=query_expansion_enabled,
                tier=tier,
                tier_trace_dir=tier_trace_dir,
            )

            all_fixture_traces.extend(tier_traces)

            summary = _build_summary(tier_results)
            per_event = evaluate_per_event(
                list(summary.per_fixture), fixtures,
            )
            print(
                f"  F2={summary.mean_quality:.3f}  "
                f"PosRecall={summary.positive_recall:.3f}  "
                f"Noise={summary.mean_noise_ratio:.3f}  "
                f"NegSil={summary.negative_silence_rate:.3f}  "
                f"p50={summary.latency_p50_ms:.0f}ms",
            )
            if per_event:
                print(format_per_event_report(per_event))

            stage_metrics = _build_stage_metrics(tier_traces)
            results_by_tier[tier] = {
                "summary": _summary_to_dict(summary),
                "stage_metrics": stage_metrics,
                "per_event": [asdict(em) for em in per_event],
                "n_fixtures_sampled": len(fixtures),
            }

            # Append tier LLM calls to flat log
            with open(llm_log_path, "a") as fp:
                for entry in tier_llm_log:
                    fp.write(json.dumps(entry, default=str) + "\n")

    finally:
        _uninstall_tracing(orig_rerank)

    # Write top-level artifacts
    config_payload = {
        "label": label,
        "seed": seed,
        "sample_ratio": sample_ratio,
        "tiers": list(results_by_tier.keys()),
        "llm_candidates": llm_candidates,
        "query_expansion_enabled": query_expansion_enabled,
        "corpus_path": corpus_path,
        "embedding_model": EMBEDDING_MODEL,
        "llm_endpoint": ENDPOINT,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (artifacts_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2),
    )

    summary_payload = {
        "label": label,
        "seed": seed,
        "sample_ratio": sample_ratio,
        "results": results_by_tier,
    }
    (artifacts_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
    )

    report_md = _build_report_md(
        label=label,
        config=config_payload,
        results_by_tier=results_by_tier,
        all_fixture_traces=all_fixture_traces,
    )
    (artifacts_dir / "report.md").write_text(report_md)

    print(f"\nArtifacts written to {artifacts_dir}")
    print("  report.md       — human summary")
    print("  summary.json    — machine metrics")
    print("  config.json     — run config")
    print("  traces/         — per-fixture JSON (all tiers)")
    print("  llm_calls.jsonl — flat LLM interaction log")

    return results_by_tier


def _eval_tier(
    *,
    fixtures: list[Fixture],
    index: Any,
    embedding_model: Any,
    eval_config: ResolvedConfig,
    affinity: object,
    query_expansion_enabled: bool,
    tier: str,
    tier_trace_dir: Path,
) -> tuple[
    list[FixtureResult],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate one tier with full tracing. Returns (results, traces, llm_log)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from tqdm import tqdm
        pbar: Any = tqdm(total=len(fixtures), desc=tier, unit="fix")
    except ImportError:
        pbar = None

    results: list[FixtureResult] = []
    traces: list[dict[str, Any]] = []
    llm_log: list[dict[str, Any]] = []

    def _eval_one(fx: Fixture) -> tuple[
        FixtureResult, dict[str, Any], dict[str, Any] | None,
    ]:
        _TRACE.set_fixture(fx.id)
        try:
            start = time.perf_counter()
            tool_name = _extract_tool_name(fx.event or "", fx.query)
            fixture_mask_stats = _count_rules_masked(
                index, affinity, fx.event or "", tool_name or None,
            )
            pr = run_pipeline(
                fx.query,
                index,
                eval_config,
                embedding_model=embedding_model,
                mode="llm-local",
                event=fx.event or "",
                tool_name=tool_name,
                affinity=affinity,  # type: ignore[arg-type]
                query_expansion_enabled=query_expansion_enabled,
                query_expansion_endpoint=ENDPOINT,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            retrieved_texts = [r.rule.text for r in pr.results]
            relevant = set(fx.should_match)
            anti_rel = set(fx.should_not_match)
            is_negative = fx.difficulty == "negative"

            fr = FixtureResult(
                fixture_id=fx.id,
                query=fx.query,
                difficulty=fx.difficulty,
                retrieved=tuple(retrieved_texts),
                precision_at_k=precision_at_k(retrieved_texts, relevant),
                recall_at_k=recall_at_k(retrieved_texts, relevant),
                mrr=mrr(retrieved_texts, relevant),
                ndcg_at_k=ndcg_at_k(retrieved_texts, relevant),
                anti_precision=anti_precision(retrieved_texts, anti_rel),
                noise_ratio=noise_ratio(retrieved_texts, relevant),
                context_waste_ratio=context_waste_ratio(
                    retrieved_texts, relevant,
                ),
                retrieved_count=len(retrieved_texts),
                latency_ms=elapsed_ms,
                quality_score=quality_score(
                    retrieved_texts, relevant, is_negative,
                ),
            )

            llm_capture = _TRACE.pop(fx.id)
            expected_mask_stats = _count_expected_rules_visible(
                fx, index, affinity, fx.event or "", tool_name or None,
            )
            trace_obj = _build_fixture_trace(
                tier,
                tool_name,
                fx,
                pr,
                fr,
                llm_capture,
                mask_stats=fixture_mask_stats,
                expected_mask_stats=expected_mask_stats,
            )

            out_file = tier_trace_dir / f"{_safe_filename(fx.id)}.json"
            out_file.write_text(json.dumps(trace_obj, indent=2, default=str))

            return fr, trace_obj, llm_capture
        finally:
            _TRACE.clear_fixture()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_eval_one, fx): fx for fx in fixtures}
        for fut in as_completed(futures):
            fr, trace_obj, llm_capture = fut.result()
            results.append(fr)
            traces.append(trace_obj)
            if llm_capture is not None:
                llm_log.append({
                    "tier": tier,
                    "fixture_id": fr.fixture_id,
                    "event": trace_obj["event"],
                    "tool_name": trace_obj["tool_name"],
                    "difficulty": trace_obj["difficulty"],
                    "query": fr.query,
                    "n_candidates": len(
                        llm_capture.get("stage1_candidates") or [],
                    ),
                    "selected_indices": llm_capture.get("selected_indices"),
                    "reasoning": llm_capture.get("reasoning"),
                    "raw_response": llm_capture.get("raw_response"),
                    "error": llm_capture.get("error"),
                })
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return results, traces, llm_log


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _truncate(text: str, n: int = 140) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _build_report_md(
    *,
    label: str,
    config: dict[str, Any],
    results_by_tier: dict[str, Any],
    all_fixture_traces: list[dict[str, Any]],
) -> str:
    """Build a human-readable markdown report with metrics + gap analysis."""
    lines: list[str] = []
    lines.append(f"# Benchmark Report — {label}")
    lines.append("")
    lines.append(f"- **Generated:** {config['generated_at']}")
    lines.append(f"- **Seed:** {config['seed']}")
    lines.append(f"- **Sample ratio:** {config['sample_ratio']}")
    lines.append(f"- **Corpus:** `{config['corpus_path']}`")
    lines.append(f"- **Embedding model:** `{config['embedding_model']}`")
    lines.append(f"- **LLM endpoint:** `{config['llm_endpoint']}`")
    lines.append(f"- **llm_candidates override:** {config['llm_candidates']}")
    lines.append(
        f"- **Query expansion:** {config['query_expansion_enabled']}",
    )
    lines.append("")

    # Top-level metrics per tier
    lines.append("## Per-Tier Summary")
    lines.append("")
    lines.append(
        "| Tier | N | F2 | PosRecall | Noise | NegSil | p50ms | p95ms |",
    )
    lines.append(
        "|------|--:|---:|----------:|------:|-------:|------:|------:|",
    )
    for tier, tdata in results_by_tier.items():
        s = tdata["summary"]
        lines.append(
            f"| {tier} | {tdata['n_fixtures_sampled']} | "
            f"{s['mean_quality']:.3f} | "
            f"{s['positive_recall']:.3f} | "
            f"{s['mean_noise_ratio']:.3f} | "
            f"{s['negative_silence_rate']:.3f} | "
            f"{s['latency_p50_ms']:.0f} | "
            f"{s['latency_p95_ms']:.0f} |",
        )
    lines.append("")

    # Per-event breakdown
    lines.append("## Per-Event Breakdown")
    lines.append("")
    lines.append("| Tier | Event | N | F2 | PosRecall | Noise | NegSil |")
    lines.append("|------|-------|--:|---:|----------:|------:|-------:|")
    for tier, tdata in results_by_tier.items():
        for em in tdata.get("per_event", []):
            lines.append(
                f"| {tier} | {em['event']} | {em['fixture_count']} | "
                f"{em['quality']:.3f} | "
                f"{em['positive_recall']:.3f} | "
                f"{em['noise_ratio']:.3f} | "
                f"{em['negative_silence']:.3f} |",
            )
    lines.append("")

    lines.append("## Stage Quality")
    lines.append("")
    lines.append(
        "| Tier | Stage | F2 | PosRecall | Noise | NegSil | "
        "Precision | Recall | MRR | mean k |",
    )
    lines.append(
        "|------|-------|---:|----------:|------:|-------:|----------:|"
        "-------:|----:|------:|",
    )
    for tier, tdata in results_by_tier.items():
        stage_metrics = tdata.get("stage_metrics", {})
        for stage_key, stage_label in (
            ("stage1_dense", "dense"),
            ("stage1_sparse", "sparse"),
            ("stage1_raw_query", "raw-query"),
            ("stage1_fused", "fused"),
            ("stage3_final", "final"),
        ):
            s = stage_metrics.get(stage_key, {})
            if not s:
                continue
            lines.append(
                f"| {tier} | {stage_label} | "
                f"{s['mean_quality']:.3f} | "
                f"{s['positive_recall']:.3f} | "
                f"{s['mean_noise_ratio']:.3f} | "
                f"{s['negative_silence_rate']:.3f} | "
                f"{s['mean_precision']:.3f} | "
                f"{s['mean_recall']:.3f} | "
                f"{s['mean_mrr']:.3f} | "
                f"{s['mean_retrieved_count']:.2f} |",
            )
    lines.append("")

    lines.append("## Query Expansion Attribution")
    lines.append("")
    lines.append(
        "| Tier | QX fixtures | Help rate | Hurt rate | Avg new hits | "
        "Avg new noise |",
    )
    lines.append(
        "|------|-----------:|----------:|----------:|-------------:|"
        "--------------:|",
    )
    for tier, tdata in results_by_tier.items():
        trans = tdata.get("stage_metrics", {}).get("transitions", {})
        if not trans:
            continue
        lines.append(
            f"| {tier} | "
            f"{trans['query_expansion_fixture_rate']:.3f} | "
            f"{trans['query_expansion_help_rate']:.3f} | "
            f"{trans['query_expansion_hurt_rate']:.3f} | "
            f"{trans['query_expansion_avg_new_hits']:.3f} | "
            f"{trans['query_expansion_avg_new_noise']:.3f} |",
        )
    lines.append("")

    lines.append("## Failure Attribution")
    lines.append("")
    lines.append(
        "| Tier | Stage0 visible | ExpMasked | "
        "Stage1 any-hit | Stage1 full-hit | Stage1 neg-sil | Sparse ran | "
        "LLM keep rel | LLM prune irr | LLM empty-from-nonempty | "
        "LLM error |",
    )
    lines.append(
        "|------|---------------:|----------:|------------:|---------------:|"
        "---------------:|-----------:|-------------:|--------------:|----------------------:|"
        "---------:|",
    )
    for tier, tdata in results_by_tier.items():
        stage_metrics = tdata.get("stage_metrics", {})
        stage0 = stage_metrics.get("stage0_mask", {})
        trans = stage_metrics.get("transitions", {})
        if not stage0 or not trans:
            continue
        lines.append(
            f"| {tier} | "
            f"{stage0['rule_visibility_rate']:.3f} | "
            f"{stage0['expected_rule_mask_rate']:.3f} | "
            f"{trans['stage1_positive_any_hit_rate']:.3f} | "
            f"{trans['stage1_positive_full_recall_rate']:.3f} | "
            f"{trans['stage1_negative_silence_rate']:.3f} | "
            f"{trans['sparse_ran_rate']:.3f} | "
            f"{trans['llm_relevant_keep_rate']:.3f} | "
            f"{trans['llm_irrelevant_prune_rate']:.3f} | "
            f"{trans['llm_abstain_from_nonempty_rate']:.3f} | "
            f"{trans['llm_error_rate']:.3f} |",
        )
    lines.append("")

    # Per-tier stage latency breakdown (from traces)
    lines.append("## Stage Latency (median per fixture)")
    lines.append("")
    lines.append("| Tier | Stage | median ms |")
    lines.append("|------|-------|----------:|")
    by_tier_traces: dict[str, list[dict[str, Any]]] = {}
    for t in all_fixture_traces:
        tier_guess = t.get("tier") or _infer_tier(t["event"])
        by_tier_traces.setdefault(tier_guess, []).append(t)

    for tier, tlist in by_tier_traces.items():
        stage_ms: dict[str, list[float]] = {}
        for t in tlist:
            for s in t["pipeline"]["stages"]:
                stage_ms.setdefault(s["stage"], []).append(s["latency_ms"])
        for stage, vals in stage_ms.items():
            vals_sorted = sorted(vals)
            median = vals_sorted[len(vals_sorted) // 2]
            lines.append(f"| {tier} | {stage} | {median:.0f} |")
    lines.append("")

    # Gap analysis: FN (positive fixtures with missed rules)
    lines.append("## Gap Analysis")
    lines.append("")

    fn_cases = [
        t for t in all_fixture_traces
        if t["classification"] in {"fn", "partial"}
           and t["difficulty"] != "negative"
    ]
    fn_cases.sort(key=lambda t: (
        -len(t["misses"]),  # most misses first
        len(t["extras"]),
    ))

    lines.append(f"### False Negatives / Partial Hits ({len(fn_cases)})")
    lines.append("")
    lines.append(
        "Positive fixtures where one or more expected rules were NOT in "
        "the final retrieved set. Shows what the LLM saw and why it "
        "excluded the missed rules.",
    )
    lines.append("")
    for t in fn_cases[:15]:
        lines.append(f"#### `{t['fixture_id']}` — {t['difficulty']}")
        lines.append(f"- **Event:** {t['event']}")
        lines.append(f"- **Query:** `{_truncate(t['query'])}`")
        lines.append(
            f"- **Missed ({len(t['misses'])}):**",
        )
        for m in t["misses"][:4]:
            lines.append(f"  - `{_truncate(m)}`")
        lines.append(
            f"- **Retrieved ({len(t['final_retrieved'])}):**",
        )
        for r in t["final_retrieved"][:4]:
            lines.append(f"  - `{_truncate(r)}`")

        llm = t.get("llm") or {}
        if llm.get("reasoning"):
            lines.append(
                f"- **LLM reasoning:** {_truncate(llm['reasoning'], 260)}",
            )
        stage1 = llm.get("stage1_candidates") or []
        expected_set = set(t["should_match"])
        missed_in_stage1 = [
            c for c in stage1 if c["text"] in expected_set
        ]
        if missed_in_stage1:
            lines.append(
                "- **Missed rules WERE in stage 1 "
                f"({len(missed_in_stage1)}):** LLM reranker dropped them",
            )
            for c in missed_in_stage1[:3]:
                lines.append(
                    f"  - idx={c['idx']} score={c['stage1_score']:.3f} "
                    f"`{_truncate(c['text'])}`",
                )
        elif stage1:
            lines.append(
                "- **Missed rules NOT in stage 1:** embedding recall "
                "failure (the LLM never saw them)",
            )
        lines.append("")

    # Gap analysis: FP (negative fixtures where rules fired)
    fp_cases = [
        t for t in all_fixture_traces
        if t["classification"] == "fp"
    ]
    fp_cases.sort(key=lambda t: -len(t["final_retrieved"]))

    lines.append(f"### False Positives on Negatives ({len(fp_cases)})")
    lines.append("")
    lines.append(
        "Negative fixtures (expected to be silent) where the pipeline "
        "fired rules anyway.",
    )
    lines.append("")
    for t in fp_cases[:15]:
        lines.append(f"#### `{t['fixture_id']}` — negative")
        lines.append(f"- **Event:** {t['event']}")
        lines.append(f"- **Query:** `{_truncate(t['query'])}`")
        lines.append(
            f"- **Fired ({len(t['final_retrieved'])}):**",
        )
        for r in t["final_retrieved"][:4]:
            lines.append(f"  - `{_truncate(r)}`")
        llm = t.get("llm") or {}
        if llm.get("reasoning"):
            lines.append(
                f"- **LLM reasoning:** {_truncate(llm['reasoning'], 260)}",
            )
        lines.append("")

    # Noisiest TP cases
    noisy_tp = [
        t for t in all_fixture_traces
        if t["classification"] == "tp_noisy"
    ]
    noisy_tp.sort(key=lambda t: -len(t["extras"]))
    lines.append(f"### Noisy True Positives ({len(noisy_tp)})")
    lines.append("")
    lines.append(
        "Positive fixtures where all expected rules were retrieved, "
        "but with extra rules on top (precision cost).",
    )
    lines.append("")
    for t in noisy_tp[:10]:
        lines.append(f"- `{t['fixture_id']}` — "
                     f"{len(t['extras'])} extras over "
                     f"{len(t['should_match'])} expected "
                     f"(`{_truncate(t['query'], 80)}`)")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `traces/{tier}/{fixture_id}.json` — full per-fixture trace")
    lines.append("- `llm_calls.jsonl` — flat LLM interaction log")
    lines.append("- `summary.json` — machine-readable metrics")
    lines.append("- `config.json` — run configuration")
    lines.append("")

    return "\n".join(lines)


def _infer_tier(event: str) -> str:
    """Map event name back to tier key for grouping in the report."""
    return {
        "PreToolUse": "pre_tool_use",
        "UserPromptSubmit": "user_prompt_submit",
        "Stop": "stop",
        "SubagentStart": "subagent_start",
    }.get(event, event or "unknown")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "E2E benchmark with full per-stage tracing: model generates "
            "its own expansions AND reranks."
        ),
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to GGUF model file",
    )
    parser.add_argument(
        "--label", required=True,
        help="Label for corpus+results (e.g. gemma-e4b-s32)",
    )
    parser.add_argument(
        "--suffix", default=None,
        help=(
            "Optional suffix appended to corpus+result names "
            "(e.g. promptv2 -> {label}-promptv2)."
        ),
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
        help="Fraction of fixtures to evaluate",
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
        "--artifacts-dir", type=str, default=None,
        help=(
            "Directory for traces, report, summary. "
            "Default: eval/results/{label-with-suffix}-traced-seed{seed}/"
        ),
    )
    parser.add_argument(
        "--tiers", type=str, default=None,
        help=(
            "Comma-separated tiers to benchmark "
            f"(available: {','.join(SOURCE_FILES)}). Default: all."
        ),
    )
    parser.add_argument(
        "--llm-candidates", type=int, default=None,
        help="Override llm_candidates (rules sent to LLM reranker).",
    )
    parser.add_argument(
        "--query-expansion", action="store_true",
        help="Enable query-side expansion (experimental).",
    )
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Model not found: {args.model_path}")
        sys.exit(1)

    if not re.match(r"^[a-zA-Z0-9_\-]+$", args.label):
        print(
            f"Invalid label: {args.label!r} — only alphanumeric, "
            "dash, underscore",
        )
        sys.exit(1)
    if args.suffix and not re.match(r"^[a-zA-Z0-9_\-]+$", args.suffix):
        print(
            f"Invalid suffix: {args.suffix!r} — only alphanumeric, "
            "dash, underscore",
        )
        sys.exit(1)

    run_label = _apply_suffix(args.label, args.suffix)

    tiers: tuple[str, ...] | None = None
    if args.tiers:
        tiers = tuple(t.strip() for t in args.tiers.split(","))
        invalid = [t for t in tiers if t not in SOURCE_FILES]
        if invalid:
            print(
                f"Unknown tiers: {invalid}. "
                f"Available: {list(SOURCE_FILES)}",
            )
            sys.exit(1)
        print(f"Benchmarking tiers: {', '.join(tiers)}")

    os.environ.setdefault("CUECARD_LLM_ENDPOINT", ENDPOINT)

    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        artifacts_dir = RESULTS_DIR / f"{run_label}-traced-seed{args.seed}"

    proc = None
    if not args.no_server:
        proc = start_server(args.model_path, ngl=args.ngl)

    try:
        ping_seconds = ping_server()
        if ping_seconds > 10:
            print(
                f"WARNING: PING took {ping_seconds:.1f}s — endpoint may "
                "be slow. Proceeding anyway.",
            )

        print("\n=== Phase 0: Infer affinity (once for all events) ===")
        _rules, affinity = _infer_affinity_once()

        print(f"\n=== Phase 1: Generate expansions for {run_label} ===")
        generate_expansions_for_label(
            run_label, affinity, force=args.force_expand,
        )

        print(f"\n=== Phase 2: Traced benchmark {run_label} ===")
        run_benchmark_traced(
            run_label,
            affinity,
            sample_ratio=args.sample_ratio,
            seed=args.seed,
            tiers=tiers,
            llm_candidates=args.llm_candidates,
            query_expansion_enabled=args.query_expansion,
            artifacts_dir=artifacts_dir,
        )

    finally:
        if proc is not None:
            kill_server(proc)


if __name__ == "__main__":
    main()
