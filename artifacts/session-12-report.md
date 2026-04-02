# Session 12 Report (2026-04-01)

## Executive Summary

Session 12 completed the multi-stage retrieval pipeline (Stages 2 & 3), expanded the eval framework with precision/context efficiency metrics, scaled the golden fixture dataset to 226, ran comprehensive benchmarks, and integrated pipeline modes into the CLI/adapter/config. The project is now feature-complete for v0.1 — only PyPI publishing remains.

## Commits (14 this session)

| # | Hash | Description |
|---|------|-------------|
| 1 | `87c2945` | feat: cross-encoder + LLM re-ranking (Stages 2 & 3) |
| 2 | `eecf1a5` | feat: expand golden fixtures to 90 (20 negatives) |
| 3 | `74b3008` | fix: security findings (SSRF @-bypass, nonce, haiku allowlist) |
| 4 | `fefa9e5` | refactor: remove duplicate L2Normalize tests |
| 5 | `733f35b` | refactor: deduplicate test classes in test_cli.py |
| 6 | `647c9e3` | feat: precision/context efficiency metrics in eval |
| 7 | `3cab986` | feat: notebook with context efficiency visualizations |
| 8 | `845d612` | fix: lower default threshold 0.35 → 0.30 |
| 9 | `834b1a4` | docs: update CLAUDE.md |
| 10 | `16b795d` | chore: update .gitignore |
| 11 | `4145533` | fix: add missing fields to 8 negative fixtures |
| 12 | `70c79e1` | feat: integrate pipeline modes into CLI, eval, adapter (Cody) |
| 13 | `efecb73` | feat: expand golden fixtures to 226 (Ralph) |
| 14 | `0123ece` | fix: mypy errors + e2e integration test |

## Test Status

- **516 tests, 100% coverage** (1739 statements)
- ruff clean, mypy strict clean
- 8 new e2e slow tests with deterministic BoW embedder

## New Modules

### reranker.py — Cross-Encoder Re-Ranking (Stage 2)
- fastembed `TextCrossEncoder` integration
- Model allowlist (`ALLOWED_RERANKER_MODELS`)
- Opt-in only — benchmarks showed MiniLM hurts precision on code (-17.3%)

### llm_reranker.py — LLM Re-Ranking (Stage 3)
- Two backends: local (httpx → llama-server) + Haiku (claude-agent-sdk)
- Nonce-based prompt injection defense
- SSRF validation (scheme + userinfo + hostname checks)
- Secrets scrubbing before LLM calls
- Ordinal scoring, graceful degradation on all failures
- Haiku model allowlist

### PipelineConfig — Config Integration
- New `[pipeline]` TOML section with mode, llm.local_endpoint, llm.haiku_model, llm.thinking
- Global/project layering (project wins)
- CLI `--mode` flag on `retrieve` and `eval` commands
- Adapter reads pipeline mode from config

## Pipeline Mode Benchmarks (226 fixtures, jina-code)

### Mode Comparison

| Mode | Recall | Precision | MRR | Noise | NegSilence | AvgRet | Latency p50 |
|------|--------|-----------|-----|-------|------------|--------|-------------|
| **embedding** (t=0.30) | 0.394 | 0.233 | 0.434 | 0.607 | 0.342 | 2.8 | 15ms |
| **rerank** (cross-encoder) | 0.337 | 0.123 | 0.311 | 0.877 | 0.000 | 5.0 | 110ms |
| **llm-local** (Qwen3.5-35B) | 0.458 | 0.161 | — | 0.839 | 0.000 | 5.0 | ~2s |

**Key finding: cross-encoder is a regression.** MiniLM was trained on web search, not code. It reorders results incorrectly for code queries, dropping recall -5.7% and precision -11%. New `llm-local`/`llm-haiku` modes skip the cross-encoder entirely.

### Critical Bug Fixed: Cross-Encoder Was Silently Failing
fastembed's `TextCrossEncoder.rerank()` returns raw floats, not objects with `.index`/`.score`. The reranker crashed on every call, and the pipeline's graceful degradation masked it — falling back to 20 unfiltered embedding candidates. Fixed in `f6c14cc`.

## Embedding-Only Benchmark (226 fixtures, threshold=0.30)

### Aggregate

| Metric | Value |
|--------|-------|
| Mean Recall@k | **0.394** |
| Mean Precision@k | 0.233 |
| Mean MRR | 0.434 |
| Mean Noise Ratio | 0.607 |
| Mean Context Waste | 0.608 |
| Neg Silence Rate | 0.342 |
| Mean Retrieved Count | 2.8 |
| Latency p50 | 14.6 ms |

### Per-Tier

| Tier | N | Recall | Precision | MRR | Noise | Silence |
|------|---|--------|-----------|-----|-------|---------|
| **easy** | 36 | **0.913** | 0.472 | 0.931 | 0.528 | 0.000 |
| **medium** | 82 | 0.583 | 0.382 | 0.673 | 0.581 | 0.037 |
| **hard** | 29 | 0.287 | 0.152 | 0.328 | 0.641 | 0.207 |
| **negative** | 79 | 0.000 | 0.000 | 0.000 | 0.658 | 0.342 |

### Model Comparison (82 fixtures, earlier benchmark)

| Metric | jina-code (0.30) | BGE-small (any) |
|--------|-----------------|-----------------|
| Recall | **0.581** | 0.579 |
| Precision | **0.251** | 0.141 |
| MRR | **0.565** | 0.494 |
| Noise | **0.676** | 0.859 |
| Neg Silence | **0.250** | 0.000 |

**BGE-small has zero discrimination** — all scores > 0.40, threshold has no effect. jina-code is the clear winner.

## Key Findings

1. **Easy tier is production-ready**: 91.3% recall, 93.1% MRR — the model finds the right rules for straightforward queries.

2. **Noise is the dominant problem**: Even easy tier has 52.8% noise ratio. Over half of injected rules are irrelevant across all tiers. This is where the re-ranker stages add value.

3. **Negative silence is weak at 34.2%**: Only 34.2% of "no rules should match" queries correctly return nothing. The rest inject irrelevant context. Higher thresholds help (50% at 0.40) but cost recall.

4. **Hard tier needs re-ranking**: 28.7% recall confirms that embeddings alone can't handle indirect/contextual queries. LLM re-ranking is the quality lever here.

5. **jina-code >> BGE-small**: BGE can't discriminate at all. jina-code responds to thresholds and provides meaningful signal/noise separation.

6. **Latency is excellent**: p50=14.6ms, p99=70ms — well within the 500ms budget.

## Security Review Findings (from earlier this session)

All MEDIUM findings fixed and committed:
- MEDIUM-2: SSRF @-bypass in `validate_endpoint` → now rejects userinfo in URLs
- MEDIUM-3: Nonce stripped from query string (was only stripped from rules)
- LOW-2: Added `ALLOWED_HAIKU_MODELS` allowlist

## Golden Fixture Dataset

Expanded from 90 → 226 fixtures:
- **36 easy**: keyword overlap queries
- **82 medium**: semantic gap queries
- **29 hard**: deep reasoning / indirect queries
- **79 negative**: queries where no rules should match
- Sources: 90 existing + 50 edge cases + 30 negatives + 56 mined from real delulu sessions

## Quality Iteration Findings

### Approaches Tried and Results

| Approach | Result | Verdict |
|----------|--------|---------|
| Query normalization (strip tool prefix) | -6.9% hard recall | **Reverted** — jina-code benefits from tool prefix |
| Cross-encoder (MiniLM) | -5.7% recall, -11% precision | **Regression** — trained on web search, not code |
| LLM reranker (Qwen3.5-35B) | +21% recall, but +22% noise | **Promising** — needs prompt improvement |
| Adaptive score gap filter | No improvement over threshold | **Not worth complexity** |
| Score ratio filter | Marginal noise reduction | **Not worth complexity** |

### Fundamental Insight

The embedding model (jina-code) has a **hard quality ceiling** for this task:
- Easy queries: ~91% recall (keyword overlap → embeddings work well)
- Medium queries: ~58% recall (semantic gap → embeddings struggle)
- Hard queries: ~29% recall (indirect reasoning → embeddings can't do this)
- Noise: ~61% at any threshold that preserves recall

**No amount of post-retrieval filtering can fix this.** The problem is that relevant and irrelevant rules score too similarly in the embedding space. The score distributions overlap heavily (match median=0.350, non-match max=0.703).

### LLM Reranker Results (Qwen3.5-35B, 226 fixtures)

**Critical bug fixed:** Empty `{"rules": []}` was treated as unparseable, causing fallback to ALL candidates for negative queries.

| Tier | Recall | Precision | Noise | Silence | AvgRet |
|------|--------|-----------|-------|---------|--------|
| easy | 84.0% | 87.5% | **12.5%** | 0% | 1.4 |
| medium | 55.6% | 62.4% | **29.1%** | 8.5% | 1.7 |
| hard | 31.6% | 40.2% | **32.2%** | 27.6% | 1.1 |
| negative | — | — | 24.1% | **75.9%** | 0.4 |

**Overall noise: 25.1%** (target <30% ✅), **neg silence: 75.9%** (target 80% ~close)

The LLM reranker is conservative — avg 1.1 results, almost no noise. Latency p50=281ms, p99=3s.

### Recommended Configuration

| Use Case | Mode | Threshold | Strengths |
|----------|------|-----------|-----------|
| **Low latency** | embedding | 0.30 | 15ms, 91% easy recall |
| **Low noise** | llm-local | 0.30 | 25% noise, 76% neg silence |
| **Balanced** | llm-local | 0.30 | Best overall quality |

### Improved Prompt Results (357 fixtures, few-shot + guidelines)

| Tier | Recall | Noise | Silence |
|------|--------|-------|---------|
| easy | 84.0% | **35.6%** | — |
| medium | **62.4%** | 52.9% | — |
| hard | **34.1%** | 51.4% | — |
| negative | — | — | **86.3%** |

**Neg silence: 86.3%** (was 24.2%) — target 80% **MET** ✅
Few-shot examples + DO/DONT guidelines massively improved negative handling.

### Ralph's Research Findings

1. **CodeRankEmbed (137M)** — beats jina-code by +10.7 MRR on CodeSearchNet. Worth evaluating.
2. **BM25 hybrid REGRESSES** — tool-call queries are hostile to keyword matching (-5% to -16% recall).
3. **Rule augmentation (category prefix)** — only clean win: +1.7% recall, +3.5% MRR, no regressions.
4. **False negative analysis** — 78 missed instances: 29% reasonable miss, 28% wrong fixture expectations, 23% narrow corpus rules, 19% need better embedding.
5. **Qwen3-Reranker-0.6B** — cross-encoder, NOT instruction-following. Can't generate JSON. Use via fastembed only.

### DEFINITIVE RESULTS (357 fixtures, thinking=OFF, server -c 16384 --jinja)

| Tier | Recall | Precision | Noise | Silence | AvgRet |
|------|--------|-----------|-------|---------|--------|
| easy | **80.9%** | 76.7% | **18.4%** | — | 1.5 |
| medium | 59.7% | 61.4% | **17.8%** | — | 1.6 |
| hard | 33.1% | 41.3% | **25.3%** | — | 1.5 |
| negative | — | — | **4.0%** | **96.0%** | 0.0 |

**Overall noise: 14.2%** (was 64%) — 78% reduction
**Neg silence: 96.0%** (was 24.2%) — 4x improvement
**Precision: 39.9%** (was 23.9%) — 67% improvement
**Latency p50: 889ms** (thinking disabled via chat_template_kwargs)

### Key Breakthrough: `chat_template_kwargs: {"enable_thinking": false}`
Disabling Qwen3.5's thinking mode via Jinja template gives 10x latency reduction
(7s → 889ms) while maintaining quality. Server needs `--jinja` flag.

### Path Forward

1. **Easy recall gap** — 80.9% vs 90.2% embedding. LLM too conservative. Prompt tuning.
2. **Two-stage thinking** — budget-capped thinking for hard queries (~1s total)
3. **Rule augmentation** — category prefix, +1.7% recall, +3.5% MRR
4. **CodeRankEmbed** — 137M, +10.7 MRR potential (SOTA research)
5. **Phase 4: Publish** — PyPI, GitHub CI, README

## What's Next

1. **Run re-ranker benchmarks** — evaluate cross-encoder + LLM modes on 226 fixtures to measure noise reduction
2. **Tune threshold** — find optimal operating point for recall vs noise tradeoff
3. **Phase 4: Publish** — PyPI package, GitHub CI, README with badges
4. **Real-world test** — install as Claude Code plugin and dogfood
5. **Markdown chunking** (v0.2) — support `.md` rule files with section-aware chunking
