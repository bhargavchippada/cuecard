# Benchmark: Closed-Loop Hooks Phase 6

**Date:** 2026-04-07
**Commit:** ab18740 (fix: code review findings)
**Mode:** embedding (dense + sparse + RRF, no LLM reranker)
**Model:** BAAI/bge-small-en-v1.5 (384-dim)
**Sample:** 30% stratified, seed=42

## Context

This is the first benchmark of the closed-loop hooks implementation. It establishes
baseline quality metrics for all 5 event types using embedding-only retrieval.

**LLM reranker (Gemma E4B) returned 400 errors** — a pre-existing compatibility issue
with the reranker's `chat_template_kwargs` and Gemma's Jinja template. The LLM reranker
was tested and working with Qwen models previously. This benchmark shows embedding-only
baseline; LLM reranker results will follow once the Gemma compat is resolved.

## Per-Event Results (embedding mode, 30% sample)

| Event | Fixtures | F2 | PosRecall | Noise | NegSil |
|-------|----------|------|-----------|-------|--------|
| PreToolUse (basic) | 105 | 0.306 | 0.751 | 0.846 | 0.000 |
| UserPromptSubmit (workflow) | 23 | 0.327 | 0.677 | 0.843 | 0.000 |
| PostToolUse | 9 | 0.332 | 0.714 | 0.867 | 0.000 |
| Stop | 7 | 0.433 | 0.900 | 0.800 | 0.000 |
| SubagentStart | 8 | 0.228 | 0.417 | 0.875 | 0.000 |

## Analysis

### Recall
- **Stop** has best recall (0.900) — stop queries are broad, matching many rules semantically
- **PreToolUse** strong at 0.751 — tool+command context gives good signal
- **PostToolUse** at 0.714 — tool output adds verification signal
- **UserPromptSubmit** at 0.677 — natural language queries are harder for code embeddings
- **SubagentStart** weakest at 0.417 — agent type descriptions need better expansion matching

### Noise
- ~85% noise across all events — expected without LLM reranker
- The LLM reranker typically reduces noise from 85% to 15-25% (based on prior benchmarks)
- Event mask (not yet active in this benchmark) should further reduce cross-event noise

### Negative Silence
- 0.000 across all events — embedding retrieval always returns top-k=5 regardless
- LLM reranker is the only mechanism that achieves silence (drops all candidates when irrelevant)
- This is the expected baseline; LLM reranker benchmarks will show the real silence rate

## LLM Reranker Results (Gemma E4B, 20% sample, seed=42)

**Root cause of initial 400 error:** llama-server was started with `-np 5` (5 parallel slots),
dividing 16384 context into 3276 tokens per slot. The reranker's 12K char system prompt
exceeds 3276 tokens. Fixed by restarting with `-np 1`.

| Event | F2 | PosRecall | Noise | NegSil | p50ms |
|-------|------|-----------|-------|--------|-------|
| PreToolUse | 0.786 | 0.793 | 0.235 | 0.852 | 1086 |
| UserPromptSubmit | 0.704 | 0.712 | 0.344 | 0.750 | 1086 |
| PostToolUse | 0.635 | 1.000 | 0.583 | 0.000 | 1356 |
| Stop | 0.859 | 0.875 | 0.227 | 1.000 | 1332 |
| SubagentStart | 0.764 | 0.750 | 0.333 | 1.000 | 1169 |

## Embedding vs LLM Comparison

| Event | Embed F2 | LLM F2 | Delta | Embed Noise | LLM Noise | LLM NegSil |
|-------|----------|--------|-------|-------------|-----------|-------------|
| PreToolUse | 0.306 | 0.786 | +157% | 0.846 | 0.235 | 0.852 |
| UserPromptSubmit | 0.327 | 0.704 | +115% | 0.843 | 0.344 | 0.750 |
| PostToolUse | 0.332 | 0.635 | +91% | 0.867 | 0.583 | 0.000 |
| Stop | 0.433 | 0.859 | +98% | 0.800 | 0.227 | 1.000 |
| SubagentStart | 0.228 | 0.764 | +235% | 0.875 | 0.333 | 1.000 |

## Key Findings

1. **Stop is strongest** — F2=0.859, perfect NegSil=1.000
2. **SubagentStart improved most** — +235% F2, LLM understands agent type context
3. **PostToolUse has 100% recall** but 0% NegSil — needs more negative fixtures
4. **Noise dropped 33-72%** across all events via LLM reranker
5. **Latency ~1-1.4s** — acceptable for daemon mode

## Next Steps

1. **Benchmark with event mask** — build affinity index, measure additional noise reduction
2. **More PostToolUse negative fixtures** — current set has too few negatives for silence
3. **SubagentStart expansions** — targeted vocab for agent types could push recall higher
4. **Production validation** — test 5-event hooks in a live Claude Code session
