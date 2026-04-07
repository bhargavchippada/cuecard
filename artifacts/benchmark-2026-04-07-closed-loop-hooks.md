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

## Next Steps

1. **Fix Gemma reranker compat** — resolve 400 error from `chat_template_kwargs`
2. **Benchmark with LLM reranker** — expect noise drop from 85% to ~20%, silence > 80%
3. **Benchmark with event mask** — build affinity index, measure noise reduction from event filtering
4. **Benchmark event mask + LLM** — combined quality
5. **SubagentStart expansion** — may need targeted expansions for agent-type vocabulary
