# Session 16 Progress State

> Date: 2026-04-02
> For resumption in next session

## What Was Accomplished

### 1. Codex Review Fixes (6 findings)
All 6 issues from `artifacts/enriched-retrieval-implementation-review-2026-04-02.md` fixed:
1. `cuecard index` now merges cached rules.json (preserves expansions)
2. All retrieval paths route through `run_pipeline()` including embedding mode
3. `_EvalConfig` includes `sparse_enabled` and `fusion_k` fields
4. `rules expand` parses fresh source files before expanding
5. `hooks status` reports both global and project-scoped indexes
6. Docs updated

### 2. Comprehensive Benchmarking
Results in `eval/results/`:
- `raw-baseline-2026-04-02.json` — 5 fixture sets, embedding-only
- `enriched-embedding-2026-04-02.json` — with expansions + BM25
- `enriched-v3-embedding-2026-04-02.json` — v3 expansion prompt
- `enriched-llm-local-2026-04-02.json` — with Qwen3.5-35B reranker
- `model-comparison-2026-04-02.json` — 6 embedding models
- `small-model-qwen3-0.6b-2026-04-02.json` — 0.6B reranker attempt

### 3. Expansion Prompt v3
- Event-type aware (PreToolUse vs UserPromptSubmit)
- Variable count 3-10 per rule (was fixed 10)
- Cross-domain noise prevention
- Anti-template, trigger direction, indirect triggers
- Semantic dedup (cosine > 0.85)
- Workflow medium recall +26 pts (33% → 59%)

### 4. LLM Robustness
- `"stop": ["\n\n"]` for all local LLM calls (prevents repetition)
- First JSON block extraction via regex (handles trailing garbage)
- Single retry on parse failure before fallback

### 5. Small Model Investigation
- Qwen3-0.6B: **not viable** (recall halved, 12.6% parse failures)
- Root cause: 0.6B can't reason about relevance, not just a prompt issue
- Cross-encoder Qwen3-Reranker-0.6B with `/v1/rerank` endpoint identified as alternative

### 6. SOTA Model Research
Reports in `artifacts/`:
- `small-model-research-2026-04-02.md` — cross-encoders, tiny rerankers
- `sota-llm-research-2026-04-02.md` — 12 models ranked (1B-30B)
- `llama-cpp-inference-benchmarks-2026-04-02.md` — real measured latency data
- `small-model-investigation-plan.md` — full investigation plan
- `prompt-engineering-direction.md` — larger prompts via caching

### 7. Phase 5 PRD Draft
`artifacts/phase5-multi-source-prd-draft.md` — markdown/YAML parsing, CLAUDE.md ingestion

### 8. Notebook Updated
15 new cells: enriched index visualization, raw vs enriched comparison, model comparison, LLM reranker impact, full pipeline quality stack

## Key Benchmark Results

### PreToolUse / Basic (354 fixtures)

| Stage | Easy | Medium | Hard | Noise | Neg Silence |
|-------|------|--------|------|-------|-------------|
| Raw bge-small | ~83% | ~51% | ~39% | 86% | 0% |
| + v1 expansions + BM25 | 94.2% | 74.1% | 51.1% | 84% | 0% |
| + jina-code model | 95.3% | 76.8% | 60.0% | 83% | 0% |
| + LLM reranker (35B) | 85.2% | 73.1% | 50.1% | 21.4% | 91.0% |

### Embedding Model Comparison (enriched, basic fixtures)

| Model | Hard Recall | Overall Recall | p50ms |
|-------|-----------|---------------|-------|
| jina-code-v2 | 60.0% | 49.0% | 22ms |
| mxbai-embed-large | 57.7% | 49.1% | 45ms |
| snowflake-arctic | 57.5% | 47.5% | 14ms |
| bge-small | 51.1% | 46.8% | 4ms |

### Quality vs Targets

| Metric | Target | Best Result | Gap |
|--------|--------|------------|-----|
| Easy recall | >90% | 95.3% (embedding) / 85.2% (LLM) | LLM over-filters |
| Medium recall | >70% | 76.8% (embedding) / 73.1% (LLM) | Met |
| Hard recall | >50% | 60.0% (embedding) / 50.1% (LLM) | Met (barely) |
| Neg silence | >95% | 91.0% (LLM) | -4 pts |
| Noise | <25% | 21.4% (LLM) | Met |

## Key Architectural Decisions

1. **LLM reranker is essential** — embedding mode alone has 0% negative silence
2. **No source-side tagging** — users point at any file (CLAUDE.md, rules.txt), LLM reranker discriminates
3. **Prompt caching makes prompt size free** — invest in prompt quality, not brevity
4. **jina-code for PreToolUse, snowflake-arctic for UserPromptSubmit** — no single model wins both
5. **Qwen3.5-4B/9B are the replacement candidates** for 35B (same family, 4-8x smaller)

## What's Next (Priority Order)

### Immediate (Next Session)
1. **Download and benchmark Qwen3.5-4B** — most promising CPU/laptop model
2. **Download and benchmark Qwen3.5-9B** — quality ceiling, 8GB GPU viable
3. **Test Qwen3-Reranker-0.6B** in cross-encoder mode (`/v1/rerank`)
4. **Exhaustive prompt engineering** — 15-20 few-shot examples, loss-pattern driven

### Medium Term
5. **Fix LLM over-filtering on easy tier** (94% → 85% — add "keep obvious matches" examples)
6. **Close neg silence gap** (91% → 95% — add more negative examples to prompt)
7. **Phase 5.1**: Markdown parser for CLAUDE.md ingestion
8. **`cuecard serve` daemon** for model persistence (avoid load/unload per hook)

### Longer Term
9. Phase 4: PyPI publish
10. Phase 5.2: YAML skill parsing
11. Phase 5.3: Auto-expand on setup when LLM available
12. Fine-tune cross-encoder on our 587 fixtures

## Files Changed This Session

### Source Code
- `src/cuecard/cli.py` — index rebuild merge, all retrieval through pipeline
- `src/cuecard/cli_rules.py` — expand freshness fix
- `src/cuecard/cli_hooks.py` — scoped cache status
- `src/cuecard/eval.py` — _EvalConfig fields, always use pipeline
- `src/cuecard/adapters/claude_code.py` — always use pipeline
- `src/cuecard/expander.py` — v3 prompt, event_type, semantic dedup
- `src/cuecard/llm_reranker.py` — JSON extraction, retry, robustness
- `src/cuecard/llm_utils.py` — stop sequence

### Tests
- `tests/test_adapter.py` — pipeline routing mocks
- `tests/test_cli.py` — merge test, status tests, pipeline mocks
- `tests/test_expander.py` — v3 prompt, dedup, event_type
- `tests/test_llm_reranker.py` — retry, JSON extraction, robustness
- `tests/test_llm_utils.py` — stop sequence

### Artifacts
- `artifacts/enriched-retrieval-implementation-review-2026-04-02.md` (Codex review)
- `artifacts/post-fix-review-and-golden-eval-2026-04-02.md` (Codex post-fix)
- `artifacts/expansion-quality-review-2026-04-02.md`
- `artifacts/phase5-multi-source-prd-draft.md`
- `artifacts/small-model-investigation-plan.md`
- `artifacts/small-model-research-2026-04-02.md`
- `artifacts/sota-llm-research-2026-04-02.md`
- `artifacts/llama-cpp-inference-benchmarks-2026-04-02.md`
- `artifacts/prompt-engineering-direction.md`

### Models Downloaded
- `~/models/Qwen3-0.6B-Q4_K_M.gguf` (379MB)
- `~/models/Qwen3-Reranker-0.6B-Q8_0.gguf` (610MB)

## Git State
- Branch: master
- Latest commit: `caaac96`
- 826 tests, 100% coverage, ruff clean, mypy strict
