# Session 11 Progress (2026-04-01)

## What Was Done

### 1. cuecard Project Created
- New project at ~/projects/cuecard
- Git initialized, 7 commits on master

### 2. PRD v1.2 — Main PRD (Converged)
- `artifacts/prd-v1.md` — 20 sections, 27 decisions
- 2 rounds × 3 reviews (arch + security + ML) = converged
- Covers: core pipeline, config, security, CLI, adapter, evaluation

### 3. Multi-Stage Retrieval PRD v1.1 (Converged)
- `artifacts/multi-stage-retrieval-prd.md` — 14 sections, 14 decisions
- 2 rounds × 3 reviews = converged
- 3-stage pipeline: embedding → cross-encoder (opt-in) → LLM re-ranking

### 4. Phase 1: Core Pipeline — COMPLETE
- 12 modules: models, _math, security, config, parser, indexer, freshness, retriever, formatter, cli, adapters, py.typed
- 2 rounds code + security review, all HIGH fixed

### 5. Phase 2+3: Adapter + Logger + Eval — COMPLETE
- adapter/claude_code.py, logger.py, eval.py, notebook.ipynb
- CLI: install/uninstall, status, log, eval commands
- 1 round code + security review, all HIGH fixed

### 6. Pipeline Orchestrator — COMPLETE
- pipeline.py with run_pipeline(), StageTrace, PipelineResult
- Mode dispatch: embedding, rerank, rerank-llm-local, rerank-llm-haiku

### 7. Golden Evaluation Fixtures — COMPLETE
- 82 fixtures: easy=29, medium=38, hard=15, negative=12
- 8 unreasonable fixtures fixed, 14 new added
- Difficulty-tagged for quality analysis

### 8. Model Benchmarks
**Embedding models (20 fixtures):**
| Model | Recall@5 | Precision@5 | MRR |
|-------|----------|-------------|-----|
| BGE-small | 0.675 | 0.160 | 0.602 |
| jina-code | **0.825** | **0.455** | **0.742** |
| nomic-v1.5 | 0.825 | 0.190 | 0.622 |

**Cross-encoder re-rankers (68 fixtures):**
| Re-ranker | Recall Delta | Precision Delta | MRR Delta |
|-----------|-------------|-----------------|-----------|
| MiniLM-L-6 | +1.96% | **-17.3%** | -10.7% |
| Jina v2 | +5.15% | **-17.3%** | -5.7% |

**Score distribution analysis (jina-code, 80 relevant rules):**
- threshold=0.10 → 97.5% recall (2 lost)
- threshold=0.30 → 73.8% recall (recommended default, was 0.35)
- threshold=0.35 → 66.2% recall (too aggressive!)

**Qwen3-Reranker-0.6B research:**
- MTEB-Code: 73.42 (crushes everything in size class)
- Promptable with custom instructions (FollowIR: 5.41)
- Apache-2.0, GGUF Q4_K_M at 370MB
- Latency: 500ms-2s CPU → Stage 3 option, not Stage 2

### 9. Test Status
- **400 tests, 100% coverage, 1442 statements**
- ruff clean, mypy strict clean

## Agents Still Working
- **Cody:** reranker.py (cross-encoder re-ranking) — subagent running
- **Ralph:** llm_reranker.py (LLM re-ranking: local + Haiku) — shell running
- **Cron:** 6b9ddeb6 every 7 min

## What's Next (Next Session)
1. Wait for cody (reranker.py) and ralph (llm_reranker.py) to complete
2. Verify tests + coverage, lint, types
3. Code + security review until convergence
4. Commit reranker + llm_reranker
5. Integrate pipeline modes into CLI + adapter
6. Download Qwen3-Reranker-0.6B GGUF, benchmark via llama-server
7. Run full benchmarks across all modes on 82 fixtures
8. End-to-end test: install cuecard as Claude Code plugin, verify it works
9. Phase 4: publish (PyPI, GitHub)

## Key Architectural Decisions Made This Session
- jina-code is the recommended embedding model (not BGE-small)
- Cross-encoder (MiniLM) is opt-in, not default — regressions on code content
- LLM re-ranking is the quality lever — understands semantics embeddings miss
- Qwen3-Reranker 0.6B is a lightweight Stage 3 option (promptable, code-aware)
- Each pipeline stage is independently valuable — lightweight first, LLMs last
- Threshold lowered: 0.30 (embedding-only), 0.10 (re-ranking enabled)
- 90%+ quality target at end of pipeline

## Quality Principles Established
1. Quality at every stage — don't rely on downstream to rescue upstream
2. Lightweight first, LLMs last — deterministic CPU before non-deterministic GPU
3. Every stage independently valuable — turning off any stage leaves working system
4. Robustness and reproducibility — deterministic for foundation, non-deterministic for boost
5. Provenance through every stage — traceable, loggable, debuggable
6. 90%+ quality target — measured, iterated, measured again
