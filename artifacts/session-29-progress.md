# Session 29 Progress — Test Speed, Network Guard, Directory Restructure, Config Centralization

**Date:** 2026-04-08 → 2026-04-09
**Branch:** master

## What Was Done

### 1. Test Suite Speedup (11.3s → 4.3s)
- `test_serve_stop_running`: mock `os.kill` to simulate process death on liveness check — 5.21s → 0.03s
- `test_serve_http.py`: class-scoped server fixture shares one HTTPServer across 9 tests — 4.5s → 0.57s
- `test_external_host_raises`: mock `socket.getaddrinfo` to avoid real DNS lookup — 0.21s → <0.01s

### 2. Network Guard (conftest autouse fixture)
- Blocks `socket.create_connection` and `socket.getaddrinfo` for non-localhost hosts
- Raises `RuntimeError("Test attempted real network connection to {host}")`
- Caught and fixed 5 tests making real DNS lookups

### 3. 1s Timeout Per Test (pytest-timeout)
- `timeout = 1` and `timeout_method = "thread"` in pyproject.toml
- Slow tests excluded via `addopts = "-m 'not slow'"`

### 4. Source Directory Restructure (25 files → 4 subdirectories)
```
src/cuecard/
├── models.py, config.py, security.py, logger.py, serve.py  # root
├── indexing/     — parser, indexer, freshness, loader, expander
├── retrieval/    — pipeline, retriever, dense, sparse, fusion, reranker, llm_reranker, llm_utils, affinity, formatter
├── eval/         — harness, metrics, report
├── cli/          — main, setup, rules, hooks, eval_cmd
└── adapters/     — claude_code
```
Key fix: CLI submodules import `cuecard.cli.main` directly (not via package re-export).

### 5. Test Directory Restructure (42 files → 6 subdirectories)
Tests mirror src/: indexing/, retrieval/, eval/, cli/, adapters/, serve/

### 6. Remove Parameter Defaults, Centralize Constants
- Removed defaults from 7 functions (retrieve, dense/sparse.retrieve, fuse, rerank, rerank_llm, expand_rules, _semantic_dedup)
- All callers now pass values explicitly from config — no drift bugs
- 9 constants moved to models.py: DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_DIM, DEFAULT_RERANKER_MODEL, DEFAULT_LLM_ENDPOINT, DEFAULT_HAIKU_MODEL, DEFAULT_SERVE_PORT, DEFAULT_HOOK_EVENT, LLM_MAX_TOKENS, LLM_TIMEOUT

### 7. New Configurable Fields (4)
- `[retrieval] reranker_model` — cross-encoder model
- `[serve] port` — daemon port
- `[pipeline.llm] max_tokens` — LLM response token limit
- `[pipeline.llm] timeout` — HTTP timeout for LLM calls

### 8. Eval Harness Fix
- Replaced `_EvalConfig` stub with `ResolvedConfig` — stub was missing `llm_recall_threshold` and `reranker_model`, silently degrading benchmark quality

### 9. Benchmark Fixes
- Benchmark always infers affinity via LLM, never uses ground truth labels (data leakage fix)
- Ground truth (`rules_global_tagged.json`) is for accuracy measurement only

### 10. Benchmark Results (Gemma E4B, inferred affinity, 20% sample, seed=42)

| Event | F2 | PosRecall | Noise | NegSil |
|-------|------|-----------|-------|--------|
| PreToolUse | 0.547 | 0.624 | 0.492 | 0.523 |
| UserPromptSubmit | 0.616 | 0.616 | 0.396 | 0.652 |
| PostToolUse | 0.577 | 0.808 | 0.532 | 0.462 |
| Stop | 0.425 | 0.333 | 0.619 | 0.571 |
| SubagentStart | 0.362 | 0.643 | 0.727 | 0.263 |

Affinity accuracy: 90.7% (97/107)

Per-stage analysis (PreToolUse):
- Stage 1 (Embedding): F2=0.203, PosRecall=0.581, Noise=0.897
- Stage 1+2 (+LLM): F2=0.495, PosRecall=0.585, Noise=0.531
- Stage 1+2+3 (+Affinity): F2=0.507, PosRecall=0.597, Noise=0.536

## Commits (12 pushed)
1. `f9eb878` — perf: fix test suite slowness — 11.3s → 5.6s
2. `cfa912e` — test: add network guard and 1s timeout
3. `a12e3fa` — test: exclude slow tests from default run
4. `c3835e8` — refactor: restructure src/cuecard/ into feature subdirectories
5. `f6f1ab1` — refactor: restructure tests/ mirroring src/
6. `e4ccbeb` — refactor: update notebook imports
7. `7ca7ef1` — docs: update CLAUDE.md for session 29
8. `bdeed8e` — refactor: remove parameter defaults, centralize constants
9. `15bc6bf` — feat: make reranker_model, serve_port, llm_max_tokens, llm_timeout configurable
10. `407fe98` — fix: replace _EvalConfig stub with ResolvedConfig
11. `a4c6c8c` + `177ac3f` + `72fd569` — fix: benchmark always infers affinity via LLM

## Current State
- `master` — 1153 fast tests in 4.3s, 8 slow in 1s, lint/mypy clean
- Network guard + 1s timeout active
- All constants in models.py, no parameter defaults on configurable functions
- Benchmark correctly infers affinity, no ground truth leakage

## Remaining Work
1. **Phase 7 Rule Quality** — 46/109 rules still need trigger-condition rewrites. Biggest lever for PreToolUse F2.
2. **Re-install cuecard** — `uv tool install -e .` after confirming stability
3. **Phase 6 Completion Gate** — PRD converged, implementation pending
4. **Better embedding model** — jina-code-v2 at 60% hard recall; explore alternatives
5. **Full-sample benchmark** — run with `--sample-ratio 1.0` for publication-grade numbers
