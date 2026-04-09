# Session 29 Progress — Test Speed, Network Guard, Directory Restructure

**Date:** 2026-04-08
**Branch:** master

## What Was Done

### 1. Test Suite Speedup (11.3s → 4.3s)
- `test_serve_stop_running`: mock `os.kill` to simulate process death on liveness check — 5.21s → 0.03s (was polling 50x with `time.sleep(0.1)`)
- `test_serve_http.py`: class-scoped server fixture shares one HTTPServer across 9 tests — 4.5s → 0.57s (one shutdown instead of nine)
- `test_external_host_raises`: mock `socket.getaddrinfo` to avoid real DNS lookup — 0.21s → <0.01s

### 2. Network Guard (conftest autouse fixture)
- Blocks `socket.create_connection` for non-localhost hosts → `RuntimeError("Test attempted real network connection to {host}")`
- Blocks `socket.getaddrinfo` for non-localhost hosts → `RuntimeError("Test attempted DNS lookup for {host}")`
- Caught 5 tests making real DNS lookups (evil.com, 10.0.0.1) — all fixed with `cuecard.retrieval.llm_utils.socket.getaddrinfo` mocks

### 3. 1s Timeout Per Test (pytest-timeout)
- `timeout = 1` and `timeout_method = "thread"` in pyproject.toml
- Thread method handles serve tests with background threads
- Slow tests excluded via `addopts = "-m 'not slow'"`

### 4. Source Directory Restructure
Moved 25 source files into 4 subdirectories:
```
src/cuecard/
├── models.py, config.py, security.py, logger.py, serve.py, _entry.py, _math.py
├── indexing/     — parser, indexer, freshness, loader, expander
├── retrieval/    — pipeline, retriever, dense, sparse, fusion, reranker, llm_reranker, llm_utils, affinity, formatter
├── eval/         — harness, metrics, report
├── cli/          — main, setup, rules, hooks, eval_cmd
└── adapters/     — claude_code (unchanged)
```

Key fix: CLI submodules import `cuecard.cli.main` directly (not via package re-export) so monkeypatch targets hit the defining module. No re-exports in `cli/__init__.py`.

### 5. Test Directory Restructure
Moved 42 test files into 6 subdirectories mirroring src/:
```
tests/
├── conftest.py + 6 root files (models, config, security, logger, math, e2e)
├── indexing/    — 9 files
├── retrieval/   — 15 files
├── eval/        — 3 files
├── cli/         — 8 files
├── adapters/    — 1 file
└── serve/       — 6 files
```

### 6. Tool Updates
- `tools/notebook.ipynb` — all imports updated to new paths
- `tools/bench_e2e.py` — imports were already updated during source restructure

### 7. Docs Updated
- CLAUDE.md — project structure, test conventions (network guard, timeout, monkeypatch rule), key commands

## Commits (7 pushed)
1. `f9eb878` — perf: fix test suite slowness — 11.3s → 5.6s
2. `cfa912e` — test: add network guard and 1s timeout
3. `a12e3fa` — test: exclude slow tests from default run
4. `c3835e8` — refactor: restructure src/cuecard/ into feature subdirectories
5. `f6f1ab1` — refactor: restructure tests/ into subdirectories mirroring src/
6. `e4ccbeb` — refactor: update notebook imports for new directory structure
7. (pending) — docs: update CLAUDE.md for session 29

## Current State
- `master` — 1153 fast tests in 4.3s, 8 slow tests via `-m slow`
- Network guard active, 1s timeout per test
- Lint/mypy clean
- 99.16% coverage (100% with slow tests)

## Remaining Work (from session 28)
1. **Remove scattered parameter defaults** — force all configurable params to be explicit (no defaults). top_k=5 drift bug found.
2. **Re-install cuecard** — `uv tool install -e .` after confirming restructure is stable
3. **Benchmark** — verify no performance regression after refactoring
4. **Phase 6 Completion Gate** — PRD converged, implementation pending
5. **Phase 7 Rule Quality** — 63/109 rules rewritten with trigger conditions, 46 remaining
