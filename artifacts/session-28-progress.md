# Session 28 Progress — Code Review, Refactoring, Restructure

**Date:** 2026-04-09
**Branch:** master

## What Was Done

### 1. Full Code Review (all 25+ source files)
- Reviewed every source file in `src/cuecard/` for correctness, conventions, and quality
- Used parallel subagents for code review and security review at each milestone
- Convergence-based reviews: 0 CRITICAL, 0 HIGH remaining after fixes

### 2. Config Architecture — Single Source of Truth
- `ResolvedConfig` in models.py is now the single source of truth for all defaults + validation
- Field metadata `{"min": 1, "max": 50}` drives validators — no separate `_VALIDATORS` dict
- Eliminated `_DEFAULTS` and `_enriched_defaults` dicts from config.py
- `_TOML_DEFAULTS` and `_VALIDATORS` derived from `dataclasses.fields(ResolvedConfig)`
- Adding a new config field: add to ResolvedConfig + add TOML key in `_extract_flat()`

### 3. Constants Centralized in models.py
- `MAX_RULE_LENGTH = 500`
- `MAX_EXPANSION_LENGTH = MAX_RULE_LENGTH` (unified, was 200)
- `MAX_EXPANSIONS_PER_RULE = 10` (safety cap, operational default = 5)
- `MAX_RULES_PER_FILE = 500` (was `_MAX_RULES_PER_FILE` in parser.py only)
- `MAX_REQUEST_BYTES = 1_000_000` (was scattered in serve.py + adapter)
- `MAX_TOOL_NAME_LENGTH = 200` (was `_MAX_TOOL_NAME` in adapter)
- `VALID_AFFINITY_SOURCES` (was inline tuple in indexer.py)

### 4. New Config Fields
- `expansion_dedup_threshold: float = 0.80` — configurable via `[expansion] dedup_threshold`
- `expansion_max_per_rule: int = 5` — default reduced from 10 (LLM self-limits to 4-5)
- `expansion_max_length: int = 500` — unified with MAX_RULE_LENGTH
- `llm_recall_threshold: float = 0.25` — was hardcoded in pipeline.py

### 5. Security Hardening
- SSH/PEM private key pattern added to scrub_secrets
- OpenAI `sk-` pattern: negative lookahead to avoid matching `sk-ant-`/`sk-proj-`
- DNS-based loopback validation (`socket.getaddrinfo` + `ipaddress.is_loopback`) replacing string hostname check
- `logger.exception()` for traceback in hook adapter (was swallowed)

### 6. Function Extractions (under 50-line convention)
- `llm_reranker.py`: `_call_and_parse()` — eliminated retry code duplication
- `affinity.py`: `_infer_single_rule()` — extracted from `infer_affinities` loop
- `pipeline.py`: `_retrieval_params()`, `_run_sparse()`, `_build_retriever_traces()` — split 117-line function
- `expander.py`: `_expand_single_rule()` — extracted from `expand_rules` loop
- `adapters/claude_code.py`: `_run_pipeline_path()` — extracted from `main()`, fixes fallback state bug

### 7. File Splits (under 800-line convention)
- `eval.py` (903) → `eval.py` (518) + `eval_metrics.py` (163) + `eval_report.py` (307)
- `cli.py` (858) → `cli.py` (522) + `cli_setup.py` (352)
- Re-exports in `__init__` files for backwards compatibility

### 8. Directory Restructure (IN PROGRESS — subagent running)
Target structure:
```
src/cuecard/
├── models.py, config.py, security.py, logger.py, serve.py  # root
├── indexing/     # parser, indexer, freshness, loader, expander
├── retrieval/    # pipeline, retriever, dense, sparse, fusion, reranker, llm_reranker, llm_utils, affinity, formatter
├── eval/         # harness, metrics, report
├── cli/          # main, setup, rules, hooks, eval_cmd
└── adapters/     # claude_code
```

### 9. Miscellaneous Fixes
- `Index.__hash__ = None` for consistency with AffinityIndex
- `Rule.MAX_LENGTH` → `ClassVar[int]` (was polluting `fields()`)
- `rule_map` truthy → `is not None` checks in models.py + indexer.py
- `AffinityIndex._items` removed (reconstruct from `_lookup` on demand)
- `fuse()` default k 60→10, docstring updated
- Sparse threshold operator comment (`>` vs `>=`)
- `formatter.py` lazy import → top-level
- `loader.py` truthy string check → explicit `!= ""`
- `cli_hooks.py` lazy stdlib imports → top-level, JSONDecodeError handling
- `retriever.py` BM25 drop now logs warning
- `logger.py` TOCTOU fix in `rotate_log`
- `serve.py` `getattr` chain removed
- Dead `hasattr` guards removed from pipeline.py
- Model name `"qwen"` → `"default"` in llm_utils.py
- `_hash_rule_text` used instead of inline hashlib in indexer.py
- `compute_file_hash` from freshness.py replaces duplicate `_compute_checksum` in indexer.py
- Notebook rewritten as 35-cell debugging tool

### 10. Global Rules Added (4 new)
- No lazy imports for stdlib/internal modules
- Single source of truth for config, user-configurable
- Subdirectory structure when >15 files
- No parameter defaults for configurable values

## Commits (4 pushed + restructure pending)
1. `9324b4b` — refactor: code review pass — single source of truth, security hardening, function extraction
2. `9cb331e` — refactor: file splits, function extraction, review findings
3. `2051595` — refactor: centralize safety cap constants in models.py
4. `ab2adbc` — docs: update CLAUDE.md for session 28

## Current State
- `master` at `ab2adbc` — 4 commits pushed, 1161 tests green, 12.47s
- Directory restructure in `git stash` — new dirs created but test patches broken
- Stashed untracked dirs were removed (`rm -rf src/cuecard/cli/ eval/ indexing/ retrieval/`)
- cuecard was uninstalled (`uv tool uninstall cuecard`) due to hook spinning at 1563% CPU

## Remaining Work (for next session)
1. **Directory restructure** — `git stash pop`, then fix the root cause:
   - CLI submodules use `_cli._home_dir()` (reads `cuecard.cli.__init__` re-export)
   - But tests patch `cuecard.cli.main._home_dir` (the defining module)
   - Fix: make ALL submodules import directly from `cuecard.cli.main` not `_cli`
   - Then test patches target the defining module and work correctly
   - Also need: `addopts = "-m 'not slow'"` in pyproject.toml
   - Also need: conftest.py 2s SIGALRM timeout guard per test
2. **Remove scattered parameter defaults** — option C (force explicit params from config)
   - top_k=5 in retriever/dense/sparse != config top_k=7 (drift bug found)
   - All configurable function params should have NO defaults
3. **Update CLAUDE.md** — final project structure after restructure
4. **Re-install cuecard** — `uv tool install -e .` after restructure is green
5. **Benchmark** — verify no performance regression after refactoring
