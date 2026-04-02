# Session 14 Progress (2026-04-02)

**Status:** COMPLETE — 60 commits, 560 tests, 100% coverage

## What Was Done

Resolved all 5 findings from Codex deep code review (`artifacts/deep-code-review-2026-04-01.md`):

1. **CRITICAL — Cross-project rule leakage** → Fixed
   - `ResolvedConfig` now has `global_source_paths` and `project_source_paths`
   - Adapter uses `load_config(project_dir=Path.cwd())`
   - New `loader.py` with `load_or_build()` composes scoped indexes via `merge_indexes()`
   - CLI `index` builds separate indexes per scope
   - Cross-project isolation test added

2. **HIGH — Freshness contract missing** → Fixed
   - `loader.py` checks freshness per scope, auto-rebuilds stale indexes
   - Adapter + CLI retrieve/format use `load_or_build()` instead of `load_index()`
   - `_load_or_rebuild_scope()` handles new/stale/fresh cases

3. **MEDIUM — normalize_query dead code** → Fixed
   - One-line fix: `query = normalize_query(query)` in `retrieve()` before truncation

4. **MEDIUM — Docs/API drift** → Fixed
   - CLAUDE.md: `__init__.py` exports corrected, `loader.py` added, scoped caching documented
   - PRD: `load_or_build` annotated as internal, hook code example updated

5. **MEDIUM — CLI 950 lines** → Fixed
   - Split into: cli.py (485), cli_rules.py (234), cli_hooks.py (258), cli_eval.py (90)
   - Submodules use module-level indirection for monkeypatch compatibility

## Files Changed

**New files:**
- `src/cuecard/loader.py` — Unified index loading with freshness + scope composition
- `src/cuecard/cli_rules.py` — Rules subcommands
- `src/cuecard/cli_hooks.py` — Install/uninstall/status/log commands
- `src/cuecard/cli_eval.py` — Eval command
- `tests/test_loader.py` — 17 tests for loader module
- `artifacts/deep-code-review-2026-04-01.md` — Original review from Codex

**Modified files:**
- `src/cuecard/models.py` — Added `global_source_paths`, `project_source_paths` to ResolvedConfig
- `src/cuecard/config.py` — Populates scoped source paths
- `src/cuecard/adapters/claude_code.py` — Uses `load_or_build` + `project_dir=Path.cwd()`
- `src/cuecard/cli.py` — Trimmed to 485 lines, scoped index builds
- `src/cuecard/retriever.py` — Wired `normalize_query()` into `retrieve()`
- `CLAUDE.md` — Updated exports, architecture, design decisions
- `artifacts/prd-v1.md` — Annotated internal vs public API
- 4 test files updated for new ResolvedConfig fields and mock targets

## Cody's Benchmark Findings (from tmux)

Cody ran 4-config LLM prompt benchmark before session 13:
- 256 tokens: 76% parse failures (prompt too long)
- 512 tokens: best quality (94.4% neg silence) but 4.2s latency
- 2048 tokens: 0% parse failures but 55% noise (over-returns)
- 1024 tokens (our current): sweet spot — validated by these findings

## Pending Next Steps

1. Prompt tuning for unified index — reduce cross-domain noise (31.6% → <25%)
2. Rule augmentation (category prefix) — help embeddings separate domains
3. recall_top_k 20→30 for larger corpus
4. Benchmark jina-code vs bge-base on workflow fixtures
5. Phase 4: Publish (PyPI, GitHub CI)
6. Markdown parsing (v0.3)
7. Dogfood as Claude Code hook
