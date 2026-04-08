# Session 21 Progress State

> Date: 2026-04-06
> For resumption in next session

## What Was Accomplished

### 1. Gemma E4B Production Switch
- Stopped Qwen 9B, started Gemma E4B on port 8081
- Generated 133 expansions (avg 4.4/rule, matches Gemma profile)
- Verified retrieval quality on diverse queries
- Known weakness: mongo --eval still confuses CLI flag with eval() (accepted tradeoff)

### 2. Daemon Reliability Fixes (7 issues)
- **Config-aware timeout**: 5s for llm-local/llm-haiku, 0.5s for embedding (was hardcoded 0.5s)
- **Fast hook dispatch**: `_entry.py` routes `cuecard hook` via argv check, skips Typer (~1.2s vs ~2.5s)
- **stop_server**: polls for exit after SIGTERM, SIGKILL fallback after 5s
- **SO_REUSEADDR**: `_ReuseHTTPServer` subclass for immediate port rebind
- **Daemon output**: always sets hookEventName + permissionDecision
- **Daemon input**: reads hook_event_name (was reading wrong key "event")
- **Security**: POST path validation (/retrieve only), negative Content-Length guard

### 3. Code Review (2 rounds, converged)
- Round 1: 3 HIGH (security), 3 MEDIUM (code), 2 LOW — all fixed
- Round 2: 0 CRITICAL, 0 HIGH, 0 MEDIUM — clean convergence
- Deferred: H2 daemon auth, H3 PID reuse SIGKILL (design decisions for Phase 4)

### 4. Test File Splits (7 files → 27 files)
- test_cli.py (2580) → 7 files (max 799)
- test_serve.py (1110) → 6 files (max 304)
- test_llm_reranker.py (969) → 4 files (max 410)
- test_indexer.py (958) → 3 files (max 505)
- test_eval.py (851) → 2 files (max 635)
- test_pipeline.py (831) → 2 files (max 578)
- test_expander.py (806) → 3 files (max 356)

### 5. Rule Update
- Updated global rule 15: "files under 400 lines" → "source files under 800 lines (400 preferred)"
- Rebuilt index and generated expansions for updated rule

### 6. New Project: alphaloom
- Financial market analysis/prediction project
- Name chosen: alphaloom ("Weave alpha from market signals")
- Available on PyPI
- Directory created at ~/projects/alphaloom
- Scoping discussion in progress

## Current State
- Branch: master
- Latest commit: 22d9aa1 (test file splits)
- llama-server: Gemma E4B on port 8081
- cuecard daemon: running on port 8452
- 951 tests, 100% coverage, ruff clean, mypy clean
- Hook latency: ~1.1s (PreToolUse), ~1.7s (UserPromptSubmit)

## alphaloom Project (new)
- PRD v0.1 complete at ~/projects/alphaloom/artifacts/prd-v0.1.md
- 3 rounds of review (architect + security + code quality), all R1/R2 findings fixed
- R3 not yet run (pane limit hit) — PRD is solid, ready for implementation
- CLAUDE.md and README created
- ai-hedge-fund cloned at ~/projects/ai-hedge-fund/ for reference
- 38 cuecard global rules (8 new this session)
- Research artifacts saved: landscape, OSS survey, architecture comparison

## What's Next
- alphaloom Phase 0: spike-test OpenBB + claude-agent-sdk
- alphaloom Phase 1: foundation (models, config, data layer)
- Run R3 PRD review for formal convergence
- Phase 4: PyPI publish for cuecard
- Phase 5.1: Markdown parser for CLAUDE.md ingestion
