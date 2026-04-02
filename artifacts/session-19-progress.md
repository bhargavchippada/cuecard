# Session 19 Progress State

> Date: 2026-04-02
> For resumption in next session

## What Was Accomplished

### 1. Production-Ready CLI
- `cuecard configure` — interactive setup (pipeline mode, endpoint, thresholds)
- `cuecard serve` — persistent daemon with PID management, 500ms fallback
- `cuecard hook` — CLI entry point for global install via `uv tool install`
- Auto-rebuild index after `cuecard rules expand`
- Rich progress bar for expand command
- `cuecard config` shows all fields (pipeline_mode, endpoint, sparse_enabled)
- Setup template includes all config options as commented examples

### 2. Hook Format Fix (CRITICAL)
- Claude Code requires `hookEventName` + `permissionDecision: "allow"` for PreToolUse
- UserPromptSubmit must NOT include `permissionDecision`
- Input field is `hook_event_name` (not `event`) — fixed adapter detection
- Stderr causes "hook error" display — suppressed with `2>/dev/null`
- "PreToolUse hook error" was caused by missing format fields, not actual errors

### 3. Live Compliance Testing
- Installed cuecard globally, set up 23 rules, verified hooks fire
- Tested with separate Claude Code agent (Cody) in tmux

**Compliance results:**
| Rule | Phrasing | Complied? |
|------|----------|-----------|
| Use uv not pip | Command style | Yes (via UserPromptSubmit) |
| Use uv not pip | Command style | Yes (via UserPromptSubmit) |
| Send Enter after tmux | Command style | No |
| Send Enter after tmux | Command style | No |
| Send Enter after tmux | Stronger framing only | No |
| Append Enter...without it text is pasted but never submitted | Explains WHY | **Yes** |
| Append Enter...without it text is pasted but never submitted | Explains WHY | **Yes** |

### 4. Landmark Finding: Rules That Explain WHY Work
- Command-style rules ("Send Enter after tmux send-keys"): 0/3 compliance
- Explanatory rules ("...without it text is pasted but never submitted"): 2/2 compliance
- Agent follows rules it understands, not rules it's told to obey
- This mirrors the reranker prompt finding: reasoning principles > command lists

### 5. Full Dataset Benchmark (35B)
| Event Type | Quality (F2) | Pos Recall | Noise | Neg Silence |
|-----------|-------------|-----------|-------|-------------|
| Basic (354) | 0.782 | 76.4% | 20.0% | 87.6% |
| Workflow (84) | 0.776 | 74.4% | 24.8% | 95.8% |

20% sample was ~2 pts optimistic. Full dataset confirms honest numbers.

## Key Findings

### Hook Timing
- **UserPromptSubmit** fires before agent plans → high influence on behavior
- **PreToolUse** fires after tool call is decided → can't change current call
- `additionalContext` is advisory, not blocking — `permissionDecision: "deny"` would block
- Advisory mode is correct default (12.4% false positive rate too high for enforcement)

### Rule Writing Best Practice
- Explain the **consequence of violation**, not just the action
- Include the **failure mode** — what goes wrong if ignored
- Reasoning principles outperform command lists
- The agent internalizes explanatory rules (Cody parroted the WHY back)

## Current State
- Branch: master, commit 8ee1093 pushed
- 946 tests, 100% coverage, ruff clean, mypy strict
- cuecard installed globally via `uv tool install`
- 23 global rules with 173 expansions
- Both PreToolUse + UserPromptSubmit hooks registered
- Boundary label: "RULES you must follow for this action to avoid failures"
- 35B llama-server running on port 8081

## What's Next
1. Expand rules with updated text (`cuecard rules expand`)
2. Rewrite all global rules to explain WHY (not just WHAT)
3. Compliance test suite — automated A/B testing with/without cuecard
4. `cuecard serve` daemon testing in production
5. Phase 4: PyPI publish
6. Phase 5.1: Markdown parser for CLAUDE.md ingestion

## Files Changed This Session

### Source Code
- `src/cuecard/cli.py` — configure command, serve command, hook command, config display
- `src/cuecard/cli_hooks.py` — both events, nested hook format, entry_has_cuecard
- `src/cuecard/cli_rules.py` — expand progress bar, auto-index after expand
- `src/cuecard/adapters/claude_code.py` — hook format fix, hook_event_name, daemon fast path
- `src/cuecard/serve.py` — NEW: daemon server module
- `src/cuecard/formatter.py` — boundary label update
- `src/cuecard/expander.py` — on_progress callback
- `src/cuecard/models.py` — ExpandProgress dataclass
- `src/cuecard/__init__.py` — ExpandProgress export

### Tests
- `tests/test_cli.py` — configure, hook, nested format tests
- `tests/test_adapter.py` — hook format assertions
- `tests/test_serve.py` — NEW: 56 daemon tests
- `tests/test_expander.py` — progress callback tests
- `tests/test_formatter.py` — boundary label update

### Docs
- `README.md` — CLI reference, configure, serve, benchmarks, rule writing guide
- `CLAUDE.md` — hook format docs, serve daemon, implementation status
