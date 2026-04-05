# Session 19 Progress State

> Date: 2026-04-02 through 2026-04-04
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
- Before fix: every tool call showed "PreToolUse hook error"
- After fix: hooks fire cleanly as "additional context"

### 3. Live Compliance Testing
- Installed cuecard globally via `uv tool install`
- Set up 30 rules, verified hooks fire for both events
- Tested with separate Claude Code agent (Cody) in tmux

**Compliance results:**
| Rule | Phrasing | Complied? |
|------|----------|-----------|
| Use uv not pip | Command style | Yes (UserPromptSubmit) |
| Use uv not pip | Command style | Yes (UserPromptSubmit) |
| Send Enter after tmux | Command style | No |
| Send Enter after tmux | Command style | No |
| Send Enter after tmux | Stronger framing only | No |
| Append Enter...without it text is pasted but never submitted | WHY format | **Yes** |
| Append Enter...without it text is pasted but never submitted | WHY format | **Yes** |

### 4. Landmark Finding: Rules That Explain WHY Achieve Compliance
- Command-style rules: 0/3 compliance
- Explanatory rules (with consequence): 2/2 compliance
- Agent even parroted back the reasoning: "without the trailing Enter, the text only gets pasted but never submitted"
- **Consequences beat commands for rule writing** — same lesson as reasoning principles vs command lists in reranker prompts

### 5. Curated 30 Global Rules in WHY Format
Organized by category (Security, Testing, Code Quality, Git, Orchestration, Methodology, Search):
- Mined from SOUL.md (19 arcs), 587 eval fixtures, ~/.claude/rules/, and instincts
- Reviewed one-by-one with Bhargav, 1 removed (SQL parameterized queries — not applicable)
- Each rule explains the CONSEQUENCE of violation, not just the action
- 30 rules, 250 expansions (avg 8.3/rule)
- Location: `/home/turiya/.cuecard/rules/global.txt`

### 6. Full Dataset Benchmark (35B)
| Event Type | Quality (F2) | Pos Recall | Noise | Neg Silence |
|-----------|-------------|-----------|-------|-------------|
| Basic (354) | 0.782 | 76.4% | 20.0% | 87.6% |
| Workflow (84) | 0.776 | 74.4% | 24.8% | 95.8% |

20% sample was ~2 pts optimistic. Full dataset confirms honest numbers.

## Key Findings

### Hook Timing (Fundamental Architecture)
- **UserPromptSubmit** fires before agent plans → high influence on behavior
- **PreToolUse** fires after tool call is decided → can't change current call
- `additionalContext` is advisory, not blocking — `permissionDecision: "deny"` would block
- Advisory mode is correct default (12.4% false positive rate too high for enforcement)

### Rule Writing Best Practice (Documented in README)
- Explain the **consequence of violation**, not just the action
- Include the **failure mode** — what goes wrong if ignored
- Reasoning principles outperform command lists
- The agent internalizes explanatory rules (Cody parroted the WHY back)

### Boundary Label
- Changed from "user-defined guidelines relevant to this action"
- To "RULES you must follow for this action to avoid failures"
- Stronger framing alone didn't change compliance (0/3 still)
- Only when combined with explanatory rules did compliance jump to 2/2

## Current State
- Branch: master, commits cf63db0 + ed62e07 pushed
- 946 tests, 100% coverage, ruff clean, mypy strict
- cuecard installed globally via `uv tool install`
- 30 curated global rules with 250 expansions
- Both PreToolUse + UserPromptSubmit hooks registered
- Boundary label: "RULES you must follow for this action to avoid failures"
- 35B llama-server running on port 8081 (or may need restart)

## What's Next (Priority Order)

### Near-term
1. More compliance A/B testing with Cody (automate it?)
2. Test rule retrieval on your real day-to-day tasks
3. Add rules as they're needed from real workflows

### Medium-term
4. `cuecard compliance-test` command — automated A/B with/without cuecard
5. Phase 4: PyPI publish
6. Phase 5.1: Markdown parser for CLAUDE.md ingestion

### Future
7. Async hooks (if Claude Code supports) to reduce blocking latency
8. Confidence-based enforcement mode (deny only at score > 0.95)
9. Production daemon (`cuecard serve`) reliability testing

## Files Changed This Session

### Source Code (cuecard)
- `src/cuecard/cli.py` — configure, serve, hook commands, config display
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

### Docs & Artifacts
- `README.md` — CLI reference, configure, serve, benchmarks, rule writing guide
- `CLAUDE.md` — hook format docs, serve daemon, implementation status
- `artifacts/session-19-progress.md` — this file
- `artifacts/global-rules-extraction.md` — mined from claude rules
- `artifacts/soul-rules-extraction.md` — mined from SOUL.md
- `.gitignore` — mutants/, enriched corpora, .claude/

### User Environment
- `~/.cuecard/rules/global.txt` — 30 curated rules in WHY format
- `~/.cuecard/config.toml` — llm-local mode configured
- `~/.cuecard/index/` — rebuilt with 30 rules + 250 expansions
- `~/.claude/settings.json` — cuecard hooks registered (both events)
