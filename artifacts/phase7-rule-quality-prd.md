# Phase 7: Rule Quality — Trigger-Aware Rewriting PRD v1.0

**Author:** Turiya
**Date:** 2026-04-08
**Status:** DRAFT

## Objective

Rewrite all 109 eval rules (and production global.txt rules) to include **trigger conditions** — explicit descriptions of WHEN each rule should fire. This is the single highest-ROI improvement for retrieval quality because the rule text is the only signal the embedding model sees.

### Success Criteria

- [ ] All 109 rules rewritten with trigger conditions ("When X: do Y — because Z")
- [ ] Each rule re-categorized as tool_use / workflow / both based on trigger context
- [ ] Rules are concise (under 300 chars preferred, 400 max) to avoid context bloat
- [ ] Re-expanded with new trigger vocabulary as seeds
- [ ] PreToolUse F2 improves by ≥10 points (0.566 → 0.666+)
- [ ] Affinity accuracy improves from 86.2% to 90%+
- [ ] Fixtures updated to match rewritten rule text
- [ ] Production global.txt updated with rewritten rules

## Problem Statement

Current rules describe WHAT to do but not WHEN:
- `"Use uv for all Python package operations, never pip"` — when? Every tool call? Only Bash?
- `"Run convergence reviews after each milestone"` — when? UserPromptSubmit? Stop? Both?
- `"Keep functions under 50 lines"` — when? Only Edit? Write too? Read?

The embedding model sees these rules and maps them to a broad semantic neighborhood. `"Run convergence reviews"` lands near testing, quality, methodology, reviews — matching 15+ queries that are topically similar but not actually trigger-relevant.

**Root cause:** The retrieval system has no trigger signal. It can only match TOPIC, not TIMING. Adding trigger conditions to rule text gives the embedding model the vocabulary it needs to match specific queries.

**Evidence:**
- Session 19: "rules that explain WHY work, rules that just say WHAT don't" — consequences beat commands
- Session 24: binary classification with golden examples >> multi-class — simpler abstractions win
- Session 25: affinity accuracy 86.2% — but 78 rules fire on PreToolUse when maybe 40 should
- 109-rule corpus caused 30% F2 drop from 32-rule baseline — semantic overlap is the core problem

## Design Principles

### P1: "When X: do Y — because Z" format
Every rule should answer three questions:
1. **When** — trigger condition (tool name, action type, context)
2. **What** — the instruction
3. **Why** — the consequence of not following

### P2: Conciseness over completeness
Rules injected into agent context compete for attention. Shorter rules are read; longer rules are skimmed.
- Target: 150-250 chars (sweet spot)
- Maximum: 400 chars (hard limit for eval)
- If a rule needs >400 chars, split into two focused rules

### P3: Tool names as trigger vocabulary
PreToolUse queries are `"{tool_name}: {input}"`. Rules that mention tool names explicitly will score higher:
- "When running Bash commands with `git commit`" matches `"Bash: git commit -m 'fix'"` 
- "Before committing" matches weakly — no tool name overlap
- "When editing Python files (Edit, Write)" matches `"Edit: src/auth.py"`

### P4: Event-specific language
Different events need different trigger vocabulary:
- **PreToolUse**: "When running X", "Before executing", "When editing", "When writing"
- **PostToolUse**: "After running X", "When output shows", "If the result contains"
- **UserPromptSubmit**: "When the user asks to", "Before starting", "When receiving a request to"
- **SubagentStart**: "When delegating to", "Before spawning", "When launching a subagent for"
- **Stop**: "Before stopping", "At turn end", "Before presenting results"

### P5: One rule, one trigger
Avoid compound rules that fire on multiple unrelated triggers. Instead of:
> "Use uv not pip AND run ruff before committing"

Split into:
> "When running pip/pip install/pip freeze: use uv instead"
> "When running git commit on Python code: run ruff and mypy first"

### P6: Re-categorize based on trigger
The trigger condition makes the category obvious:
- "When running Bash: git push --force" → `tool_use` (constrains a specific tool)
- "When the user asks to build a feature" → `workflow` (guides process)
- "When editing Python files: add type hints" → `both` (constrains tool + guides process)

## Rewrite Examples

### Security rules (mostly tool_use — constrain specific actions)

| Original | Rewritten | Chars | Category |
|----------|-----------|-------|----------|
| Never commit secrets (API keys, tokens, passwords, connection strings) to git — once pushed, secrets are in the history forever... | When running git add, git commit, or writing .env files: never include secrets (API keys, tokens, passwords) — once pushed, they're in history forever and require rotation | 198 | tool_use |
| Never use eval(), exec(), or shell=True in production code — these execute arbitrary input... | When writing Python code (Edit, Write): never use eval(), exec(), or shell=True — these execute arbitrary input as code, enabling RCE attacks | 165 | tool_use |
| Always validate user input at system boundaries — unvalidated input propagates... | When writing API endpoints, form handlers, or CLI parsers (Edit, Write): validate all user input — unvalidated input propagates corrupted data through the entire system | 188 | tool_use |

### Testing rules (mostly both — constrain tools + guide process)

| Original | Rewritten | Chars | Category |
|----------|-----------|-------|----------|
| Require 100% test coverage on all new code before committing — uncovered code is untested... | When running git commit after writing new code: verify 100% test coverage first — uncovered code breaks silently in production | 140 | both |
| Always mock external calls in unit tests — real calls make tests slow... | When writing pytest tests (Edit, Write): mock all external calls (LLM, APIs, network) — real calls make tests slow, flaky, and non-deterministic | 167 | tool_use |
| The full unit test suite must complete in under 5 seconds... | When running pytest (Bash): the full suite must complete in under 5 seconds — if slower, something is unmocked; investigate immediately | 145 | tool_use |

### Methodology rules (mostly workflow — guide process decisions)

| Original | Rewritten | Chars | Category |
|----------|-----------|-------|----------|
| Search GitHub for existing implementations before writing from scratch... | Before writing new utility code: search GitHub for existing implementations — 80% of problems are already solved with battle-tested code | 145 | workflow |
| Run convergence reviews after each implementation milestone... | After completing an implementation phase or PRD revision: run convergence reviews with parallel agents — per-phase reviews catch bugs when they're cheap to fix | 175 | workflow |
| COMPLEX tasks (10+ files) require a full PRD... | When the user requests a feature touching 10+ files or new architecture: write a full PRD first — complex tasks without specs produce rework and missed requirements | 185 | workflow |

### Agent rules (mostly workflow — guide orchestration)

| Original | Rewritten | Chars | Category |
|----------|-----------|-------|----------|
| Set subagent scope boundaries and terminate stale agents... | When launching subagents (SubagentStart): set explicit scope boundaries — unbounded reviewers find infinite speculative findings, consuming context without improving quality | 190 | both |
| Delegate implementation to worker agents via tmux... | When delegating to worker agents: use tmux sessions with cron-based monitoring — direct delegation without monitoring leads to stale or stuck agents | 160 | both |

## Implementation Phases

### Phase 1: Rewrite all 109 rules
- Apply "When X: do Y — because Z" format
- Re-categorize each as tool_use / workflow / both
- Target 150-250 chars, max 400
- Produce `eval/corpora/rules_global_v2.txt` (don't overwrite v1 yet)
- Produce updated `eval/corpora/rules_global_tagged_v2.json`

### Phase 2: Validate rewrites
- Launch 3 subagents to cross-check:
  - Agent A: verify no meaning was lost (original vs rewritten)
  - Agent B: verify trigger conditions are specific (no vague "when appropriate")
  - Agent C: verify categories match trigger conditions
- Iterate until convergence

### Phase 3: Update fixtures
- All fixture `should_match` rules must use the rewritten text (exact match)
- This is a mechanical find-replace: old text → new text across all fixture files
- Verify no broken references

### Phase 4: Re-expand
- Generate new expansions using rewritten rules as source
- Trigger vocabulary becomes expansion seeds ("git commit", "Edit: src/", "pytest")
- Rebuild index with new expansions

### Phase 5: Benchmark
- Run E2E benchmark with rewritten rules + new fixtures
- Compare against session 24 baseline (old rules, old fixtures)
- Diagnostic: run with top_k=30 to check recall ceiling
- Per-event breakdown: did trigger conditions help PreToolUse most?

### Phase 6: Audit mismatches
- For every false positive/negative in the benchmark:
  - Was the model wrong? → the rule needs a better trigger condition
  - Was the fixture wrong? → fix the expectation
  - Was the category wrong? → re-classify
- This is the closed-loop feedback that improves quality iteratively

### Phase 7: Deploy to production
- Update `~/.cuecard/rules/global.txt` with rewritten rules
- Update `eval/corpora/rules_global.txt` with rewritten rules
- Re-expand production rules
- Rebuild production index

## Conciseness Guidelines

Rules are injected as `additionalContext` in hook output. Every char costs attention.

| Length | Quality | Target % |
|--------|---------|----------|
| < 150 chars | Excellent — short, punchy, memorable | 30% |
| 150-250 chars | Good — detailed but scannable | 50% |
| 250-350 chars | Acceptable — thorough but long | 15% |
| 350-400 chars | Maximum — only for complex rules | 5% |
| > 400 chars | Split into two rules | 0% |

**Current distribution:** Average ~220 chars, max 398 chars. Many rules have verbose "because" clauses that repeat the obvious. Trim these:

BEFORE (285 chars):
> "Always close file handles, database connections, and HTTP sessions after use — unclosed resources leak memory and exhaust connection pools, eventually crashing the service"

AFTER (175 chars):
> "When opening files, DB connections, or HTTP sessions (Edit, Bash): always close after use — unclosed resources leak memory and exhaust connection pools"

Cut 110 chars by:
1. Added trigger condition (+30 chars)
2. Removed "eventually crashing the service" (-40 chars, obvious consequence)
3. Tightened phrasing (-100 chars)

## Rule Writing Checklist

For each rule, verify:
- [ ] Has explicit trigger ("When X", "Before Y", "After Z")
- [ ] Mentions specific tools or actions when applicable (Bash, Edit, Write, git, pytest, pip)
- [ ] Under 300 chars (ideally 150-250)
- [ ] Consequence is specific, not generic ("causes X" not "is bad practice")
- [ ] Category matches trigger (tool names → tool_use, process words → workflow)
- [ ] No compound instructions (one rule, one trigger)
- [ ] No redundancy with other rules (check for overlapping triggers)

## Metrics

### Primary: PreToolUse F2 (target ≥0.666)
The main hook we're optimizing for. 10+ point improvement from trigger conditions.

### Secondary
- Affinity accuracy: 86.2% → 90%+ (trigger conditions make categories obvious)
- Avg chars per rule: 220 → 180 (conciseness)
- Rules per event (via affinity): PreToolUse 78 → 50-60 (more precise categorization)
- Positive recall on rewritten rules ≥ original (no meaning lost)

## Risks

1. **Meaning loss**: Rewriting could drop nuance. Mitigation: validation subagent compares original vs rewritten.
2. **Over-specificity**: Trigger conditions too narrow → rules don't fire when they should. Mitigation: benchmark recall must not drop.
3. **Fixture breakage**: All fixtures reference exact rule text. Mechanical replace should work, but verify zero broken references.
4. **Expansion quality**: New expansions based on rewritten rules may differ from v5 expansions. Full re-expansion needed.

## Connection to Other Phases

- **Phase 6 (Completion Gate)**: Stop hook queries will use Phase 6 format. Rewritten rules with "At turn end: verify X" will match better.
- **Affinity**: Re-categorized rules update the affinity index. More precise categories → better event masks → less noise.
- **Fixtures**: Updated fixtures from session 25 (984 total) need rule text replacement. The mined Stop fixtures (41) already use Phase 6 queries and will benefit from rewritten rules.
