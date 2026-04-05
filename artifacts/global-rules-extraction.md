# Global Rules Extraction from Bhargav's Claude Code Rules

Extracted from common rules files, organized by impact. These are the principles that prevent real bugs and should apply to every cuecard phase.

---

## CRITICAL: Foundation Rules (Security, Correctness, Design)

### 1. **Never Mutate State**
- **Source:** common/coding-style.md — Immutability section
- **Core Principle:** ALWAYS create new objects, NEVER mutate existing ones
- **Why:** Immutability prevents hidden side effects, makes debugging easier, and enables safe concurrency. Mutations cause bugs that are hard to track because the original object's change propagates to unexpected places. With immutability, you can always trace where a value came from.
- **How to apply:** Every assignment should create a new object (`update(original, field, value)` returns a new copy, not `modify(original, field, value)`). Use frozen dataclasses, immutable patterns, `.copy()`, spreading, etc. Applied at: code review, every file edit.

### 2. **Research Before Coding**
- **Source:** common/development-workflow.md — Feature Implementation Workflow Step 0
- **Core Principle:** Search GitHub, library docs, package registries BEFORE writing any new implementation
- **Why:** Without research, you'll reinvent solved problems, miss battle-tested libraries, and build worse solutions from scratch. GitHub code search + Context7 takes 5-10 minutes and saves hours of implementation. Prefering proven libraries prevents maintenance burden and security issues.
- **How to apply:** Before writing code: (1) `gh search repos` + `gh search code` for existing implementations, (2) Context7 for library docs, (3) package registries (npm, PyPI, crates.io), (4) only then write new code. Applied at: task intake.

### 3. **Understand Existing Code Before Writing**
- **Source:** common/task-intake.md — Step 1, common/tool-failure-recovery.md — Step 2c
- **Core Principle:** Read CLAUDE.md, README, existing patterns — understand what exists before writing anything
- **Why:** Without understanding existing code, you'll write code that conflicts with established patterns, miss critical context, and make wrong architectural decisions. Reading 2-3 similar files takes 5 minutes and prevents hours of rework. This is about learning the "why" behind decisions, not just the "what".
- **How to apply:** On every task: (1) Read CLAUDE.md first, (2) `git status`/`git log` to understand state, (3) read 2-3 similar files in codebase, (4) follow established patterns exactly, don't invent new ones. Applied at: task intake.

### 4. **Input Validation at System Boundaries**
- **Source:** common/coding-style.md — Input Validation section, common/security.md
- **Core Principle:** ALWAYS validate all user input, API responses, file content — never trust external data
- **Why:** Without validation, invalid data propagates through the system causing cascading failures that are hard to debug. Schema-based validation lets you fail fast with clear errors. This is your last defense — everything beyond this boundary might be corrupted.
- **How to apply:** At every entry point (user input, API calls, file reads, environment variables): validate data type, structure, and constraints. Fail fast with clear error messages. Applied at: code review, quality checks.

### 5. **Secrets Never in Source Code**
- **Source:** common/security.md, common/quality-checks-post-edit.md
- **Core Principle:** NEVER hardcode secrets (API keys, tokens, passwords). ALWAYS use environment variables or secret managers
- **Why:** Hardcoded secrets get committed to git history forever — even if you delete them, they're in backups and git logs. You can't "unclone" a repo. The remediation cost (rotating all exposed keys) is massive compared to the 30-second cost of using env vars.
- **How to apply:** Check every commit: no `.env`, credentials, API keys, connection strings. Extract to env vars. Applied at: pre-commit hooks.

### 6. **Error Handling at Every Level**
- **Source:** common/coding-style.md — Error Handling section
- **Core Principle:** Handle errors explicitly at EVERY level. Never silently swallow errors
- **Why:** Silent failures are worse than loud failures — they hide bugs that seem fine until production. Graceful degradation that silently falls back can mask total failure of a stage (e.g., reranker returning no parses, yet pipeline continues with embedding results). Log loudly when falling back.
- **How to apply:** Every try/catch, every async operation, every HTTP call must have explicit error handling. Log the error + context. Provide user-friendly messages in UI, detailed context in server logs. Applied at: code review, quality checks.

---

## HIGH: Prevention Rules (Bugs, Failures, Hidden Issues)

### 7. **Don't Retry Blindly on Failure**
- **Source:** common/tool-failure-recovery.md — Step 1, Step 3
- **Core Principle:** When ANY tool fails, READ THE ERROR first, then diagnose root cause. Don't retry the same thing twice
- **Why:** Retrying a failed command rarely fixes the underlying problem — you're just spinning wheels. If it failed once, it will fail again with the same parameters. Root cause investigation is the only durable fix. This saves 30+ minutes per failure.
- **How to apply:** On tool failure: (1) Read FULL error message (not just first line), (2) identify error TYPE, (3) search CLAUDE.md → README → patterns → rules → docs in order, (4) understand root cause before attempting fix. Applied at: every tool failure.

### 8. **Convergence-Based Quality Review**
- **Source:** common/quality-checks-stop.md — Step 5 (Convergence-Based)
- **Core Principle:** Keep running review rounds until TWO CONSECUTIVE ROUNDS produce 0 CRITICAL, 0 HIGH, 0 MEDIUM findings. Never use fixed round count
- **Why:** A single clean round can be luck — the second round confirms you actually fixed the problems. Fixed round counts (e.g., "do 3 rounds") ship with unknown bugs. Convergence proves the system is stable. Fixes from one round often introduce new issues — the counter resets.
- **How to apply:** Review → fix → review again. Track rounds. Only stop when rounds N and N+1 both come back clean. Security > logic > style > tests (priority order). Applied at: after code changes, before committing.

### 9. **Surgical Changes Only — No Scope Creep**
- **Source:** common/coding-style.md — Surgical Changes section (marked CRITICAL)
- **Core Principle:** ONLY modify what the task requires. Nothing else
- **Why:** Scope creep turns a 30-minute task into a 3-hour task. "While I'm here" refactoring hides the actual changes, makes reviews harder, introduces unplanned bugs. The user asked for X — do X. If you see something else broken, mention it, don't fix it.
- **How to apply:** On every task, before starting: list the files that WILL change. Stick to that list. Don't refactor adjacent code, don't add abstractions for hypothetical future use, don't rename variables you didn't change. Applied at: task intake, code review.

### 10. **Don't Normalize Slow — Investigate Immediately**
- **Source:** common/performance.md — SDK / LLM Call Performance section
- **Core Principle:** If something takes 30s and should take 3s, that's a 10x signal. Don't normalize slow — investigate immediately
- **Why:** Performance problems are symptoms of deeper issues: wrong algorithm, missing cache, inefficient query, model thrashing. A 10x slowdown means something fundamental is broken. Ignoring it bakes slow into every future use. Investigation takes 15-30 minutes; the payoff is every subsequent call being 10x faster.
- **How to apply:** Before any pipeline run, run a PING test (trivial prompt, measure latency). If >10s, investigate. Benchmark before full runs. If a stage regresses, don't patch downstream — fix the stage. Applied at: performance validation, pipeline design.

### 11. **Audit Ground Truth as Aggressively as Predictions**
- **Source:** SOUL.md (session 13, 18), CLAUDE.md session notes
- **Core Principle:** Wrong test fixtures are indistinguishable from wrong model outputs. Audit the ruler before measuring the table
- **Why:** You can spend days "fixing" the model when the problem is wrong test expectations. A fixture that expects `git rebase` to mean "force push" will mark correct outputs as wrong. Auditing fixtures is as important as auditing predictions — both can be wrong.
- **How to apply:** When evaluation results look wrong: check the fixture expectations first (not the model output). Mine real sessions — synthetic data overweights positives. Applied at: eval setup.

### 12. **The Right Abstraction > No Abstraction > Wrong Abstraction**
- **Source:** SOUL.md (session 12), CLAUDE.md
- **Core Principle:** A wrong abstraction is worse than duplication. Three similar lines is better than a premature abstraction
- **Why:** Wrong abstractions force you to work around them, add complexity, and break on edge cases. Three lines of duplication costs 5 minutes to understand. A wrong abstraction costs 30 minutes every time you hit its limitations. Build only what the task requires.
- **How to apply:** Before extracting a shared function: (1) benchmark the abstraction first, (2) confirm it helps, (3) only then build. Don't build for hypothetical future requirements. Applied at: code design, quality review.

### 13. **Graceful Degradation Masks Failures**
- **Source:** SOUL.md (session 12), CLAUDE.md
- **Core Principle:** Graceful fallback can completely hide that a stage failed. Test the actual output, not just code paths
- **Why:** A silent fallback from Stage 2 to Stage 1 results hides that Stage 2 is broken. Tests pass (the code ran), coverage is 100% (the path executed), but Stage 2 is dead. You only discover this in production. Log loudly on fallback, benchmark the actual output.
- **How to apply:** When a stage is optional, log at WARNING when it falls back. Benchmark actual outputs, not code paths. Applied at: optional stages, E2E validation.

---

## MEDIUM: Workflow Rules (Efficiency, Clarity, Maintainability)

### 14. **Measurement Infrastructure Pays for Itself**
- **Source:** SOUL.md (session 16), CLAUDE.md
- **Core Principle:** Build observability at each stage. Track metrics that inform decisions within the same session
- **Why:** Blind optimization wastes time. Metrics tell you where the problem actually is. "Per-tier recall breakdown" + "noise ratio" revealed that the issue was cross-domain confusion, not overall model capacity. Built once, used repeatedly.
- **How to apply:** For each pipeline stage: define 2-3 metrics that matter (recall, precision, latency, noise). Track per-tier if applicable. Use these metrics to make decisions. Applied at: pipeline design.

### 15. **Test Coverage is Necessary but Not Sufficient**
- **Source:** SOUL.md (session 16), CLAUDE.md, common/testing.md
- **Core Principle:** 100% line coverage doesn't mean the logic is tested. Use mutation testing to verify tests actually catch bugs
- **Why:** Line coverage only says "this code ran." It doesn't say "the test would catch a bug in this code." Mutation testing flips operators and shows which tests fail to detect the change. The gap is where bugs hide.
- **How to apply:** Reach 100% line coverage first. Then run `mutmut` to find surviving mutants (mutations tests don't catch). Focus on dangerous mutants (formula operators, boundary conditions, error handling). Applied at: pre-commit, quality review.

### 16. **Consistency Over Clever**
- **Source:** common/tool-failure-recovery.md — Step 2c
- **Core Principle:** Follow the existing pattern exactly. Don't introduce a new pattern alongside old ones
- **Why:** One pattern everywhere is easier to learn and maintain. Two patterns double the mental load for everyone reading the code. If a better pattern exists, refactor existing code to match — don't let two coexist.
- **How to apply:** Before writing new code, read 2-3 similar files. Copy their structure exactly. If you think a better pattern is needed, refactor existing code to use it first. Applied at: code design, review.

### 17. **Document Decisions, Not Just Code**
- **Source:** common/pre-compaction.md, CLAUDE.md updates
- **Core Principle:** Update CLAUDE.md when structure/architecture changes. Update README when user-facing behavior changes
- **Why:** Documentation saves the next session hours of reverse-engineering. CLAUDE.md captures "why did we do it this way" — the decision rationale is more valuable than the code itself.
- **How to apply:** On every task completion: (1) Update CLAUDE.md if structure/architecture/conventions changed, (2) Update README if user-facing behavior changed, (3) Update env examples if new variables added. Applied at: task completion.

### 18. **Make Decisions Observable**
- **Source:** SOUL.md (session 19), CLAUDE.md
- **Core Principle:** Reasoning principles beat command lists. Teach HOW to think, not WHAT to match
- **Why:** "Does this modify state?" generalizes to unseen queries. "git commit is not a secret operation" doesn't handle similar cases. Principles scale; command lists don't. Plus, observable reasoning helps debug when decisions go wrong.
- **How to apply:** When creating rules/prompts: frame as principles ("when state changes, log context") not commands ("log git commits"). In code: add comments explaining the reasoning, not just what happens. Applied at: prompt engineering, code review.

### 19. **Consequences Beat Commands for Rule Injection**
- **Source:** SOUL.md (session 19), CLAUDE.md
- **Core Principle:** "Always do X" fails. "Do X because without it, Y happens" succeeds
- **Why:** Agents (and humans) follow rules they understand. "Send Enter after tmux send-keys" fails 3/3 times. "Without Enter, text is pasted but never submitted" succeeds 2/2 times. The agent internalizes the WHY and generalizes to similar cases.
- **How to apply:** When writing rules for injection: explain the consequence, not just the action. Format: "Do X — without it, Y happens (which breaks Z)." Applied at: rule writing.

### 20. **Search Tools Route to Right Answer**
- **Source:** common/search-routing.md
- **Core Principle:** Use the right tool for the right query. ripgrep for exact symbols, mgrep for concepts, Context7 for library docs, exa for research
- **Why:** Searching with the wrong tool wastes time. ripgrep on "what is error handling" returns nothing useful. mgrep finds the pattern immediately. Context7 avoids needing web search for library docs.
- **How to apply:** For exact strings/symbols → Grep/Glob. For concepts/semantics → mgrep. For library docs → Context7. For research → exa. For URLs → firecrawl. Applied at: research phase, debugging.

---

## LOW-MEDIUM: Infrastructure Rules (Tests, Code Quality)

### 21. **100% Test Coverage on Changed Files**
- **Source:** common/testing.md, common/quality-checks-stop.md
- **Core Principle:** Every new code path must be tested. Minimum 80% project coverage, 100% on changed files
- **Why:** Coverage is a floor, not a ceiling. Untested paths are where bugs hide. Session 12 proved: "tests passed, coverage was 100%" — but reranker was completely broken. Tests need to validate actual output, not just code paths.
- **How to apply:** Write tests before implementation (TDD). Verify coverage on changed files reaches 100%. Use mutation testing to confirm tests actually catch bugs. Applied at: code review, pre-commit.

### 22. **Fix Implementation, Not Tests (Unless Test is Wrong)**
- **Source:** common/tool-failure-recovery.md — Step 3: Test Failures
- **Core Principle:** When a test fails, fix the code, not the test. Unless the test itself is wrong
- **Why:** Tests are ground truth. If the test fails, the code is likely wrong. If you fix the test, you're just hiding the bug. The only exception is when the test expectation itself is wrong — audit that first.
- **How to apply:** On test failure: (1) read the full output (expected vs actual), (2) read the test to understand intent, (3) fix the implementation, (4) verify test passes, (5) only change test if it was genuinely wrong. Applied at: quality review.

### 23. **Run Build + Type Check + Tests After Every Fix**
- **Source:** common/quality-checks-post-edit.md, common/tool-failure-recovery.md
- **Core Principle:** After every fix, verify build succeeds with zero warnings, type checker passes, all tests pass
- **Why:** Partial verification hides problems. A fix that passes tests but breaks the build wastes 30 minutes later. Run the full suite after each change, not in batches.
- **How to apply:** Fix → build → type check → tests. If any fails, fix and repeat. Never batch fixes. Applied at: code review, pre-commit.

### 24. **Never Disable Lint/Type Rules Without Documentation**
- **Source:** common/quality-checks-post-edit.md, common/tool-failure-recovery.md — Type/Lint Errors
- **Core Principle:** Fix the code to satisfy the rule. Never add `eslint-disable`, `@ts-ignore`, `@SuppressWarnings` unless there's a genuine false positive (document why)
- **Why:** Disabled rules accumulate silently. A future developer doesn't know why it was disabled — was it safe or a bomb? Disabling rules is a form of technical debt that compounds.
- **How to apply:** When lint/type error occurs: (1) understand WHY the rule exists, (2) fix the code to satisfy it, (3) only disable if proven false positive + document why. Applied at: linting, type checking.

---

## Summary Table: Rules by Prevention Value

| # | Rule | Prevents | Impact |
|----|------|----------|--------|
| 1 | Never Mutate State | Hidden side effects, hard-to-debug cascades | Every session |
| 2 | Research Before Coding | Reinventing, missing libraries, worse solutions | Every feature |
| 3 | Understand Existing Code | Wrong assumptions, architecture conflicts | Every task |
| 4 | Input Validation at Boundaries | Cascading failures, corrupted data | Every integration |
| 5 | Secrets Never in Code | Exposed keys, massive remediation cost | Every commit |
| 6 | Error Handling at Every Level | Silent failures, hidden bugs, production surprises | Every feature |
| 7 | Don't Retry Blindly | Wasted time diagnosing, spinning wheels | Every failure |
| 8 | Convergence-Based Quality | Shipping with unknown bugs | Every code change |
| 9 | Surgical Changes | Scope creep, hidden bugs, longer reviews | Every task |
| 10 | Don't Normalize Slow | Baked-in slowness, undiagnosed problems | Every pipeline |
| 11 | Audit Ground Truth | Wrong conclusions, wasted fix attempts | Every eval |
| 12 | Right > No > Wrong Abstraction | Workarounds, broken edge cases, rework | Every design |
| 13 | Graceful Degradation Masks Failures | Dead stages, hidden bugs, production failures | Every optional stage |
| 14 | Measurement Infrastructure | Blind optimization, wasted time | Every pipeline |
| 15 | Test Coverage + Mutation Testing | False confidence, bugs that tests miss | Every test suite |
| 16 | Consistency Over Clever | High mental load, duplicated knowledge | Every codebase |
| 17 | Document Decisions | Lost context, reverse-engineering, repeated mistakes | Every session |
| 18 | Make Decisions Observable | Brittle rules, no generalization | Every rule injection |
| 19 | Consequences > Commands | Rule failures, low compliance | Every rule |
| 20 | Right Search Tool | Wasted research time, wrong answers | Every investigation |
| 21 | 100% Coverage on Changes | Untested paths, hidden bugs | Every commit |
| 22 | Fix Code, Not Tests | Hidden bugs, false confidence | Every test failure |
| 23 | Full Verification Cycle | Partial bugs, discover in production | Every fix |
| 24 | Never Silently Disable Rules | Technical debt accumulation, unknown bombs | Every codebase |

---

## How These Apply to cuecard

These principles should become cuecard's global rules:

1. **Code Quality Enforcement**: Rules 1, 6, 21-24 (never ship untested, uncovered, error-silenced code)
2. **Architecture Safety**: Rules 2-4, 12 (search + understand + validate before building)
3. **Research & Reuse**: Rule 2 (GitHub → docs → build)
4. **Observability**: Rules 10, 14 (measure, don't guess; don't normalize slow)
5. **Decision Quality**: Rules 7-8, 11, 18-19 (converge on clean, understand ground truth, make decisions observable)
6. **Maintainability**: Rules 16-17 (one pattern, document decisions)

### Proposed cuecard Global Rules File
Should include the CRITICAL rules (1-6) and HIGH rules (7-13) as mandatory enforcement. MEDIUM rules (14-20) as best practices. LOW rules (21-24) as code quality hooks.
