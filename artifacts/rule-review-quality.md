# Rule Rewrite Quality Review

109 rules total: 83 rewrites, 26 keeps. Reviewed against 6 criteria: general enough, specific enough, no meaning lost, concise, category correctness, over-specification.

## Summary

- **GOOD:** 52 rewrites
- **FIX:** 25 rewrites
- **REVERT:** 2 rewrites
- **KEEP issues:** 4 (flagged despite being keeps)

---

## Rewrites with Issues

### FIX: Over-specification (library/tool enumeration)

**Rule 7 — uv over pip**
`"When running pip, pip install, pip freeze, or any Python package command (Bash): use uv instead"`
FIX: Enumerating `pip, pip install, pip freeze` is fragile. Simplify trigger to "When running Python package commands (Bash):" — the rule body already says "use uv instead of pip."

**Rule 10 — close resources**
`"When writing code that opens files, database connections, or HTTP sessions (Edit, Write): always close them after use with context managers (with/async with)"`
FIX: Added "with context managers (with/async with)" which is implementation-specific (Python). Original said "close after use" which is language-agnostic. Drop the parenthetical — the rule should work for any language.

**Rule 16 — .gitignore for build artifacts**
`"When running build commands (Bash: npm run build, cargo build, python setup.py) or adding new generated output directories:"`
FIX: Enumerates `npm run build, cargo build, python setup.py`. Simplify to "When running build commands or adding generated output directories:" — the examples add nothing for retrieval and will miss `make`, `gradle`, `go build`, etc.

**Rule 43 — asyncio over threads**
`"When writing concurrent I/O code (Edit, Write) with threads, threading.Thread, or ThreadPoolExecutor:"`
FIX: Enumerating `threading.Thread, ThreadPoolExecutor` is fragile. Simplify to "When writing concurrent I/O code with threads:" — the body already says "use asyncio instead."

**Rule 46 — add dependencies**
`"When adding new dependencies (Bash: uv add, npm install, pip install, cargo add):"`
FIX: Enumerating package manager commands is fragile. Simplify to "When adding new dependencies:" — the Bash tool prefix already narrows the context.

**Rule 48 — quality checks before commit**
`"When running git commit (Bash): run quality checks (ruff, mypy, pytest) before every commit"`
FIX: Enumerating `ruff, mypy, pytest` is Python-specific. Original said "lint, type, and test failures." Keep the general categories: "run quality checks (lint, type check, tests) before every commit."

**Rule 55 — research new libraries**
`"When adding a new library or API dependency (Bash: uv add, npm install, pip install):"`
FIX: Same issue as rule 46. Drop the enumerated commands. "When adding a new library or API dependency:" is sufficient.

**Rule 44 — parameterized queries**
`"When writing database queries (Edit, Write) with string formatting, f-strings, or concatenation:"`
FIX: "f-strings" is Python-specific. Simplify to "When writing database queries with string interpolation or concatenation:" — covers all languages.

### FIX: Trigger too vague / matches everything

**Rule 9 — error handling**
`"When writing or editing try/except, catch, .catch(), or error handling code (Edit, Write):"`
FIX: This trigger fires on ALL error handling code, but the rule is about EMPTY catch blocks specifically. Better trigger: "When writing try/except or catch blocks:" — shorter and equally specific.

**Rule 38 — debug statements**
`"When writing or editing code (Edit, Write): remove console.log, print(), and debug statements"`
FIX: Trigger "When writing or editing code" matches literally everything. Better: "Before committing code:" — that's when debug statements should be caught. The original "No console.log or print debug statements in production code" was already clear about when.

**Rule 39 — immutable data structures**
`"When writing data structures, classes, or containers (Edit, Write): prefer immutable data structures (frozen dataclasses, tuples, frozensets)"`
FIX: Trigger matches almost all code. Also "(frozen dataclasses, tuples, frozensets)" is Python-specific. Simplify trigger to "When writing data structures or containers:" and drop the Python-specific examples from the trigger (they're fine in the body as examples).

**Rule 20 — functions under 50 lines**
`"When writing or editing functions (Edit, Write): keep functions under 50 lines"`
FIX: Trigger "When writing or editing functions" matches every function edit. This is really a review/commit-time check. Better: "When a function exceeds 50 lines:" — specific to the violation condition.

### FIX: Meaning lost or changed

**Rule 70 — reasoning-first JSON (index ~513)**
Original: "Put reasoning BEFORE decision in LLM JSON output format — when the signal/action field comes first, the model commits to a default before thinking"
Rewritten: "When choosing LLM reasoning mode: thinking mode costs 10x but is most decisive; reasoning-first JSON recovers 80% of quality at 3x speed"
FIX: The rewrite completely changes the instruction. The original is about JSON field ordering. The rewrite is about choosing between thinking mode and reasoning-first. These are DIFFERENT rules. The original rule 71 ("Use reasoning-first JSON...") is kept and covers field ordering, but this rewrite (rule 70) lost its original meaning and became a duplicate of rule 72 (thinking mode). **Merge rules 70 and 72 since the rewrite made them identical, and ensure the field-ordering instruction from rule 70's original is preserved (it IS preserved in rule 71's keep).**

**Rule 72 — thinking mode (index ~525)**
`"When choosing LLM reasoning mode: thinking mode costs 10x but is most decisive; reasoning-first JSON recovers 80% of quality at 3x speed — match compute to difficulty"`
FIX: This is now IDENTICAL to rule 70's rewrite. One of these two must be removed — they are exact duplicates. Keep rule 72 (better fit), remove rule 70 (which lost its original meaning anyway).

**Rule 29 — mock external calls in tests**
`"When writing unit tests that involve LLM calls, HTTP requests, or subprocesses (Edit, Write test files): always mock external calls — real calls make tests slow, flaky, and non-deterministic; 18-minute test suites get skipped"`
FIX: Minor — lost the original's advice about what to mock ("at the boundary"). The original said "mock external calls (LLM, APIs, network, subprocesses)" which was already well-scoped. The rewrite is fine but slightly less instructive. Acceptable, but note the nuance loss.

### FIX: Category miscategorization

**Rule 40 — extract magic numbers**
`category: "both"`
FIX: Should be `tool_use`. This constrains a specific code-editing action (replacing literals with constants). It has nothing to do with workflow/process decisions.

**Rule 56 — validate phases against real data**
`category: "workflow"`
FIX: Could be `both` — it applies to both process decisions (when to validate) AND tool actions (running real tests). But workflow is defensible. Borderline.

**Rule 76 — MEDIUM tasks inline plan**
`category: "both"`
FIX: Should be `workflow`. This is purely about process — when to create a plan. No tool action is constrained.

**Rule 77 — COMPLEX tasks PRD**
`category: "both"`
FIX: Should be `workflow`. Same reasoning — this is process guidance about when to write a PRD.

**Rule 75 — search GitHub**
`category: "both"`
FIX: Should be `workflow`. "Search GitHub before writing from scratch" is a process decision, not a tool constraint.

### FIX: Conciseness (over 250 chars, could be shorter)

**Rule 4 — sensitive file permissions**
`"When creating or writing sensitive files (keys, configs, tokens, .env, credentials) via Bash or Write: set file permissions to 0o600 — world-readable secrets are the most common cloud breach vector"`
FIX: 168 chars, fine. Actually no issue here — retracted.

**Rule 11 — environment variables**
`"When writing code with hardcoded URLs, ports, API endpoints, database strings, or feature flags (Edit, Write): use environment variables instead — hardcoded config prevents deployment to different environments and leaks secrets into source control"`
FIX: 248 chars. The enumeration "URLs, ports, API endpoints, database strings, or feature flags" could be shortened to "hardcoded configuration values" — the examples don't help retrieval.

**Rule 12 — no mutation**
`"When writing or editing functions that receive objects, dicts, or lists as arguments (Edit, Write): never mutate function arguments or shared state — mutations cause action-at-a-distance bugs where changing one variable silently breaks code elsewhere"`
FIX: 265 chars. "objects, dicts, or lists" is Python-specific. Simplify to "When writing functions that receive mutable arguments:" — shorter and language-agnostic.

**Rule 54 — save artifacts before overwriting**
`"When about to overwrite model artifacts, pickles, checkpoints, or baseline files (Bash: mv, cp, Write): save LLM-generated artifacts before overwriting — never destroy a baseline during experiments, because non-deterministic pipelines cannot reproduce the exact same output"`
FIX: 280 chars. Too long. The trigger already says "overwrite model artifacts" and then the body repeats "save before overwriting." Trim to: "When about to overwrite model artifacts or baseline files: save a copy first — non-deterministic pipelines cannot reproduce the exact same output."

### FIX: Redundancy with other rules

**Rule 52 — write tests before implementation (TDD)**
`"When starting a new feature or fixing a bug (Edit, Write): write tests before implementation following TDD red-green-refactor cycle — test-first development catches design issues early and guarantees coverage"`
FIX: This is a near-duplicate of Rule 49 ("When starting a new feature or bug fix: write tests before implementation (TDD red-green-refactor)"). One should be removed. Rule 49 has a better consequence clause; rule 52 is the duplicate. **Remove rule 52.**

---

## Rewrites: REVERT

**Rule 64 — prefer mature frameworks**
Original: "Prefer mature extensible frameworks over building from scratch — if LangGraph, FastAPI, or similar battle-tested projects provide the feature (routing, checkpointing, streaming), use them even if a simpler stdlib approach works today, because refactoring from simple to framework later costs more than starting with the framework"
Rewritten: "When choosing between a simple stdlib approach and a framework (LangGraph, FastAPI): prefer the framework — refactoring from simple to framework later costs more"
REVERT: The rewrite still names specific frameworks (LangGraph, FastAPI) — violating criterion 1. But worse, it lost the nuance about WHEN to prefer frameworks ("when battle-tested projects provide the feature"). The rewrite makes it sound like you should ALWAYS use a framework, which is wrong. Suggested fix: "When a mature framework provides the needed functionality: prefer it over building from scratch — refactoring from simple to framework later costs more than starting with the framework."

**Rule 63 — reuse open-source**
Original: "Reuse popular open-source components for each layer of the system — only build custom code for what doesn't exist, because mature projects provide stability, community support, and extensibility that hand-rolled code cannot match"
Rewritten: "When building system components: reuse popular open-source projects — only build custom code for what doesn't exist; mature projects provide stability and community support"
REVERT: The trigger "When building system components" is extremely vague — it matches everything. The original was already a clear principle statement. The rewrite added a meaningless trigger and lost "extensibility that hand-rolled code cannot match." Keep the original.

---

## Keeps with Issues

**Rule 14 (keep) — force push**
`"Never force-push to main or master branch"`
`category: "both"`
FIX: Category should be `tool_use`. This constrains a specific git command. It's not a workflow/process decision.

**Rule 51 (keep) — delegate via tmux**
`"Delegate implementation to worker agents via tmux sessions and monitor progress with cron-based polling"`
`category: "both"`
OK — defensible. The rule involves both tool use (tmux) and workflow (delegation pattern).

**Rule 71 (keep) — reasoning-first JSON**
GOOD as-is, but note it now overlaps with rules 70/72 due to the rewrite making 70 and 72 identical. See rule 70 note above.

**Rule 31 (keep) — single slow test**
`"If a single test takes more than 1 second, it is either testing too much or hitting a real service"`
`category: "both"`
FIX: Should be `tool_use` — this fires when running tests and seeing slow results. Not a workflow decision.

---

## Rewrites: GOOD (52 rules)

The following rewrites pass all 6 criteria:

1, 2, 3, 5, 6, 8, 13, 14 (readme), 15, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 37, 41, 42, 45, 47, 49, 50, 53, 57, 58, 59, 60, 61, 62, 65, 66, 67, 68, 69, 73, 74, 78, 79, 80, 81, 82

(Indexed by position in the JSON array, 1-based)

---

## Action Items (prioritized)

1. **Remove duplicate:** Rules 70 and 72 are now identical rewrites. Remove one (preferably 70, since its original meaning is already covered by keep rule 71).
2. **Remove duplicate:** Rule 52 is a near-duplicate of rule 49. Remove 52.
3. **De-enumerate 7 rules:** Rules 7, 16, 43, 44, 46, 48, 55 enumerate specific libraries/commands. Replace with general categories.
4. **Fix 3 vague triggers:** Rules 9, 38, 39 have triggers that match everything. Narrow them.
5. **Fix 5 category errors:** Rules 40 (both->tool_use), 75 (both->workflow), 76 (both->workflow), 77 (both->workflow), 14-keep (both->tool_use), 31-keep (both->tool_use).
6. **Revert 2 rules:** Rules 63 and 64 lost meaning or gained vague triggers.
7. **Trim 3 long rules:** Rules 11, 12, 54 exceed 250 chars and can be shortened.
