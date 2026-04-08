# Fixture Verification Round 1 — UserPromptSubmit (workflow)

## CRITICAL FINDING: Complete corpus mismatch

**Every single rule text in all 84 fixtures references the OLD `rules_workflow.txt` corpus. Zero texts match the new `rules_global.txt` corpus.** The `"corpus": "rules_global.txt"` field is set correctly, but the `should_match` and `should_not_match` texts are all from the old workflow corpus.

- **46 unique rule texts** referenced across all fixtures
- **0** match the global corpus exactly
- **17** can be remapped (same concept, evolved wording with added rationale)
- **29** have NO equivalent in the global corpus (rules that existed only in the workflow corpus)
- **43 of 60 positive fixtures** reference at least one dropped rule

## Decision needed from Bhargav

There are two paths forward:

**Option A: Add the 29 missing workflow rules to `rules_global.txt`** — the global corpus was 32 basic rules + these workflow rules should be merged in, bringing the total to ~94 rules. This preserves all fixture expectations.

**Option B: Accept that the global corpus has different rules and rewrite fixtures** — remove fixtures that test rules no longer in the corpus, remap the 17 that evolved, and add new fixtures for the ~20 new global rules relevant to UserPromptSubmit.

**Recommendation: Option A** — the workflow rules represent real, tested process knowledge. They were deliberately authored and fixture-validated. Dropping them loses coverage of agent orchestration, benchmarking methodology, and LLM pipeline operations.

---

## Part 1: Remappable rules (17 rules — text evolved, same concept)

These old workflow rule texts have equivalent rules in the global corpus with expanded rationale. Every occurrence in should_match AND should_not_match needs updating.

| # | Old workflow rule | New global rule |
|---|---|---|
| R1 | "Review PRDs with multiple specialist subagents (architect, security, database, code quality) in parallel before implementing" | "Review PRDs with multiple specialist subagents (architect, security, database, code quality) in parallel — a single reviewer misses entire categories of issues that other specialists catch immediately" |
| R2 | "Keep running review rounds until two consecutive rounds produce zero CRITICAL, HIGH, or MEDIUM findings — never use a fixed round count" | "Keep running review rounds until two consecutive rounds produce zero CRITICAL, HIGH, or MEDIUM findings — fixed round counts stop too early when issues cascade, and too late when code is already clean" |
| R3 | "Validate every phase against real data before moving to the next phase — synthetic tests alone are insufficient" | "Validate every phase against real data before moving to the next — synthetic tests pass when the real thing fails because mocks diverge from production behavior" |
| R4 | "Always benchmark a single LLM call before launching a full pipeline run — a 2-second PING test can save hours" | "Always benchmark a single LLM call before launching a full pipeline run — a 2-second PING test saves hours of wasted compute when the endpoint is misconfigured or slow" |
| R5 | "Establish baseline metrics before implementing any improvement — you cannot claim X is better without before/after numbers" | "Establish baseline metrics before implementing any improvement — without before/after numbers you cannot distinguish real improvement from noise" |
| R6 | "Fix the pipeline stage that produces bad output, do not patch the output with manual rules downstream" | "Fix the pipeline stage that produces bad output, do not patch downstream — downstream patches accumulate into unmaintainable complexity while the root cause keeps generating bad data" |
| R7 | "Each pipeline stage must be independently high quality — do not rely on downstream stages to rescue upstream failures" | "Each pipeline stage must be independently high quality — relying on downstream stages to rescue upstream failures means any stage failure cascades through the entire system" |
| R8 | "Update CLAUDE.md when project structure, architecture, or conventions change during a session" | "Update CLAUDE.md when project structure, architecture, or conventions change — without this, the next session starts with stale context and makes wrong assumptions" |
| R9 | "Update README when user-facing behavior, setup steps, or dependencies change" | "Update README when user-facing behavior, setup steps, or dependencies change — outdated README causes every new contributor to fail on setup" |
| R10 | "Search GitHub for existing implementations and proven patterns before writing anything from scratch" | "Search GitHub for existing implementations and proven patterns before writing anything from scratch — 80% of problems are already solved, and battle-tested code has fewer bugs than fresh code" |
| R11 | "Use Context7 for library documentation lookups, Exa for broader web research, mgrep for semantic code search" | "Use Context7 for library documentation lookups, Exa for broader web research, mgrep for semantic code search — wrong search tool returns irrelevant results and wastes time on false leads" |
| R12 | "When a tool or command fails, read the full error message and search project docs before retrying — never retry blindly" | "When a tool fails, read the full error and search docs before retrying — blind retries waste time and hide the root cause, while the fix is usually in the error message" |
| R13 | "Require 100 percent test coverage on all new code — not 80 percent, not 90 percent, always 100 percent" | "Require 100% test coverage on all new code before committing — uncovered code is untested code, and untested code breaks silently in production" |
| R14 | "Never implement a feature without reading 2-3 similar files in the codebase first to understand existing patterns" | "Read 2-3 similar files in the codebase before writing new code — introducing a second pattern where one exists creates permanent inconsistency that confuses every future reader" |
| R15 | "Use conventional commit format with types feat, fix, refactor, docs, test, chore, perf, ci" | "Use conventional commit format (feat:, fix:, refactor:, docs:, test:) — without structured messages, changelogs are impossible to generate and git blame is useless" |
| R16 | "If a single LLM call takes more than 5 seconds for a trivial prompt, investigate SDK overhead before proceeding" | "If a tool call or LLM response takes 10x longer than expected, investigate immediately — normalized slowness compounds across thousands of calls into hours of wasted time" |
| R17 | "Run security review before every commit — check for hardcoded secrets, SQL injection, XSS, missing input validation" | "Never commit secrets (API keys, tokens, passwords, connection strings) to git — once pushed, secrets are in the history forever and require rotation across all environments" (partial — narrower scope) |

### Fixtures needing R1-R17 remaps

All 84 fixtures need at least one remap. Here are the specific should_match remaps per fixture:

- `wf-easy-run-reviews`: R2
- `wf-easy-write-tests-first`: R13
- `wf-easy-commit-push`: R15, R17
- `wf-easy-update-docs`: R8, R9
- `wf-easy-benchmark-first`: R4
- `wf-easy-search-existing`: R10
- `wf-easy-look-at-patterns`: R14
- `wf-easy-prd-review`: R1
- `wf-easy-real-data`: R3
- `wf-easy-convergence`: R2
- `wf-easy-lookup-docs`: R11
- `wf-medium-plan-feature`: (no remaps, all dropped)
- `wf-medium-new-feature`: R14
- `wf-medium-iterate-quality`: R3, R5
- `wf-medium-add-new-library`: R11
- `wf-medium-classification-pipeline`: R4
- `wf-medium-refactor-arch`: (no remaps, all dropped)
- `wf-medium-implement-improvement`: R5
- `wf-medium-fix-pipeline-bug`: R6, R7
- `wf-medium-multi-file-feature`: R13
- `wf-medium-cdt-rebuild`: (all dropped)
- `wf-medium-phase-transition`: R3, R13
- `wf-medium-slow-pipeline`: R4
- `wf-medium-add-reranking`: (all dropped)
- `wf-medium-save-state`: R8
- `wf-medium-full-review-after-code`: R2, R17
- `wf-hard-error-loop`: R12
- `wf-hard-scores-differ`: (all dropped)
- `wf-hard-new-approach-untested`: R5
- `wf-hard-low-quality-output`: R6, R7
- `wf-hard-context-running-low`: (all dropped)
- `wf-hard-pilot-looks-great`: R3
- `wf-hard-wrong-tool-for-search`: R11
- `wf-hard-coverage-not-enough`: R13
- `wf-hard-adding-more-stages`: R7
- `wf-hard-skipped-error-reading`: R12
- `wf-hard-paper-code-vs-ours`: (all dropped)

---

## Part 2: Dropped rules (29 rules — no equivalent in global corpus)

These rules existed in `rules_workflow.txt` but have NO equivalent in `rules_global.txt`. They cover agent orchestration, task classification, benchmarking methodology, and LLM pipeline operations.

### Category: Task classification (4 rules, ~8 fixtures)
1. "Classify every task as SIMPLE, MEDIUM, or COMPLEX before starting — state the assessment explicitly with reasoning"
2. "SIMPLE tasks (1-2 files, low risk) can proceed without a plan — just implement and verify"
3. "MEDIUM tasks (3-10 files) require an inline plan, research, tests alongside code, and agent review after implementation"
4. "COMPLEX tasks (10+ files, architecture changes) require a full PRD written to artifacts/ before any implementation begins"

Fixtures affected: `wf-easy-complexity-assess`, `wf-medium-plan-feature`, `wf-medium-new-feature`, `wf-medium-refactor-arch`, `wf-medium-quick-fix`, `wf-medium-multi-file-feature`

### Category: Agent orchestration (5 rules, ~5 fixtures)
5. "Use the planner agent for complex features, architect agent for system design decisions, tdd-guide agent for new features and bug fixes"
6. "Launch parallel subagents for independent review tasks — never run them sequentially when they can run simultaneously"
7. "Run code review with 3 parallel agents (code-reviewer, security-reviewer, python-reviewer) and iterate until clean"
8. "Delegate implementation to worker agents via tmux sessions and monitor progress with cron-based polling"
9. "Tell delegated agents to report back via tmux send-keys to your session instead of polling for completion"

Fixtures affected: `wf-easy-delegate-tmux`, `wf-easy-parallel-agents`, `wf-easy-run-reviews`, `wf-medium-agent-stalled`, `wf-medium-full-review-after-code`

### Category: Agent monitoring (2 rules, ~3 fixtures)
10. "Monitor agent context usage in tmux — compact when context remaining drops below 67 percent"
11. "Run iterative PRD reviews until a round comes back clean — typical convergence is 4 rounds"

Fixtures affected: `wf-easy-check-context`, `wf-easy-prd-review`, `wf-medium-agent-stalled`, `wf-hard-context-running-low`

### Category: TDD and testing (1 rule, ~4 fixtures)
12. "Write tests before implementation following TDD red-green-refactor cycle"

Fixtures affected: `wf-easy-write-tests-first`, `wf-medium-new-feature`, `wf-medium-refactor-arch`, `wf-medium-multi-file-feature`

### Category: LLM pipeline operations (6 rules, ~8 fixtures)
13. "Manually inspect LLM output after every classification wave — read actual output, do not trust confidence scores"
14. "Never trust small sample benchmark results — run the full dataset before drawing conclusions"
15. "Never build infrastructure around unproven techniques — test hypotheses in controlled experiments first, implement only winners"
16. "Use lightweight deterministic approaches first, add LLM stages only when they demonstrably improve quality"
17. "After the parallel LLM phase, retry failed items sequentially one at a time to avoid rate limiting contention"
18. "After every batch LLM response, diff sent IDs against returned IDs to detect silently dropped items"

Fixtures affected: `wf-easy-inspect-output`, `wf-easy-benchmark-first`, `wf-medium-classification-pipeline`, `wf-medium-implement-improvement`, `wf-medium-retry-failed`, `wf-medium-add-reranking`, `wf-medium-early-results`, `wf-hard-items-silently-missing`, `wf-hard-adding-more-stages`, `wf-hard-low-quality-output`, `wf-hard-new-approach-untested`, `wf-hard-pilot-looks-great`

### Category: Data/artifact management (5 rules, ~6 fixtures)
19. "Use semantic similarity (embeddings or LLMs) for text deduplication — never use lexical or word-based matching"
20. "Save LLM-generated artifacts (pickles, checkpoints) before overwriting — never destroy a baseline during experiments"
21. "Store all predictions, reasoning, and grounding in benchmark results for later reanalysis — not just numeric scores"
22. "Filter pipeline noise at multiple independent stages with configurable thresholds — no single filter point"
23. "When schema or cardinality changes, audit every consumer (views, queries, CLI, exports, tests) before implementing"

Fixtures affected: `wf-easy-store-predictions`, `wf-easy-save-artifacts`, `wf-easy-schema-audit`, `wf-medium-dedup-text`, `wf-medium-cdt-rebuild`, `wf-medium-noise-filtering`, `wf-hard-lexical-dedup-failing`, `wf-hard-nondeterministic-results`, `wf-hard-items-silently-missing`, `wf-hard-schema-change-subtle`

### Category: Session/compaction (1 rule, ~3 fixtures)
24. "Before compaction, save task state to artifacts/ including what was done, what remains, key decisions, and blockers"

Fixtures affected: `wf-easy-compact`, `wf-medium-save-state`, `wf-hard-context-running-low`

### Category: Benchmarking methodology (3 rules, ~4 fixtures)
25. "When running A/B benchmark comparisons on LLM artifacts, use the same pre-built artifact and change only one variable"
26. "Run the original paper code first when reproducing published results — do not theorize about score gaps without evidence"
27. "Score gaps between models on routine tasks usually indicate code or config differences, not model quality differences"

Fixtures affected: `wf-medium-cdt-rebuild`, `wf-medium-reproduce-paper`, `wf-hard-scores-differ`, `wf-hard-nondeterministic-results`, `wf-hard-paper-code-vs-ours`

### Category: Dependency research (1 rule, ~2 fixtures)
28. "For new libraries or APIs, research alternatives, verify documentation, audit maintenance status, and benchmark before adopting"

Fixtures affected: `wf-medium-add-new-library`, `wf-medium-add-reranking`

### Category: Error recovery (1 rule, ~1 fixture)
29. "After 2 failed fix attempts on the same error, stop and ask the user for guidance instead of spiraling"

Fixtures affected: `wf-hard-error-loop`

---

## Part 3: New global rules potentially missing from existing fixtures

These rules are in the global corpus but were NOT in the old workflow corpus. Some are relevant to UserPromptSubmit queries that already exist but don't reference them:

### Additions for existing positive fixtures

- `wf-easy-commit-push`: ADD "Commit after each implementation phase completes — atomic tested commits prevent compound debt..."
- `wf-easy-commit-push`: ADD "Run ruff and mypy before committing Python code — type errors and lint violations caught at commit time..."
- `wf-medium-new-feature`: ADD "Search GitHub for existing implementations and proven patterns before writing anything from scratch..."
- `wf-medium-plan-feature`: ADD "Follow the three-layer quality gate: PRD + reviews -> Implementation + reviews -> Test plan + reviews..."
- `wf-medium-phase-transition`: ADD "Commit after each implementation phase completes — atomic tested commits prevent compound debt..."
- `wf-medium-phase-transition`: ADD "Run convergence reviews (code + security agents in parallel) after each implementation milestone..."
- `wf-medium-full-review-after-code`: ADD "Run convergence reviews (code + security agents in parallel) after each implementation milestone..."
- `wf-easy-prd-review`: ADD "Run convergence reviews on PRD changes too, not just code — a design gap that survives review becomes an implementation bug..."
- `wf-medium-add-new-library`: ADD "Spike-test risky integrations before building on them..."
- `wf-medium-add-new-library`: ADD "Reuse popular open-source components for each layer of the system..."
- `wf-medium-implement-improvement`: ADD "Spike-test risky integrations before building on them..."
- `wf-medium-add-reranking`: ADD "Spike-test risky integrations before building on them..."
- `wf-easy-inspect-output`: ADD "Never trust LLM-generated confidence scores — use evidence counts, ensemble agreement, or historical track record instead..."
- `wf-medium-classification-pipeline`: ADD "Never use silent fallbacks — if a pipeline stage falls back to an alternative..."
- `wf-medium-noise-filtering`: ADD "Never use silent fallbacks — if a pipeline stage falls back to an alternative..."
- `wf-medium-slow-pipeline`: ADD "If a tool call or LLM response takes 10x longer than expected, investigate immediately..."
- `wf-hard-low-quality-output`: ADD "Never use silent fallbacks — if a pipeline stage falls back to an alternative..."
- `wf-medium-iterate-quality`: ADD "Control your evaluation methodology before trusting results..."
- `wf-hard-scores-differ`: ADD "Control your evaluation methodology before trusting results..."
- `wf-easy-write-tests-first`: ADD "Always mock external calls (LLM, APIs, network, subprocesses) in unit tests..."
- `wf-medium-multi-file-feature`: ADD "Always mock external calls (LLM, APIs, network, subprocesses) in unit tests..."

---

## Part 4: Correctness issues in should_match

- `wf-easy-security-review`: "Run security review before every commit" maps to "Never commit secrets..." which is a NARROWER rule. The old rule was about a full security review process; the new global corpus doesn't have a direct equivalent for "run security review". This is a LOSSY remap.
- `wf-medium-add-reranking`: "Use lightweight deterministic approaches first" — this rule has NO equivalent in the global corpus. The fixture tests a valid concept but the rule was dropped.
- `wf-medium-refactor-arch`: "When schema or cardinality changes, audit every consumer" — dropped from global corpus. The fixture is testing database refactoring which the global corpus doesn't cover.

---

## Part 5: Corpus field check

All 84 fixtures already have `"corpus": "rules_global.txt"` — no changes needed here.

---

## Summary

| Category | Count |
|----------|-------|
| Rules needing REMAP (old text -> new text) | 17 rules across ~55 fixture references |
| Rules DROPPED (no global equivalent) | 29 rules across ~43 fixtures |
| Potential ADDITIONS (new global rules for existing fixtures) | ~21 additions across ~15 fixtures |
| Corpus field fixes | 0 (all correct) |
| Total fixtures requiring changes | **84 of 84** (every single fixture) |

### Immediate action items

1. **DECISION NEEDED**: Add the 29 dropped workflow rules to `rules_global.txt`, or rewrite/remove the 43 affected fixtures
2. **REMAP**: Update all 17 remappable rule texts to their global corpus equivalents (mechanical find-replace)
3. **ADD**: Add ~21 new global rule references to existing positive fixtures where they clearly apply
4. **REMOVE**: Consider removing should_not_match references to dropped rules (these become meaningless if the rule isn't in the corpus)
