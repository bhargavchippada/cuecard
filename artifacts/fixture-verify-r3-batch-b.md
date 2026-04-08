# Fixture Verification Round 3 — Batch B (workflow, stop, subagent_start)

## Scope
- **workflow.json**: 124 fixtures reviewed (94 positive, 30 negative)
- **stop.json**: 77 fixtures reviewed (50 positive, 27 negative)
- **subagent_start.json**: 73 fixtures reviewed (56 positive, 17 negative)

## Changes Made

### workflow.json (5 fixes)

| Fixture ID | Change | Reasoning |
|---|---|---|
| `wf-easy-commit-push` | Replaced `"Run ruff and mypy before committing Python code"` with `"Run quality checks before every commit"` | Python-specific tool guidance is PreToolUse-level, not UserPromptSubmit process guidance. The general "run quality checks" rule is the correct process rule for "commit and push". |
| `wf-medium-full-review-after-code` | Removed `"Never commit secrets..."` | User asked to "review everything", not to commit. Secrets rule is about committing, not code review. 3 rules remain (parallel agents, convergence, convergence reviews). |
| `wf-medium-phase-transition` | Removed `"Require 100% test coverage..."` | 100% coverage is a coding standard, not phase-transition process guidance. The 3 remaining rules (validate real data, commit per phase, convergence reviews) are the correct phase transition checklist. |
| `wf-medium-noise-filtering` | Removed `"Never use silent fallbacks..."` | User complains about noise in extracted sessions. This is a filtering problem, not a fallback problem. Kept filter + stage quality rules. |
| `wf-medium-add-new-library` | Removed `"Reuse popular open-source components..."` | User already chose to add fastembed (an OSS library). The "reuse OSS" rule is about not building from scratch; it doesn't apply when the user is already adding an OSS dependency. |
| `v3-user-web-interface-broken` | Replaced `"Always handle errors explicitly..."` with `"Set subagent scope boundaries..."` | User reports a UI bug and says "ask a subagent to test and verify everything." Error handling is a coding standard. The process guidance is about scoping the subagent ("everything" = unbounded). |

### stop.json (0 fixes)

All stop fixtures verified as correct. Key observations:
- Positive fixtures correctly match actions taken (commits, code edits, security violations)
- Negative fixtures correctly have empty should_match for read-only operations
- `stop-neg-created-plan` and `stop-neg-compared-libraries` have "neg" in name but ARE positive (have should_match) — naming inconsistency only, expectations are correct
- Multi-violation fixtures (stop-multi-violation-commit, stop-multi-security-violations) correctly list all applicable rules

### subagent_start.json (3 fixes)

| Fixture ID | Change | Reasoning |
|---|---|---|
| `sub-database-migration` | Replaced `"Always close file handles..."` with `"Use parameterized queries to prevent SQL injection..."` | Writing a migration script is about generating correct SQL, not managing persistent connections. SQL injection prevention is the specific risk for migration code. |
| `sub-hard-ambiguous-script` | Removed `"Use type hints on all function signatures..."` | Too generic for SubagentStart. When delegating "create utility script", the specific guidance is about error handling (processing config files can fail). Type hints is a PreToolUse concern when actually editing code. |
| `sub-database-reviewer-schema` | Removed `"Always close file handles..."` | A database reviewer is reviewing code, not writing resource-managing code. SQL injection is the specific concern mentioned in the task ("check for injection risks"). Close resources would fire via PreToolUse when the reviewer reads the code. |

## Fixtures Verified as Correct (notable confirmations)

### workflow.json
- **wf-easy-write-tests-first**: Mock rule kept — testing methodology guidance is valid UserPromptSubmit process guidance
- **wf-medium-multi-file-feature**: 4 rules kept — all are process/methodology guidance for building features with tests
- **wf-hard-low-quality-output**: Silent fallbacks kept — plausible diagnostic cause for generic/flat output
- **wf-neg-debug-print**: Correctly has should_match despite being in "neg" naming — the rule legitimately fires when user requests adding debug prints
- All 30 negative workflow fixtures confirmed empty should_match

### stop.json  
- All audit rules correctly match actions described in stop summaries
- Read-only stop summaries (read files, git log, answered questions) correctly negative
- v3 fixtures from real sessions correctly mapped

### subagent_start.json
- All fixtures respect the 1-2 rule limit for SubagentStart guidance
- Negative fixtures correctly have empty should_match for read-only/monitoring agents
- "neg" labeled fixtures with should_match (sub-neg-search-agent, sub-neg-docs-reader, sub-neg-analyze-benchmark, sub-neg-plan-migration) are correctly positive — naming reflects original expectation, not current state

## Summary
- **Total fixtures reviewed**: 274
- **Fixes applied**: 8 (5 workflow, 0 stop, 3 subagent_start)
- **Fix types**: 4 rule removals (wrong applicability), 2 rule replacements (better match), 2 rule removals (too generic for SubagentStart)
- **Rule text verification**: All remaining should_match rules verified present in rules_global.txt
