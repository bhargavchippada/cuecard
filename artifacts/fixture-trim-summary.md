# Fixture Trim Summary

## Issue 1: SubagentStart Over-Specification

**Problem:** SubagentStart positive fixtures averaged 2.6 expected rules per fixture (146 total across 56 positives). With LLM reranker returning max 5 results and F2 scoring, fixtures with 4+ expected rules can never achieve perfect recall.

**Fix:** Trimmed fixtures with 3+ rules to the 1-2 most task-specific rules. Generic coding standards (type hints, error handling, test coverage, immutability, etc.) that any implementation agent gets via PreToolUse were deprioritized in favor of rules specific to the SubagentStart task context.

**Decision criteria:**
- KEEP: Rules specific to the agent's task (e.g., "Use parameterized queries" for SQL review, "Use asyncio" for async HTTP service)
- REMOVE: Generic coding standards any developer follows (e.g., "Use type hints" for an implementation agent)

### Results

| Metric | Before | After |
|--------|--------|-------|
| Positive fixtures | 56 | 56 |
| Total expected rules | 146 | 87 |
| Average rules/fixture | 2.6 | 1.6 |
| Fixtures trimmed | — | 22 |
| Rules removed | — | 59 |

### Trimmed fixtures

| ID | Before | After | Kept rules (abbreviated) |
|----|--------|-------|--------------------------|
| sub-security-reviewer | 4 | 2 | secrets, SQL injection |
| sub-code-reviewer | 7 | 2 | type hints, file size |
| sub-tdd-guide | 5 | 2 | TDD, mock external calls |
| sub-implement-feature | 7 | 2 | close resources, cache TTL |
| sub-implement-api-endpoint | 8 | 2 | input validation, SQL injection |
| sub-git-commit-prep | 4 | 2 | ruff/mypy, conventional commits |
| sub-async-service | 6 | 2 | asyncio, close resources |
| sub-install-deps | 3 | 2 | uv, vulnerability audit |
| sub-eval-script | 4 | 2 | no eval(), input validation |
| sub-file-processor | 4 | 2 | close resources, error handling |
| sub-planner-feature | 7 | 2 | close resources, log API calls |
| sub-e2e-runner-login-flow | 4 | 2 | CSRF, XSS |
| sub-multi-task-agent | 6 | 2 | uv, test coverage |
| sub-hard-ambiguous-script | 3 | 2 | read patterns, error handling |
| sub-hard-config-agent | 4 | 2 | env vars, frozen dataclasses |
| sub-credential-manager | 4 | 2 | secrets, file permissions |
| sub-hard-async-crawler | 4 | 2 | asyncio, close resources |
| sub-hard-secret-rotation-ambiguous | 3 | 2 | secrets, file permissions |
| sub-hard-git-branch-commit | 5 | 2 | feature branch, test coverage |
| v3-subagent-code-reviewer | 3 | 2 | type hints, file size |
| v3-subagent-security-reviewer | 4 | 2 | secrets, input validation |
| v3-subagent-build-phase | 4 | 2 | type hints, frozen dataclasses |

## Issue 2: Noisy Negatives

**Problem:** Some negative fixtures genuinely match rules in the 109-rule corpus.

**Approach:** Programmatic scan of all negative fixtures across 5 files, followed by manual review of each candidate conversion. Conservative — only clear matches converted.

### Results

| File | Negatives scanned | Converted | Details |
|------|-------------------|-----------|---------|
| basic.json | 197 | 0 | All negatives are genuinely negative (read-only, monitoring, compliance) |
| stop.json | 27 | 0 | All negatives are genuinely negative (reading, explaining, reviewing) |
| subagent_start.json | 17 | 0 | All negatives are genuinely negative (monitoring, explaining, read-only) |
| post_tool_use.json | ~30 | 1 | `post-neg-bash-pip-list`: using `pip list` matches "use uv" rule |
| workflow.json | ~40 | 0 | All negatives are genuinely negative |

**Total negatives converted: 1**

The initial heuristic flagged 13 candidates, but manual review found 12 were false positives:
- Compliance != violation (e.g., `os.getenv()` is FOLLOWING the env vars rule)
- Read-only operations don't trigger action rules (e.g., `git status` != committing)
- Force push to feature branch != force push to main
- `uv sync` is correct behavior, not missing audit
- Code using `with open()` is USING context managers correctly
- "Format this JSON" has nothing to do with CSRF

## Combined.json Rebuild

Rebuilt `eval/fixtures/combined.json` from all fixture files (deduped by ID).

| Metric | Before (stale) | After |
|--------|----------------|-------|
| Total fixtures | 438 | 980 |
| PreToolUse | 354 | 560 |
| UserPromptSubmit | 84 | 155 |
| PostToolUse | 0 | 115 |
| Stop | 0 | 77 |
| SubagentStart | 0 | 73 |

Note: The previous combined.json was stale — it only included a subset of basic.json and workflow.json, predating the addition of PostToolUse, Stop, and SubagentStart events.
