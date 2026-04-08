# Fixture Verification Round 3 — Batch A

**Date:** 2026-04-07
**Reviewer:** Turiya (Opus 4.6)
**Files:** `eval/fixtures/basic.json`, `eval/fixtures/post_tool_use.json`

## Summary

- **31 total changes** across both files (28 basic, 3 post_tool_use)
- **24 positives converted to negatives** (fixtures where ALL should_match rules were wrong)
- **4 positives trimmed** (some rules removed, fixture stays positive)
- **1 contradiction fixed** (same rule in should_match AND should_not_match)
- **2 new negatives from trimming** (v3-post fixtures)

## Root Cause Pattern: Compliance Treated as Violation

**The #1 error pattern (20 of 31 changes):** A fixture matched a rule by TOPIC when the query shows COMPLIANCE with the rule, not a violation. The retrieval system should NOT fire a rule when the code is already following it.

Examples:
- `ruff-check`: Running `uv run ruff check` IS compliance with "Run ruff and mypy before committing"
- `new-branch`: Creating `git checkout -b feature/add-login` IS compliance with "Always create a new branch"
- `mypy-check`: Running mypy IS the type checking the rule recommends
- `async-code`: Using `async with aiohttp.ClientSession()` IS proper async + resource cleanup
- `embedded-code-python-class`: Code has `@dataclass(frozen=True)` — IS compliance with frozen dataclass rule
- `long-write-django-settings`: Uses `os.getenv()` and `CSRF_COOKIE_SECURE = True` — IS compliance

## Changes — basic.json (28 changes)

### Converted to Negative (compliance, not violation)

| ID | Removed Rule(s) | Reason |
|----|-----------------|--------|
| `new-branch` | create branch rule | Creating a branch IS compliance |
| `ruff-check` | ruff/mypy rule | Running ruff IS compliance |
| `mypy-check` | ruff/mypy + type hints | Running mypy IS compliance |
| `npm-audit` | vulnerability review | Running audit IS the review |
| `eslint-check` | quality checks | Running eslint IS the quality check |
| `prettier-format` | quality checks | Running prettier IS the quality check |
| `cargo-fmt` | quality checks | Running cargo fmt IS the quality check |
| `async-code` | asyncio + close resources | Code uses async with correctly |
| `long-edit-async-db-handler` | parameterized queries, asyncio, close resources | New code fixes all issues (uses $1/$2, async, async with) |
| `long-write-django-settings` | env vars + CSRF | Code uses os.getenv and enables CSRF |
| `embedded-code-decorator-pattern` | input validation + parameterized queries | Code has @validate_input and uses $1/$2 |
| `ambiguous-async-db-connections` | asyncio | Code IS using asyncio correctly |
| `special-chars-unicode-comment` | input validation | Code IS validating input |

### Converted to Negative (topic mismatch)

| ID | Removed Rule(s) | Reason |
|----|-----------------|--------|
| `write-typescript-file` | XSS/sanitize HTML | JSX auto-escapes; `<button>{label}</button>` is not an XSS vector |
| `docker-build` | vulnerability review | Docker build doesn't add dependencies |
| `docker-run` | env vars | Port mapping is not a config violation |
| `docker-compose-up` | env vars | Operational command, not config |
| `write-to-etc` | file permissions 0o600 | Cron file is not sensitive keys/configs/tokens |
| `secret-rotation` | secrets + rotation | Grepping for AKIA is investigation, not violation |
| `mined-edit-test-file-python` | 100% coverage | Editing existing test file is not a coverage issue |
| `mined-edit-store-test-version` | 100% coverage | Same |
| `mined-edit-cuecard-test-indexer` | 100% coverage | Same |
| `mined-edit-delulu-loader-remove-commit` | close resources | Removing commit() != resource leak |
| `mined-write-forceatlas2-claude-md` | docstrings | Docstring rule is for code functions, not doc files |
| `mined-write-advanced-docs` | docstrings | Same |

### Trimmed (some rules removed, stays positive)

| ID | Removed | Kept | Reason |
|----|---------|------|--------|
| `embedded-code-python-class` | frozen dataclass | docstrings | Code already has frozen=True |
| `long-edit-flask-route` | error handling | docstrings | Code HAS try/except with logging |
| `mined-write-postgres-ts-connection` | console.log | env vars, close resources | Cannot tell from query if code has console.log |

## Changes — post_tool_use.json (3 changes)

| ID | Change | Reason |
|----|--------|--------|
| `post-neg-bash-pip-list` | Removed uv rule from `should_not_match` | Same rule was in BOTH should_match AND should_not_match (contradiction) |
| `v3-post-edit-add-frozen-dataclass` | Converted to negative | Edit "adds frozen dataclass" — compliance, not violation |
| `v3-post-bash-pytest-failures` | Converted to negative | Test assertion failures are not error handling violations |

## Impact

### Before
- basic.json: 442 total, ~245 positive, ~197 negative
- post_tool_use.json: 115 total, ~67 positive, ~48 negative

### After
- basic.json: 442 total, 220 positive, 222 negative
- post_tool_use.json: 115 total, 66 positive, 49 negative

**Net effect:** 25 fewer positive fixtures (false expectations removed), which should improve:
- **F2 scores:** Model no longer penalized for correct behavior (not firing on compliance)
- **Negative silence rate:** More negatives to measure abstention quality
- **Noise ratio:** Fewer false positive expectations inflating noise measurement
