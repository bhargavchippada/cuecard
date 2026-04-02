# Fixture Audit: 2026-04-02

**Auditor:** Task #14 (enriched-retrieval team)
**Based on:** `artifacts/false-negative-analysis.md`
**Scope:** Medium and hard tier fixtures in basic.json (354 fixtures)

## Summary

| Action | Count |
|--------|------:|
| should_match entries removed | 25 |
| Fixtures reclassified to negative | 6 |
| Corpus rules reworded | 2 |
| Fixture rule text updated (to match corpus) | 31 |

Net effect: 4 medium fixtures reclassified to negative, 2 hard fixtures reclassified to negative.

## Fixture Changes

### 1. "Use type hints" removed (10 instances)

The rule "Use type hints on all function signatures" was the most over-expected rule (14 of 22 FIXABLE_FIXTURE instances in the analysis). Removed where:

| Fixture ID | Reason |
|------------|--------|
| ambiguous-password-reset | Code already has type hints (`email: str -> bool`) |
| mined-edit-auth-ts | TypeScript file; Python-specific rule |
| mined-write-mcp-client-stub | TypeScript file; Python-specific rule |
| long-function | Tangential; primary concern is function length |
| ambiguous-mutable-config | Tangential; primary concerns are mutability and magic numbers |
| ambiguous-auth-middleware | Least relevant of 5 should_match entries |
| hard-cron-job-setup | Tangential; primary concerns are resource closing and error handling |
| hard-migration-script | Tangential; migration scripts are simple utility functions |
| hard-logging-config-with-secrets | Tangential; primary concerns are hardcoded secrets |
| mined-edit-benchmark-device-setup | No semantic link between device setup and type hints |
| embedded-code-decorator-pattern | Code already has type hints (`request: Request -> Response`) |

### 2. "Document all public API functions" removed (3 instances)

| Fixture ID | Reason |
|------------|--------|
| hard-migration-script | `upgrade(db)` / `downgrade(db)` are not public API |
| hard-logging-config-with-secrets | `setup_logging()` is a utility, not public API |
| embedded-code-decorator-pattern | Code already has docstring (`"""Handle user update requests."""`) |

### 3. "Set file permissions to 0o600" removed (3 instances)

| Fixture ID | Reason |
|------------|--------|
| docker-privileged | Docker --privileged is unrelated to file permission settings |
| rm-rf-system | Destructive operation, not about permission settings |
| write-dockerfile | Dockerfile has no file permission context |

### 4. Code already follows the rule (3 instances)

| Fixture ID | Rule Removed | Reason |
|------------|-------------|--------|
| ambiguous-password-reset | Use parameterized queries | Code uses `$1, $2` parameterized queries |
| ambiguous-file-upload | Always close file handles... | Code uses `with open()` for proper cleanup |
| embedded-code-decorator-pattern | Document all public API functions | Code already has docstring |

### 5. No semantic path exists (5 instances)

| Fixture ID | Rule Removed | Reason |
|------------|-------------|--------|
| embedded-code-fastapi-endpoint | Enable CSRF protection | REST API uses tokens, not browser forms |
| multi-tool-eval-or-fallback | Always handle errors explicitly | Shell `\|\|` is shell error handling, not Python |
| mined-edit-workflow-engine | Always handle errors explicitly | "Fixing stall bug" has no link to error handling |
| mined-edit-cluster-input-validation | Always handle errors explicitly | Guard clause is not about exception swallowing |
| mined-edit-auth-ts | Always validate user input | Auth session is authorization, not input validation |
| mined-write-signal-engine-tests | Always validate user input | Pydantic model tests have no link to input validation |

### 6. Fixtures reclassified to negative (6 total)

These fixtures had all should_match entries removed:

| Fixture ID | Was | Entries Removed |
|------------|-----|----------------|
| docker-privileged | hard | "Set file permissions to 0o600" |
| rm-rf-system | hard | "Set file permissions to 0o600" |
| mined-write-mcp-client-stub | medium | "Use type hints" (TypeScript file) |
| mined-edit-benchmark-device-setup | medium | "Use type hints" (no semantic link) |
| mined-edit-workflow-engine | medium | "Always handle errors explicitly" (no semantic link) |
| mined-edit-auth-ts | medium | "Use type hints" + "Always validate user input" (TypeScript + authz != validation) |

## Corpus Rule Changes

### 1. Magic numbers rule broadened

**Before:** `Extract magic numbers and hardcoded values into named constants`
**After:** `Extract magic numbers (hardcoded timeouts, limits, sizes, counts) into named constants`

**Rationale:** Queries containing timeout/limit values (e.g., `timeout = 30`, `timeout 90 -> 120`) should match this rule. Adding concrete examples helps embedding models bridge the semantic gap (analysis #2, #41).

### 2. Resource closing rule broadened

**Before:** `Always close file handles, database connections, network sockets, and other resources`
**After:** `Always close file handles, database connections, network sockets, HTTP sessions, and other resources`

**Rationale:** aiohttp.ClientSession and similar HTTP clients are resources needing explicit close. Adding "HTTP sessions" helps embedding models match queries involving `aiohttp.ClientSession`, `requests.Session`, etc. (analysis #5, #20b).

## What Was NOT Changed

- **Easy tier fixtures:** Not audited (out of scope).
- **Negative tier fixtures:** Not modified (already correct, mined-grep-db-pool-acquire was already negative).
- **Workflow fixtures:** No changes needed; workflow fixtures have well-targeted expectations.
- **FIXABLE_EMBEDDING entries:** These are real semantic links that the embedding model misses. They need better models or query augmentation, not fixture changes.
- **REASONABLE_MISS entries** that represent genuinely hard retrieval: kept as hard-tier fixtures to track improvement over time.

## Fixture Count Changes

| Tier | Before | After | Delta |
|------|--------|-------|-------|
| easy | 62 | 62 | 0 |
| medium | 117 | 113 | -4 |
| hard | 48 | 46 | -2 |
| negative | 127 | 133 | +6 |
| **Total** | **354** | **354** | **0** |
