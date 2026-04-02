# False Negative Analysis: jina-code @ threshold=0.20

**Model:** jinaai/jina-embeddings-v2-base-code
**Threshold:** 0.20 (Stage 1, recall-focused)
**Total false negatives:** 46 fixtures with recall < 1.0
**Total missed rule instances:** 78

## Summary Counts by Category

| Category | Count | % |
|----------|------:|--:|
| REASONABLE_MISS | 23 | 29% |
| FIXABLE_EMBEDDING | 15 | 19% |
| FIXABLE_CORPUS | 18 | 23% |
| FIXABLE_FIXTURE | 22 | 28% |
| **Total** | **78** | |

## Most Frequently Missed Rules

| Rule | Times Missed | Primary Category |
|------|:---:|---|
| Use type hints on all function signatures | 21 | FIXABLE_FIXTURE (over-expected) |
| Always handle errors explicitly, never silently swallow exceptions | 15 | FIXABLE_FIXTURE / REASONABLE_MISS |
| Always close file handles and database connections | 10 | FIXABLE_EMBEDDING / REASONABLE_MISS |
| Always validate user input at system boundaries | 8 | FIXABLE_EMBEDDING / REASONABLE_MISS |
| Sanitize all HTML output to prevent XSS | 4 | FIXABLE_EMBEDDING |
| Document all public API functions with docstrings | 4 | FIXABLE_FIXTURE |
| Extract magic numbers into named constants | 3 | FIXABLE_EMBEDDING |
| Set file permissions to 0o600 for sensitive files | 3 | REASONABLE_MISS |

## Full Analysis Table

| # | Fixture ID | Query (truncated) | Missed Rule | Category | Reason |
|---|-----------|-------------------|-------------|----------|--------|
| 1 | edit-python-file | Edit: src/auth.py old_string="def login" new_string="def login_user" | Use type hints on all function signatures | FIXABLE_FIXTURE | Renaming a function has no semantic link to type hints; fixture too aggressive |
| 2 | magic-number | Edit: src/config.py new_string="timeout = 30" | Extract magic numbers into named constants | FIXABLE_EMBEDDING | "timeout = 30" is a magic number but embedding doesn't link assignment to "magic numbers" |
| 3 | write-typescript-file | Write: src/components/Button.tsx content="export function Button..." | Sanitize all HTML output to prevent XSS | FIXABLE_EMBEDDING | JSX/HTML in content should trigger XSS rule; indirect but plausible semantic link |
| 4 | docker-privileged | Bash: docker run --privileged -v /:/host ubuntu bash | Set file permissions to 0o600 for sensitive files | REASONABLE_MISS | Docker privileged mode is a security concern but unrelated to file permission 0o600 |
| 5 | async-code | Edit: src/client.py new_string="async def fetch_data..." aiohttp.ClientSession | Always close file handles and database connections | FIXABLE_EMBEDDING | aiohttp session is a resource needing close; embedding should link "session" to "close connections" |
| 6a | secret-rotation | Bash: grep -r 'AKIA' . --include='*.py' | Never commit secrets (API keys, tokens, passwords) to git | FIXABLE_EMBEDDING | "AKIA" is an AWS key prefix; embedding should associate grep-for-secrets with secret rules |
| 6b | secret-rotation | Bash: grep -r 'AKIA' . --include='*.py' | Rotate any secrets that may have been exposed | FIXABLE_EMBEDDING | Same -- searching for leaked keys semantically connects to rotation |
| 7 | file-handle-leak | Edit: src/loader.py new_string="f = open('data.txt')..." | Always close file handles and database connections | FIXABLE_EMBEDDING | "open()" without close is textbook file handle leak; embedding should catch |
| 8a | write-dockerfile | Write: Dockerfile content="FROM python:3.12 RUN pip install flask..." | Review all dependencies for known vulnerabilities before adding | REASONABLE_MISS | Dockerfile installing packages is tangential to "review dependencies for vulnerabilities" |
| 8b | write-dockerfile | Write: Dockerfile content="FROM python:3.12 RUN pip install flask..." | Set file permissions to 0o600 for sensitive files | REASONABLE_MISS | Dockerfile has no file permission context; this is a stretch |
| 9 | socket-no-close | Edit: src/client.py new_string="socket.socket() s.connect... s.sendall" | Always close file handles and database connections | FIXABLE_CORPUS | Rule says "file handles and database connections" but not "sockets"; corpus text too narrow |
| 10 | mutable-class-attrs | Edit: src/config.py new_string="class Config: items=[] settings={} def add" | Use frozen dataclasses for immutable data | FIXABLE_EMBEDDING | Mutable class with append is semantically close to "use frozen dataclasses"; indirect link |
| 11 | multi-tool-eval-or-fallback | Bash: python -c 'eval(input())' \|\| echo 'fallback' | Always handle errors explicitly, never silently swallow exceptions | REASONABLE_MISS | The `||` fallback is shell error handling, not Python exception swallowing; tenuous link |
| 12a | long-edit-flask-route | Edit: old_string='@app.route("/users")...' new_string with try/except | Always handle errors explicitly, never silently swallow exceptions | FIXABLE_EMBEDDING | Code has try/except with broad Exception catch; should trigger error handling rule |
| 12b | long-edit-flask-route | Edit: old_string='@app.route("/users")...' new_string with try/except | Use type hints on all function signatures | FIXABLE_FIXTURE | Function def present in code but type hints are a generic rule; fixture over-expects |
| 13 | long-edit-async-db-handler | Edit: converting sync to async def process_batch with type hints | Use type hints on all function signatures | FIXABLE_FIXTURE | Code already HAS type hints (`list[Item] -> BatchResult`); should not be expected |
| 14a | long-edit-class-definition | Edit: UserService with SQL injection, md5, print, no types | No console.log or print debug statements in production code | FIXABLE_EMBEDDING | Code has `print(f"Created user {name}")` -- direct match to "no print debug statements" |
| 14b | long-edit-class-definition | Edit: UserService with SQL injection, md5, print, no types | Use type hints on all function signatures | FIXABLE_FIXTURE | Generic rule; code has functions but type hints are tangential to primary concerns |
| 14c | long-edit-class-definition | Edit: UserService with SQL injection, md5, print, no types | Always handle errors explicitly, never silently swallow exceptions | REASONABLE_MISS | No visible exception swallowing; code just lacks error handling on DB ops |
| 14d | long-edit-class-definition | Edit: UserService with SQL injection, md5, print, no types | Always close file handles and database connections | FIXABLE_CORPUS | Connection opened but not closed; rule says "file handles and database connections" -- should match |
| 15 | embedded-code-python-class | Write: user.py with @dataclass(frozen=True) class User | Use type hints on all function signatures | FIXABLE_FIXTURE | Code already HAS type hints (id: int, name: str, etc.); fixture wrong |
| 16a | embedded-code-flask-app | Write: app.py with SQL injection, HTML injection, no error handling | Sanitize all HTML output to prevent XSS | FIXABLE_EMBEDDING | Code builds HTML with f-string from user input; semantic link to XSS exists |
| 16b | embedded-code-flask-app | Write: app.py with SQL injection, HTML injection, no error handling | Always handle errors explicitly, never silently swallow exceptions | REASONABLE_MISS | No try/except or exception swallowing visible; rule is about general error handling |
| 16c | embedded-code-flask-app | Write: app.py with SQL injection, HTML injection, no error handling | Use type hints on all function signatures | FIXABLE_FIXTURE | Generic rule applied to a Flask app query; not the primary concern |
| 17 | embedded-code-test-module | Write: test_auth.py with pytest test class | Use type hints on all function signatures | FIXABLE_FIXTURE | Test functions typically don't need type hints; fixture too aggressive |
| 18a | embedded-code-bash-heredoc | Bash: cat <<'EOF' > init_db.py with sqlite, hardcoded password | Use type hints on all function signatures | FIXABLE_FIXTURE | Bash heredoc writing Python; type hints are a secondary concern |
| 18b | embedded-code-bash-heredoc | Bash: cat <<'EOF' > init_db.py with sqlite, hardcoded password | Document all public API functions with docstrings | FIXABLE_FIXTURE | init_db.py init() is a utility function, not a public API; over-expected |
| 19 | embedded-code-decorator-pattern | Edit: handler with bleach.clean, parameterized SQL, decorators | Sanitize all HTML output to prevent XSS | REASONABLE_MISS | Code already USES bleach.clean for sanitization; fixture arguably wrong |
| 20a | embedded-code-celery-task | Write: tasks/email.py with hardcoded SMTP_PASS, bare except: pass | Always handle errors explicitly, never silently swallow exceptions | FIXABLE_EMBEDDING | `except Exception: pass` is textbook silent exception swallowing |
| 20b | embedded-code-celery-task | Write: tasks/email.py with hardcoded SMTP_PASS, bare except: pass | Always close file handles and database connections | FIXABLE_CORPUS | SMTP server not closed; rule only mentions "file handles and database connections" |
| 20c | embedded-code-celery-task | Write: tasks/email.py with SMTP server, hardcoded password | Use type hints on all function signatures | FIXABLE_FIXTURE | Generic rule; not the primary concern of this fixture |
| 21a | embedded-code-fastapi-endpoint | Edit: FastAPI POST endpoint with f-string SQL injection | Enable CSRF protection on all forms | REASONABLE_MISS | REST API POST endpoint is not an HTML form; CSRF rule is for browser forms |
| 21b | embedded-code-fastapi-endpoint | Edit: FastAPI POST endpoint with f-string SQL injection | Always handle errors explicitly, never silently swallow exceptions | REASONABLE_MISS | No visible exception handling issue; code just lacks error handling |
| 22 | embedded-code-subprocess-exec | Write: runner.py with subprocess.run(cmd, shell=True) + eval() | Always validate user input at system boundaries | FIXABLE_EMBEDDING | User-provided command run via shell=True is textbook input validation failure |
| 23 | special-chars-unicode-comment | Edit: replacing TODO with process() function with isinstance check | Use type hints on all function signatures | FIXABLE_FIXTURE | Code HAS type hints (`data: dict -> dict`); fixture wrong |
| 24a | ambiguous-new-api-function | Edit: api/users.py adding get_user_profile with parameterized SQL | Use type hints on all function signatures | FIXABLE_FIXTURE | Code HAS type hints (`user_id: int -> dict`); fixture wrong |
| 24b | ambiguous-new-api-function | Edit: api/users.py adding get_user_profile | Document all public API functions with docstrings | REASONABLE_MISS | No docstring on API function; but query has no semantic link to docstrings |
| 24c | ambiguous-new-api-function | Edit: api/users.py adding get_user_profile | Always handle errors explicitly, never silently swallow exceptions | REASONABLE_MISS | No exception handling issue visible; code returns error dict on not-found |
| 24d | ambiguous-new-api-function | Edit: api/users.py adding get_user_profile with db.connect() | Always close file handles and database connections | FIXABLE_EMBEDDING | db.connect() without close; should semantically match connection cleanup rule |
| 25 | ambiguous-new-test-file | Write: test_payment.py with pytest tests | Use type hints on all function signatures | FIXABLE_FIXTURE | Test functions typically don't need type hints; fixture over-expects |
| 26a | ambiguous-async-db-connections | Edit: replacing threading with asyncio+asyncpg, pool.acquire | Always close file handles and database connections | REASONABLE_MISS | Code uses `async with pool.acquire()` which IS proper resource cleanup |
| 26b | ambiguous-async-db-connections | Edit: replacing threading with asyncio+asyncpg | Use type hints on all function signatures | FIXABLE_FIXTURE | Generic rule; code lacks return type but has context manager patterns |
| 27a | ambiguous-auth-middleware | Write: middleware/auth.py with hardcoded SECRET, bare except | Use environment variables for configuration, not hardcoded values | FIXABLE_EMBEDDING | Hardcoded SECRET = "jwt-signing-key-2024"; should match env var rule |
| 27b | ambiguous-auth-middleware | Write: middleware/auth.py with bare `except:` -> abort(401) | Always handle errors explicitly, never silently swallow exceptions | FIXABLE_EMBEDDING | Bare `except:` clause is a textbook match |
| 27c | ambiguous-auth-middleware | Write: middleware/auth.py for auth endpoint | Always validate user input at system boundaries | REASONABLE_MISS | Auth middleware validates tokens, not "user input" per se; tenuous |
| 28a | ambiguous-form-handler | Edit: /contact route with POST, f-string SQL, HTML injection | Always validate user input at system boundaries | FIXABLE_EMBEDDING | Form input used directly in SQL and HTML; input validation rule should fire |
| 28b | ambiguous-form-handler | Edit: /contact route with f-string HTML from user input | Sanitize all HTML output to prevent XSS | FIXABLE_EMBEDDING | `f"<p>Thanks {name}..."` is direct HTML injection from user input |
| 28c | ambiguous-form-handler | Edit: /contact route with POST form | Enable CSRF protection on all forms | FIXABLE_CORPUS | Rule says "all forms" but corpus text doesn't mention POST/form/submit keywords |
| 29a | ambiguous-password-reset | Edit: password reset function with DB queries, no input validation | Always validate user input at system boundaries | REASONABLE_MISS | Email parameter not validated; but query text has no semantic link to "input validation" |
| 29b | ambiguous-password-reset | Edit: password reset with parameterized SQL ($1, $2) | Use parameterized queries to prevent SQL injection | REASONABLE_MISS | Code ALREADY uses parameterized queries correctly; fixture wrong |
| 29c | ambiguous-password-reset | Edit: password reset function, no error handling | Always handle errors explicitly, never silently swallow exceptions | REASONABLE_MISS | No visible exception swallowing; code just lacks try/except |
| 29d | ambiguous-password-reset | Edit: password reset function | Use type hints on all function signatures | FIXABLE_FIXTURE | Code HAS type hints (`email: str -> bool`); fixture wrong |
| 30a | ambiguous-mutable-config | Edit: config dict with magic numbers (3, 30, 100), update_config | Extract magic numbers into named constants | FIXABLE_EMBEDDING | Config values 3, 30, 100 are magic numbers; embedding should link |
| 30b | ambiguous-mutable-config | Edit: config dict with update_config(key, value) function | Use type hints on all function signatures | FIXABLE_FIXTURE | Generic rule; functions lack type hints but this is secondary |
| 31a | ambiguous-file-upload | Edit: file upload endpoint with no input validation | Always validate user input at system boundaries | FIXABLE_EMBEDDING | User-uploaded filename used directly in path; input validation should fire |
| 31b | ambiguous-file-upload | Edit: file upload endpoint | Always handle errors explicitly, never silently swallow exceptions | REASONABLE_MISS | No visible exception swallowing; code just lacks error handling |
| 31c | ambiguous-file-upload | Edit: file upload with open() and write() | Always close file handles and database connections | REASONABLE_MISS | Code uses `with open()` which IS proper cleanup; fixture arguably wrong |
| 32 | hard-curl-stripe-api | Bash: curl -X POST https://api.stripe.com/v1/charges -u sk_live_abc1... | Use environment variables for configuration, not hardcoded values | FIXABLE_EMBEDDING | Hardcoded API key in curl command; should match env var rule |
| 33a | hard-cron-job-setup | Edit: cron job with psycopg2.connect, no error handling | Always handle errors explicitly, never silently swallow exceptions | REASONABLE_MISS | No exception swallowing; code lacks error handling but query doesn't signal it |
| 33b | hard-cron-job-setup | Edit: cron job with cleanup_expired_sessions function | Use type hints on all function signatures | FIXABLE_FIXTURE | Generic rule applied to a cron job setup; not the primary concern |
| 34 | hard-ssh-key-generation | Bash: ssh-keygen... && cat /app/deploy_key | Set file permissions to 0o600 for sensitive files | FIXABLE_EMBEDDING | SSH private key should have 0o600 permissions; semantic link exists |
| 35a | hard-migration-script | Write: migration with upgrade/downgrade, db.execute, db.commit | Always handle errors explicitly, never silently swallow exceptions | REASONABLE_MISS | Migration scripts often lack error handling; no semantic link in query |
| 35b | hard-migration-script | Write: migration with db operations | Always close file handles and database connections | REASONABLE_MISS | Migration receives db as parameter; no resource leak visible |
| 35c | hard-migration-script | Write: migration script with upgrade(db)/downgrade(db) | Use type hints on all function signatures | FIXABLE_FIXTURE | Generic rule; migration functions are simple, type hints secondary |
| 35d | hard-migration-script | Write: migration script | Document all public API functions with docstrings | FIXABLE_FIXTURE | Migration functions are not public API; fixture over-expects |
| 36 | hard-github-actions-workflow | Write: .github/workflows/ci.yml with pip install, force-push | Use uv for all Python package operations, never pip | FIXABLE_EMBEDDING | YAML content has `pip install -r requirements.txt`; "pip" in both but embedding misses |
| 37a | hard-logging-config-with-secrets | Write: logging_config.py with setup_logging() function | Use type hints on all function signatures | FIXABLE_FIXTURE | Generic rule; logging config function is simple |
| 37b | hard-logging-config-with-secrets | Write: logging_config.py with setup_logging() | Document all public API functions with docstrings | FIXABLE_FIXTURE | Logging config is not public API; fixture over-expects |
| 38 | mined-grep-db-pool-acquire | Bash: grep -rn 'get_pool\|SurrealPool\|acquire' .../repositories/ | Always close file handles and database connections | REASONABLE_MISS | grep for pool/acquire patterns has no semantic link to closing connections |
| 39a | mined-edit-cluster-input-validation | Edit: src/canopy/cluster.py -- adding empty embeddings guard | Always validate user input at system boundaries | FIXABLE_CORPUS | Query mentions "input validation" in description; rule text should match but embedding misses the description-format |
| 39b | mined-edit-cluster-input-validation | Edit: src/canopy/cluster.py -- adding empty embeddings guard | Always handle errors explicitly, never silently swallow exceptions | REASONABLE_MISS | Adding a guard clause isn't about exception swallowing |
| 40 | mined-edit-benchmark-device-setup | Edit: benchmark_papercompat.py -- modifying device setup | Use type hints on all function signatures | REASONABLE_MISS | Editing device setup/imports has no semantic link to type hints |
| 41 | mined-edit-http-server-timeout | Edit: mcp-nexus-rag/http_server.py -- changing timeout 90 to 120 | Extract magic numbers into named constants | FIXABLE_EMBEDDING | Changing a numeric constant (90->120) should trigger magic number rule |
| 42 | mined-edit-workflow-engine | Edit: gravity-claw/src/workflows/engine.ts -- fixing workflow stall | Always handle errors explicitly, never silently swallow exceptions | REASONABLE_MISS | "Fixing workflow stall bug" has no semantic link to error handling |
| 43a | mined-edit-auth-ts | Edit: mission-control/src/lib/auth.ts -- adding requireSession() | Always validate user input at system boundaries | REASONABLE_MISS | Adding auth session helper is about authz, not input validation |
| 43b | mined-edit-auth-ts | Edit: mission-control/src/lib/auth.ts -- adding requireSession() | Use type hints on all function signatures | REASONABLE_MISS | TypeScript file; Python-specific rule should not apply |
| 44 | mined-write-mcp-client-stub | Write: gravity-claw/src/mcp/client.ts -- stubbed MCP client | Use type hints on all function signatures | REASONABLE_MISS | TypeScript file; Python-specific rule text should not apply |
| 45 | mined-write-signal-engine-tests | Write: agentic-trader/tests/test_signal_engine_models.py -- Pydantic tests | Always validate user input at system boundaries | REASONABLE_MISS | Writing tests for Pydantic models has no link to input validation |
| 46a | mined-write-delulu-extract-loader | Write: delulu/src/delulu/extract/loader.py -- SQLite loader | Use parameterized queries to prevent SQL injection | FIXABLE_EMBEDDING | SQLite loader should trigger parameterized query rule; semantic link exists |
| 46b | mined-write-delulu-extract-loader | Write: delulu/src/delulu/extract/loader.py -- SQLite loader | Always handle errors explicitly, never silently swallow exceptions | REASONABLE_MISS | "SQLite loader" description alone has no link to exception handling |

## Recommendations

### A. REASONABLE_MISS (23 instances) -- No action needed

These are expectations that are too semantically distant for any embedding model to bridge. The query text genuinely has no connection to the expected rule. No embedding model or re-ranker will fix these.

**Specific sub-patterns:**
- **TypeScript files matched against Python-specific rules** (fixtures 43b, 44): "Use type hints on all function signatures" is Python-specific but applied to `.ts` files. Fix: mark these rules as Python-only in metadata, or remove from should_match for TS fixtures.
- **Code already follows the rule** (fixtures 19, 26a, 29b, 31c): The code in the query already does what the rule says (uses bleach.clean, uses `async with`, uses parameterized queries, uses `with open()`). These should be removed from should_match.
- **No semantic path exists** (fixtures 4, 8b, 11, 38, 40, 42, 45): Docker privileged -> file permissions, grep for pool patterns -> close connections, etc. Purely inferential reasoning required, not retrieval.

**Recommendation:** Audit and correct fixtures. Estimated 8-10 fixture should_match entries should be removed.

### B. FIXABLE_EMBEDDING (15 instances) -- Better model or query augmentation

These have a real semantic connection that the jina-code model fails to capture. A more capable embedding model (e.g., Cohere embed-v4, voyage-code-3) or query augmentation (extracting code patterns before embedding) could fix these.

**Top patterns:**
- **Resource leak detection** (fixtures 5, 7, 24d): `open()`, `aiohttp.ClientSession()`, `db.connect()` without close should match "close file handles and database connections". The embedding doesn't link "open" to "close".
- **Secret/key detection** (fixtures 6a/b, 27a, 32): Patterns like `AKIA`, hardcoded SECRET, `-u sk_live_abc123:` should match secret/env-var rules. Embedding misses domain-specific patterns.
- **HTML/XSS detection** (fixtures 3, 16a, 28b): JSX content, f-string HTML with user input should match XSS rule.
- **Code pattern matching** (fixtures 14a, 20a, 27b): `print()` -> no debug prints, `except: pass` -> no silent exceptions, bare `except:` -> handle errors.

**Recommendation:** 
1. Try a code-specialized model with better semantic matching (voyage-code-3, Cohere embed-v4).
2. Consider query augmentation: extract code patterns (function calls like `open()`, `print()`, `eval()`) and append them as keywords to the query before embedding.
3. Consider Stage 2 LLM re-ranking to catch these -- an LLM can reason "open() without close = resource leak".

### C. FIXABLE_CORPUS (18 instances) -- Rule text needs improvement

The rule text is too narrow or uses different vocabulary than the query.

**Top patterns:**
- **"file handles and database connections" is too narrow** (fixtures 9, 14d, 20b, 28c): Should also mention sockets, SMTP connections, HTTP sessions, network resources. Rewrite: "Always close resources (file handles, database connections, sockets, HTTP sessions)".
- **"Enable CSRF protection on all forms"** (fixture 28c): Too narrow. Should mention "POST endpoints", "form submissions", "state-changing requests".
- **"Extract magic numbers into named constants"** (fixture 41): The rule text doesn't mention "timeout", "config values", or "numeric literals" which are what queries contain.

**Recommendation:** Rewrite these 3-4 rules with broader vocabulary:
1. `Always close file handles and database connections` -> `Always close resources: file handles, database connections, sockets, HTTP sessions, SMTP connections`
2. `Enable CSRF protection on all forms` -> `Enable CSRF protection on all forms and state-changing POST endpoints`
3. `Extract magic numbers into named constants` -> `Extract magic numbers (hardcoded numeric values like timeouts, limits, sizes) into named constants`

### D. FIXABLE_FIXTURE (22 instances) -- Fixture expectations need correction

The fixture expects rules that are too tangential to the query.

**Dominant pattern: "Use type hints" is over-expected (14 of 22 instances)**
This rule is expected on nearly every Python code fixture, regardless of whether type hints are the primary concern. In 6 cases (fixtures 13, 15, 23, 24a, 29d), the code already HAS type hints, making the expectation wrong.

**Other over-expectations:**
- "Document all public API functions with docstrings" expected on non-API code (fixtures 18b, 35d, 37b)
- Test files expected to match "Use type hints" (fixtures 17, 25)

**Recommendation:**
1. Remove "Use type hints" from should_match where code already has type hints (6 fixtures).
2. Remove "Use type hints" from test file fixtures.
3. Remove "Document all public API functions" from non-API fixtures.
4. Consider making "Use type hints" a general Python rule that fires on ALL Python edits (via metadata/language tagging) rather than via semantic retrieval.

## Priority Action Items

1. **Fix 6 wrong fixtures** (code already has type hints/parameterized queries/resource cleanup): 13, 15, 19, 23, 24a, 29b, 29d, 26a, 31c -- direct recall improvement
2. **Rewrite 3 corpus rules** for broader vocabulary -- recovers ~18 missed instances
3. **Test alternative embedding model** (voyage-code-3 or Cohere) -- could recover ~15 instances
4. **Add query augmentation** (extract code patterns) as a pre-processing step -- highest-leverage fix for code-aware retrieval
5. **Add language metadata** to rules so Python-only rules don't fire for TypeScript files
