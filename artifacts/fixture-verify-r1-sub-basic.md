# Fixture Verification Round 1 — SubagentStart + PreToolUse(basic)

## Critical Finding: 15 Old Rules Missing from Expanded Corpus

Before fixture-level issues, a corpus-level problem: **15 rules referenced by fixtures are NOT in the 65-rule `rules_global.txt` corpus**. These were in the old 32-rule corpus but were dropped during expansion to 65 rules. Fixtures referencing them will always fail retrieval because the rules don't exist in the corpus.

**Missing old rules (still referenced by 100+ fixtures):**
1. "Always create a new branch for feature work"
2. "Document all public API functions with docstrings"
3. "Enable CSRF protection on all forms and state-changing endpoints"
4. "Extract magic numbers (hardcoded timeouts, limits, sizes, counts) into named constants"
5. "No console.log or print debug statements in production code"
6. "Prefer immutable data structures over mutable ones"
7. "Review all dependencies for known vulnerabilities before adding"
8. "Rotate any secrets that may have been exposed"
9. "Run quality checks before every commit"
10. "Sanitize all HTML output to prevent XSS"
11. "Use asyncio for I/O-bound operations, not threads"
12. "Use frozen dataclasses for immutable data"
13. "Use parameterized queries to prevent SQL injection"
14. "Use type hints on all function signatures"
15. "Write tests before implementation (TDD red-green-refactor)"

**Action needed:** Add these 15 rules to `eval/corpora/rules_global.txt` OR remove all references from fixtures. Adding them to the corpus is strongly preferred — they are all legitimate coding rules.

---

## SubagentStart Fixes Needed

### Additions — New rules from expanded corpus that should match

- `sub-security-reviewer`: ADD — "Always validate user input at system boundaries" because security reviewers checking for vulnerabilities need input validation rules
- `sub-code-reviewer`: ADD — "Never mutate function arguments or shared state" because code reviewers checking for patterns/best practices need immutability rules
- `sub-code-reviewer`: ADD — "Keep functions under 50 lines and source files under 800 lines" because code quality review includes file/function size (note: fixtures reference old text "Keep functions under 50 lines, files under 400 lines" which is not in the 65-rule corpus)
- `sub-tdd-guide`: ADD — "Always mock external calls (LLM, APIs, network, subprocesses) in unit tests" because TDD guide writing tests needs to know mock requirements
- `sub-tdd-guide`: ADD — "The full unit test suite must complete in under 5 seconds" because TDD guide needs test speed awareness
- `sub-implement-feature`: ADD — "Never mutate function arguments or shared state" because implementing a caching layer should follow immutability principles
- `sub-implement-feature`: ADD — "Cache expensive external data with explicit TTL" because the task literally mentions "caching layer with TTL support"
- `sub-implement-api-endpoint`: ADD — "Never mutate function arguments or shared state" because API endpoint implementation should follow immutability
- `sub-implement-api-endpoint`: ADD — "Always mock external calls (LLM, APIs, network, subprocesses) in unit tests" because building an API endpoint with tests needs mock rules
- `sub-async-service`: ADD — "Log every external API call with latency, status code, and response size" because an async HTTP client fetching from upstream APIs needs API observability
- `sub-async-service`: ADD — "Never use silent fallbacks" because an async service needs explicit error surfacing
- `sub-install-deps`: ADD — "Spike-test risky integrations before building on them" because installing new packages for a new service benefits from validation
- `sub-planner-feature`: ADD — "Log every external API call with latency, status code, and response size" because a webhook notification system makes HTTP POST callbacks
- `sub-planner-feature`: ADD — "Never use silent fallbacks" because webhook system needs explicit failure surfacing
- `sub-planner-feature`: ADD — "Design module interfaces as Protocols" because building a new subsystem benefits from protocol-based interfaces
- `sub-e2e-runner-login-flow`: ADD — "Never use silent fallbacks" because login flow testing should verify no silent degradation
- `sub-multi-task-agent`: ADD — "Never commit secrets (API keys, tokens, passwords) to git" because the task includes committing, which needs secret checking
- `sub-implement-feature`: ADD — "Read 2-3 similar files in the codebase before writing new code" because implementing new code should follow existing patterns
- `sub-implement-api-endpoint`: ADD — "Read 2-3 similar files in the codebase before writing new code" because building a new endpoint should follow existing patterns
- `sub-hard-ambiguous-script`: ADD — "Read 2-3 similar files in the codebase before writing new code" because creating a utility script should follow existing patterns
- `sub-hard-config-agent`: ADD — "Never mutate function arguments or shared state" because a configuration system should be immutable
- `sub-file-processor`: ADD — "Never mutate function arguments or shared state" because a data processing pipeline should use immutable patterns
- `sub-hard-async-crawler`: ADD — "Log every external API call with latency, status code, and response size" because a web crawler makes external HTTP requests
- `sub-credential-manager`: ADD — "Never use silent fallbacks" because credential management must be explicit about failures

### Removals

- `sub-doc-updater`: This is currently a negative fixture (should_match=[]) but the task says "Update README.md and CLAUDE.md". The new rules "Update CLAUDE.md when project structure changes" (rule 17) and "Update README when user-facing behavior changes" (rule 18) are directly relevant. REMOVE negative status — convert to positive.
- `sub-doc-updater`: ADD — "Update CLAUDE.md when project structure, architecture, or conventions change" because the task explicitly says to update CLAUDE.md
- `sub-doc-updater`: ADD — "Update README when user-facing behavior, setup steps, or dependencies change" because the task explicitly says to update README.md
- `sub-doc-updater`: DIFFICULTY change from "negative" to "easy"

### Difficulty Changes

- `sub-data-model-design`: DIFFICULTY — should be "medium" not "hard". Designing a data model with immutability is a direct match to immutability rules, not ambiguous.
- `sub-hard-build-agent-python`: DIFFICULTY — should be "medium" not "hard". Fixing build errors and adding dependencies directly maps to uv and vulnerability rules.

---

## PreToolUse (basic) Fixes Needed

### Additions — New rules from expanded corpus that should match

- `sub-implement-feature` and many Edit/Write fixtures that create new code: The new rule "Read 2-3 similar files in the codebase before writing new code" (rule 24) is relevant to many Write/Edit fixtures. However, for PreToolUse semantics, the action is the Edit/Write itself — the rule says to read similar files BEFORE writing, which means the reading should have happened already. This rule is better suited for UserPromptSubmit ("I'm going to implement X") than for the actual Edit tool call. **Not adding to PreToolUse Edit/Write fixtures** — this is a methodological rule, not a tool-action constraint.

- `git-commit-secrets`: ADD — "Run ruff and mypy before committing Python code" because committing should trigger linting rules (assuming Python context from other fixtures)
- `git-commit-quality`: ADD — "Never commit secrets (API keys, tokens, passwords) to git" because any git commit should check for secrets
- `write-env-file`: ADD — "Set file permissions to 0o600 for sensitive files" because writing a .env file with API keys needs secure permissions
- `write-yaml-secrets`: ADD — "Set file permissions to 0o600 for sensitive files" because writing secrets.yaml needs secure permissions
- `write-yaml-secrets`: ADD — "Use environment variables for configuration, not hardcoded values" because hardcoded secrets in YAML should use env vars
- `subprocess-shell-true`: ADD — "Never use eval(), exec(), or shell=True in production code" because `shell=True` is explicitly called out in the expanded rule text. Currently only matches "Always validate user input" which is weaker. This is actually the PRIMARY match.
- `password-in-code`: ADD — "Use environment variables for configuration, not hardcoded values" because hardcoded passwords should be env vars
- `write-dockerfile`: ADD — "Use uv for all Python package operations, never pip" because the Dockerfile content uses `pip install`
- `embedded-code-flask-app`: ADD — "Use environment variables for configuration, not hardcoded values" because `DB_PATH = "app.db"` is hardcoded config
- `ambiguous-async-db-connections`: This fixture shows code using parameterized queries (`$1`) which is COMPLIANCE not violation. REMOVE — "Use parameterized queries to prevent SQL injection" because the new code is FOLLOWING the rule correctly
- `long-write-config-module`: ADD — "Set file permissions to 0o600 for sensitive files" because writing a config file with hardcoded secrets and passwords needs secure permissions
- `long-edit-class-definition`: ADD — "Never commit secrets (API keys, tokens, passwords) to git" because `hashlib.md5(password)` is storing password insecurely
- `embedded-code-celery-task`: ADD — "Never mutate function arguments or shared state" because global SMTP_PASS is shared state (borderline, skip)
- `docker-run`: This fixture is borderline — running a container doesn't inherently need "Use environment variables" more than any other command. The rule match is weak. Keep as-is (hard difficulty is appropriate).
- `write-markdown-docs`: This is currently negative (should_match=[]). With new rules, "Update README when user-facing behavior changes" (rule 18) could apply if writing to docs, but writing API reference docs is not README. Correct as-is, keep negative but change difficulty from "medium" to "negative" (it IS negative already; difficulty should match).
- `mined-uv-run-classify`: REMOVE — "Use uv for all Python package operations, never pip" because the command IS using uv (compliance, not violation). This is running a tool via uv, not installing packages.
- `embedded-code-python-class`: The fixture shows a frozen dataclass with methods but no docstrings. The "Document all public API functions with docstrings" rule is in the old corpus but NOT in the 65-rule corpus. If added back to corpus, the match is correct.
- `hard-read-package-lock`: REMOVE — "Review all dependencies for known vulnerabilities before adding" because READING a lock file is a read-only operation. No action to constrain. Change to negative.
- `mined-delulu-tmux-ralph-fix`: REMOVE — "Use type hints on all function signatures" because sending a tmux message about fixing mypy errors is not the same as writing code that needs type hints. The tmux send-keys rule is correct though.
- `borderline-subprocess-run`: The fixture shows code replacing os.system with subprocess.run (which is an IMPROVEMENT). "Never use eval() or exec() in production code" is debatable since subprocess.run without shell=True is the safe version. The old code used f-string interpolation. Keep as-is — it's correctly rated "hard" difficulty.
- `mined-edit-docker-compose-model`: REMOVE — "Review all dependencies for known vulnerabilities before adding" because changing a model tag in docker-compose is not adding a new dependency. It's updating a config value.
- `mined-edit-docker-compose-model`: DIFFICULTY — change from "hard" to "negative" (no rules should match for changing a model tag)

### Compliance-vs-Violation Issues (fixtures matching rules the code FOLLOWS)

- `embedded-code-decorator-pattern`: The code shows `@validate_input`, `@rate_limit`, parameterized queries with `$1, $2`, and input sanitization with `bleach.clean()`. The should_match includes "Always validate user input" and "Use parameterized queries" — but the code is FOLLOWING these rules. For PreToolUse, we're checking the code being written, and even compliant code should have the rules injected as context. Keep as-is.

### Difficulty Changes

- `hard-read-package-lock`: DIFFICULTY — change from "hard" to "negative" (reading a file is read-only, should not match any action rules)
- `mined-edit-docker-compose-model`: DIFFICULTY — change from "hard" to "negative"
- `mined-uv-run-classify`: DIFFICULTY — change from "easy" to "negative" and clear should_match (running a CLI tool via uv is not a package install operation)
- `write-markdown-docs`: DIFFICULTY is already "medium" which seems wrong for a negative fixture; if no rules apply, difficulty should be "negative"

### New Rule Matches for Specific Fixtures

- `embedded-code-bash-heredoc`: ADD — "Always handle errors explicitly, never silently swallow exceptions" because the init_db.py code has no error handling
- `long-write-config-module`: ADD — "Never mutate function arguments or shared state" — skip, borderline
- `hard-github-actions-workflow`: ADD — "Run ruff and mypy before committing Python code" because CI workflow should include linting
- `hard-makefile-creation`: ADD — "Never commit secrets (API keys, tokens, passwords) to git" — no, Makefile has no secrets. Skip.
- `borderline-write-config-with-defaults`: ADD — "Never mutate function arguments or shared state" because a mutable dataclass can be mutated. Already matches "Prefer immutable" (old corpus rule).
- `ambiguous-mutable-config`: ADD — "Never mutate function arguments or shared state" because `config[key] = value` mutates shared state directly

---

## Summary

- **Critical corpus issue:** 15 rules from old corpus missing in expanded 65-rule corpus, affecting 100+ fixtures
- **SubagentStart additions needed:** 22
- **SubagentStart removals needed:** 1 (sub-doc-updater reclassified from negative to positive)
- **SubagentStart difficulty changes needed:** 3 (sub-doc-updater, sub-data-model-design, sub-hard-build-agent-python)
- **PreToolUse (basic) additions needed:** 12
- **PreToolUse (basic) removals needed:** 4 (ambiguous-async-db-connections, mined-uv-run-classify, hard-read-package-lock, mined-edit-docker-compose-model)
- **PreToolUse (basic) difficulty changes needed:** 4 (hard-read-package-lock, mined-edit-docker-compose-model, mined-uv-run-classify, write-markdown-docs)

### Priority

1. **HIGHEST:** Add the 15 missing rules back to `rules_global.txt` — without this, ~40% of positive fixtures are broken
2. **HIGH:** Fix the compliance-vs-violation removals (fixtures matching rules the code follows)
3. **MEDIUM:** Add new rule matches from expanded corpus
4. **LOW:** Difficulty adjustments
