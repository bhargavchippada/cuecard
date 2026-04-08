# Fixture Audit Findings

Audit of PostToolUse, Stop, and SubagentStart fixture expectations against the 30-rule basic corpus.

## PostToolUse

### Wrong expectations (should_match contains wrong rules)

- `post-bash-git-commit-clean`: "Use conventional commit format" should NOT be in should_match because the output shows `feat: add search` which IS conventional commit format. This is compliance, not violation. PostToolUse VERIFY means checking for violations in the output — the output shows correct behavior, so no rule should fire. This should be a negative fixture.

- `post-bash-test-failures`: "100% test coverage on all new code" should NOT be in should_match because the output shows test FAILURES (AssertionError), not coverage issues. Test failures and test coverage are different things. The output says "3 failed, 12 passed" — it does not show any coverage report. There is no rule in the corpus about "all tests must pass." The coverage rule is about coverage percentage, which is not shown here.

- `post-bash-ruff-errors`: "Run ruff and mypy before committing Python code" should NOT be in should_match because the output shows ruff WAS run (the user ran `uv run ruff check src/`). The rule says to run ruff before committing — the developer IS running ruff. The output shows lint errors exist, but the rule is about running the tool, not about having zero errors. This is compliance with the rule (they ran ruff), not violation.

- `post-bash-branch-on-main`: "Always create a new branch for feature work" should NOT be in should_match because `git branch` just lists branches — it is a read-only informational command. The output shows `* main` meaning the user is on main, but there is no evidence of feature work happening on main. Listing branches is not a violation. This should be a negative fixture or at minimum downgraded to hard with a more specific query showing feature work on main.

### Missing expectations (should_match is missing rules)

- `post-edit-no-input-validation`: "Use type hints on all function signatures" should BE in should_match because `def create_user(request)` has no type hints on the parameter or return type. The violation is clearly visible in the output.

- `post-edit-sql-injection`: "Always validate user input at system boundaries" should BE in should_match because the `get_user(name)` function takes user input (`name`) and passes it directly to a SQL query without any validation.

- `post-write-no-type-hints`: "Document all public API functions with docstrings" could arguably be in should_match because the functions in utils.py have no docstrings, though this is debatable since utils may not be "public API."

- `post-edit-hardcoded-config`: "Extract magic numbers (hardcoded timeouts, limits, sizes, counts) into named constants" should BE in should_match because `TIMEOUT = 30` and `MAX_RETRIES = 3` are hardcoded numeric values. Though they are assigned to named constants already (TIMEOUT, MAX_RETRIES), the values should come from environment variables since they are configuration. Rule 13 (env vars) is already matched. Rule 20 (magic numbers) is borderline since they ARE named constants — this one is actually fine as-is on reflection.

- `post-edit-console-log`: "Use type hints on all function signatures" should BE in should_match because `def fetch_data(url)` has no type hints on the parameter or return type.

- `post-edit-swallowed-exception`: The code `except Exception: pass` also implicitly violates "Use type hints on all function signatures" if we could see the full function signature, but the snippet is too minimal to judge.

- `post-write-sensitive-file-perms`: "Use environment variables for configuration, not hardcoded values" should BE in should_match because the file contains hardcoded `api_key`, `db_password`, and `aws_secret` values that should be environment variables, not written to a YAML file.

- `post-edit-secret-rotation`: "Never commit secrets (API keys, tokens, passwords) to git" should BE in should_match because the code is logging the API key, which means the secret is exposed. Additionally, "No console.log or print debug statements in production code" is relevant since `logger.info(f'Using key: {api_key}')` is effectively logging sensitive debug information.

- `post-bash-pip-audit-vuln`: "Extract magic numbers (hardcoded timeouts, limits, sizes, counts) into named constants" should NOT be added — just noting that installing an old version (1.7.0) is the vulnerability signal, and the existing expectations are correct.

- `post-edit-long-function`: "Extract magic numbers (hardcoded timeouts, limits, sizes, counts) into named constants" is not visible enough to add. But the deeply nested code also violates common style principles not in the corpus, so the existing match is fine.

### Difficulty mismatches

- `post-bash-git-commit-clean`: rated "easy" but should be "negative" because the output shows compliance with conventional commit format, not a violation (see wrong expectations above).

- `post-bash-branch-on-main`: rated "hard" — this is correct IF the expectation is kept, since inferring "you should create a branch" from a `git branch` listing requires significant reasoning. But the fixture itself is problematic (see wrong expectations).

- `post-bash-ruff-errors`: rated "medium" but if the expectation is wrong (see above), it should be "negative" — ruff was run successfully.

- `post-bash-test-failures`: rated "medium" but the match is wrong (coverage != test failures). If kept, the difficulty should be "hard" since the connection between test failures and the coverage rule is tenuous.

## Stop

### Wrong expectations (should_match contains wrong rules)

- `stop-file-edits-python`: "Always validate user input at system boundaries" should NOT be in should_match because the Stop summary says "add new user creation endpoint" but we cannot see whether input validation was omitted. The Stop event should only match rules where the violation is described or strongly implied. Merely mentioning "endpoint" does not mean validation was skipped. "Use type hints on all function signatures" is similarly speculative — we do not know the code lacks type hints.

- `stop-wrote-new-module`: "Use type hints on all function signatures" and "Document all public API functions with docstrings" are speculative. The Stop summary says a new module was created. We do not know whether it has type hints or docstrings. These rules MIGHT apply but are not clearly violated. "100% test coverage on all new code" is valid — new code was written, and the rule should remind to write tests.

- `stop-tests-written`: "Write tests before implementation (TDD red-green-refactor)" should NOT be in should_match because the Stop summary says "Wrote test_api.py with 15 test cases" — writing tests IS compliance with TDD principles. We don't know the order (tests before or after implementation). The assumption that tests came after implementation is not stated. "100% test coverage on all new code" is valid since coverage is 95%, not 100%.

- `stop-secret-exposed-in-logs`: "Never commit secrets (API keys, tokens, passwords) to git" should NOT be in should_match because logging the Authorization header is not committing secrets to git. The exposure is in logs, not in a git commit. Rule 24 (rotate secrets) is the correct match. Rule 1 is about git commits specifically.

- `stop-refactored-dataclasses`: "Use frozen dataclasses for immutable data" and "Prefer immutable data structures" — the Stop summary says the developer refactored TO dataclasses. This could be compliance (they are adopting dataclasses) or a reminder to make them frozen. This is borderline. The rules are relevant as reminders during the audit, but the developer is moving in the right direction.

- `stop-git-commit-made`: "Use conventional commit format" — the Stop summary shows the commit message IS `fix: resolve auth bug`, which follows conventional format. This is compliance, not violation. Only "Run quality checks before every commit" is clearly applicable as an audit reminder.

### Missing expectations (should_match is missing rules)

- `stop-git-commit-made`: "Run ruff and mypy before committing Python code" should BE in should_match because a commit was made, and the rule about running linters before committing applies. (We do not know if it is a Python project, but the corpus is Python-focused.)

- `stop-git-commit-secrets-risk`: "Add .gitignore entries for generated files and build artifacts" — not directly applicable, but `.env` files should be in .gitignore. However, rule 29 is about generated/build files, not .env. So this is not a missing match.

- `stop-pip-install-used`: "Use uv for all Python package operations, never pip" is correctly matched. Could also match "Run quality checks before every commit" if the install is part of a commit workflow, but this is a stretch.

- `stop-sensitive-file-created`: "Use environment variables for configuration, not hardcoded values" should BE in should_match because database passwords and API keys should be environment variables, not stored in a YAML file.

- `stop-linting-skipped`: "Use conventional commit format (feat:, fix:, refactor:, etc.)" could be in should_match since commits were made, but this is speculative without seeing the commit message.

### Difficulty mismatches

- `stop-git-commit-made`: rated "easy" but should be "medium" or "negative" for the conventional commit rule, since the commit message shown IS in conventional format (compliance).

- `stop-tests-written`: rated "medium" — if the TDD expectation is wrong (see above), the difficulty should reflect only the coverage match, which is "easy" since 95% < 100% is obvious.

- `stop-file-edits-python`: rated "medium" but should be "hard" because matching type hints and input validation from a vague summary requires significant inference.

- `stop-secret-exposed-in-logs`: rated "hard" which is appropriate for the rotate-secrets match, but the "never commit secrets" match is wrong.

## SubagentStart

### Wrong expectations (should_match contains wrong rules)

- `sub-doc-updater`: "Document all public API functions with docstrings" should NOT be in should_match because the agent is updating README.md and CLAUDE.md, not writing function docstrings. The rule is about code-level docstrings on functions, not project documentation files. A doc-updater that updates markdown files does not need the docstrings rule.

- `sub-implement-api-endpoint`: "Enable CSRF protection on all forms and state-changing endpoints" should NOT be in should_match because a REST API endpoint (`POST /api/users`) is typically consumed by API clients (not browsers with forms), and CSRF protection is for form-based browser requests. REST APIs typically use token-based auth, not cookies, so CSRF is not applicable. "Always validate user input" and "Use type hints" are correct.

- `sub-feature-branch`: "Never force-push to main or master branch" should NOT be in should_match because the task is "create a feature branch, implement, push for review." Force-pushing is not part of this workflow. The rule is a prohibition, not general git guidance. "Always create a new branch" is the correct match. The force-push rule is only relevant if force-pushing is being done or could be done by accident, which is not the case here.

- `sub-database-migration`: "Use parameterized queries to prevent SQL injection" should NOT be in should_match because writing a database migration (DDL: CREATE TABLE, ADD COLUMN) does not involve user-input-driven queries. Migrations are static schema changes, not dynamic queries. The SQL injection rule is about queries that incorporate user input. "Always close resources" is borderline — migrations usually run in a managed context.

- `sub-build-error-resolver`: "Run ruff and mypy before committing Python code" should NOT be in should_match because the agent is FIXING mypy errors, not committing code. The rule is about running tools before commits. The agent already has mypy errors to fix — telling it to "run mypy before committing" is the wrong framing. "Use type hints on all function signatures" is correct since the agent needs to fix type errors.

- `sub-write-html-templates`: "Enable CSRF protection on all forms and state-changing endpoints" is borderline. Creating HTML templates for an admin dashboard with "user data display" does not necessarily involve forms. If it is a display-only dashboard, CSRF is not relevant. "Sanitize all HTML output to prevent XSS" is correct.

- `sub-security-audit-secrets`: "Set file permissions to 0o600 for sensitive files" is borderline — an audit for hardcoded secrets is about finding secrets in code, not about file permissions. The agent is looking for secrets to move to environment variables, not checking file permission modes. Rules 1 and 13 are correct.

### Missing expectations (should_match is missing rules)

- `sub-implement-feature`: "100% test coverage on all new code" should BE in should_match because implementing a new caching layer is writing new code that needs tests.

- `sub-implement-feature`: "Use frozen dataclasses for immutable data" could BE in should_match since a caching layer likely involves data models.

- `sub-implement-api-endpoint`: "Use parameterized queries to prevent SQL injection" should BE in should_match if the API endpoint interacts with a database (likely for user creation).

- `sub-implement-api-endpoint`: "Always handle errors explicitly, never silently swallow exceptions" should BE in should_match since building an API endpoint requires error handling.

- `sub-implement-api-endpoint`: "100% test coverage on all new code" should BE in should_match for new feature code.

- `sub-code-reviewer`: "No console.log or print debug statements in production code" should BE in should_match since a code reviewer should check for debug statements.

- `sub-code-reviewer`: "Always handle errors explicitly, never silently swallow exceptions" should BE in should_match since a code reviewer should check error handling patterns.

- `sub-code-reviewer`: "Document all public API functions with docstrings" should BE in should_match since a code reviewer reviewing api.py and models.py should check for documentation.

- `sub-tdd-guide`: "Use type hints on all function signatures" should BE in should_match since writing tests for an auth module means writing typed test code and verifying the module has types.

- `sub-git-commit-prep`: "Never commit secrets (API keys, tokens, passwords) to git" should BE in should_match because the task explicitly says "check for secrets."

- `sub-eval-script`: "Use type hints on all function signatures" and "Always handle errors explicitly" should BE in should_match since implementing a script requires proper typing and error handling.

- `sub-file-processor`: "Use type hints on all function signatures" should BE in should_match for implementing new code.

- `sub-async-service`: "Use type hints on all function signatures" and "Always close file handles, database connections, network sockets, HTTP sessions, and other resources" should BE in should_match because an async HTTP client service must close HTTP sessions/connections.

### Difficulty mismatches

- `sub-database-migration`: rated "hard" — if the SQL injection expectation is removed, the remaining "close resources" match is genuinely hard to infer from a migration task, so "hard" is correct.

- `sub-doc-updater`: rated "medium" — if the docstring expectation is removed, this becomes a negative fixture (no rules clearly apply to updating markdown docs), making the difficulty "negative."

- `sub-eval-script`: rated "hard" which is correct — the word "eval" in the task name is coincidental with the eval() rule, and the model must reason about the domain (processing user-submitted expressions = eval() risk).

- `sub-build-error-resolver`: rated "medium" but if "Run ruff and mypy" is removed, only "Use type hints" remains, which is an easy match for fixing mypy errors. Should be "easy."

- `sub-write-html-templates`: rated "medium" — if CSRF is removed, only XSS remains, which is "easy" for HTML templates.

## Summary

- **14 wrong expectations found** (rules in should_match that do not actually apply)
  - PostToolUse: 4 (git-commit-clean, test-failures, ruff-errors, branch-on-main)
  - Stop: 4 (file-edits-python x2 speculative, tests-written TDD, secret-exposed-in-logs git rule, git-commit-made conventional)
  - SubagentStart: 6 (doc-updater, implement-api-endpoint CSRF, feature-branch force-push, database-migration SQL injection, build-error-resolver ruff, security-audit-secrets file perms)

- **15+ missing expectations found** (rules that should be in should_match but are not)
  - PostToolUse: 3-4 (type hints on several edit fixtures, input validation on SQL fixture, env vars on secrets file)
  - Stop: 2 (ruff/mypy on git-commit-made, env vars on sensitive-file-created)
  - SubagentStart: 10+ (test coverage, error handling, type hints, docstrings on multiple implementation agents)

- **7 difficulty mismatches**
  - PostToolUse: 3 (git-commit-clean should be negative, ruff-errors should be negative, test-failures should be hard)
  - Stop: 2 (git-commit-made should be medium/negative, file-edits-python should be hard)
  - SubagentStart: 2 (build-error-resolver should be easy, write-html-templates should be easy)
