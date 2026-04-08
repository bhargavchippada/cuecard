# Fixture Verification Round 1 — PostToolUse + Stop

## CRITICAL SYSTEMIC ISSUE

All 143 fixtures in both files set `"corpus": "rules_global.txt"` but use short-form rule texts from the OLD `rules_basic.txt` (32 rules) in their `should_match` / `should_not_match` arrays. The eval framework uses **exact string equality** (`r in relevant`), so none of these short-form texts will match the full-form texts in `rules_global.txt` (which include `— explanation` suffixes).

### Two classes of problems:

**Class A — Text mismatch:** Rules that exist in both corpora but with different wording. Every should_match entry must be updated to the EXACT full text from `rules_global.txt`.

**Class B — Rule absent from global:** 17 rules from `rules_basic.txt` have NO equivalent in `rules_global.txt`. Fixtures referencing these rules will always fail. These rules must either be added to `rules_global.txt` or the should_match entries removed.

### Rules ABSENT from rules_global.txt (Class B)

These rules exist in `rules_basic.txt` but have NO equivalent in `rules_global.txt`:

1. "Run quality checks before every commit"
2. "Use frozen dataclasses for immutable data"
3. "No console.log or print debug statements in production code"
4. "Use parameterized queries to prevent SQL injection"
5. "Sanitize all HTML output to prevent XSS"
6. "Enable CSRF protection on all forms and state-changing endpoints"
7. "Write tests before implementation (TDD red-green-refactor)"
8. "Use type hints on all function signatures"
9. "Extract magic numbers (hardcoded timeouts, limits, sizes, counts) into named constants"
10. "Prefer immutable data structures over mutable ones"
11. "Use asyncio for I/O-bound operations, not threads"
12. "Rotate any secrets that may have been exposed"
13. "Review all dependencies for known vulnerabilities before adding"
14. "Always create a new branch for feature work"
15. "Document all public API functions with docstrings"

### Rules requiring TEXT UPDATE (Class A) — old → new mapping

| Old (basic) short form | New (global) full form |
|---|---|
| "Never commit secrets (API keys, tokens, passwords) to git" | "Never commit secrets (API keys, tokens, passwords, connection strings) to git — once pushed, secrets are in the history forever and require rotation across all environments" |
| "Use uv for all Python package operations, never pip" | "Use uv for all Python package operations, never pip — pip doesn't respect lock files, creates inconsistent environments, and lacks dependency resolution" |
| "Send Enter after every tmux send-keys command" | "Always append Enter after tmux send-keys text — without it the text is pasted into the prompt but never submitted, so the target session never receives or executes the message" |
| "Always validate user input at system boundaries" | "Always validate user input at system boundaries — unvalidated input propagates corrupted data through the entire system, causing failures far from the source" |
| "100% test coverage on all new code" | "Require 100% test coverage on all new code before committing — uncovered code is untested code, and untested code breaks silently in production" |
| "Set file permissions to 0o600 for sensitive files" | "Set file permissions to 0o600 for sensitive files (keys, configs, tokens) — world-readable secrets are the most common cloud breach vector" |
| "Use environment variables for configuration, not hardcoded values" | "Use environment variables for configuration, not hardcoded values — hardcoded config prevents deployment to different environments and leaks secrets into source control" |
| "Run ruff and mypy before committing Python code" | "Run ruff and mypy before committing Python code — type errors and lint violations caught at commit time cost 10 minutes to fix, but in production they cost hours to debug" |
| "Never use eval() or exec() in production code" | "Never use eval(), exec(), or shell=True in production code — these execute arbitrary input as code, enabling remote code execution attacks" |
| "Always handle errors explicitly, never silently swallow exceptions" | "Always handle errors explicitly at every level, never silently swallow exceptions — empty catch blocks hide bugs that surface later as mysterious failures with no stack trace" |
| "Always close file handles, database connections, network sockets, HTTP sessions, and other resources" | "Always close file handles, database connections, and HTTP sessions after use — unclosed resources leak memory and exhaust connection pools, eventually crashing the service" |
| "Never force-push to main or master branch" | "Never force-push to main or master branch — force-push overwrites other developers' commits with no recovery path" |
| "Use conventional commit format (feat:, fix:, refactor:, etc.)" | "Use conventional commit format (feat:, fix:, refactor:, docs:, test:) — without structured messages, changelogs are impossible to generate and git blame is useless" |
| "Add .gitignore entries for generated files and build artifacts" | "Add .gitignore entries for generated files and build artifacts — committed build artifacts bloat the repo and cause merge conflicts on every build" |
| "Keep functions under 50 lines, files under 400 lines" | "Keep functions under 50 lines and source files under 800 lines (400 preferred) — large functions hide bugs in nested logic, and large files make it impossible to understand scope of changes" |

---

## PostToolUse fixes needed

### Text updates (Class A) — every should_match/should_not_match entry using old short-form text

- `post-bash-git-commit-env`: UPDATE — "Never commit secrets (API keys, tokens, passwords) to git" → full global form
- `post-bash-git-commit-env`: UPDATE should_not_match — "Send Enter after every tmux send-keys command" → full global form
- `post-edit-eval-usage`: UPDATE — "Never use eval() or exec() in production code" → full global form; UPDATE should_not_match
- `post-bash-pip-install`: UPDATE — "Use uv for all Python package operations, never pip" → full global form; UPDATE should_not_match
- `post-write-no-type-hints`: REMOVE — "Use type hints on all function signatures" does not exist in global corpus
- `post-bash-test-low-coverage`: UPDATE — "100% test coverage on all new code" → full global form; UPDATE should_not_match
- `post-edit-console-log`: REMOVE — "No console.log or print debug statements in production code" not in global; REMOVE — "Use type hints on all function signatures" not in global
- `post-edit-sql-injection`: REMOVE — "Use parameterized queries to prevent SQL injection" not in global; UPDATE — "Always validate user input at system boundaries" → full global form
- `post-write-xss-vector`: REMOVE — "Sanitize all HTML output to prevent XSS" not in global
- `post-edit-hardcoded-config`: UPDATE — "Use environment variables for configuration, not hardcoded values" → full global form
- `post-edit-mutable-dataclass`: REMOVE — "Use frozen dataclasses for immutable data" not in global. ADD — "Never mutate function arguments or shared state — mutations cause action-at-a-distance bugs..." (closest equivalent)
- `post-edit-swallowed-exception`: UPDATE — "Always handle errors explicitly, never silently swallow exceptions" → full global form
- `post-bash-force-push-master`: UPDATE — "Never force-push to main or master branch" → full global form
- `post-edit-magic-numbers`: REMOVE — "Extract magic numbers..." not in global
- `post-write-no-docstring`: REMOVE — "Document all public API functions with docstrings" not in global. Fixture becomes negative or needs different rule.
- `post-edit-unclosed-resource`: UPDATE — "Always close file handles, database connections, network sockets, HTTP sessions, and other resources" → full global form (note: wording differs slightly)
- `post-edit-no-input-validation`: UPDATE — "Always validate user input at system boundaries" → full global form. REMOVE — "Use type hints on all function signatures" not in global.
- `post-edit-csrf-missing`: REMOVE — "Enable CSRF protection on all forms and state-changing endpoints" not in global. Fixture becomes negative or needs re-mapping.
- `post-write-sensitive-file-perms`: UPDATE all three should_match entries to full global forms
- `post-bash-pip-audit-vuln`: REMOVE — "Review all dependencies for known vulnerabilities before adding" not in global. UPDATE — "Use uv..." → full global form.
- `post-edit-thread-for-io`: REMOVE — "Use asyncio for I/O-bound operations, not threads" not in global
- `post-bash-git-commit-no-conventional`: UPDATE — "Use conventional commit format..." → full global form
- `post-edit-long-function`: UPDATE — "Keep functions under 50 lines, files under 400 lines" → full global form
- `post-write-gitignore-missing`: UPDATE — "Add .gitignore entries for generated files and build artifacts" → full global form
- `post-edit-secret-rotation`: REMOVE — "Rotate any secrets that may have been exposed" not in global. REMOVE — "No console.log or print debug statements..." not in global. UPDATE — "Never commit secrets..." → full global form.
- `post-bash-git-commit-on-main-no-tests`: UPDATE — "Use conventional commit format..." → full global form. REMOVE — "Always create a new branch for feature work" not in global.
- `post-edit-sql-concat-with-unclosed-conn`: REMOVE — "Use parameterized queries..." not in global. UPDATE — "Always close file handles..." → full global form. REMOVE — "Use type hints..." not in global.
- `post-bash-tmux-no-enter`: UPDATE — "Send Enter after every tmux send-keys command" → full global form
- `post-bash-pip-install-multiple`: UPDATE — "Use uv..." → full global form. REMOVE — "Review all dependencies..." not in global.
- `post-read-env-file-with-real-secrets`: UPDATE — "Never commit secrets..." → full global form. REMOVE — "Rotate any secrets..." not in global.
- `post-edit-almost-correct-missing-docstring`: REMOVE — "Document all public API functions with docstrings" not in global. Fixture becomes negative.
- `post-write-exec-disguised`: UPDATE — "Never use eval()..." → full global form
- `post-bash-test-zero-coverage`: UPDATE — "100% test coverage..." → full global form. REMOVE — "Write tests before implementation..." not in global.
- `post-edit-hardcoded-secret-in-env-var-name`: UPDATE both should_match entries → full global forms. REMOVE should_not_match entries that reference missing rules.
- `post-bash-force-push-main-verbose`: UPDATE — "Never force-push..." → full global form
- `post-write-thread-pool-io`: REMOVE — "Use asyncio for I/O-bound operations, not threads" not in global
- `post-bash-git-commit-env-credentials`: UPDATE — "Never commit secrets..." → full global form. UPDATE should_not_match entries.
- `post-edit-bare-except-with-print`: UPDATE — "Always handle errors explicitly..." → full global form. REMOVE — "No console.log or print debug statements..." not in global. REMOVE — "Use type hints..." not in global.
- `post-write-mutable-plus-no-hints-plus-magic`: REMOVE — "Use frozen dataclasses..." not in global. REMOVE — "Use type hints..." not in global. REMOVE — "Extract magic numbers..." not in global. ADD — "Never mutate function arguments or shared state..." (partial match for the mutable dataclass).
- `post-bash-mypy-errors`: UPDATE — "Run ruff and mypy..." → full global form. REMOVE — "Use type hints..." not in global.
- `post-write-html-no-sanitize-no-csrf`: REMOVE — "Sanitize all HTML output..." not in global. REMOVE — "Enable CSRF protection..." not in global.
- `post-glob-sensitive-paths`: UPDATE both should_match entries → full global forms.
- `post-edit-eval-in-template`: UPDATE — "Never use eval()..." → full global form. UPDATE should_not_match entries.
- `post-read-secrets-in-source`: UPDATE both should_match entries → full global forms.

### All negative fixtures — should_not_match text updates needed

Every negative fixture that references old short-form rules in should_not_match also needs text updates. These are less critical (should_not_match controls anti-precision, not recall), but still need fixing for correctness. Affected:

- `post-bash-git-commit-clean`, `post-bash-test-failures`, `post-bash-ruff-errors`, `post-bash-branch-on-main`, `post-neg-read-config`, `post-neg-glob-listing`, `post-neg-grep-search`, `post-neg-read-readme`, `post-neg-bash-ls`, `post-neg-bash-git-status`, `post-neg-bash-uv-run-pytest-pass`, `post-neg-read-test-file`, `post-neg-bash-python-version`, `post-neg-bash-nvidia-smi`, `post-neg-bash-df`, `post-neg-bash-whoami`, `post-neg-bash-uv-sync`, `post-neg-bash-git-log`, `post-neg-bash-git-diff-clean`, `post-neg-bash-docker-build`, `post-neg-bash-git-pull`, `post-neg-bash-ruff-clean`, `post-neg-bash-mypy-clean`, `post-neg-read-clean-code`, `post-neg-edit-clean-refactor`, `post-neg-write-normal-config`, `post-neg-bash-pip-list`, `post-neg-bash-cat-env-example`, `post-neg-bash-git-checkout-branch`, `post-neg-grep-import-search`, `post-neg-bash-wc-lines`, `post-neg-bash-git-push-feature`, `post-neg-write-gitignore`, `post-neg-bash-curl-health`, `post-neg-edit-add-test`

### Missing NEW rule additions (for fixtures that reference global corpus)

These new rules from the 65-rule corpus should be ADDED to existing fixtures:

- `post-edit-hardcoded-config`: ADD — "Never mutate function arguments or shared state..." if the code mutates shared config state (borderline — skip)
- `post-edit-no-input-validation`: ADD — "Always validate user input at system boundaries..." already present (just needs text update)
- `post-bash-git-commit-env`: ADD — "Run ruff and mypy before committing Python code..." (commit was made, linting status unknown). Actually NO — PostToolUse verifies output, and the output doesn't show linting was skipped. Skip.
- `post-edit-swallowed-exception`: ADD — "Never use silent fallbacks — if a pipeline stage falls back to an alternative..." — NO, this is about error handling, not silent fallbacks. Skip.

No new rules from the expanded corpus need to be added to PostToolUse fixtures beyond what's already listed, because:
- Most new rules are about PROCESS (benchmark before building, convergence reviews, PRD reviews) which don't apply to PostToolUse output verification
- LLM-specific rules (reasoning-first JSON, thinking mode) are domain-specific and only apply if the output shows LLM usage
- Testing infrastructure rules (mock externals, 5-second suite) are process rules, not output verification

### Difficulty changes

- `post-write-no-type-hints`: If "Use type hints" is removed (not in global), becomes NEGATIVE → DIFFICULTY change to negative
- `post-edit-console-log`: If both should_match rules are removed, becomes NEGATIVE → DIFFICULTY change to negative
- `post-write-xss-vector`: If should_match is removed, becomes NEGATIVE → DIFFICULTY change to negative
- `post-edit-magic-numbers`: If should_match is removed, becomes NEGATIVE → DIFFICULTY change to negative (unless remapped)
- `post-write-no-docstring`: If should_match is removed, becomes NEGATIVE → DIFFICULTY change to negative
- `post-edit-csrf-missing`: If should_match is removed, becomes NEGATIVE → DIFFICULTY change to negative
- `post-edit-thread-for-io`: If should_match is removed, becomes NEGATIVE → DIFFICULTY change to negative
- `post-write-thread-pool-io`: If should_match is removed, becomes NEGATIVE → DIFFICULTY change to negative
- `post-write-html-no-sanitize-no-csrf`: If both removed, becomes NEGATIVE → DIFFICULTY change to negative

---

## Stop fixes needed

### Text updates (Class A) — same systemic issue

Every Stop fixture with should_match/should_not_match entries referencing old short-form rules needs text updates. Affected fixtures:

- `stop-git-commit-made`: UPDATE — "Run quality checks before every commit" NOT IN GLOBAL (REMOVE). UPDATE — "Run ruff and mypy before committing Python code" → full global form.
- `stop-git-commit-secrets-risk`: UPDATE — "Never commit secrets..." → full global form. REMOVE — "Run quality checks before every commit" not in global.
- `stop-file-edits-python`: UPDATE should_not_match — "Send Enter after every tmux send-keys command" → full global form.
- `stop-wrote-new-module`: UPDATE — "100% test coverage on all new code" → full global form. UPDATE should_not_match.
- `stop-tests-written`: UPDATE — same as above.
- `stop-pip-install-used`: UPDATE — "Use uv..." → full global form. REMOVE — "Review all dependencies..." not in global.
- `stop-database-query-added`: REMOVE — "Use parameterized queries..." not in global.
- `stop-html-template-created`: REMOVE — "Sanitize all HTML output..." not in global.
- `stop-force-pushed-main`: UPDATE — "Never force-push..." → full global form.
- `stop-refactored-dataclasses`: REMOVE — "Use frozen dataclasses for immutable data" not in global. REMOVE — "Prefer immutable data structures over mutable ones" not in global. ADD — "Never mutate function arguments or shared state..." as partial replacement.
- `stop-form-endpoint-added`: REMOVE — "Enable CSRF protection..." not in global. UPDATE — "Always validate user input..." → full global form.
- `stop-env-vars-hardcoded`: UPDATE — "Use environment variables..." → full global form. REMOVE — "Extract magic numbers..." not in global.
- `stop-error-handling-added`: UPDATE — "Always handle errors explicitly..." → full global form.
- `stop-tmux-delegation`: UPDATE — "Send Enter after every tmux send-keys command" → full global form.
- `stop-sensitive-file-created`: UPDATE all three should_match entries → full global forms.
- `stop-async-network-calls`: REMOVE — "Use asyncio..." not in global.
- `stop-linting-skipped`: UPDATE — "Run ruff and mypy..." → full global form. REMOVE — "Run quality checks..." not in global.
- `stop-branch-not-created`: REMOVE — "Always create a new branch for feature work" not in global.
- `stop-dependency-added-no-audit`: REMOVE — "Review all dependencies..." not in global.
- `stop-resource-leak`: UPDATE — "Always close file handles..." → full global form.
- `stop-gitignore-missing`: UPDATE — "Add .gitignore entries..." → full global form.
- `stop-secret-exposed-in-logs`: REMOVE — "Rotate any secrets..." not in global.
- `stop-multi-violation-commit`: UPDATE — "Use uv..." → full global form. REMOVE — "Always create a new branch..." not in global. UPDATE — "100% test coverage..." → full global form. UPDATE — "Run ruff and mypy..." → full global form. REMOVE — "Run quality checks..." not in global.
- `stop-multi-security-violations`: UPDATE — "Always validate user input..." → full global form. REMOVE — "Use parameterized queries..." not in global. REMOVE — "Sanitize all HTML..." not in global. REMOVE — "Enable CSRF protection..." not in global.
- `stop-eval-usage-in-code`: UPDATE — "Never use eval()..." → full global form. UPDATE — "Always validate user input..." → full global form.
- `stop-large-function-written`: UPDATE — "Keep functions under 50 lines, files under 400 lines" → full global form.
- `stop-mutable-dict-config`: REMOVE — "Use frozen dataclasses..." not in global. REMOVE — "Prefer immutable data structures over mutable ones" not in global. ADD — "Never mutate function arguments or shared state..." as replacement.
- `stop-debug-prints-left`: REMOVE — "No console.log or print debug statements..." not in global.
- `stop-hard-new-module-no-tests`: UPDATE — "100% test coverage..." → full global form. REMOVE — "Use type hints..." not in global. REMOVE — "Document all public API functions..." not in global.
- `stop-hard-db-pool-no-cleanup`: UPDATE — "Always close file handles..." → full global form.
- `stop-hard-committed-nonfmt`: UPDATE — "Run ruff and mypy..." → full global form. REMOVE — "Run quality checks..." not in global.
- `stop-hard-network-with-threads`: REMOVE — "Use asyncio..." not in global. UPDATE — "Always close file handles..." → full global form.
- `stop-hard-partial-gitignore`: UPDATE — "Add .gitignore entries..." → full global form. REMOVE should_not_match — "Run quality checks..." not in global.
- `stop-hard-env-in-code`: UPDATE — "Use environment variables..." → full global form.
- `stop-hard-test-after-code`: REMOVE — "Write tests before implementation..." not in global.
- `stop-hard-ambiguous-function-untyped`: REMOVE — "Use type hints..." not in global.
- `stop-commit-no-conventional`: UPDATE — "Use conventional commit format..." → full global form. REMOVE — "Run quality checks..." not in global (was in should_match).
- `stop-sensitive-perms-wrong`: UPDATE — "Set file permissions..." → full global form.
- `stop-hard-swallowed-error`: UPDATE — "Always handle errors explicitly..." → full global form.

### All negative Stop fixtures — should_not_match text updates

All 20 negative Stop fixtures reference old short-form rules in should_not_match that need text updates or removal (same Class A/B issues). These include: `stop-neg-read-only`, `stop-neg-grep-search`, `stop-neg-listed-files`, `stop-neg-git-status-check`, `stop-neg-read-docs`, `stop-neg-answered-question`, `stop-neg-browsed-tests`, `stop-neg-git-log-review`, `stop-neg-explained-architecture`, `stop-neg-created-plan`, `stop-neg-ran-tests-passed`, `stop-neg-checked-server-logs`, `stop-neg-compared-libraries`, `stop-neg-reviewed-pr`, `stop-neg-ran-benchmark`, `stop-neg-discussed-design`, `stop-neg-read-error-output`, `stop-neg-checked-ci-status`, `stop-neg-inspected-docker`, `stop-neg-checked-git-blame`, `stop-neg-listed-processes`, `stop-neg-summarized-session`, `stop-neg-curl-health-check`, `stop-neg-read-test-output`.

### Missing NEW rule additions for Stop fixtures

Stop audits the turn, so process rules ARE relevant. New rules to consider:

- `stop-git-commit-made`: ADD — "Use conventional commit format..." (commit was made, should audit format). Actually the commit message shown IS conventional format ("fix:"), so this is compliance — skip.
- `stop-git-commit-secrets-risk`: ADD — "Add .gitignore entries for generated files and build artifacts..." — .env should be in .gitignore. Borderline.
- `stop-wrote-new-module`: ADD — "Read 2-3 similar files in the codebase before writing new code..." — relevant when creating new modules. ADD — "Search GitHub for existing implementations..." — relevant when writing from scratch. Both are hard-tier additions.
- `stop-file-edits-python`: Currently negative. ADD — "Read 2-3 similar files in the codebase before writing new code..." — arguable since it's editing existing files. Keep negative — editing existing files doesn't trigger "before writing NEW code."
- `stop-tests-written`: No new rules apply beyond the existing coverage check.
- `stop-pip-install-used`: ADD — "Spike-test risky integrations before building on them..." — borderline, this is about installing known packages not risky integrations. Skip.
- `stop-multi-violation-commit`: ADD — "Use conventional commit format..." — the commit on main doesn't show conventional format explicitly. Borderline — the commit message isn't shown. Skip.
- `stop-linting-skipped`: ADD — "Use conventional commit format..." — commit was made. But focus should stay on the linting violation. Skip.
- `stop-hard-test-after-code`: With "Write tests before implementation" removed, this becomes NEGATIVE unless another rule applies. "Require 100% test coverage..." could apply since tests were written AFTER (process violation). But 100% coverage is about coverage quantity, not TDD order. This fixture likely becomes negative or needs "Validate every phase against real data before moving to the next" — but that's a stretch.

### Difficulty changes for Stop

- `stop-database-query-added`: If "Use parameterized queries" removed → NEGATIVE
- `stop-html-template-created`: If "Sanitize all HTML" removed → NEGATIVE
- `stop-refactored-dataclasses`: With both immutability rules removed, ADD "Never mutate function arguments or shared state..." keeps it positive but DIFFICULTY should be hard (semantic gap between "refactored to dataclasses" and "never mutate")
- `stop-form-endpoint-added`: With CSRF removed, only "validate user input" remains → DIFFICULTY downgrade from hard to medium
- `stop-async-network-calls`: If "Use asyncio" removed → NEGATIVE
- `stop-branch-not-created`: If "create a new branch" removed → NEGATIVE
- `stop-dependency-added-no-audit`: If "Review all dependencies" removed → NEGATIVE
- `stop-secret-exposed-in-logs`: If "Rotate any secrets" removed → NEGATIVE
- `stop-debug-prints-left`: If "No console.log" removed → NEGATIVE
- `stop-hard-test-after-code`: If "Write tests before implementation" removed → NEGATIVE
- `stop-hard-ambiguous-function-untyped`: If "Use type hints" removed → NEGATIVE
- `stop-hard-network-with-threads`: With "asyncio" removed, only "close resources" remains → keep positive but lower to medium
- `stop-multi-security-violations`: With 3 of 4 rules removed, only "validate user input" remains → DIFFICULTY downgrade from medium to easy (single obvious rule)
- `stop-env-vars-hardcoded`: With "Extract magic numbers" removed, only "Use environment variables" remains → DIFFICULTY downgrade from medium to easy

---

## Summary

### Systemic issue
- **ALL 143 fixtures** need should_match/should_not_match text updates from short-form (basic) to full-form (global) rule texts — this is a bulk find-and-replace operation

### Per-fixture changes
- **~85 text updates needed** (Class A: rule exists in global but with different text)
- **~55 removals needed** (Class B: rule does not exist in global corpus at all)
- **~5 additions needed** (new rules from expanded corpus that apply)
- **~15 difficulty changes needed** (fixtures become negative when all their should_match rules are absent from global)

### Recommendation

Before doing per-fixture fixes, the corpus problem must be resolved first. Two options:

**Option 1 (recommended):** Add the 15 missing basic rules to `rules_global.txt` WITH explanation suffixes (matching the existing pattern). This preserves all existing fixtures and only requires the text-update pass (Class A).

**Option 2:** Accept the reduced corpus and remove/reclassify all fixtures that reference absent rules. This loses ~15 positive test cases that become negatives.

Either way, ALL should_match/should_not_match entries must be updated to use the EXACT full rule text from `rules_global.txt`.
