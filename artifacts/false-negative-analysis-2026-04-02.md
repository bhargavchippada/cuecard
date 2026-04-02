# False Negative Analysis — Session 18

> 9B LLM, v5 corpora, 20% sample of basic fixtures (69 total, 43 positive)
> 13 false negatives (30% of positives have recall=0)
> 11 high-noise fixtures (noise > 50%)

## Pattern 1: Preparatory Actions Ignored (5 cases)

The LLM treats preparatory/search actions as "no rules apply" but they have indirect consequences:

| Fixture | Query | Expected Rule | Why |
|---------|-------|---------------|-----|
| git-add-all | `git add -A` | secrets, gitignore | Staging precedes commit — secrets check applies |
| grep-api-key | `Grep: API_KEY\|SECRET_KEY` | secrets, env vars | Searching for secrets = should review secrets policy |
| grep-sql-fstring | `Grep: f"SELECT\|f"INSERT` | parameterized queries | Searching for SQL injection patterns |
| git-tag | `git tag v1.0.0` | quality checks | Tagging a release should have quality checks |
| hard-read-package-lock | `Read: package-lock.json` | dependency review | Reading deps = should review for vulnerabilities |

**Root cause:** Reranker principle #2 ("does this modify state?") causes the model to dismiss read/search operations. But these actions PREPARE for state changes and should trigger anticipatory rules.

**Fix:** Add a reasoning principle about preparatory actions: "Staging, searching, and reviewing are precursors to action — rules about the downstream action still apply."

## Pattern 2: Test Commands Not Linked to Coverage (3 cases)

| Fixture | Query | Expected Rule |
|---------|-------|---------------|
| go-run-tests | `go test ./... -v` | 100% coverage |
| mined-npx-vitest-run | `npx vitest run tests/...` | 100% coverage |
| multi-tool-cargo-test-push | `cargo test && git push` | 100% coverage |

**Root cause:** Running tests IS the rule's domain — "100% test coverage" applies when you run tests. The model sees "running tests" as already doing the right thing, not as a moment to enforce coverage standards.

**Fix:** Add reasoning principle or few-shot example: "Running tests is when coverage rules apply — this is the enforcement point."

## Pattern 3: Code Editing Not Inspected for Style (2 cases)

| Fixture | Query | Expected Rule |
|---------|-------|---------------|
| mined-gravity-edit-test-ts | `Edit: tests/...test.ts` | coverage, type hints |
| ts-write-react-component | `Write: Button.tsx` | type hints, docstrings |

**Root cause:** Reranker doesn't inspect the embedded code content deeply enough. The file extension (.tsx) and the code content should trigger TypeScript/React rules.

**Fix:** Add reasoning principle: "For Edit/Write, consider both the action AND the content being written. File extension and code patterns indicate which style rules apply."

## Pattern 4: Indirect/Hard Matches (3 cases)

| Fixture | Query | Expected Rule | Indirect Connection |
|---------|-------|---------------|---------------------|
| mined-node-bcrypt-compare | `node -e bcrypt.compare('admin')` | never commit secrets | Password in CLI args |
| edit-python-file | `Edit: src/auth.py` | (none expected) | N/A — but model fires 3 wrong rules |
| edge-git-force-push-feature | `git push --force feature/my-branch` | (none expected) | Correctly empty |

## High Noise Pattern: Code Edits Fire Everything

When Edit/Write contains embedded code, the model matches 3-5 rules where 1-2 are relevant:
- subprocess + exec code → fires SQL injection, input validation, env vars (only eval/exec rule needed)
- File upload code → fires file handles, async, magic numbers (only validation + permissions needed)
- Dataclass code → fires docstrings, type hints, magic numbers (only frozen dataclass needed)

**Root cause:** Principle #4 ("scan for ALL violations") is too aggressive. The model sees code and fires every remotely related rule.

**Fix:** Refine principle #4: "Scan for violations IN the code, but only rules that the code ACTUALLY violates, not rules about topics the code touches."

## Summary: Reranker Prompt Changes Needed

1. **New principle: Preparatory actions** — staging/searching/reviewing are precursors; downstream rules apply
2. **New principle: Enforcement points** — running tests = when coverage rules apply
3. **New principle: Code content inspection** — for Edit/Write, inspect what's being written
4. **Refine principle #4** — "scan for violations" should mean actual violations, not topic adjacency
5. **New few-shot example: test running** — `go test` should match coverage rules
6. **New few-shot example: git add** — `git add -A` should match secrets rules
