# Fixture False-Negative Fixes (2026-04-07)

## Problem

The corpus expanded from 32 to 109 rules. Many fixtures written as negatives (should_match=[]) when only 32 rules existed now have genuinely matching rules in the expanded corpus. This caused artificially low NegSil scores because the model correctly retrieved rules for these "negative" fixtures.

## Changes Summary

| File | Negatives Converted | Over-Broad Positives Trimmed |
|------|:-------------------:|:----------------------------:|
| basic.json | 0 | 1 (reclassified) |
| post_tool_use.json | 0 | 0 |
| stop.json | 3 | 0 |
| subagent_start.json | 5 | 0 |
| workflow.json | 5 | 0 |
| **Total** | **13** | **1** |

## Specific Changes

### stop.json (3 conversions)

1. **stop-neg-compared-libraries**: "Compared fastembed, sentence-transformers, and openai embeddings for latency and quality tradeoffs"
   - Was: negative (no rules)
   - Now: hard positive, should_match: "For new libraries or APIs, research alternatives..."
   - Rationale: Comparing libraries IS the research-alternatives workflow

2. **stop-neg-created-plan**: "Created an implementation plan for the new caching layer, outlining phases, risks, and dependencies"
   - Was: negative (no rules)
   - Now: hard positive, should_match: "Classify every task as SIMPLE, MEDIUM, or COMPLEX..."
   - Rationale: Creating a plan implies task classification applies

3. **stop-file-edits-python**: "Edited src/api.py and src/models.py to add new user creation endpoint"
   - Was: negative (no rules)
   - Now: hard positive, should_match: "Require 100% test coverage...", "Use type hints..."
   - Rationale: Adding a new endpoint = new code that needs tests and type hints

### subagent_start.json (5 conversions)

4. **sub-neg-compare-frameworks**: "Compare Django, FastAPI, and Flask for the new API service and summarize tradeoffs"
   - Was: negative
   - Now: hard positive, should_match: "For new libraries or APIs, research alternatives..."
   - Rationale: Comparing frameworks IS the research-alternatives workflow

5. **sub-neg-analyze-benchmark**: "Analyze the benchmark results from eval/results/ and identify regression patterns"
   - Was: negative
   - Now: hard positive, should_match: "Store all predictions, reasoning, and grounding..."
   - Rationale: Analyzing benchmarks should trigger the store-predictions rule

6. **sub-neg-search-agent**: "Search GitHub for existing BM25 implementations in Python to evaluate"
   - Was: negative
   - Now: hard positive, should_match: "Search GitHub for existing implementations..."
   - Rationale: The query directly matches the rule's action

7. **sub-neg-plan-migration**: "Plan the migration from txt rule format to TOML format including phases, risks, and rollback strategy"
   - Was: negative
   - Now: hard positive, should_match: "Classify every task as SIMPLE, MEDIUM, or COMPLEX..."
   - Rationale: Planning a migration requires task classification

8. **sub-neg-docs-reader**: "Read the FastAPI documentation to understand dependency injection patterns"
   - Was: negative
   - Now: hard positive, should_match: "Use Context7 for library documentation lookups..."
   - Rationale: Reading library docs is exactly what Context7 is for

### workflow.json (5 conversions)

9. **wf-neg-debug-print**: "add a print statement to debug this"
   - Was: negative
   - Now: hard positive, should_match: "No console.log or print debug statements in production code..."
   - Rationale: User asking to add debug prints should trigger the no-debug-prints rule

10. **v3-user-research-hooks**: "Currently we have two hooks pretooluse and userpromptsubmit, claude code has more hooks..."
    - Was: negative
    - Now: hard positive, should_match: "Use Context7...", "Search GitHub..."
    - Rationale: Researching API capabilities = docs lookup + GitHub search

11. **v3-user-use-toml-and-migrate**: "This is not right design, lets use toml and migrate"
    - Was: negative
    - Now: hard positive, should_match: "Classify every task as SIMPLE, MEDIUM, or COMPLEX..."
    - Rationale: Design change + migration requires complexity assessment

12. **v3-user-benchmarking-important**: "Option B + C sounds good but we need to evaluate quality metrics"
    - Was: negative
    - Now: hard positive, should_match: "Control your evaluation methodology...", "Establish baseline metrics..."
    - Rationale: Evaluating quality metrics triggers eval methodology rules

13. **v3-user-migrate-corpus-eval-data**: "We need to migrate our corpus / golden eval data to have event"
    - Was: negative
    - Now: hard positive, should_match: "When schema or cardinality changes, audit every consumer..."
    - Rationale: Migrating eval data schema requires consumer audit

## Negatives Reviewed and Kept as Negative

Reviewed all 331 original negatives. The remaining 318 are genuinely negative:

- **Pure read operations** (Read, Grep, Glob, cat, head, tail): ~80 fixtures. Reading doesn't trigger rules.
- **Informational commands** (whoami, date, nvidia-smi, df, uptime, etc.): ~30 fixtures. System info queries.
- **Clean git operations** (git status, git log, git diff, git branch, git pull): ~25 fixtures. Read-only git.
- **Git history operations** (git stash, git rebase, git cherry-pick, git tag, git blame, git reset): ~10 fixtures. No specific rules for these.
- **Compliance scenarios** (clean test output, clean lint output, proper frozen dataclasses, env vars used correctly): ~40 fixtures. Code already follows the rules.
- **Non-coding tasks** (greetings, math, translations, explanations, summaries): ~30 fixtures in workflow.
- **Monitoring/observing** (read logs, check CI, watch processes, profile latency): ~25 fixtures in stop/subagent.
- **Mined real sessions** (specific project edits, grep patterns, file operations): ~60 fixtures that are genuinely context-specific with no matching global rules.

## Over-Broad Positives Reviewed

Reviewed the mentioned over-broad fixtures:

- **git-merge-feature**: Expects "Run quality checks before every commit" for `git merge --no-ff`. A merge creates a merge commit, so this is arguably applicable. **Kept as-is**.
- **edge-git-force-push-feature**: Was difficulty=hard with empty should_match (invalid). Force-pushing a feature branch (not main/master) has no matching rule. **Reclassified to negative**.

## Post-Fix Statistics

| File | Total | Negatives | Positives |
|------|:-----:|:---------:|:---------:|
| basic.json | 442 | 197 | 245 |
| post_tool_use.json | 115 | 48 | 67 |
| stop.json | 77 | 27 | 50 |
| subagent_start.json | 73 | 17 | 56 |
| workflow.json | 124 | 30 | 94 |
| **combined.json** | **831** | **319** | **512** |
