# Fixture Verification Round 2 Summary

## Context

Round 1 verification reports identified issues across all fixture files after the corpus expansion from 32 to 109 rules. Three reports covered:
- `fixture-verify-r1-post-stop.md` (PostToolUse + Stop)
- `fixture-verify-r1-sub-basic.md` (SubagentStart + PreToolUse basic)
- `fixture-verify-r1-workflow.md` (UserPromptSubmit workflow)

## Key Finding: Reports Were Based on 65-Rule Corpus

The R1 reports identified 15 "missing" rules (Class B) that existed in the old basic corpus but not in the global corpus. However, the corpus was subsequently expanded from 65 to 109 rules, which added all 15 missing rules WITH full rationale suffixes. This eliminated the need for:
- Converting positive fixtures to negative (the reports recommended ~15 difficulty changes to "negative")
- Removing should_match entries for absent rules (~55 removals recommended)

## Changes Applied

### PASS 1: Text Remapping (Global)
- **0 fixtures needed remapping** -- all should_match/should_not_match entries already used full-form rule text from the 109-rule corpus (this was done in a prior session)

### PASS 2: Fixture-Level Changes

#### basic.json (18 changes)
- **Additions (12 fixtures):**
  - `subprocess-shell-true`: Added "Never use eval(), exec(), or shell=True" (primary match for shell=True)
  - `git-commit-secrets`: Added "Run ruff and mypy" (commit should trigger linting)
  - `git-commit-quality`: Added "Never commit secrets" (any commit needs secret check)
  - `write-env-file`: Added "Set file permissions to 0o600" (sensitive file)
  - `write-yaml-secrets`: Added "Set file permissions" + "Use environment variables" (secrets in YAML)
  - `password-in-code`: Added "Use environment variables" (hardcoded password)
  - `write-dockerfile`: Added "Use uv" (Dockerfile uses pip install)
  - `embedded-code-flask-app`: Added "Use environment variables" (hardcoded DB_PATH)
  - `long-write-config-module`: Added "Set file permissions" (config with secrets)
  - `embedded-code-bash-heredoc`: Added "Always handle errors explicitly" (no error handling in init_db.py)
  - `hard-github-actions-workflow`: Added "Run ruff and mypy" (CI should include linting)
  - `ambiguous-mutable-config`: Added "Never mutate function arguments" (mutates shared state)
- **Removals (2 fixtures):**
  - `ambiguous-async-db-connections`: Removed "Use parameterized queries" (code uses $1 -- compliance, not violation)
  - `mined-delulu-tmux-ralph-fix`: Removed "Use type hints" (tmux message, not code writing)
- **Negative conversions (4 fixtures):**
  - `mined-uv-run-classify`: Converted to negative (running via uv is compliance)
  - `hard-read-package-lock`: Converted to negative (reading a file is read-only)
  - `mined-edit-docker-compose-model`: Converted to negative (config change, not dep addition)
  - `write-markdown-docs`: Difficulty set to "negative" (no rules apply)

#### subagent_start.json (18 changes)
- **Reclassification (1 fixture):**
  - `sub-doc-updater`: Negative -> positive (easy) with "Update CLAUDE.md" + "Update README" rules
- **Additions (15 fixtures):** Added rules including mutation prevention, mock requirements, test speed, API logging, silent fallbacks, spike-testing, Protocols, read-patterns, secrets checking, and cache TTL
- **Difficulty changes (2 fixtures):**
  - `sub-data-model-design`: hard -> medium (direct match to immutability rules)
  - `sub-hard-build-agent-python`: hard -> medium (direct match to uv/vulnerability rules)

#### workflow.json (18 changes)
- **Additions across 18 fixtures** of new global rules:
  - "Commit after each implementation phase" (2 fixtures)
  - "Run convergence reviews" (2 fixtures)
  - "Run convergence reviews on PRD changes" (1 fixture)
  - "Follow the three-layer quality gate" (1 fixture)
  - "Spike-test risky integrations" (3 fixtures)
  - "Reuse popular open-source components" (1 fixture)
  - "Never use silent fallbacks" (3 fixtures)
  - "Never trust LLM-generated confidence scores" (1 fixture)
  - "Control your evaluation methodology" (2 fixtures)
  - "Always mock external calls in unit tests" (2 fixtures)
  - "If a tool call takes 10x longer, investigate" (1 fixture)
  - "Search GitHub for existing implementations" (1 fixture)
  - "Run ruff and mypy before committing" (1 fixture)

#### post_tool_use.json (2 changes)
- `post-edit-mutable-dataclass`: Added "Never mutate function arguments"
- `post-write-mutable-plus-no-hints-plus-magic`: Added "Never mutate function arguments"

#### stop.json (2 changes)
- `stop-refactored-dataclasses`: Added "Never mutate function arguments"
- `stop-mutable-dict-config`: Added "Never mutate function arguments"

## Validation Results

### Corpus Integrity Check
- **ALL rule texts in ALL fixtures match the corpus exactly** -- 0 missing rules across 831 fixtures
- 109 rules in corpus, all referenced correctly

### combined.json
- Rebuilt from all 5 fixture files
- 831 total fixtures (0 duplicate IDs)
- Breakdown: 442 basic, 115 post_tool_use, 77 stop, 73 subagent_start, 124 workflow

### Round 2 Spot-Check (20 fixtures, 4 per file)
All 20 randomly sampled fixtures (seed=42) verified correct:
- Positive fixtures have appropriate should_match rules for their queries
- Negative fixtures correctly have empty should_match
- should_not_match entries are plausible anti-matches
- Difficulty ratings align with semantic gap between query and rules
- No compliance-as-violation issues found in the sample

## Fixtures NOT Changed (Report Recommendations Skipped)

The R1 reports recommended some changes that were NOT applied because:
1. **Class B removals** -- All 15 "missing" rules are now present in the 109-rule corpus, so no removals needed
2. **Difficulty downgrades to "negative"** -- Same reason; rules exist, fixtures remain positive
3. **Borderline additions** -- Report explicitly marked as "skip" (e.g., `post-edit-hardcoded-config` mutation rule)
4. **PostToolUse/Stop new rule additions** -- Report concluded most new process rules don't apply to output verification events

## Outstanding Considerations

- `post-neg-bash-pip-list`: Uses `pip list` but is classified as negative. The query includes `pip` which could trigger the uv rule. Debatable -- PostToolUse evaluates output, and the output (package list) doesn't show a violation. Left as-is.
- Some workflow fixtures reference dropped workflow-specific rules that are now in the 109-rule corpus but were originally from `rules_workflow.txt`. These are correctly present.
