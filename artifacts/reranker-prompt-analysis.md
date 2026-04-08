# Reranker Prompt Analysis: 109-Rule Corpus

## Problem Statement

The eval corpus expanded from 32 to 109 rules. The LLM reranker's F2 dropped ~30%.
The primary issue is false positives: the reranker returns rules on negative fixtures
where it should stay silent. This analysis identifies the root causes and recommends
specific prompt changes.

---

## 1. Root Cause Analysis

### 1a. "When in doubt, include" is catastrophic at 109 rules

The prompt's Reasoning Principle #6 says:

> When in doubt, include. A missed rule (false negative) is worse than an extra
> rule (false positive). The agent can ignore an extra rule but cannot follow a
> rule it never sees.

At 32 rules, this was correct — the candidate set was small, and the LLM saw 3-5
candidates max. The cost of including one extra was low.

At 109 rules, this is the primary noise driver. With 20 candidates reaching the LLM
(pipeline.py line 163: `top_k = 20, threshold = 0.20` for LLM modes), the model now
sees rules from 6+ different categories. The "when in doubt, include" instruction
causes the model to include tangentially related rules from every category it
recognizes.

**Concrete example:** For `Bash: uv run pytest --cov=cuecard`, the LLM sees candidates
from testing (coverage, mocking, TDD), methodology (quality gates, convergence reviews),
LLM infrastructure (benchmark before building), and code style (keep files small). The
"when in doubt, include" instruction causes it to return 5+ rules when only 1-2 apply.

### 1b. New rule categories have overlapping semantic neighborhoods

The 109 rules break down as:

| Category | Count | Was in original 32? |
|----------|------:|---------------------|
| Security | 7 | Yes |
| Git | 10 | Yes |
| Testing/QA | 23 | Partially (3-4) |
| Code style | 8 | Yes |
| Methodology/Process | 19 | No |
| LLM infrastructure | 20 | No |
| Data quality | 5 | No |
| Other | 17 | Partially |

The NEW categories (methodology, LLM infra, data quality = 44 rules) share keywords
with the OLD categories. "Benchmark" appears in both testing and LLM infrastructure.
"Review" appears in code review, PRD review, and convergence review. "Quality" appears
in code quality, quality gates, and quality checks. The model cannot distinguish
"run pytest" (testing rule) from "benchmark before building" (LLM methodology rule)
because both share the semantic space of "verification before proceeding."

### 1c. Few-shot examples don't cover the new rule categories

The 13 few-shot examples cover:

| Example | Category | Positive/Negative |
|---------|----------|-------------------|
| 1. pip install | Package manager | Positive |
| 2. npm install | Dependency audit | Positive |
| 3. git commit | Multi-match | Positive |
| 4. Read file | Read-only | Negative |
| 5. git rebase | Tricky negative | Negative |
| 6. UserPromptSubmit complex task | Workflow | Positive |
| 7. UserPromptSubmit question | Workflow negative | Negative |
| 8. Edit code | Style rules | Positive |
| 9. docker build | Indirect match | Positive |
| 10. nvidia-smi | System command | Negative |
| 11. git stash | Local operation | Negative |
| 12. tmux send-keys | Direct match | Positive |
| 13. Edit debug print | Obvious match | Positive |

**Missing coverage:**
- No example showing a methodology/process rule being EXCLUDED from a code action
- No example showing an LLM infrastructure rule being EXCLUDED from a routine action
- No example showing a testing rule being EXCLUDED from a non-testing action
- No example where rules share keywords but only one applies (disambiguation)
- No example with 8+ candidate rules (current examples show 3-4 candidates max)

### 1d. Candidate count is too high for discrimination

Pipeline sends top_k=20, threshold=0.20 to the LLM. At 109 rules with enriched
expansions, many more rules pass the 0.20 threshold. The LLM receives ~15-20
candidates when it was optimized for ~8-10. More candidates = more noise to sift
through = more "when in doubt, include" triggers.

### 1e. Long-form rules with explanations create keyword pollution

The new rules have a "-- explanation" suffix:

> Never commit secrets (API keys, tokens, passwords, connection strings) to git
> -- once pushed, secrets are in the history forever and require rotation across
> all environments

The explanation adds keywords ("environments", "rotation", "history") that expand
the rule's embedding neighborhood beyond its actual applicability. Combined with
expansions that embed these explanation-derived keywords, the rule matches queries
it shouldn't.

---

## 2. Expansion Quality Assessment

### Good expansions (tight, actionable triggers):

Rule 1 (secrets): `"git add . and commit containing AWS_SECRET_ACCESS_KEY"` -- specific, actionable
Rule 9 (uv): `"pip install requests"` (would need to be in expansions) -- direct trigger
Rule 30 (file size): `"function body exceeds 50 lines during review"` -- specific

### Problematic expansions (too broad, cross-domain):

Rule 5 (100% coverage): `"git commit after adding new feature files"` -- triggers on ANY git commit
Rule 6 (validate with real data): `"running integration tests against mock API responses"` -- triggers on ANY test run
Rule 7 (benchmark LLM call): `"executing full data processing job before confirming LLM endpoint latency"` -- too abstract
Rule 42 (subagent scope): `"agent loop running indefinitely on a single file"` -- matches any long-running process
Rule 44 (audit ground truth): `"debugging a test case that fails consistently"` -- matches ANY test debug
Rule 52 (mock external calls): `"real calls make tests slow, flaky, and non-deterministic"` -- description, not trigger
Rule 94 (small sample): `"running performance tests on a subset of data"` -- matches any test run
Rule 95 (unproven techniques): `"deploying a new ML model architecture without A/B testing"` -- matches any deployment

**Pattern:** Methodology and LLM infrastructure rules generate expansions that overlap
with routine developer actions. The expansion prompt does a good job for concrete rules
(secrets, package managers, file permissions) but produces overly abstract expansions
for process-oriented rules.

---

## 3. Specific Recommendations

### R1. Replace "when in doubt, include" with category-aware guidance

**Before (Principle #6):**
```
6. **When in doubt, include.** A missed rule (false negative) is worse
than an extra rule (false positive). The agent can ignore an extra rule
but cannot follow a rule it never sees.
```

**After:**
```
6. **When in doubt about CONCRETE rules (security, code style, tool usage),
include.** A missed security or style rule is worse than an extra one.
**When in doubt about PROCESS rules (methodology, benchmarking, review
workflow, testing philosophy), exclude.** Process rules apply to the
overall approach, not to individual tool calls. A git commit needs the
secrets rule; it does NOT need "run convergence reviews after each
milestone" unless the commit is part of a milestone workflow.
```

### R2. Add 3 negative few-shot examples for new rule categories

Add these after Example 13:

```
Example 14 -- Methodology rules do NOT apply to routine actions:
RULES:
1. <rule_data_EXAMPLE>Run convergence reviews after each implementation milestone</rule_data_EXAMPLE>
2. <rule_data_EXAMPLE>Classify every task as SIMPLE, MEDIUM, or COMPLEX before starting</rule_data_EXAMPLE>
3. <rule_data_EXAMPLE>Never commit secrets to git</rule_data_EXAMPLE>
4. <rule_data_EXAMPLE>Require 100% test coverage on all new code</rule_data_EXAMPLE>
5. <rule_data_EXAMPLE>Always benchmark a single LLM call before launching a full pipeline</rule_data_EXAMPLE>
ACTION: <query_data_EXAMPLE>Bash: uv run pytest tests/test_config.py -v</query_data_EXAMPLE>
RESPONSE: {{"reasoning": "The action runs a specific test file. Rule 4 is tangentially related (coverage), but running a single test file is a routine development action, not a commit or milestone. Rules 1, 2, and 5 are process/methodology rules that apply to task planning and workflow, not to executing a test. Rule 3 is about secrets, irrelevant here. No rules directly constrain running pytest.", "rules": []}}

Example 15 -- LLM/benchmark rules do NOT apply to normal code operations:
RULES:
1. <rule_data_EXAMPLE>Always benchmark a single LLM call before launching a full pipeline run</rule_data_EXAMPLE>
2. <rule_data_EXAMPLE>Never trust small sample benchmark results</rule_data_EXAMPLE>
3. <rule_data_EXAMPLE>Use type hints on all function signatures</rule_data_EXAMPLE>
4. <rule_data_EXAMPLE>Keep functions under 50 lines</rule_data_EXAMPLE>
ACTION: <query_data_EXAMPLE>Edit: {{"file_path": "src/utils.py", "new_string": "def calculate_score(items: list[float]) -> float:\n    return sum(items) / len(items)"}}</query_data_EXAMPLE>
RESPONSE: {{"reasoning": "The edit adds a function to a utility file. Rule 3 applies -- the function has type hints, which is correct, but we should verify. Rule 4 applies -- the function should stay under 50 lines. Rules 1 and 2 are about LLM benchmarking methodology, completely unrelated to editing a utility function.", "rules": [3, 4]}}

Example 16 -- Many candidate rules but most are irrelevant (high-candidate discrimination):
RULES:
1. <rule_data_EXAMPLE>Review all dependencies for known vulnerabilities</rule_data_EXAMPLE>
2. <rule_data_EXAMPLE>Run convergence reviews after each milestone</rule_data_EXAMPLE>
3. <rule_data_EXAMPLE>Validate every phase against real data</rule_data_EXAMPLE>
4. <rule_data_EXAMPLE>Run quality checks before every commit</rule_data_EXAMPLE>
5. <rule_data_EXAMPLE>Require 100% test coverage on all new code</rule_data_EXAMPLE>
6. <rule_data_EXAMPLE>Always benchmark a single LLM call before pipeline runs</rule_data_EXAMPLE>
7. <rule_data_EXAMPLE>Follow the three-layer quality gate</rule_data_EXAMPLE>
8. <rule_data_EXAMPLE>Use uv for all Python package operations</rule_data_EXAMPLE>
ACTION: <query_data_EXAMPLE>Bash: git diff HEAD~1</query_data_EXAMPLE>
RESPONSE: {{"reasoning": "The action is git diff, which shows changes between commits. This is a read-only inspection command -- it does not modify code, install packages, commit, or run pipelines. None of these rules constrain viewing a diff.", "rules": []}}
```

### R3. Reduce LLM candidate count from 20 to 12

In `pipeline.py`, the LLM modes send `top_k=20, threshold=0.20`. With 109 rules:
- Reduce to `top_k=12, threshold=0.25`
- This reduces the noise surface the LLM must discriminate against
- 12 candidates is still 3-4x more than the final top_k=5, giving ample room for discrimination

### R4. Add a "RULE CATEGORY AWARENESS" section to the prompt

Add after the REASONING PRINCIPLES section:

```
RULE CATEGORIES — match the right category to the event:

- **Concrete action rules** (security, code style, tool usage, package managers):
  Apply when the action DIRECTLY involves the rule's domain. "pip install" →
  package manager rule. "def foo():" without types → type hints rule.

- **Process/methodology rules** (task classification, PRD, convergence reviews,
  quality gates, subagent delegation): Apply ONLY to UserPromptSubmit events
  where the user is starting a task or making a process decision. Do NOT apply
  to individual PreToolUse events like running tests, editing files, or git
  operations — those are routine actions, not process decisions.

- **LLM/ML infrastructure rules** (benchmarking, prompt format, model selection,
  sample size): Apply ONLY when the action involves LLM calls, model evaluation,
  or prompt engineering. Do NOT apply to normal code editing, testing, or git
  operations.

- **Testing philosophy rules** (mock externals, test speed, TDD): Apply when
  the action involves writing or modifying test code. Do NOT apply when merely
  running existing tests.
```

### R5. Tighten expansion prompt for process/methodology rules

The expansion prompt should generate narrower expansions for abstract rules.

**Before (in expander.py, after the reasoning principles):**
(No category-specific guidance)

**After — add to the system prompt before the examples block:**
```
CATEGORY-SPECIFIC EXPANSION GUIDANCE:
- For CONCRETE rules (security, code style, tool commands): generate specific
  tool commands and code patterns as expansions.
- For PROCESS/METHODOLOGY rules (task classification, PRD, reviews, quality gates):
  generate ONLY UserPromptSubmit-style queries. Do NOT generate tool commands
  like "git commit" or "pytest" — those are routine actions that happen to
  occur during the process but are not what triggers the methodology rule.
- For LLM/ML rules (benchmarking, prompt format): generate only LLM-specific
  actions (calling models, evaluating results, tuning prompts). Do NOT generate
  generic testing or deployment actions.
```

### R6. Do NOT reduce final top_k from 5 to 3

The final `top_k=5` in the reranker call (`rerank_llm(..., top_k=5)`) is the maximum
the LLM can return. The LLM already returns `[]` (empty) when no rules apply, and
returns fewer than 5 when only 1-2 apply. Reducing to 3 would hurt recall on the
multi-match fixtures (e.g., git commit matching secrets + quality checks + conventional
format = 3 rules). Keep at 5.

---

## 4. Summary of Answers to Key Questions

### Q1. What specific prompt changes would reduce false positives without hurting recall?
- Replace "when in doubt, include" with category-aware inclusion guidance (R1)
- Add 3 negative few-shot examples for methodology/LLM/high-candidate scenarios (R2)
- Add rule category awareness section (R4)
- Reduce LLM candidate input from 20 to 12 (R3)

### Q2. Should "when in doubt, include" become "when in doubt, exclude" for methodology/process rules?
**Yes, but with nuance.** Keep "include" for concrete rules (security, code style, tools).
Switch to "exclude" for process rules on PreToolUse events. Process rules should only
fire on UserPromptSubmit (task-level decisions), not on individual tool calls.

### Q3. Are the few-shot negative examples adequate for the new rule categories?
**No.** All 5 current negatives are "read-only/local operation" patterns (Read file,
git rebase, nvidia-smi, git stash, simple question). None demonstrate excluding
methodology, LLM infrastructure, or testing philosophy rules from routine actions.
The model has never seen an example of "here are 8 candidates, most are process rules,
and none apply to this routine git diff."

### Q4. Should top_k be reduced from 5 to 3 for 109 rules?
**No.** Final top_k=5 is fine — the LLM already returns fewer when appropriate. The
problem is the INPUT candidate count (top_k=20 in Stage 1 for LLM modes), which should
be reduced to 12. The discrimination problem is at the input, not the output.

### Q5. What's the expansion quality like for the new long-form rules?
**Mixed.** Concrete rules have good expansions. Process/methodology rules have
expansions that are too broad — they overlap with routine developer actions. The
expansion prompt needs category-specific guidance to generate narrower expansions
for abstract rules (R5). Specific offenders:
- Rule 5 expansion "git commit after adding new feature files" matches ALL commits
- Rule 6 expansion "running integration tests against mock API responses" matches ALL test runs
- Rule 44 expansion "debugging a test case that fails consistently" matches ALL debugging
- Rule 94 expansion "running performance tests on a subset of data" matches ALL test runs

---

## 5. Implementation Priority

1. **R1 (rewrite Principle #6)** — highest impact, lowest effort. This is the single
   biggest driver of false positives.
2. **R2 (add 3 negative examples)** — high impact. Models learn from examples more
   than from instructions.
3. **R4 (category awareness section)** — medium impact. Gives the model a framework
   for thinking about rule relevance by category.
4. **R3 (reduce candidate count)** — medium impact. Reduces the noise surface.
5. **R5 (expansion prompt tightening)** — medium impact but requires re-expansion
   of all rules (slow). Do after prompt changes are validated.
6. **R6 (keep final top_k=5)** — no change needed.

**Expected outcome:** R1 + R2 + R4 together should recover most of the NegSil regression.
The "when in doubt, include" + missing negative examples is the dominant failure mode.
