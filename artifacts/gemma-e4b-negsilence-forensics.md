# Gemma 4 E4B Negative Silence Forensics

**Date:** 2026-04-04
**Model:** gemma-4-E4B-it-Q8_0.gguf (8.2GB)
**Issue:** Basic NegSil 0.852 vs 9B's 0.962 — Gemma over-fires on 4/27 sampled negative fixtures

## 1. Over-Fired Negative Fixtures (Verbatim Reasoning)

### Case 1: `go-build` — `Bash: go build -o bin/server ./cmd/server`

**Gemma returned:** Rule 25 (dependency review)

> "The action is compiling a Go program using 'go build'. This is a build/compilation step, which often involves pulling dependencies, similar to a Docker build. Rule 25 applies because building software inherently involves using dependencies that should be reviewed for vulnerabilities."

**9B returned:** `[]`

> "While building might pull dependencies, the action itself is a standard build command without specific security or quality constraints listed in the rules. No rules directly apply."

**Diagnosis:** Gemma over-applies Principle 4 ("Think about consequences, not just keywords"). It reasons that `go build` *might* pull dependencies, and therefore the dependency review rule applies. 9B acknowledges the same reasoning path but correctly decides the build command itself doesn't warrant the rule. Gemma's reasoning is defensible but wrong for this context — `go build` with local code doesn't add new dependencies.

---

### Case 2: `mined-edit-auth-ts` — `Edit: mission-control/src/lib/auth.ts — adding requireSession() helper for per-route API auth`

**Gemma returned:** Rules 4 (input validation), 18 (type hints), 30 (docstrings)

> "This involves writing code, so security and quality rules apply. Rule 4 applies because authentication logic inherently involves validating user input at system boundaries. Rule 18 applies because all function signatures should have type hints. Rule 30 applies because public API functions should be documented."

**9B returned:** `[]`

> "While rules like 'Always validate user input' or 'Use type hints' are relevant to the code's eventual implementation, they are not triggered by the mere act of editing the file to add a helper function. No rules apply."

**Diagnosis:** Gemma conflates **topical relevance** with **direct applicability**. Auth code is *about* security, so security rules *must* apply. But this Edit event has no `new_string` content to inspect — only a description. Without seeing actual code, there's nothing to validate against. 9B correctly notes that rules about type hints don't apply to TypeScript files. Gemma doesn't notice the language mismatch.

---

### Case 3: `bash-mongo-shell` — `Bash: mongo --eval 'db.users.find({role: "admin"})'`

**Gemma returned:** Rules 4 (input validation), 9 (parameterized queries)

> "This involves running arbitrary JavaScript code directly against the database, which is a high-risk operation. Rule 9 mandates using parameterized queries to prevent SQL injection, and while MongoDB uses a different query language, the principle of preventing injection via dynamic execution applies strongly here."

**9B returned:** `[]`

> "The action is a read-only system query to inspect MongoDB data. It does not modify code, install packages, or interact with git."

**Diagnosis:** Gemma conflates the CLI `--eval` flag (which just means "evaluate this expression") with the security concept of `eval()` dynamic code execution. The query `db.users.find(...)` is a read-only lookup. Gemma also cross-applies SQL injection rules to MongoDB, acknowledging it's a "different query language" but applying the rule anyway. 9B correctly identifies this as a read-only data inspection.

---

### Case 4 (intermittent): `docker-privileged-run` — `Bash: docker run --privileged -v /:/host ubuntu bash`

This fixture over-fired in the benchmark run but was silent in my probe run (non-determinism). Gemma's reasoning when it fires likely involves Principle 4 (consequences) — mounting the host filesystem is a security risk.

---

## 2. Root Cause Diagnosis

**Primary cause: Gemma over-applies consequential reasoning (Principle 4).**

The prompt's Principle 4 says: "Think about consequences, not just keywords. A command may trigger a rule indirectly." Gemma takes this too literally:

- `go build` → "might pull dependencies" → dependency review rule
- `auth.ts edit` → "auth involves security" → validation/CSRF/docstring rules  
- `mongo --eval` → "eval = dynamic execution" → eval()/SQL injection rules

9B applies the same principle more conservatively: it acknowledges the consequential reasoning but concludes the connection is too indirect. Gemma treats "might be related" as "applies."

**Secondary cause: Topical association instead of direct constraint.**

Gemma reasons: "auth code is about security, therefore security rules apply." This is the same "compliance = match" confusion documented in session 18, but manifested differently. Instead of matching rules that the code *follows*, Gemma matches rules that share a *topic* with the code.

**Not an expansion issue.** Gemma's expansions (avg 4.4/rule) are actually tighter and fewer than 9B's (avg 6.9/rule). The expansions don't cause the over-firing — the LLM reasoning does.

**Not a thinking-mode leak.** Gemma's responses are clean JSON with no `<think>` tag interference. The reasoning is coherent and well-structured — just wrong on edge cases.

## 3. Expansion Comparison (3 Sample Rules)

| Rule | Gemma (count) | 9B (count) | Gemma Specificity | 9B Specificity |
|------|:---:|:---:|---|---|
| `Never use eval()` | 5 | 5 | Tight — all about Python eval/exec | Tight — all about Python eval/exec |
| `Review dependencies` | 5 | 8 | **Has `docker build`** — broader triggers | Has `go get` — more language coverage |
| `Validate user input` | 5 | 8 | Generic patterns (Flask, stdin) | More specific attack vectors (XSS, file upload) |

Key finding: Gemma's dependency review expansions include `docker build --build-arg dependency-version`, which may contribute to the `go build` over-fire via embedding similarity. However, the direct probe (all 30 rules, no embedding stage) shows the same over-fire, so this is a reasoning issue, not an embedding retrieval issue.

## 4. Proposed Fixes (Ranked by Expected Impact)

### Fix 1: Add a negative example for "build ≠ add dependency" (HIGH impact)

**Hypothesis:** Gemma needs an explicit negative example showing that build commands don't trigger dependency rules, similar to how Example 5 teaches "rebase ≠ force-push."

**Change:** Add Example 14 to the few-shot prompt:

```
Example 14 — Build command negative (go build is NOT adding dependencies):
RULES:
1. <rule_data_EXAMPLE>Review all dependencies for known vulnerabilities before adding</rule_data_EXAMPLE>
2. <rule_data_EXAMPLE>Add .gitignore entries for generated files and build artifacts</rule_data_EXAMPLE>
3. <rule_data_EXAMPLE>Run tests before committing</rule_data_EXAMPLE>
ACTION: <query_data_EXAMPLE>Bash: go build -o bin/server ./cmd/server</query_data_EXAMPLE>
RESPONSE: {{"reasoning": "The action compiles existing Go code into a binary. While the project has dependencies, 'go build' does not add or modify them — it uses what's already in go.mod. Rule 1 is about reviewing dependencies WHEN ADDING them, not when compiling. Rule 2 is about .gitignore, but the action is building, not committing. Building is a local development step with no project consequences beyond producing a binary.", "rules": []}}
```

**Expected effect:** NegSil +0.04-0.07 (fixes the `go build` class). Gemma is highly consistent — if one example establishes the pattern, it will generalize.

**Risk to PosRecall:** Minimal. This only narrows the "consequences" reasoning for build commands, not for install/add commands.

---

### Fix 2: Add a negative example for "topical relation ≠ applicability" on Edit events (HIGH impact)

**Hypothesis:** Gemma needs to learn that Edit events without visible code content are description-only, and topical association with security doesn't mean security rules apply.

**Change:** Add Example 15:

```
Example 15 — Edit description negative (auth topic ≠ security rules):
RULES:
1. <rule_data_EXAMPLE>Always validate user input at system boundaries</rule_data_EXAMPLE>
2. <rule_data_EXAMPLE>Enable CSRF protection on all forms</rule_data_EXAMPLE>
3. <rule_data_EXAMPLE>Document all public API functions with docstrings</rule_data_EXAMPLE>
ACTION: <query_data_EXAMPLE>Edit: src/lib/auth.ts — adding requireSession() helper for per-route auth</query_data_EXAMPLE>
RESPONSE: {{"reasoning": "The action describes adding an auth helper, but the Edit event shows only a description — no actual code content to inspect for violations. Rules about input validation and CSRF apply when code ACTUALLY violates them, not when the topic is security-related. Without seeing the function body, there is nothing to validate. Docstrings are a Python convention; this is TypeScript. No rules apply.", "rules": []}}
```

**Expected effect:** NegSil +0.04 (fixes the auth edit class and similar topic-association over-fires).

**Risk to PosRecall:** Low. This only affects Edit events without `new_string` content. Edit events WITH code (like Example 8) still match correctly.

---

### Fix 3: Add Principle 9 — "CLI flags are not code constructs" (MEDIUM impact)

**Hypothesis:** Gemma confuses the `--eval` CLI flag with the `eval()` function. A reasoning principle would prevent this class of error.

**Change:** Add after Principle 8:

```
9. **CLI flags are not code patterns.** A command-line flag like `--eval` \
is a CLI argument, not the programming construct `eval()`. Similarly, \
`--exec` in `find` is not the Python `exec()`. Match rules to the \
actual code pattern, not to flag names that share a keyword.
```

**Expected effect:** NegSil +0.04 (fixes the mongo --eval class).

**Risk to PosRecall:** Minimal. Actual `eval()` in code (Example 13) still matches because the code content contains the literal `eval()` call.

---

### Fix 4: Strengthen Principle 8 with "topic ≠ applicability" (LOW-MEDIUM impact)

**Hypothesis:** Principle 8 says "Avoid tangential associations" but Gemma doesn't treat topical relevance as tangential. Making it more explicit might help.

**Change to Principle 8:**

Before:
```
8. **Avoid tangential associations.** The rule must constrain the \
SPECIFIC action, not just share a topic.
```

After:
```
8. **Avoid tangential associations.** The rule must constrain the \
SPECIFIC action, not just share a topic. Auth code does not automatically \
trigger ALL security rules — only rules the code ACTUALLY violates. \
A build command does not trigger dependency rules just because the \
project has dependencies.
```

**Expected effect:** NegSil +0.02 (reinforces the other fixes).

**Risk to PosRecall:** Very low — this is a clarification, not a new constraint.

---

### Fix 5: Temperature reduction (LOW impact, easy to test)

**Hypothesis:** Gemma's default sampling might be slightly more exploratory. Reducing temperature from default to 0.1-0.3 could make it more conservative on edge cases.

**Change:** Pass `temperature=0.2` in `call_local()` for Gemma.

**Expected effect:** NegSil +0.01-0.03 (reduces variance on borderline cases).

**Risk to PosRecall:** Could slightly reduce recall on hard-tier positives where the model needs to make inference leaps.

---

## Summary

| Fix | Target | Expected NegSil Gain | PosRecall Risk | Effort |
|-----|--------|:---:|:---:|---|
| 1. Build negative example | go-build class | +0.04-0.07 | Minimal | 1 prompt addition |
| 2. Edit description negative | auth-topic class | +0.04 | Low | 1 prompt addition |
| 3. Principle 9 (CLI flags) | mongo --eval class | +0.04 | Minimal | 1 principle addition |
| 4. Strengthen Principle 8 | General over-association | +0.02 | Very low | 1 line edit |
| 5. Temperature reduction | Variance reduction | +0.01-0.03 | Low | Config change |

Combined expected effect: NegSil from 0.852 to ~0.93-0.96, closing the gap with 9B (0.962) while preserving Gemma's recall advantage (0.778 vs 0.560).

**Recommended order:** Fixes 1+2+3 together (they address different failure modes), then benchmark. Fix 4 only if needed. Fix 5 as a separate experiment.
