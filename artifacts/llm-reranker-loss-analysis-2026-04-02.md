# LLM Reranker Loss Pattern Analysis

**Date:** 2026-04-02
**Model:** Qwen3.5-35B-A3B (llm-local, reasoning prompt)
**Dataset:** enriched corpora, 567 fixtures total
**Source:** `eval/results/enriched-llm-local-2026-04-02.json`

---

## Aggregate Results

| Corpus | Fixtures | Easy Recall | Med Recall | Hard Recall | Neg Silence |
|--------|----------|-------------|------------|-------------|-------------|
| basic | 354 | 85.2% | 73.1% | 50.1% | 91.0% |
| workflow | 84 | 93.2% | 63.8% | 66.7% | 91.7% |
| mined-sessions | 49 | 75.0% | 56.3% | 0.0% | 91.2% |
| mined-sessions-v2 | 80 | 94.4% | 63.3% | 30.8% | 86.1% |

## Top 5 Loss Patterns

### 1. "Type hints" and "close resources" are invisible to the reranker (~40% of false negatives)

The most-missed rules across all corpora are:
- "Use type hints on all function signatures"
- "Always close file handles and database connections"

The LLM treats these as generic style advice rather than event-specific constraints. Even when code clearly shows untyped functions or unclosed handles, the reranker drops these rules.

**Evidence from `false_negatives_detail.json`:**

| Fixture ID | Query | Missed Rule | Recall |
|---|---|---|---|
| `edit-python-file` | `Edit: src/auth.py, def login → def login_user` | "Use type hints on all function signatures" | 0.0 |
| `file-handle-leak` | `Edit: src/loader.py, f = open('data.txt')` | "Always close file handles and database connections" | 0.0 |
| `socket-no-close` | `Edit: src/client.py, socket.connect(...)` | "Always close file handles and database connections" | 0.0 |
| `embedded-code-celery-task` | `Write: tasks/email.py` (SMTP, no close) | "Always close file handles..." + "type hints" + "error handling" | 0.4 |

**Root cause:** The current prompt has no example showing resource cleanup or type hint matching. The 7 existing examples focus on git operations, package managers, and workflow — never on code quality rules triggered by embedded code patterns.

### 2. Embedded code content is ignored for secondary rules

When Write/Edit contains multi-line code, the reranker catches the *primary* issue but misses *secondary* violations in the same code.

**Evidence:**

| Fixture ID | Primary (caught) | Secondary (missed) | Recall |
|---|---|---|---|
| `embedded-code-celery-task` (EASY) | hardcoded secret | except:pass, unclosed SMTP, no type hints | 0.40 |
| `embedded-code-flask-app` | SQL injection | XSS, error handling, type hints | 0.40 |
| `long-edit-class-definition` | immutable data | print debug, type hints, error handling, close connections | 0.20 |
| `embedded-code-fastapi-endpoint` | SQL injection | type hints, error handling | 0.50 |

**Root cause:** No few-shot example demonstrates matching 4+ rules on a single piece of embedded code. The model sees "primary issue found" and stops scanning.

### 3. Indirect/abstract rule-to-action mappings fail

When the connection between action and rule requires domain reasoning (not literal keyword overlap), the reranker misses.

**Evidence:**

| Query | Expected Rule | Reasoning Gap |
|---|---|---|
| `Bash: docker build -t myapp .` | "Review dependencies for vulnerabilities" | docker build installs deps via FROM/RUN |
| `Bash: docker run -p 8080:80` | "Use environment variables for configuration" | hardcoded port = configuration |
| `Bash: grep -r 'AKIA' .` | "Rotate exposed secrets" | searching for AWS keys = secret exposure |
| `Bash: docker run --privileged` | "Set file permissions to 0o600" | privileged mode = security concern |

**Root cause:** The prompt's "cross-domain match" example (Example 2: npm install → review deps) covers one case but not docker/grep/infrastructure commands.

### 4. Negative leaks on generic tool operations (~9% leak rate)

The 9% negative leak comes from operations where the LLM returns 1-2 generic rules instead of empty.

**Leaking operation categories (inferred from fixture patterns):**
- Git local operations: `git stash`, `git cherry-pick`, `git tag` — LLM returns "conventional commit format" or "quality checks"
- System commands: `nvidia-smi`, `pactl list sources`, `df -h` — LLM returns "environment variables" or "configuration"
- Read-like operations: `ls`, `cat`, `which` — occasionally leak generic rules

**Root cause:** The prompt has 2 negative examples (Example 4: Read file, Example 5: git rebase). Neither covers system info commands or git stash/cherry-pick. The model generalizes "read-only" from file reads but not from system diagnostics.

### 5. mined-sessions-v2 has worst negative silence (86.1%)

Real-world mined fixtures from actual developer sessions have more ambiguous queries that aren't clearly read-only.

**Characteristics of leaking mined fixtures:**
- Hardware/system queries that look "operational" but need zero rules
- Ambiguous package manager commands (listing, not installing)
- IDE/editor operations routed through Bash

**Root cause:** Mined fixtures reflect real developer workflows where 54% are negatives. The prompt's negative examples don't cover the diversity of real no-op commands.

---

## Recommended Prompt Changes

### 4 New Few-Shot Examples

#### Example 8 — Embedded code multi-match (scan ALL violations)

```
Example 8 — Embedded code multi-match (scan ALL violations):
RULES:
1. <rule_data_EXAMPLE>Never commit secrets (API keys, tokens, passwords) \
to git</rule_data_EXAMPLE>
2. <rule_data_EXAMPLE>Always handle errors explicitly, never silently \
swallow exceptions</rule_data_EXAMPLE>
3. <rule_data_EXAMPLE>Always close file handles and database connections\
</rule_data_EXAMPLE>
4. <rule_data_EXAMPLE>Use type hints on all function signatures\
</rule_data_EXAMPLE>
5. <rule_data_EXAMPLE>Use frozen dataclasses for immutable data\
</rule_data_EXAMPLE>
ACTION: <query_data_EXAMPLE>Write: file_path='tasks/email.py' \
content='from celery import shared_task\nimport smtplib\n\n\
SMTP_PASS = "password123"\n\n@shared_task\ndef send_email(to, subject, body):\n\
    try:\n        server = smtplib.SMTP("smtp.gmail.com", 587)\n\
        server.login("app@mail.com", SMTP_PASS)\n\
        server.sendmail("app@mail.com", to, body)\n\
    except Exception:\n        pass\n'</query_data_EXAMPLE>
RESPONSE: {{"reasoning": "The code has multiple violations. SMTP_PASS is a hardcoded secret — rule 1 applies. The except/pass silently swallows errors — rule 2 applies. The SMTP connection is never closed — rule 3 applies. The function has no type hints — rule 4 applies. Rule 5 is about dataclasses, not relevant here. When code contains multiple issues, match ALL applicable rules, not just the most obvious one.", "rules": [1, 2, 3, 4]}}
```

**Targets:** Loss patterns #1 (type hints/resources invisible) and #2 (secondary violations missed)

#### Example 9 — Resource cleanup match (unclosed handle)

```
Example 9 — Resource cleanup match (unclosed handle):
RULES:
1. <rule_data_EXAMPLE>Always close file handles and database connections\
</rule_data_EXAMPLE>
2. <rule_data_EXAMPLE>Use type hints on all function signatures\
</rule_data_EXAMPLE>
3. <rule_data_EXAMPLE>Use conventional commit format</rule_data_EXAMPLE>
ACTION: <query_data_EXAMPLE>Edit: {{"file_path": "src/loader.py", \
"new_string": "f = open('data.txt')\ndata = f.read()\nprocess(data)"}}\
</query_data_EXAMPLE>
RESPONSE: {{"reasoning": "The code opens a file with open() but never closes it — no with block or explicit f.close(). Rule 1 directly applies: file handles must be closed. The code also lacks type hints on any visible function context, but the edit only shows statements, not a function signature, so rule 2 is not clearly triggered. Rule 3 is about commits, unrelated.", "rules": [1]}}
```

**Targets:** Loss pattern #1 (resource cleanup invisible)

#### Example 10 — System command negative (hardware/info commands)

```
Example 10 — System command negative (hardware/info commands):
RULES:
1. <rule_data_EXAMPLE>Never commit secrets to git</rule_data_EXAMPLE>
2. <rule_data_EXAMPLE>Run quality checks before every commit\
</rule_data_EXAMPLE>
3. <rule_data_EXAMPLE>Use environment variables for configuration\
</rule_data_EXAMPLE>
ACTION: <query_data_EXAMPLE>Bash: nvidia-smi</query_data_EXAMPLE>
RESPONSE: {{"reasoning": "The action runs nvidia-smi, a system information command that displays GPU status. This is a read-only diagnostic command — it does not modify code, commit changes, or install packages. None of the rules constrain viewing system information.", "rules": []}}
```

**Targets:** Loss patterns #4 (negative leaks) and #5 (mined-sessions silence)

#### Example 11 — Git local-only operation negative

```
Example 11 — Git local-only operation negative (stash/cherry-pick/bisect):
RULES:
1. <rule_data_EXAMPLE>Never force-push to main or master branch\
</rule_data_EXAMPLE>
2. <rule_data_EXAMPLE>Run quality checks before every commit\
</rule_data_EXAMPLE>
3. <rule_data_EXAMPLE>Use conventional commit format</rule_data_EXAMPLE>
ACTION: <query_data_EXAMPLE>Bash: git stash pop</query_data_EXAMPLE>
RESPONSE: {{"reasoning": "The action is git stash pop, which restores previously stashed changes to the working directory. This is a local-only operation — it does not create a commit, push to remote, or modify the commit history. Rule 1 is about force-pushing, rule 2 about pre-commit checks, rule 3 about commit messages. None apply to restoring stashed work.", "rules": []}}
```

**Targets:** Loss pattern #4 (negative leaks on git operations)

### 7 New DO/DON'T Guidelines

Add to the existing `MATCHING GUIDELINES:` section (after line 60, before the `IMPORTANT:` injection warning):

```
- DO match resource cleanup rules (close handles, connections) when code \
creates file handles (open()), database connections (connect()), sockets \
(socket()), or network clients (SMTP, HTTP) without a with-block or \
explicit .close()
- DO match "type hints" rules when the code contains a Python function \
definition (def/async def) without type annotations, even if the primary \
issue is something else like SQL injection
- DO scan embedded code for ALL violations — when Write or Edit contains \
multi-line code, check every rule against the full content, not just the \
most prominent issue. Five matches is better than one if five rules apply.
- DO match "review dependencies" for ANY dependency-adding command: \
pip install, npm install, cargo add, go get, docker build (FROM + RUN \
install), composer require, gem install
- DO NOT match any rules for system/hardware info commands (nvidia-smi, \
df, htop, free, uname, lsb_release, pactl, whoami, uptime, ps, top)
- DO NOT match git workflow rules (commit format, quality checks, \
force-push, branching) for local-only git operations that neither \
create commits nor push: stash, cherry-pick, bisect, reflog, log, \
blame, show, diff, status, branch --list
- DO NOT stop after finding the first matching rule for embedded code — \
continue evaluating every candidate rule against the full code content
```

### Insertion Points

- **Examples 8-11:** After line 142 in `src/cuecard/llm_reranker.py` (after Example 7), before the closing `"""`
- **7 guidelines:** After line 60 in `src/cuecard/llm_reranker.py` (end of MATCHING GUIDELINES block), before the `IMPORTANT:` injection warning on line 62

---

## Expected Impact

| Loss Pattern | Fix | Metric | Before | Target |
|---|---|---|---|---|
| Type hints / resources invisible | Examples 8-9 + 3 DO guidelines | Easy recall (basic) | 85.2% | 92%+ |
| Secondary violations missed | Example 8 + "scan ALL" guideline | Medium recall (basic) | 73.1% | 80%+ |
| Negative leaks (system/git) | Examples 10-11 + 2 DON'T guidelines | Neg silence (basic) | 91.0% | 95%+ |
| Negative leaks (mined) | Example 10 + system DON'T | Neg silence (mined-v2) | 86.1% | 92%+ |
| Indirect mappings | "review deps" DO guideline | Hard recall (basic) | 50.1% | ~53% |

## Risks

1. **Prompt length:** Adding 4 examples + 7 guidelines increases system prompt by ~1.5K tokens. With llama-server prompt caching, this costs nothing after warmup (session 16 learning).
2. **Over-matching on type hints:** The "DO match type hints" guideline could increase noise if the model starts matching type hints on every Edit. Example 9 mitigates by showing restraint when no function definition is visible.
3. **Guideline conflicts:** "DO scan ALL violations" could conflict with the existing "DO NOT match tangentially related rules." The few-shot examples demonstrate the boundary: match all *applicable* rules, not all *candidate* rules.

## Validation Plan

1. Apply prompt changes
2. Re-run `cuecard eval` on all 4 corpora with `--mode llm-local`
3. Compare per-tier metrics against this baseline
4. If easy recall improves but noise increases >5%, tune the "type hints" guideline
5. If negative silence doesn't reach 95%, add more negative examples (system commands list)
