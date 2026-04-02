# Expansion Quality Review — 2026-04-02

Reviewer: Turiya (subagent)
Corpora: `eval/corpora/enriched_basic/rules.json` (30 rules, 300 expansions), `eval/corpora/enriched_workflow/rules.json` (46 rules, 460 expansions)
Generator: Qwen3.5-35B via llama-server, v2 expansion prompt (DO/DON'T + 3 golden examples)

---

## Basic Corpus (30 coding rules, 300 expansions)

### Overall Quality Score: 4/5

### Estimated % Good Expansions: ~75%

The basic corpus is strong. Most expansions are concrete, tool-specific, and diverse. The v2 prompt clearly works well for coding rules where the vocabulary gap is between abstract principles and specific CLI commands/code patterns.

### Common Failure Patterns

| Pattern | Frequency | Example |
|---------|-----------|---------|
| **Paraphrase/restatement** | ~10% | "committing without pre-commit ruff validation" (rule: "Run ruff and mypy before committing") |
| **Same-scenario repetition** | ~8% | Rule "Use uv..." has 8/10 expansions that are all `pip install X` variants |
| **Too close to canonical** | ~5% | "dataclass decorator without frozen=True" for "Use frozen dataclasses" |
| **Positive instead of violation** | ~2% | "using os.environ.get instead of hardcoded API key" — this is the *correct* behavior, not a trigger |

### 5 Best Expansions

1. **"f-string formatting SQL query with user input"** (rule: parameterized queries)
   — Perfect: specific code pattern, concrete, different vocabulary from "SQL injection"

2. **"aiohttp.ClientSession not closed after request"** (rule: close resources)
   — Perfect: names a specific library + specific violation, exactly what a query would contain

3. **"hardcoded timeout value 30000 in socket.connect"** (rule: magic numbers)
   — Perfect: concrete code, includes a number, bridges "magic number" → actual usage

4. **"mongoose query with $where and user input"** (rule: SQL injection/parameterized queries)
   — Excellent: crosses into NoSQL territory, different vocabulary entirely from "SQL"

5. **"Vue v-html binding with raw data"** (rule: sanitize HTML/XSS)
   — Excellent: framework-specific, wouldn't match "XSS" lexically, perfect vocabulary bridge

### 5 Worst Expansions

1. **"using os.environ.get instead of hardcoded API key"** (rule: use env vars not hardcoded)
   — **Wrong direction**: describes the CORRECT behavior, not the violation that should trigger the rule

2. **"starting work on a feature branch"** (rule: always create a new branch)
   — **Paraphrase**: restates the rule rather than describing a concrete action

3. **"tmux send-keys missing final carriage return"** (rule: send Enter after tmux send-keys)
   — **Near-duplicate**: 6 of 10 expansions for this rule are minor variations of "tmux send-keys without Enter/newline"

4. **"creating a release branch from develop"** (rule: always create new branch for feature work)
   — **Tangential**: release branches from develop is a git-flow concept, not about feature branching

5. **"using type hints with typing module imports"** (rule: type hints on all functions)
   — **Positive example**: describes correct behavior, not a trigger scenario

### Rules with Weakest Expansion Quality

1. **"Send Enter after every tmux send-keys command"** — 6/10 expansions are near-duplicates ("tmux send-keys without Enter", "tmux send-keys missing newline", "tmux send-keys not followed by newline", etc.). Very low diversity. Missing: `Bash` tool context like `bash: tmux send-keys 'npm start'`, or patterns like `send-keys C-c` without Enter.

2. **"Always create a new branch for feature work"** — Multiple expansions describe the correct action (creating branches) rather than the violation (committing directly to main). 4/10 are just `git checkout -b` variants. Should include: "committing directly to main", "pushing to master without feature branch", "editing files on main branch".

3. **"Use frozen dataclasses for immutable data"** — Too tightly clustered around `dataclass(frozen=False)` variations. Missing: `@dataclass` without frozen, `class Config:` with mutable attrs, `NamedTuple` vs `dataclass` decision, dict/list used where frozen dataclass would be better.

4. **"Use type hints on all function signatures"** — Two expansions describe correct behavior ("using type hints with typing module imports"). Others are generic ("dynamic typing used instead of explicit types"). Missing tool context: `mypy --strict`, `def foo(x):` in a diff.

5. **"Keep functions under 50 lines, files under 400 lines"** — Multiple paraphrases ("refactor large method", "condense long function", "split file"). Missing concrete triggers: `wc -l showing 850`, `Read: src/utils.py` on a massive file, `Edit` adding 30 lines to an already-long function.

### Length Distribution

All 300 expansions are well within the 200-char limit. Typical length is 30-60 chars. No excessively long or terse entries. Good.

### Near-Duplicate Analysis (within rules)

Worst offenders:
- **tmux send-keys rule**: "tmux send-keys without trailing Enter" / "tmux send-keys command not followed by newline" / "tmux send-keys missing final carriage return" — 3 near-identical expansions
- **force-push rule**: "git push -f to main" / "git push --force to master" / "git push -f origin main" / "git push -f master" — 4 near-identical, only differs by branch name or flag variant
- **Use uv rule**: "pip install requests" / "python -m pip install numpy" / "pip install -r requirements.txt" — all are `pip install X`, low diversity in the *type* of pip operation

### Tool Context Coverage

Good: Many expansions include real tool names (git, pip, pytest, ruff, mypy, docker, psycopg2, SQLAlchemy, etc.)

Missing: Very few expansions reference the Claude Code tool context that queries actually arrive in. For example:
- No expansions mention `Bash:`, `Edit:`, `Read:` prefixes
- No expansions look like actual PreToolUse queries: "Bash: git commit -m 'wip'"
- This matters because at query time, cuecard sees "Bash: pip install requests", not just "pip install requests"

---

## Workflow Corpus (46 workflow rules, 460 expansions)

### Overall Quality Score: 3/5

### Estimated % Good Expansions: ~55%

The workflow corpus is notably weaker than the basic corpus. Workflow rules are more abstract ("classify every task as SIMPLE/MEDIUM/COMPLEX"), and the LLM struggles more to generate concrete trigger phrases. Many expansions drift into project management jargon or stay too abstract.

### Common Failure Patterns

| Pattern | Frequency | Example |
|---------|-----------|---------|
| **Paraphrase/restatement** | ~20% | "PRD review with unresolved feedback" (rule: run iterative PRD reviews until clean) |
| **Too abstract** | ~15% | "parallel PRD validation by multiple agents" (rule: review PRDs with specialist subagents) |
| **Same-scenario repetition** | ~10% | Rule "CLAUDE.md" has 8/10 variations of "changed X without updating CLAUDE.md" |
| **Wrong domain** | ~5% | "pip install context7" for rule about using Context7/Exa/mgrep |
| **Missing user message patterns** | ~15% | Workflow rules trigger on UserPromptSubmit but expansions look like tool calls |

### 5 Best Expansions

1. **"openai.ChatCompletion.create() takes 5s for simple prompt"** (rule: investigate SDK overhead)
   — Perfect: specific API call + specific symptom, bridges "SDK overhead" → concrete observation

2. **"alter table add column without checking existing views"** (rule: audit consumers on schema change)
   — Excellent: concrete SQL action + specific oversight, diverse from rule text

3. **"using Levenshtein distance for text deduplication"** (rule: use semantic similarity not lexical)
   — Excellent: names specific wrong approach, vocabulary bridge from "word-based matching"

4. **"overwriting model checkpoint without saving previous version"** (rule: save artifacts before overwriting)
   — Good: concrete ML workflow action, specific file pattern

5. **"retrying failed docker build without checking logs"** (rule: read error before retrying)
   — Good: concrete tool + concrete anti-pattern

### 5 Worst Expansions

1. **"pip install context7"** (rule: use Context7 for docs, Exa for web, mgrep for code)
   — **Wrong domain**: Context7 is an MCP server, not a pip package. This expansion is factually incorrect.

2. **"planner agent for complex features"** (rule: use planner/architect/tdd-guide agents)
   — **Direct restatement**: literally restates part of the rule with no vocabulary bridging

3. **"task paused before starting compaction"** (rule: save state to artifacts before compaction)
   — **Too abstract**: doesn't describe a concrete action, just a vague state description

4. **"tmux context usage exceeds 67 percent"** (rule: monitor agent context, compact at 67%)
   — **Paraphrase**: restates the rule with numbers. Should be: "claude context window 85% full", "/compact suggestion appearing"

5. **"running review agents one after another"** (rule: launch parallel subagents)
   — **Paraphrase**: just restates the anti-pattern in almost the same words as the rule

### Rules with Weakest Expansion Quality

1. **"Use the planner agent for complex features, architect agent for system design decisions, tdd-guide agent for new features and bug fixes"** — All 10 expansions are direct restatements: "planner agent for complex features", "architect agent for system design", "tdd-guide agent for new features". Zero vocabulary bridging. Should include: "I want to build a new auth system" → planner, "should we use microservices or monolith?" → architect, "fix the login bug" → tdd-guide.

2. **"Use Context7 for library documentation lookups, Exa for broader web research, mgrep for semantic code search"** — Factually wrong expansions ("pip install context7", "npm install @exa-js/api"). These are MCP tools, not packages. Expansions should be: "how does fastapi dependency injection work?", "find similar implementations on GitHub", "search codebase for auth middleware".

3. **"Tell delegated agents to report back via tmux send-keys"** — 8/10 expansions are minor restatements of "tmux send-keys for reporting". Missing: actual user messages like "have the agent notify me when done", "set up callback from worker agent".

4. **"Monitor agent context usage in tmux"** — All expansions are tautological variations of "tmux context usage at 67 percent". Missing: "session getting long", "agent responses getting worse", "running out of context window".

5. **"Review PRDs with multiple specialist subagents in parallel"** — Generic project management phrases: "architect review PRD", "run multiple specialist checks". Missing user message patterns: "review my design doc", "check this spec for security issues", "get feedback on the architecture".

6. **"Before compaction, save task state to artifacts/"** — All 10 expansions are robotic restatements: "saving progress to artifacts/ directory", "writing current state to artifacts/ before cleanup". Missing: "session is getting long", "about to run /compact", "need to preserve context".

### Length Distribution

Acceptable but slightly more varied than basic corpus. A few expansions are terse (5-6 words) which may not provide enough embedding signal. No excessively long entries.

### Near-Duplicate Analysis (within rules)

Worst offenders:
- **Planner/architect/tdd-guide rule**: all 10 expansions are effectively the same structure — "[agent name] for [task type]"
- **CLAUDE.md update rule**: 8/10 are "[action] without updating CLAUDE.md" — same template, different verbs
- **Compaction/artifacts rule**: 9/10 are "saving/writing/persisting X to artifacts/ before Y"
- **tmux send-keys reporting rule**: 7/10 are "[variant of] tmux send-keys for [variant of] reporting"

### Tool Context Coverage for Workflow Rules

**Critical gap**: Workflow rules fire on `UserPromptSubmit` events — meaning the query is a user's natural language message, NOT a tool call. But many expansions look like tool calls or code commands rather than natural language user prompts.

For example, rule "classify every task as SIMPLE/MEDIUM/COMPLEX" should have expansions like:
- "add authentication to the API" (user prompt that needs COMPLEX classification)
- "fix the typo in the README" (user prompt that is clearly SIMPLE)
- "refactor the payment module" (user prompt that needs MEDIUM classification)

Instead, expansions are: "labeling JIRA task as simple medium or complex", "estimating story points with complexity reasoning" — project management actions, not actual user prompts.

---

## Cross-Corpus Patterns

### Pattern 1: Violation vs. Correct Behavior Confusion
The prompt says "actions that should trigger this rule", which the LLM sometimes interprets as "correct actions following the rule" rather than "violations or contexts where the rule is relevant." ~5% of expansions describe the correct behavior.

**Fix**: Add to prompt: "Generate phrases that describe situations where the rule SHOULD BE SURFACED — typically violations, near-violations, or contexts where the developer needs to be reminded."

### Pattern 2: Template Repetition
The LLM falls into templates within a rule. For "CLAUDE.md" rule, it generates "[X] without updating CLAUDE.md" 8 times. For "force-push" it generates "git push [flags] [branch]" 7 times.

**Fix**: Add to prompt: "Each expansion must use DIFFERENT sentence structure. If you find yourself writing '[X] without [Y]' repeatedly, stop and diversify: include the action in a different form, from a different perspective, or in a different context."

### Pattern 3: Workflow Rules Need User Message Examples
The golden examples in the prompt are all coding/tool patterns. Workflow rules need a different kind of expansion — natural language user messages.

**Fix**: Add a workflow-specific golden example:
```
Rule: "Classify every task as SIMPLE, MEDIUM, or COMPLEX before starting"
Good expansions:
- "add authentication to the API" (complex task needing classification)
- "fix the typo on line 42" (clearly simple, but classification should happen)
- "refactor the payment processing module" (medium-complexity signal)
- "build a new microservice for notifications" (complex signal)
Bad expansions:
- "classifying task as SIMPLE or COMPLEX" (restatement)
- "labeling JIRA ticket with complexity" (project management, not user prompt)
```

### Pattern 4: Missing Indirect Triggers
The prompt says "include INDIRECT triggers" but the LLM rarely generates them. For "100% test coverage", no expansion mentions "add a new endpoint" (which implies tests are needed). For "run quality checks before commit", no expansion mentions "I'm done with the feature" (which implies commit is coming).

**Fix**: Add to prompt: "At least 2 of your expansions should be INDIRECT triggers — actions that don't mention the rule topic at all but should still surface it. For 'run tests before commit': 'I'm done, ship it' is an indirect trigger."

### Pattern 5: No Event-Type Awareness
Expansions don't reflect that basic rules fire on PreToolUse (tool context) while workflow rules fire on UserPromptSubmit (user messages). This is the single biggest quality gap.

**Fix**: Consider two prompt variants — one for PreToolUse corpora that emphasizes tool commands, and one for UserPromptSubmit corpora that emphasizes natural language user messages.

---

## Prompt Improvement Suggestions (Priority Order)

### P0: Split Prompt by Event Type
Create separate expansion prompts for PreToolUse rules (tool/code focus) and UserPromptSubmit rules (natural language focus). The golden examples should match the query distribution.

### P1: Add Anti-Template Instruction
"Each expansion must use a DIFFERENT sentence structure. Do not repeat the same template (e.g., '[X] without [Y]') more than twice."

### P2: Clarify Trigger Direction
"Generate phrases describing situations where the rule should be SURFACED — typically violations, contexts needing the reminder, or adjacent actions. Do NOT generate phrases that describe the correct/compliant behavior."

### P3: Require Indirect Triggers
"At least 2 expansions must be INDIRECT triggers — actions that don't mention the rule's topic but should still surface it."

### P4: Add Workflow Golden Example
Add a 4th golden example showing natural language user prompts as expansions for a workflow/process rule.

### P5: Increase Temperature or Nucleus Sampling
The template repetition may be partly due to temperature (currently 0.7). Consider 0.8-0.9 for more diversity, or nucleus sampling (top_p=0.9).

### P6: Post-Generation Dedup Pass
Add a semantic dedup step after generation — if two expansions have cosine similarity > 0.85, drop the shorter one and regenerate. This is cheap and catches the template problem mechanically.

---

## Summary Table

| Metric | Basic | Workflow |
|--------|-------|---------|
| Overall quality | 4/5 | 3/5 |
| % good expansions | ~75% | ~55% |
| Top failure mode | Same-scenario repetition | Paraphrase/restatement |
| Near-duplicate rate | ~12% | ~22% |
| Tool context coverage | Good (missing Bash: prefix) | Poor (missing user message patterns) |
| Indirect triggers | Rare | Almost none |
| Factual errors | 1 (positive direction) | 3 (wrong domain, factual) |
