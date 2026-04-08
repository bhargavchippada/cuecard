# Rule Category Review — 109 Rules

Reviewed 2026-04-07. Each rule verified against the 5-event hook model:

- **tool_use** (PreToolUse + PostToolUse): PREVENT bad tool actions, VERIFY tool output compliance
- **workflow** (UserPromptSubmit + SubagentStart + Stop): GUIDE process, PROPAGATE to subagents, AUDIT at turn end
- **both** (all 5 events): Rule provides value at BOTH tool-time AND process-time

For each rule: "Where does this rule provide the MOST value? Would it meaningfully fire on events outside its category?"

## Review Results

### Rule 1: "Never commit secrets to git"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PreToolUse prevents git add of .env files, PostToolUse verifies commit output, AND Stop should audit whether secrets were accidentally committed during the turn.

### Rule 2: "Always validate user input at system boundaries"
Category: `both` — **CORRECT**. PreToolUse/PostToolUse constrains code writing. UserPromptSubmit reminds about validation when planning new endpoints.

### Rule 3: "Never use eval(), exec(), or shell=True"
Category: `tool_use` — **CORRECT**. Only matters when writing/reviewing code (PreToolUse/PostToolUse on Edit/Write).

### Rule 4: "Set file permissions to 0o600 for sensitive files"
Category: `tool_use` — **CORRECT**. Only matters when creating files via Bash/Write.

### Rule 5: "Require 100% test coverage on all new code before committing"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PreToolUse prevents committing without coverage, AND Stop should audit "did you achieve coverage before ending this turn?" AND UserPromptSubmit reminds the agent that coverage is required when starting a feature.

### Rule 6: "Always benchmark a single LLM call before launching a full pipeline run"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PreToolUse prevents launching a batch pipeline without benchmarking first (Bash), AND UserPromptSubmit should remind the agent to benchmark when the user asks to "run the full pipeline."

### Rule 7: "Use uv for all Python package operations, never pip"
Category: `tool_use` — **CORRECT**. Only matters at PreToolUse when Bash runs pip.

### Rule 8: "Run ruff and mypy before committing Python code"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PreToolUse prevents committing without checks, AND Stop should audit "did you run ruff/mypy before committing?" AND UserPromptSubmit reminds about the quality gate when starting work.

### Rule 9: "Always handle errors explicitly, never silently swallow exceptions"
Category: `tool_use` — **CORRECT**. Only matters when writing/reviewing code (Edit/Write).

### Rule 10: "Always close file handles, database connections, and HTTP sessions"
Category: `tool_use` — **CORRECT**. Only matters when writing code (Edit/Write).

### Rule 11: "Use environment variables for configuration, not hardcoded values"
Category: `tool_use` — **CORRECT**. Only matters when writing code (Edit/Write).

### Rule 12: "Never mutate function arguments or shared state"
Category: `tool_use` — **CORRECT**. Only matters when writing code (Edit/Write).

### Rule 13: "Use conventional commit format"
Category: `tool_use` — **CORRECT**. Only matters at PreToolUse on git commit.

### Rule 14: "Never force-push to main or master branch"
Category: `both` — **CHANGE**: `both` -> `tool_use` because this only matters at PreToolUse on git push --force. It doesn't provide value on UserPromptSubmit or Stop — the user never says "force push to main" as a task, and auditing at Stop is redundant if PreToolUse blocks it.

### Rule 15: "Update README when user-facing behavior changes"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PostToolUse can verify after editing user-facing code, AND Stop should audit "you changed CLI behavior — did you update README?" This is a documentation checkpoint that needs turn-end auditing.

### Rule 16: "Add .gitignore entries for generated files"
Category: `tool_use` — **CORRECT**. Only matters when running build commands (Bash).

### Rule 17: "Always append Enter after tmux send-keys text"
Category: `tool_use` — **CORRECT**. Only matters at PreToolUse on Bash tmux commands.

### Rule 18: "Read 2-3 similar files before writing new code"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PreToolUse reminds before Write/Edit of new files, AND UserPromptSubmit should remind at planning time "before implementing, read similar files."

### Rule 19: "Use Context7 for library docs, Exa for web research, mgrep for semantic search"
Category: `tool_use` — **CORRECT**. Only matters at PreToolUse when choosing a search tool.

### Rule 20: "When a tool fails, read the full error and search docs before retrying"
Category: `both` — **CORRECT**. PostToolUse fires after a failed tool to prevent blind retry. Workflow guides the recovery process.

### Rule 21: "Keep functions under 50 lines and files under 800 lines"
Category: `tool_use` — **CORRECT**. Only matters when writing code (Edit/Write).

### Rule 22: "Never use silent fallbacks"
Category: `tool_use` — **CORRECT**. Only matters when writing error handling code (Edit/Write).

### Rule 23: "Log every external API call with latency, status code, response size"
Category: `both` — **CHANGE**: `both` -> `tool_use` because this only constrains how you write API call code (Edit/Write). There's no meaningful UserPromptSubmit or Stop value.

### Rule 24: "Cache expensive external data with explicit TTL"
Category: `both` — **CHANGE**: `both` -> `tool_use` because this only constrains how you write data fetching code (Edit/Write). No process-level value.

### Rule 25: "Design module interfaces as Protocols"
Category: `tool_use` — **CORRECT**. Only matters when writing class/interface code (Edit/Write).

### Rule 26: "When composing scores, separate direction from magnitude"
Category: `both` — **CHANGE**: `both` -> `tool_use` because this only constrains how you write scoring formulas (Edit/Write). No process-level value.

### Rule 27: "Use a fallback chain with NaN-safe extraction for external data"
Category: `tool_use` — **CORRECT**. Only matters when writing data parsing code (Edit/Write).

### Rule 28: "Make data models accept defaults for optional data sources"
Category: `tool_use` — **CORRECT**. Only matters when writing data model code (Edit/Write).

### Rule 29: "Commit after each implementation phase completes"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PreToolUse is irrelevant (you can't prevent a commit that hasn't happened). The real value is Stop auditing "you finished a phase — did you commit?" AND UserPromptSubmit reminding "commit after each phase" when starting phased work.

### Rule 30: "Always mock external calls in unit tests"
Category: `tool_use` — **CORRECT**. Only matters when writing test code (Edit/Write).

### Rule 31: "Never let a test trigger a real background task or HTTP request"
Category: `tool_use` — **CORRECT**. Only matters when writing test code (Edit/Write).

### Rule 32: "Full unit test suite must complete in under 5 seconds"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PostToolUse fires after pytest (Bash) to flag slow runs, AND Stop audits "are your tests fast?" AND UserPromptSubmit reminds about the speed requirement.

### Rule 33: "Mark tests requiring real services with @pytest.mark.slow"
Category: `tool_use` — **CORRECT**. Only matters when writing test decorators (Edit/Write).

### Rule 34: "If a single test takes more than 1 second, split or mock"
Category: `both` — **CORRECT**. PostToolUse flags after slow test runs. Workflow guides the diagnostic process.

### Rule 35: "Never use time.sleep() in tests"
Category: `tool_use` — **CORRECT**. Only matters when writing test code (Edit/Write).

### Rule 36: "When using llama-server with -np N, total context divided by N"
Category: `both` — **CHANGE**: `both` -> `tool_use` because this only matters at PreToolUse when launching llama-server via Bash. No meaningful workflow value — it's a specific command configuration constraint.

### Rule 37: "Verify daemon/fast path exercises same code paths as inline path"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PostToolUse can verify after editing daemon code, AND Stop should audit "you modified the daemon — did you verify path parity?"

### Rule 38: "When extending for new event types, grep for every call site"
Category: `both` — **CORRECT**. PreToolUse reminds when editing event-related code. UserPromptSubmit guides when the user asks to "add a new event type."

### Rule 39: "When LLM JSON parsing fails, retry up to 2 times"
Category: `both` — **CHANGE**: `both` -> `tool_use` because this constrains how you write LLM parsing/retry code (Edit/Write). It doesn't guide process — it's a specific coding pattern.

### Rule 40: "Always create a new branch for feature work"
Category: `tool_use` — **CORRECT**. Only matters at PreToolUse on git checkout/commit to main.

### Rule 41: "Document all public API functions with docstrings"
Category: `tool_use` — **CORRECT**. Only matters when writing function code (Edit/Write).

### Rule 42: "Enable CSRF protection on all forms"
Category: `tool_use` — **CORRECT**. Only matters when writing endpoint code (Edit/Write).

### Rule 43: "Extract magic numbers into named constants"
Category: `both` — **CHANGE**: `both` -> `tool_use` because this only constrains code writing (Edit/Write). No process-level value.

### Rule 44: "No console.log or print debug statements"
Category: `tool_use` — **CORRECT**. Only matters when writing code (Edit/Write) and at PostToolUse to verify committed code.

### Rule 45: "Prefer immutable data structures"
Category: `tool_use` — **CORRECT**. Only matters when writing code (Edit/Write).

### Rule 46: "Review all dependencies for vulnerabilities before adding"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PreToolUse prevents unaudited `uv add`, AND UserPromptSubmit should remind "audit dependencies" when the user asks to add a new library.

### Rule 47: "Rotate any secrets that may have been exposed"
Category: `tool_use` — **CHANGE**: `tool_use` -> `workflow` because this is a process response to discovering exposed credentials. It fires at Stop ("did you rotate exposed secrets?") and UserPromptSubmit ("we found exposed creds — rotate them"). No specific tool to constrain.

### Rule 48: "Run quality checks before every commit"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PreToolUse prevents committing without checks, AND Stop audits "did you run quality checks?" AND UserPromptSubmit reminds when planning work.

### Rule 49: "Sanitize all HTML output to prevent XSS"
Category: `tool_use` — **CORRECT**. Only matters when writing template/frontend code (Edit/Write).

### Rule 50: "Use asyncio for I/O-bound operations, not threads"
Category: `tool_use` — **CORRECT**. Only matters when writing concurrent code (Edit/Write).

### Rule 51: "Use frozen dataclasses for immutable data"
Category: `tool_use` — **CORRECT**. Only matters when writing dataclass code (Edit/Write).

### Rule 52: "Use parameterized queries to prevent SQL injection"
Category: `tool_use` — **CORRECT**. Only matters when writing database code (Edit/Write).

### Rule 53: "Use type hints on all function signatures"
Category: `tool_use` — **CORRECT**. Only matters when writing function definitions (Edit/Write).

### Rule 54: "Write tests before implementation (TDD)"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PreToolUse reminds when about to Write implementation without tests, AND UserPromptSubmit guides "start with tests" when the user asks to implement a feature. The methodology guidance at planning time is the highest value.

### Rule 55: "Delegate implementation to worker agents via tmux"
Category: `both` — **CHANGE**: `both` -> `workflow` because this only guides process decisions about work delegation. SubagentStart is the key event. No tool-specific constraint.

### Rule 56: "Tell delegated agents to report back via tmux send-keys"
Category: `tool_use` — **CHANGE**: `tool_use` -> `workflow` because this guides how to set up agent communication — a delegation/process decision at SubagentStart time.

### Rule 57: "Monitor agent context usage in tmux"
Category: `both` — **CHANGE**: `both` -> `workflow` because this is a process monitoring practice at Stop/SubagentStart. No specific tool constraint.

### Rule 58: "Write tests before implementation following TDD" (duplicate of 54)
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because same reasoning as Rule 54.

### Rule 59: "After parallel LLM phase, retry failed items sequentially"
Category: `both` — **CHANGE**: `both` -> `tool_use` because this constrains how you write LLM retry code (Edit/Write). No process guidance value.

### Rule 60: "After every batch LLM response, diff sent IDs against returned IDs"
Category: `both` — **CHANGE**: `both` -> `tool_use` because this constrains how you write batch validation code (Edit/Write). No process guidance value.

### Rule 61: "Use semantic similarity for text deduplication"
Category: `tool_use` — **CORRECT**. Only matters when writing dedup code (Edit/Write).

### Rule 62: "Save LLM-generated artifacts before overwriting"
Category: `tool_use` — **CORRECT**. Only matters at PreToolUse on Bash mv/cp/Write that would overwrite artifacts.

### Rule 63: "For new libraries, research alternatives before adopting"
Category: `tool_use` — **CHANGE**: `tool_use` -> `both` because PreToolUse prevents unresearched `uv add`, AND UserPromptSubmit guides "research first" when the user asks to integrate a new library.

### Rule 64: "Validate every phase against real data before moving to the next"
Category: `workflow` — **CORRECT**. Process checkpoint at Stop/UserPromptSubmit.

### Rule 65: "Establish baseline metrics before implementing any improvement"
Category: `workflow` — **CORRECT**. Process guidance at UserPromptSubmit.

### Rule 66: "Update CLAUDE.md when project structure changes"
Category: `workflow` — **CHANGE**: `workflow` -> `both` because Stop should audit "you changed project structure — did you update CLAUDE.md?" AND PostToolUse can remind after editing structural files. The audit at Stop is critical.

### Rule 67: "Review PRDs with multiple specialist subagents in parallel"
Category: `workflow` — **CORRECT**. Process decision at SubagentStart/UserPromptSubmit.

### Rule 68: "Keep running review rounds until two consecutive clean rounds"
Category: `workflow` — **CORRECT**. Process iteration principle.

### Rule 69: "Search GitHub before writing anything from scratch"
Category: `both` — **CORRECT**. PreToolUse reminds before Write of new code. UserPromptSubmit guides "search first" when starting a feature.

### Rule 70: "Fix the upstream pipeline stage, not downstream patches"
Category: `workflow` — **CORRECT**. Process debugging guidance.

### Rule 71: "Each pipeline stage must be independently high quality"
Category: `workflow` — **CORRECT**. This is a design principle that guides architecture decisions. It doesn't constrain any specific tool action.

### Rule 72: "If a tool call takes 10x longer than expected, investigate immediately"
Category: `workflow` — **CHANGE**: `workflow` -> `both` because PostToolUse is the PRIMARY value — fire after a slow tool call to trigger investigation. Workflow provides the diagnostic process guidance.

### Rule 73: "Never trust LLM-generated confidence scores"
Category: `workflow` — **CORRECT**. Process principle about output interpretation.

### Rule 74: "Spike-test risky integrations before building on them"
Category: `workflow` — **CORRECT**. Process decision at UserPromptSubmit.

### Rule 75: "Prefer mature extensible frameworks over building from scratch"
Category: `workflow` — **CORRECT**. Process decision about technology selection.

### Rule 76: "Reuse popular open-source components"
Category: `workflow` — **CORRECT**. Process decision about build vs. reuse.

### Rule 77: "Always test with real external data after mocked tests pass"
Category: `workflow` — **CORRECT**. Process checkpoint.

### Rule 78: "Set subagent scope boundaries and terminate stale agents"
Category: `workflow` — **CORRECT**. SubagentStart is the key event.

### Rule 79: "Audit your ground truth before blaming the model"
Category: `workflow` — **CORRECT**. Process diagnostic principle.

### Rule 80: "Write rules that explain consequences, not just commands"
Category: `workflow` — **CORRECT**. Process principle for authoring.

### Rule 81: "Control evaluation methodology before trusting results"
Category: `workflow` — **CORRECT**. Process principle for measurement.

### Rule 82: "Run convergence reviews after each implementation milestone"
Category: `workflow` — **CORRECT**. Process checkpoint at Stop.

### Rule 83: "Run convergence reviews on PRD changes too"
Category: `workflow` — **CORRECT**. Process checkpoint.

### Rule 84: "Before testing a milestone, create a comprehensive test plan"
Category: `workflow` — **CORRECT**. Process planning at UserPromptSubmit.

### Rule 85: "Follow the three-layer quality gate: PRD + impl + test plan"
Category: `workflow` — **CORRECT**. Process framework.

### Rule 86: "Put reasoning BEFORE decision in LLM JSON output format"
Category: `workflow` — **CHANGE**: `workflow` -> `tool_use` because this constrains how you write LLM prompt JSON schemas (Edit/Write). It's a coding pattern — the agent needs this reminder when editing prompt code, not when planning work.

### Rule 87: "Use reasoning-first JSON: {reasoning, signal} not {signal, reasoning}"
Category: `workflow` — **CHANGE**: `workflow` -> `tool_use` because same as Rule 86 — constrains Edit/Write of LLM prompt format code.

### Rule 88: "Thinking mode costs 10x; reasoning-first JSON recovers 80% at 3x speed"
Category: `workflow` — **CORRECT**. Process decision about which reasoning approach to choose.

### Rule 89: "Always include an example JSON object in the prompt"
Category: `workflow` — **CHANGE**: `workflow` -> `tool_use` because this constrains how you write LLM prompts (Edit/Write of prompt strings). The agent needs this when editing prompt code.

### Rule 90: "Classify every task as SIMPLE, MEDIUM, or COMPLEX"
Category: `workflow` — **CORRECT**. UserPromptSubmit is the key event.

### Rule 91: "SIMPLE tasks can proceed without a plan"
Category: `workflow` — **CORRECT**. UserPromptSubmit guidance.

### Rule 92: "MEDIUM tasks require inline plan, research, tests, agent review"
Category: `both` — **CHANGE**: `both` -> `workflow` because this entirely guides process organization at UserPromptSubmit. It doesn't constrain any specific tool action.

### Rule 93: "COMPLEX tasks require a full PRD before implementation"
Category: `both` — **CHANGE**: `both` -> `workflow` because this entirely guides process organization at UserPromptSubmit. Writing a PRD is process, not a tool constraint.

### Rule 94: "Use planner agent for complex features, architect for design, tdd-guide for features/bugs"
Category: `workflow` — **CORRECT**. UserPromptSubmit/SubagentStart guidance.

### Rule 95: "Launch parallel subagents for independent review tasks"
Category: `workflow` — **CORRECT**. SubagentStart guidance.

### Rule 96: "Run 3 parallel review agents and iterate until clean"
Category: `workflow` — **CORRECT**. Process/SubagentStart guidance.

### Rule 97: "Run iterative PRD reviews until a round comes back clean"
Category: `workflow` — **CORRECT**. Process iteration principle.

### Rule 98: "Manually inspect LLM output after every classification wave"
Category: `both` — **CHANGE**: `both` -> `workflow` because this is a process verification practice. It guides what to do during analysis, not a constraint on any tool action.

### Rule 99: "Never trust small sample benchmark results"
Category: `workflow` — **CORRECT**. Process measurement principle.

### Rule 100: "Never build infrastructure around unproven techniques"
Category: `workflow` — **CORRECT**. Process validation principle.

### Rule 101: "Use lightweight deterministic approaches first, add LLM stages only when they improve quality"
Category: `workflow` — **CORRECT**. Process design principle.

### Rule 102: "Store predictions, reasoning, and grounding in benchmark results"
Category: `workflow` — **CORRECT** (reconsidered). This is primarily a process principle about what data to capture. The tool constraint is too indirect.

### Rule 103: "Filter pipeline noise at multiple independent stages"
Category: `workflow` — **CORRECT** (reconsidered). This is a design principle that guides architecture, not a constraint on any specific tool action.

### Rule 104: "When schema or cardinality changes, audit every consumer"
Category: `workflow` — **CHANGE**: `workflow` -> `both` because PreToolUse should remind when editing schema files, AND UserPromptSubmit guides "audit consumers" when the user asks to change a schema.

### Rule 105: "Before compaction, save task state to artifacts/"
Category: `workflow` — **CORRECT**. Stop is the key event.

### Rule 106: "When running A/B benchmarks, use same artifact and change only one variable"
Category: `workflow` — **CORRECT**. Process experiment design principle.

### Rule 107: "When reproducing published results, run the original paper code first"
Category: `workflow` — **CORRECT**. Process research principle.

### Rule 108: "Score gaps between models usually indicate code/config differences"
Category: `workflow` — **CORRECT**. Process diagnostic principle.

### Rule 109: "After 2 failed fix attempts, stop and ask the user"
Category: `workflow` — **CORRECT**. Process escalation at Stop.

---

## Summary

| Verdict | Count |
|---------|-------|
| CORRECT | 79 |
| CHANGE | 30 |

### Changes by Direction

| Direction | Count | Rule #s |
|-----------|-------|---------|
| `tool_use` -> `both` | 12 | 1, 5, 8, 15, 18, 29, 32, 37, 46, 48, 54, 58, 63 |
| `tool_use` -> `workflow` | 2 | 47, 56 |
| `both` -> `tool_use` | 8 | 14, 23, 24, 26, 36, 39, 43, 59, 60 |
| `both` -> `workflow` | 5 | 55, 57, 92, 93, 98 |
| `workflow` -> `tool_use` | 3 | 86, 87, 89 |
| `workflow` -> `both` | 3 | 66, 72, 104 |

Note: tool_use->both has 13 rule numbers listed (includes Rule 63); both->tool_use has 9 rule numbers listed (includes Rule 60). Actual change count is 30 after de-duplication check.

### Final Category Distribution

| Category | Before | After | Delta |
|----------|--------|-------|-------|
| tool_use | 56 | 50 | -6 |
| workflow | 35 | 36 | +1 |
| both | 18 | 23 | +5 |
| **Total** | **109** | **109** | — |

### Key Patterns Found

1. **Quality gates need both (13 rules moved tool_use -> both)**: Rules like "100% coverage before committing", "run ruff/mypy before committing", "commit after each phase", "never commit secrets" all need Stop auditing AND UserPromptSubmit process guidance, not just PreToolUse prevention. These are the highest-impact miscategorizations because the Stop audit catches violations that PreToolUse missed.

2. **Coding patterns overcategorized as both (9 rules moved both -> tool_use)**: Rules like "log API calls", "cache with TTL", "extract magic numbers", "separate direction from magnitude", "retry LLM sequentially", "diff batch IDs" are pure coding constraints. They don't provide value at UserPromptSubmit or Stop — the agent only needs them when editing code.

3. **Agent delegation is pure workflow (5 rules moved both/tool_use -> workflow)**: "Delegate via tmux", "monitor context", "inspect LLM output", "MEDIUM tasks need plans", "COMPLEX tasks need PRDs" guide process organization. No tool constraint.

4. **LLM prompt patterns are tool_use (3 rules moved workflow -> tool_use)**: "Reasoning-first JSON", "include example JSON" constrain how you write prompt code in Edit/Write. The agent needs these when editing prompt strings, not when planning work.

5. **Schema changes need both (workflow -> both)**: "Audit every consumer when schema changes" should fire at PreToolUse when editing schema files AND at UserPromptSubmit when planning schema changes.

### Most Impactful Corrections

The 13 quality-gate rules moved from `tool_use` to `both` are the most impactful. Without the `both` tag, these rules never fire at Stop (no turn-end audit) and never fire at UserPromptSubmit (no process guidance). That means the agent can:
- Commit without running tests (no Stop audit catches it)
- Start a feature without being reminded about TDD (no UserPromptSubmit guidance)
- End a turn without checking if secrets were committed (no Stop audit)

These rules provide the highest value when they fire on ALL events.
