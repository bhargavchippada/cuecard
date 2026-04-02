# PRD: Unified Event System & Reasoning Prompt — v0.2

> Extends cuecard beyond PreToolUse to support UserPromptSubmit events with a unified retrieval pipeline.

## 1. Objective

Enable cuecard to inject relevant rules for **any** Claude Code hook event — not just tool calls, but also user messages. Support a unified index over multiple rule corpora (coding rules + workflow rules + future .md sources) with a single retrieval pipeline that discriminates by event context.

## 2. Success Criteria

- [ ] UserPromptSubmit events retrieve workflow/process rules with ≥80% easy recall
- [ ] PreToolUse retrieval quality does not regress (≤2% recall drop, ≤5% noise increase)
- [ ] Cross-domain noise ≤25% (workflow rules don't leak into PreToolUse, coding rules don't leak into UserPromptSubmit)
- [ ] Latency p50 ≤1.5s for LLM mode, ≤50ms for embedding-only mode
- [ ] Negative silence ≥90% for both event types
- [ ] 100% test coverage maintained
- [ ] All eval metrics tracked per event type in the eval report

## 3. Background

### Current State (v0.1)
- **Single event**: PreToolUse only
- **Single corpus**: `rules.txt` with coding/security rules
- **Query format**: `{tool_name}: {tool_input}` (e.g., `Bash: git commit`)
- **Index**: Single embedding index over coding rules
- **Eval**: 354 PreToolUse fixtures (was 357, 3 duplicates removed + 8 expectations corrected)

### Gap
Users define rules about process and workflow (complexity assessment, subagent delegation, review convergence, PRD creation) that should surface when they send messages — not when individual tools execute. These rules have no retrieval path today.

### Design Insight
The existing pipeline architecture is event-agnostic. The query string carries the event context. The LLM reranker can distinguish event types. No architectural changes needed — extend, don't rewrite.

## 4. Architecture

### 4.1 Unified Index

All rule sources combine into a single embedding index:

```
sources.rules = [
    "rules.txt",           # coding rules (30 rules)
    "workflow.txt",        # workflow rules (46 rules)
    # Future: "~/.claude/CLAUDE.md", "project/CLAUDE.md"
]

# Single parse → single embed → single index
# Index contains ~76 rules from all sources
```

**Rationale**: When .md file parsing arrives (v0.3), a single CLAUDE.md will contain both coding and workflow rules in different sections. Forcing users to split files is impractical. A unified index with smart retrieval is simpler and more natural.

### 4.2 Event-Aware Queries

The adapter prepends event type to the query:

```python
# PreToolUse events
query = f"{tool_name}: {tool_input}"
# Example: "Bash: git commit -m 'fix auth'"

# UserPromptSubmit events
query = f"UserPromptSubmit: {prompt_text}"
# Example: "UserPromptSubmit: Add authentication to the API"
```

The event type prefix serves two purposes:
1. **Embedding signal**: jina-code can learn that "UserPromptSubmit:" correlates with workflow rules
2. **LLM context**: The reranker system prompt explicitly describes both event types and how to match them

### 4.3 LLM Reranker Prompt

The system prompt is event-type aware:

```
Events have a type prefix:
- "PreToolUse:<tool>: <args>" — a tool is about to execute
- "UserPromptSubmit: <message>" — the user just sent a request

For UserPromptSubmit: match workflow/process rules that guide HOW to
approach the user's request (complexity assessment, planning, delegation)
For PreToolUse: match coding standards and tool-specific rules
```

### 4.4 Reasoning-in-Response

The LLM returns structured reasoning before rule indices:

```json
{
    "reasoning": "The user is asking to implement a complex feature. This requires complexity assessment, PRD creation, and subagent delegation.",
    "rules": [1, 3, 7]
}
```

**Rationale**: Structured CoT without thinking mode. Single LLM call, ~100 extra tokens. Benchmark showed +3 pts overall recall, +7.3 pts hard recall vs non-reasoning prompt on same fixtures. Reasoning is inspectable for debugging.

## 5. Implementation Plan

### Phase 1: Adapter & Prompt (DONE)
- [x] Adapter handles UserPromptSubmit events
- [x] Query prefixed with event type
- [x] LLM prompt updated with event-type awareness
- [x] Reasoning-in-response added to prompt
- [x] `_MAX_TOKENS` set to 1024
- [x] Test coverage for UserPromptSubmit path
- [x] Log `event` field uses actual event type (not hardcoded "PreToolUse")

### Phase 2: Corpus & Fixtures (DONE)
- [x] `eval/corpora/rules_workflow.txt` — 46 workflow rules
- [x] `eval/fixtures/workflow.json` — 84 UserPromptSubmit fixtures
- [x] All fixture references verified against corpus

### Phase 3: Unified Index Support
- [ ] Eval harness: add `--corpus` CLI flag to override fixture-level `corpus` field with a unified corpus path (or comma-separated list). This builds one index from multiple corpora and runs all fixtures against it, enabling cross-domain noise measurement.
- [ ] Eval harness: cache index per unique corpus combination (avoid rebuilding 438 times)
- [ ] Eval report: add per-event-type breakdown when fixtures have mixed event prefixes
- [ ] Eval metric: add `cross_domain_noise` — fraction of results whose source file doesn't match the fixture's primary corpus. Uses existing `Provenance.file` field.
- [ ] Add 2-3 UserPromptSubmit few-shot examples to LLM system prompt
- [ ] Benchmark: unified index (rules_basic.txt + rules_workflow.txt) on all 438 fixtures
- [ ] Benchmark: compare jina-code vs bge-base on workflow fixtures (jina-code may underperform on non-code queries)

### Phase 4: Quality Iteration
- [ ] Benchmark unified corpus on PreToolUse fixtures — verify no regression
- [ ] Benchmark unified corpus on UserPromptSubmit fixtures — target 80%+ easy recall
- [ ] Increase `recall_top_k` from 20→30 if needed for larger corpus (hardcoded in `pipeline.py:119` and `eval.py:_EvalConfig`; make configurable via `[retrieval]` TOML section)
- [ ] Prompt tuning: optimize few-shot examples for both event types
- [ ] Add UserPromptSubmit examples to LLM system prompt

### Phase 5: Production Integration
- [ ] `cuecard install` adds both PreToolUse and UserPromptSubmit hooks
- [ ] Default config includes both event types in `hook_events`
- [ ] Update CLAUDE.md and README
- [ ] Publish v0.2

## 6. Eval Strategy

### Fixture Sets
| Set | Corpus | Fixtures | Event Type |
|-----|--------|----------|------------|
| `basic.json` | `rules_basic.txt` | 354 | PreToolUse |
| `workflow.json` | `rules_workflow.txt` | 84 | UserPromptSubmit |
| **unified** (planned) | both corpora | 438 | Mixed |

### Metrics Tracked Per Event Type
- Recall@k, Precision@k, MRR, nDCG
- Noise ratio, Context waste
- Negative silence rate
- **Cross-domain noise**: fraction of results from the "wrong" corpus
- Latency p50/p95/p99

### Quality Targets
| Tier | Recall | Noise | Silence |
|------|--------|-------|---------|
| Easy (both events) | ≥85% | ≤25% | — |
| Medium | ≥65% | ≤30% | — |
| Hard | ≥40% | ≤35% | — |
| Negative | — | — | ≥90% |

## 7. Risks & Mitigations

### R1: Cross-domain noise
**Risk**: Workflow rules appear in PreToolUse results (or vice versa)
**Mitigation**: LLM reranker prompt explicitly separates event types. The reasoning step forces the model to identify the event type before selecting rules. If noise >25%, add event-type filtering in the prompt.

### R2: Embedding space collision
**Risk**: Workflow rules and coding rules occupy similar regions in embedding space
**Mitigation**: jina-code is trained on code queries. Workflow rules may embed differently. If they overlap, rule augmentation (category prefix) can separate them: `[workflow] Use subagents for complex tasks` vs `[coding] Use type hints on all functions`.

### R3: Latency increase from larger index
**Risk**: 76 rules instead of 30 → slower retrieval
**Mitigation**: Embedding retrieval is O(n) dot product. Going from 30→76 adds ~1ms. Not a concern.

### R4: UserPromptSubmit queries are longer/noisier
**Risk**: User messages are conversational, not structured like tool calls
**Mitigation**: The LLM reranker handles semantic matching well. The reasoning step helps extract intent from noisy input. `query_max_length=500` caps long messages.

## 7b. Rule Taxonomy

**Coding rules** constrain a specific tool action: what code to write, how to format, what to validate, what to avoid. They apply per-action. Example: "Use type hints on all function signatures."

**Workflow rules** guide how to approach a task: planning, delegation, review process, measurement, iteration. They apply per-request. Example: "Classify every task as SIMPLE/MEDIUM/COMPLEX before starting."

**Boundary cases**: "Run tests before committing" is workflow (process guidance) even though it involves a tool. "Use parameterized queries" is coding (per-action constraint) even though it implies a process. The test: does the rule constrain a specific action (coding) or guide overall approach (workflow)?

In a unified index, the LLM reranker uses event type + context to decide relevance — the taxonomy helps corpus authors, not the retrieval system.

## 8. Open Questions

### Q1: RESOLVED → D6
Same injection point (`additionalContext`) for both events. Provides guidelines before the agent acts, which is correct for both tool enforcement and workflow guidance.

### Q2: RESOLVED → Phase 3
Currently: All 5 examples are PreToolUse. Adding 2-3 UserPromptSubmit examples would help the model distinguish event types.
**Decision needed**: Add examples now (Phase 3) or after benchmarking?

### Q3: Config schema for multiple corpora
Currently: `sources.rules = ["path/to/rules.txt"]` — single list.
Options:
  a) Same list, all files go into one index: `sources.rules = ["rules.txt", "workflow.txt"]`
  b) Separate keys: `sources.coding_rules = [...]`, `sources.workflow_rules = [...]`
  c) Keep single list, add provenance tags based on filename/directory

**Recommendation**: Option (a) — simplest, unified. Provenance already tracks source file. The LLM reranker doesn't need to know which file a rule came from — it just needs to match relevance.

## 9. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Unified index, not separate per-event | Simpler config, natural for .md files, LLM reranker handles discrimination |
| D2 | Event type in query prefix | Zero-cost signal for both embedding and LLM stages |
| D3 | Reasoning-in-response, not thinking mode | Single call, lower latency, inspectable, +3-7% recall improvement |
| D4 | `_MAX_TOKENS=1024` | Enough for reasoning + JSON, prevents rambling |
| D5 | Separate corpus files (for now) | Clean separation during eval, merge into single config list for production |
| D6 | Same injection point for both events | `additionalContext` works for both — agent sees guidelines before acting |

## 10. Benchmark Baselines

### PreToolUse (354 fixtures, jina-code, reasoning prompt)
| Metric | Embedding | LLM-local |
|--------|-----------|-----------|
| Recall | 0.406 | 0.422 |
| Noise | 0.644 | 0.214 |
| Neg Silence | 0.236 | 0.929 |
| Precision | 0.237 | 0.362 |
| p50 | 22ms | 1143ms |

### Per-Tier (LLM-local, reasoning prompt)
| Tier | Recall | Noise | Silence |
|------|--------|-------|---------|
| easy | 0.852 | 0.254 | 0.048 |
| medium | 0.660 | 0.303 | 0.145 |
| hard | 0.404 | 0.327 | 0.250 |
| negative | — | 0.071 | 0.929 |

### UserPromptSubmit — TBD (after Phase 3 benchmark)

## 11. Rollback Plan

If the unified index causes regressions in production:
1. Set `hook_events = ["PreToolUse"]` in config to disable UserPromptSubmit
2. Remove workflow rules from `sources.rules` to shrink index back to coding-only
3. Both are config-only changes — no code rollback needed
