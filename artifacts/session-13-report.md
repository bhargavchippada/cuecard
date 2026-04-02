# Session 13 Report (2026-04-01/02)

## Executive Summary

Session 13 extended cuecard from PreToolUse-only to a unified event system supporting both PreToolUse and UserPromptSubmit hooks. Introduced reasoning-in-response prompting (+3 pts recall), fixed 15 review findings across 3 convergence rounds, created a workflow rule corpus (46 rules) and UserPromptSubmit fixtures (84), and benchmarked the unified index (438 fixtures, 76 rules).

## Commits (8 this session)

| # | Hash | Description |
|---|------|-------------|
| 1 | `ec1d379` | feat: unified event system, reasoning prompt, review fixes |
| 2 | `325486f` | feat: Phase 3 — eval corpus override, index caching, UserPromptSubmit examples |
| 3 | `d68f440` | fix: use actual event type in empty-index log path |
| 4 | `18363e7` | feat: capture LLM reasoning for debugging and eval analysis |
| 5 | `914493b` | docs: update notebook with workflow and unified index evaluation steps |
| 6 | `2f20c29` | docs: update CLAUDE.md and README with session 13 benchmarks |

## Test Status

- **541 tests, 100% coverage** (1803 statements)
- ruff clean, mypy strict clean
- 3 convergence review rounds (Round 2: 1 MEDIUM, Round 3: CLEAN PASS)

## Key Changes

### 1. Reasoning-in-Response Prompt

Changed the LLM system prompt to request structured reasoning before rule indices:

```json
{"reasoning": "The action is a git commit. Rule 1 applies because...", "rules": [1, 2, 3]}
```

- 7 few-shot examples (5 PreToolUse + 2 UserPromptSubmit)
- `_MAX_TOKENS` raised from 512 → 1024
- `LLMParseResult` dataclass captures both indices and reasoning text
- Reasoning logged at DEBUG level for inspection

**Impact (PreToolUse, 354 fixtures):**

| Metric | Old (thinking=OFF) | Reasoning Prompt | Delta |
|--------|-------------------|-----------------|-------|
| Recall | 0.392 | **0.422** | **+0.030** |
| Noise | 0.143 | 0.214 | +0.071 |
| Neg Silence | 0.968 | 0.929 | -0.039 |
| Hard Recall | 0.331 | **0.404** | **+0.073** |
| Medium Recall | 0.597 | **0.660** | **+0.063** |

### 2. UserPromptSubmit Event Support

- Adapter routes by event type: `event == "UserPromptSubmit"` reads `prompt` field
- Query prefixed: `"UserPromptSubmit: add auth to the API"`
- Event validated against `_KNOWN_HOOK_EVENTS` frozenset (prevents spoofing)
- Log `event` field uses actual event type (not hardcoded "PreToolUse")

### 3. Unified Index Design

Single index over multiple corpora, LLM reranker discriminates by event context:
- PRD: `artifacts/unified-events-prd.md`
- Config: `sources.rules = ["rules.txt", "workflow.txt"]`
- Eval: `--corpus-override` flag for unified benchmarking with index caching

### 4. Workflow Corpus & Fixtures

- `eval/corpora/rules_workflow.txt` — 46 workflow/process rules
- `eval/fixtures/workflow.json` — 84 UserPromptSubmit fixtures (22E/23M/15H/24N)
- `eval/fixtures/combined.json` — 438 merged fixtures (354 + 84)
- Sources: CLAUDE.md rules, learned instincts, SOUL.md lessons, session memory

### 5. Review Fixes (15 findings)

**Security:**
- SSRF validation covers `llm-local` and `llm-haiku` modes (was only `rerank-llm-*`)
- Stripe regex fixed: `sk_live_`/`sk_test_` (was `sk-live-` which never matched)
- ACTION field wrapped in `<query_data_{nonce}>` delimiter (closes structural injection)
- Event type validated against known set (prevents spoofing)
- `StageTrace.error` scrubbed via `scrub_secrets()`
- Removed `exc_info=True` from LLM failure log (prevents endpoint exposure)

**Type safety:**
- `PipelineResult.results`: `list` → `tuple[RankedResult, ...]`
- `FreshnessResult.updated_sources`: `dict` → `MappingProxyType`
- Function signatures use `Sequence`/`Mapping` for flexibility
- Removed dead `thinking_budget` parameter

**Other:**
- Log writes protected with `fcntl.flock`
- Reranker bounds check on cross-encoder object-return path
- Parser uses `encoding="utf-8"`
- `secure_open` append path: `O_EXCL` prevents TOCTOU

### 6. Fixture Quality Improvements

- 8 wrong expectations fixed (git-reset-hard, black-format, etc.)
- 3 duplicate queries removed
- 354 fixtures (was 357)

## Benchmark Results

### PreToolUse — Separate Index (354 fixtures, 30 rules)

| Tier | Recall | Noise | Silence | AvgRet |
|------|--------|-------|---------|--------|
| easy | **85.2%** | 25.4% | — | 1.9 |
| medium | **66.0%** | 30.3% | — | 2.1 |
| hard | **40.4%** | 32.7% | — | 1.9 |
| negative | — | 7.1% | **92.9%** | 0.1 |
| **Overall** | **42.2%** | **21.4%** | **92.9%** | 1.3 |

### UserPromptSubmit — Separate Index (84 fixtures, 46 rules)

| Tier | Recall | Noise | Silence | AvgRet |
|------|--------|-------|---------|--------|
| easy | **93.2%** | 25.0% | — | 1.8 |
| medium | 47.1% | 44.6% | — | 2.4 |
| hard | **50.0%** | 25.3% | — | 1.6 |
| negative | — | 4.2% | **95.8%** | 0.0 |
| **Overall** | **46.2%** | **24.5%** | **95.8%** | 1.4 |

### Unified Index (438 fixtures, 76 rules)

| Tier | Recall | Noise | Silence | AvgRet |
|------|--------|-------|---------|--------|
| easy | 80.6% | 35.6% | — | 2.1 |
| medium | 65.1% | 46.5% | — | 2.8 |
| hard | 33.3% | 42.3% | — | 2.1 |
| negative | — | 11.3% | 88.7% | 0.2 |
| **Overall** | 41.1% | 31.6% | 88.7% | 1.6 |

### Cross-Domain Comparison

| Metric | PreToolUse (sep) | Workflow (sep) | Unified |
|--------|-----------------|---------------|---------|
| Recall | 42.2% | 46.2% | 41.1% |
| Noise | **21.4%** | **24.5%** | 31.6% |
| Neg Silence | **92.9%** | **95.8%** | 88.7% |
| p50 | 1143ms | 3008ms | 1377ms |

**Finding:** Unified index increases cross-domain noise by ~10 pts. The LLM reranker maintains recall but returns more irrelevant rules from the "wrong" domain. Separate corpora produce cleaner results. The unified approach needs stronger event-type filtering in the prompt or rule augmentation (category prefix) to match separate-corpus quality.

## Embedding-Only Baselines

### PreToolUse (354 fixtures, jina-code)
| Tier | Recall | Noise | Silence |
|------|--------|-------|---------|
| easy | 90.2% | 50.3% | 1.6% |
| medium | 61.7% | 60.5% | 3.4% |
| hard | 32.7% | 60.7% | 14.6% |
| negative | — | 76.4% | 23.6% |

### UserPromptSubmit (84 fixtures, jina-code)
| Tier | Recall | Noise | Silence |
|------|--------|-------|---------|
| easy | 81.8% | 79.1% | 0% |
| medium | 13.0% | 93.9% | 0% |
| hard | 36.7% | 89.3% | 0% |
| negative | — | 95.8% | 4.2% |

**Finding:** jina-code embeddings handle PreToolUse well but produce massive noise for UserPromptSubmit (90%). The LLM reranker is essential for workflow queries — it transforms noise from 90% → 25% and silence from 4% → 96%.

## Architecture Decisions (Session 13)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Unified index, not separate per-event | Simpler config, natural for .md files, LLM reranker handles discrimination |
| D2 | Event type in query prefix | Zero-cost signal for both embedding and LLM stages |
| D3 | Reasoning-in-response, not thinking mode | Single call, lower latency, inspectable, +3-7% recall |
| D4 | `_MAX_TOKENS=1024` | Enough for reasoning + JSON, prevents rambling |
| D5 | Nonce delimiter on ACTION field | Prevents structural injection from user messages |
| D6 | Same injection point for both events | `additionalContext` works for both event types |
| D7 | `LLMParseResult` captures reasoning | Enables debugging and error analysis |

## What's Next

### Quality Iteration
1. **Prompt tuning for unified index** — stronger event-type filtering to reduce cross-domain noise (31.6% → <25%)
2. **Rule augmentation** — category prefix `[coding]`/`[workflow]` to help embeddings separate domains
3. **recall_top_k increase** — 20→30 for larger unified corpus
4. **Benchmark jina-code vs bge-base** on workflow fixtures

### Features
5. **Phase 4: Publish** — PyPI, GitHub CI, README with badges
6. **Markdown parsing** (v0.3) — support CLAUDE.md as a rule source
7. **Per-corpus eval metrics** — breakdown by source file in eval report
8. **Reasoning in eval output** — show LLM reasoning for failed fixtures

### Production
9. **Dogfood** — install as Claude Code hook and use in real work
10. **`cuecard install`** — add UserPromptSubmit hook alongside PreToolUse
