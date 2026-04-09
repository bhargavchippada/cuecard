# Session 26 Progress — V2 Rules, Affinity Prompt, Fixture Overhaul

**Date:** 2026-04-08
**Branch:** master

## What Was Done

### 1. V2 Corpus Activated
- `rules_global.txt` swapped from 109 v1 rules to 107 v2 trigger-aware rules
- `rules_global_tagged.json` updated with new distribution: 60 tool_use, 44 workflow, 3 both
- All fixture files verified with v2 rule text (character-for-character match)
- Stale artifacts removed: 9 old fixture files, 4 redundant corpus backups

### 2. Affinity Prompt Rewritten
- **Old**: "Most rules ARE both" → 39-49% accuracy, biased toward `both`
- **New**: "Minimum retrieval scope — surface at earliest moment that corrects the agent" → **93.5% accuracy**
- Key principle: `both` only when agent can plan AROUND the tool event entirely
- Changed response format: singular `"category"` field (backwards-compat parser for `"categories"` list)
- 4 new parser tests added (singular tool_use, workflow, both, precedence)
- Golden examples: 3 tool_use, 3 workflow, 2 both (was 0/3/4)

### 3. Ground Truth Updated (minimum retrieval scope)
- Reclassified 23 rules based on "minimum retrieval scope" principle
- 16 both→tool_use (tool-time is sufficient to correct agent)
- 4 workflow→tool_use (fire at discovery/execution time)
- 3 both→workflow (pure process decisions)
- Kept 3 genuine `both`: 100% coverage, quality checks before commit, TDD
- **Final: 60 tool_use, 44 workflow, 3 both**

### 4. Fixture Overhaul (831 → 1084+)
- **Rebalanced**: All 5 events to 44-49% positive
- **Cross-event consistency enforced**: 191 tool_use rules removed from workflow/stop/subagent fixtures
- **Compliance-as-violation fixed**: 17 fixtures across 5 rounds of verification (converged R4+R5)
- **Stop/SubagentStart rebalanced**: Added workflow-only positives after cross-event cleanup
- **SubagentStart trimmed**: Max 1 rule per positive (was 1-2), compliance-as-violation cleaned
- **Stop Phase 6 format**: IN PROGRESS — rewriting all 135 stop fixtures to `"Stop: User asked: ... | Agent said: ..."`

### 5. Recall Diagnostic
- top_k=5 vs top_k=30: only +0.8% recall improvement
- **Bottleneck is embedding/expansion quality, NOT the LLM reranker**
- The reranker is doing its job — correct rules never reach it from embeddings
- Path forward: better expansions, richer queries (Phase 6), or better embedding model

### 6. Benchmark (v2 rules + ground truth affinity + Gemma E4B, 20% sample seed=42)

| Event | S24 F2 | v2 F2 | Δ | Notes |
|-------|--------|-------|---|-------|
| PreToolUse | 0.566 | 0.573 | +0.7 | Stable, NegSil improved +6.8 pts |
| UserPromptSubmit | 0.618 | 0.649 | +3.1 | Best improvement, noise down |
| PostToolUse | 0.528 | 0.556 | +2.8 | Good gains across all metrics |
| Stop | 0.426 | 0.115 | -31.1 | Expected: minimal queries + correct affinity filtering |
| SubagentStart | 0.248 | 0.219 | -2.9 | Same embedding bottleneck |

### 7. Tests
- 1153 tests passing, 63 affinity tests (4 new)
- ruff clean, mypy clean
- All fixture verification converged (R4+R5 clean)

## Files Modified
- `src/cuecard/affinity.py` — rewritten prompt + singular category parser
- `tests/test_affinity.py` — 4 new tests for singular category format
- `eval/corpora/rules_global.txt` — v2 active (was v1)
- `eval/corpora/rules_global_tagged.json` — 60/44/3 distribution
- `eval/corpora/affinity.json` — ground truth affinity
- `eval/fixtures/basic.json` — 441 fixtures, compliance fixes
- `eval/fixtures/post_tool_use.json` — 131 fixtures, cross-event fix
- `eval/fixtures/workflow.json` — 219 fixtures, rebalanced with workflow rules
- `eval/fixtures/stop.json` — 135 fixtures (Phase 6 rewrite in progress)
- `eval/fixtures/stop_mined.json` — schema + cross-event fixes
- `eval/fixtures/subagent_start.json` — 170 fixtures, trimmed to 1 rule/pos
- `eval/results/gemma-e4b-e2e-seed42.json` — benchmark results

### 8. Reranker Prompt Rewrite + top_k Tuning
- Diagnosed reranker drops: 6/30 correct rules dropped from candidates
- Root cause: hardcoded "exclude process rules for tool calls" + top_k=5 crowding
- Rewrote prompt: 6 reasoning principles, "When in doubt include", 11 examples (was 16)
- Changed top_k default from 5 → 7
- Recall diagnostic confirmed: embedding quality is bottleneck, not reranker (top_k=30 only +0.8%)
- But reranker WAS also dropping 26% of correct candidates — now fixed

### 9. Final Benchmark (new prompt + top_k=7 + Phase 6 Stop + all fixes)

| Event | S24 F2 | Final F2 | Δ | PosRecall | NegSil |
|-------|--------|----------|---|-----------|--------|
| PreToolUse | 0.566 | 0.570 | +0.4 | 0.570 | 0.614 |
| UserPromptSubmit | 0.618 | 0.677 | +5.9 | 0.745 | 0.696 |
| PostToolUse | 0.528 | 0.623 | +9.5 | 0.885 | 0.462 |
| Stop | 0.426 | 0.486 | +6.0 | 0.542 | 0.500 |
| SubagentStart | 0.248 | 0.432 | +18.4 | 0.786 | 0.316 |

### 10. Inline Affinity in rules.json
- Removed sidecar dependency — affinity now stored inline per rule in rules.json
- `save_rules_json` accepts optional `AffinityIndex`, writes affinity dict per entry
- `load_rules_json` returns `(rules, AffinityIndex | None)` tuple
- Loader prefers inline, falls back to sidecar for backwards compat
- Security hardened: KNOWN_HOOK_EVENTS filter, reasoning scrubbed+capped, source validated

### 11. RRF fusion_k Tuning
- Swept k={5, 10, 15, 20, 30, 40, 60} — 20% sample was misleading
- Full sample (441 PreToolUse): k=10 wins on ALL metrics (+1.7 F2, +1.8 recall, -1.6 noise)
- Default changed from 60→10 — sharper fusion for 107-rule corpus

### 12. Configurable llm_candidates
- New config field `llm_candidates` (default 12) controls how many candidates reach LLM reranker
- Wired through config.py, models.py, eval.py, pipeline.py

### 13. Parallel Eval with ThreadPoolExecutor
- 5 workers matching llama-server -np 5 slots
- ~4-5x speedup on full benchmarks
- Sequential fallback for embedding-only mode

### 14. Security Review (converged)
- 3 HIGH + 4 MEDIUM findings from security reviewer
- All fixed: reasoning capped+scrubbed, events validated against KNOWN_HOOK_EVENTS,
  affinity_mode/model validated, tool names capped, bench_e2e label sanitized

## Final Commits (10 total)
1. `b0e2476` — V2 rules, affinity prompt, fixture overhaul
2. `088fc6c` — Reranker prompt rewrite + top_k 5→7
3. `ee84430` — Query-shaped expansion prompt
4. `ebd11bd` — Session progress + benchmark results
5. `5bc99f8` — Inline affinity in rules.json
6. `57a6b6d` — Security fix: validate inline affinity fields
7. `fee3e45` — Security fix: KNOWN_HOOK_EVENTS, scrub reasoning, sanitize label
8. `7f5e961` — Configurable llm_candidates, revert fusion_k to 60
9. `77695bf` — Parallel eval with ThreadPoolExecutor
10. `520fcb0` — fusion_k 60→10 (full-sample confirmed)

## Remaining Work
1. **Expansion quality** — more expansions (MAX 10→12, dedup 0.85→0.80)
2. **Flash Rank / Jina-ColBERT V2** — Stage 2 reranker upgrade (biggest expected impact)
3. **Jina-embeddings-v3** — embedding model upgrade with task LoRA
4. **Phase 6 implementation** — completion gate Stop hook (PRD converged)
5. **PostToolUse/SubagentStart** — event-specific expansion tuning (currently optimized for PreToolUse)
