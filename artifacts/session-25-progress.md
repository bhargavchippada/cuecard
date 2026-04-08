# Session 25 Progress — Fixture Realism & Quality Gap Analysis

**Date:** 2026-04-08
**Branch:** master
**Duration:** Full session

## What Was Done

### 1. Global Rules Expanded (87 → 94 rules)
- Added 7 new rules under "CLASSIFICATION & EVAL" category
- Binary classification, golden examples, compliance-vs-violation, corpus expansion diminishing returns, canonical text form, mining real sessions, post-scoring masks
- All 7 expanded via LLM (94 new expansions)

### 2. Phase 6 Completion Gate PRD (CONVERGED)
- Stop hook blocks agent from stopping early
- Reads `last_assistant_message` + `transcript_path` from Claude Code
- Option A: always block (rules are bonus), up to configurable max_stop_blocks (1-4)
- Shared `execute_stop_gate()` function for inline + daemon
- Atomic counter in `~/.cuecard/state/` with fcntl.flock
- 3 rounds of review (architect + security + code quality) — converged at R3
- PRD at `artifacts/phase6-completion-gate-prd.md` v1.2

### 3. Fixture Realism Overhaul (831 → 984 fixtures)

| Event | Before | After | Key Changes |
|-------|--------|-------|-------------|
| basic | 442 (50/50) | 442 (50/50) | Unchanged |
| post_tool_use | 115 (57/43) | 140 (46/54) | +25 negatives, 3 compliance-as-violation fixes, 5 output realism fixes, 1 duplicate removed |
| stop | 77 (58/42) | 128 (48/52) | Realistic `"Stop: end_turn"` queries, +62 audit positives |
| subagent_start | 73 (77/23) | 94 (52/48) | Trimmed 2.6→1.5 rules/pos, +21 negatives, 7 recategorized |
| workflow | 124 (76/24) | 197 (57/43) | +73 negatives + 18 under-represented rule coverage |

Additional: 41 mined Stop fixtures in Phase 6 format at `eval/fixtures/stop_mined.json`

### 4. Affinity Ground Truth Corrected
- 20 rules re-tagged from `workflow` → `both` (they DO constrain tool actions)
- Affinity accuracy: 67.9% → **86.2%** against corrected ground truth
- Remaining 15 mismatches are genuinely borderline (LLM says `both`, human says `workflow`)
- Distribution: 63 both, 46 workflow, 0 tool_use-only

### 5. Diversity Analysis
- 100% rule coverage (all 109 rules have ≥1 fixture)
- 9 under-represented rules (1 fixture each) → expanded to 3+ each
- "Never commit secrets" over-represented (59 fixtures)
- Near-duplicates identified and cleaned (post_tool_use, basic)
- LLM-specific, infrastructure, methodology domains need more coverage

### 6. Quality Gap Analysis (Discussion)

**Six dimensions identified:**
1. Where should each rule trigger for optimal efficiency? Over-triggering causes context fatigue.
2. Keep binary classification (tool_use/workflow/both) — fix ground truth, not the system.
3. Audit after benchmark: was the model wrong or the fixture expectations?
4. Close gap between fixtures and real application — richer hook context.
5. Focus on one hook at a time — PreToolUse first (highest volume, best context).
6. Rule writing quality matters — rules should describe WHEN they apply, not just WHAT.

**Key insight:** The rule text IS the retrieval signal. Rules that describe WHEN they apply ("When running pip install: use uv instead") retrieve better than rules that just say WHAT ("Use uv not pip"). Adding trigger conditions improves embeddings, classification, AND agent compliance simultaneously.

**Affinity mask analysis:**
- Applied post-scoring but pre-threshold (scores[~mask] = -inf)
- PreToolUse gets 78/109 rules (28% reduction)
- Workflow events get ALL 109 rules (no reduction)
- Current binary affinity helps PreToolUse but doesn't help workflow events

**Pipeline bottleneck hypothesis:**
- Stage 1 (embedding): puts topically-similar rules in same neighborhood
- Stage 4 (LLM reranker): must discriminate 2 right from 10 wrong-but-similar
- Need diagnostic: run with top_k=30 to check if correct rules are even in candidates

## Files Modified
- ~/.cuecard/rules/global.txt (7 new rules)
- eval/corpora/rules_global_tagged.json (20 re-tagged)
- eval/fixtures/post_tool_use.json (140 fixtures)
- eval/fixtures/stop.json (128 fixtures)
- eval/fixtures/subagent_start.json (94 fixtures)
- eval/fixtures/workflow.json (197 fixtures)
- eval/fixtures/stop_mined.json (NEW — 41 mined from real sessions)
- artifacts/phase6-completion-gate-prd.md (NEW — v1.2 converged)
- artifacts/session-25-progress.md (NEW)

### 7. Rule Rewriting (Phase 7 — in progress)
- All 109 rules rewritten with trigger conditions ("When X: do Y — because Z")
- 83 rewritten, 26 kept as-is (already had triggers)
- Merged into `artifacts/rule-rewrites-all-109.json`
- Re-categorized: 48 tool_use, 19 both, 42 workflow (was: 63 both, 46 workflow, 0 tool_use)

### 8. Rule Review (2 subagents)
- **Quality reviewer** (`artifacts/rule-review-quality.md`): 52 GOOD, 25 FIX, 2 REVERT
  - 8 over-specification issues (library names in triggers)
  - 4 triggers too vague (match everything)
  - 2 meaning lost (rules 70/72 became duplicates)
  - 2 duplicate rules to remove (52 duplicate of 49, 70 duplicate of 72)
  - 2 reverts (rules 63, 64 — originals were better)
  - 5 category corrections
- **Category reviewer** (`artifacts/rule-review-categories.md`): 79 CORRECT, 30 CHANGE
  - 13 tool_use → both (quality gates need Stop audit too)
  - 9 both → tool_use (coding patterns don't need workflow events)
  - 5 both → workflow (agent delegation is pure process)
  - Final distribution: ~50 tool_use, ~23 both, ~36 workflow

### 9. Corpus v2 Generated
- `eval/corpora/rules_global_v2.txt` — 107 rules (2 duplicates removed)
- `eval/corpora/rules_global_tagged_v2.json` — 107 rules with categories: 44 tool_use, 44 workflow, 19 both
- `artifacts/rule-rewrites-final.json` — finalized rewrites after quality + category review

### 10. Fixture Update (PARTIAL — needs redo)
- Fixture updater agent reverted to git HEAD (831 fixtures) and applied v2 rule text
- Session 25 fixture expansions (984 fixtures) were lost due to agent overwriting
- The `_new.json` files contain partial batches from earlier expansion agents
- **Next session must:** start from 831, apply v2 rules, THEN re-expand to ~984

## Remaining Work
1. ~~Apply review fixes~~ DONE — `artifacts/rule-rewrites-final.json`
2. **Generate rules_global_v2.txt** — with rewritten rules
3. **Update fixtures** — mechanical find-replace old rule text → new rule text across all fixture files
4. **Re-expand** — generate new expansions using rewritten rules as source
5. **Benchmark** — run E2E with new fixtures + rewritten rules to measure improvement
6. **Recall diagnostic** — top_k=30 to identify retrieval vs reranker bottleneck
7. **Audit mismatches** — after benchmark, check model wrong vs fixture wrong
8. **Phase 6 implementation** — completion gate Stop hook (PRD converged)
