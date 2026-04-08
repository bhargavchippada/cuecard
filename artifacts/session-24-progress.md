# Session 24 Progress — Fixture Expansion & Eval Corpus

**Date:** 2026-04-07
**Branch:** master
**Duration:** Full session

## What Was Done

### 1. SOUL.md Updated
- Added "What the Twenty-Fourth Arc Taught Me" (closed-loop hooks session)
- Updated Principles section with infrastructure lessons
- Added 3 new global rules to ~/.cuecard/rules/global.txt

### 2. Eval Corpus Expanded (32 → 109 rules)
- Merged ~/.cuecard/rules/global.txt (65 rules) into eval corpus
- Added 15 missing basic rules with explanation suffixes
- Added 29 workflow-specific rules (task classification, agent orchestration, LLM pipeline)
- Final: eval/corpora/rules_global.txt with 109 rules

### 3. Fixture Expansion (500 → 831 fixtures)
- PostToolUse: 36 → 115 (48 negatives, was 8)
- Stop: 30 → 77 (30 negatives, was 8)
- SubagentStart: 30 → 73 (23 negatives, was 7)
- PreToolUse: 354 → 442 (coverage batches + mined sessions)
- UserPromptSubmit: 84 → 124 (coverage batches + mined sessions)
- 73 fixtures mined from real alphaloom + cuecard sessions (64% negative)
- 120 fixtures from coverage batches (45 uncovered rules → all covered)

### 4. Fixture Verification (3 rounds converged)
- **Round 0 (audit):** 14 wrong expectations, 15+ missing, 7 difficulty mismatches
- **Round 1:** 3 parallel verification agents found:
  - Critical: 15 old rules missing from expanded corpus (added)
  - Critical: ALL workflow fixtures used old rule text (1103 remaps applied)
  - 22 SubagentStart additions, 12 PreToolUse additions
  - 21 workflow additions for new global rules
  - sub-doc-updater reclassified negative → positive
- **Round 2:** 58 specific fixture changes applied, spot-check of 20 random fixtures clean
- Final: 0 unmatched rule references across 831 fixtures

### 5. Benchmarks (4 iterations)

| Iter | Corpus | Fixtures | PreToolUse F2 | PostToolUse F2 | Stop F2 | SubagentStart F2 | Workflow F2 |
|------|--------|----------|---------------|----------------|---------|-------------------|-------------|
| 1 | 32 rules (basic) | ~500 | 0.759 | 0.717 | 0.877 | 0.605 | 0.661 |
| 2 | 109 rules | 831 | 0.633 | 0.423 | 0.611 | 0.332 | 0.658 |
| 3 | 109 rules (fixed) | 831 | 0.538 | 0.457 | 0.609 | 0.276 | 0.580 |
| 4 | 109 rules (fresh exp) | 831 | 0.550 | 0.564 | 0.585 | 0.338 | 0.581 |

### 6. bench_e2e.py Updated
- SOURCE_FILES now use rules_global.txt (was rules_basic.txt/rules_workflow.txt)

## Key Findings

1. **3.4x corpus expansion causes ~30% F2 drop** — expected, noise increases with more candidate rules
2. **Fixture quality is the foundation** — wrong expectations are indistinguishable from model failures
3. **Rule text exact matching is critical** — short-form vs long-form (with `— explanation`) silently breaks all evaluations
4. **Real sessions produce better negatives** — 64% of mined fixtures are negative (real dev work is mostly benign)
5. **Compliance vs violation confusion** is the #1 fixture authoring error — output showing correct behavior was matched as violation

## Files Modified
- ~/.claude/rules/SOUL.md (24th arc)
- ~/.cuecard/rules/global.txt (3 new rules)
- eval/corpora/rules_global.txt (NEW — 109 rules)
- eval/fixtures/basic.json (354 → 442)
- eval/fixtures/post_tool_use.json (36 → 115)
- eval/fixtures/stop.json (30 → 77)
- eval/fixtures/subagent_start.json (30 → 73)
- eval/fixtures/workflow.json (84 → 124)
- eval/fixtures/combined.json (rebuilt)
- eval/fixtures/mined-sessions-v3.json (NEW — 73 mined)
- eval/fixtures/coverage_batch1.json (NEW — 48)
- eval/fixtures/coverage_batch2.json (NEW — 72)
- tools/bench_e2e.py (corpus references updated)
- artifacts/fixture-audit-findings.md (NEW)
- artifacts/fixture-verify-r1-*.md (3 reports)
- artifacts/fixture-verify-r2-summary.md (NEW)

### 6. LLM Reranker Prompt Tuning (R1-R4)
- R1: Replaced "when in doubt, include" with category-aware guidance (concrete=include, process=exclude on PreToolUse/PostToolUse)
- R2: Added 3 new negative few-shot examples (process-excluded-from-pytest, LLM-excluded-from-edit, 8-candidate-discrimination)
- R3: Reduced LLM candidate count from top_k=20/threshold=0.20 to top_k=12/threshold=0.25
- R4: Added RULE CATEGORIES section mapping rule types to event types
- Refined: Process rules allowed for Stop/SubagentStart (audit/delegation), excluded from PreToolUse/PostToolUse

### 7. Benchmarks (6 iterations)

| Iter | Changes | PreToolUse F2 | Workflow F2 | PostToolUse F2 | Stop F2 | SubagentStart F2 |
|------|---------|---------------|-------------|----------------|---------|-------------------|
| 1 | 32 rules baseline | 0.759 | 0.661 | 0.717 | 0.877 | 0.605 |
| 4 | 109 rules, fresh exp | 0.550 | 0.581 | 0.564 | 0.585 | 0.338 |
| 5 | + prompt tuning R1-R4 | 0.582 | **0.661** | 0.523 | 0.507 | 0.285 |
| 6 | + refined categories | **0.598** | 0.640 | 0.548 | 0.459 | 0.312 |

Best noise: PreToolUse 0.409 (iter6), Workflow 0.312 (iter5/6)
Best NegSil: Workflow 1.000 (iter5/6), PreToolUse 0.667 (iter6)

### 8. Fixture Verification Round 3 (compliance-as-violation)
- Found 31 fixtures where compliance was treated as violation (the #1 error pattern)
- 25 positives converted to negatives (ruff-check, new-branch, async-code, etc.)
- 8 fixes in workflow/stop/subagent_start (rule replacements, topic mismatches)
- Post-R3: 486 pos → 486 pos (some R3b changes offset), 345 neg

### 9. Binary Affinity Classification
- Replaced 5-event classification with 2-category: tool_use → PreToolUse+PostToolUse, workflow → UserPromptSubmit+SubagentStart+Stop
- 7 golden examples in affinity prompt for correct reasoning
- Result: 78 both, 31 workflow-only, 0 tool-only
- PreToolUse/PostToolUse get 78/109 rules (28% filtered), others get all 109
- Blocked expected rules: PreToolUse 1%, PostToolUse 2%, others 0%

### 10. Event Mask A/B Test (40% sample, stable)
- With mask: PreToolUse F2=0.566, Workflow F2=0.618
- Without mask: PreToolUse F2=0.535, Workflow F2=0.634
- Mask helps PreToolUse (+0.031), PostToolUse (+0.034), SubagentStart (+0.015)

### Final Benchmark (40% sample, R3 fixes, with mask)
| Event | F2 | PosRecall | Noise | NegSil |
|-------|-----|-----------|-------|--------|
| PreToolUse | 0.566 | 0.319 | 0.408 | 0.523 |
| UserPromptSubmit | 0.618 | 0.444 | 0.384 | 0.833 |
| PostToolUse | 0.528 | 0.415 | 0.493 | 0.316 |
| Stop | 0.426 | 0.414 | 0.595 | 0.100 |
| SubagentStart | 0.248 | 0.296 | 0.814 | 0.000 |

## Remaining Work
1. **Expansion quality (R5)** — process rules generate too-broad expansions; needs category-specific expansion guidance and re-expansion
2. **Production validation** — test 5-event hooks with 109-rule corpus in live session
3. **Smaller event-specific corpora** — 30-40 rules per event type instead of 109 for all
4. **Stronger reranker model** — Gemma E4B may lack capacity for 109-rule discrimination
5. **TOML annotations** — explicit event/tool tags on each rule in the source file
6. **Add workflow rules to ~/.cuecard/rules/global.txt** for production use
