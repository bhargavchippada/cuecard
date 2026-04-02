# Session 18 Progress State

> Date: 2026-04-02
> For resumption in next session

## What Was Accomplished

### 1. Eval Metrics Overhaul
- **Bug found:** `recall_at_k` returned 0.0 for negative fixtures (empty should_match), diluting mean_recall
- **Quality Score (F2):** New per-fixture composite metric using F-beta (β=2, recall-weighted)
  - Positive fixtures: `F2 = 5*P*R / (4P + R)` — rewards recall 4x over precision
  - Negative fixtures: `1.0 if silent, 0.0 otherwise` — correct-abstention convention
- **positive_recall:** Recall averaged only over positive fixtures (excludes negatives)
- **positive_quality:** F2 averaged only over positive fixtures (comparable across datasets)
- **mean_quality:** F2 averaged over ALL fixtures (unified metric)
- Refactored `_compute_tier_summaries` into `_make_tier_summary` helper
- Updated `format_eval_report` to show Quality, Positive Quality, Positive Recall, per-tier F2
- 843 tests, 100% coverage, ruff clean, mypy strict

### 2. Expansion Prompt v5 (Reasoning Field)
- Added `"reasoning"` field to expansion prompt — model reasons about vocabulary gap before generating
- Examples in system prompt now include reasoning analysis
- User prompt asks model to "first reason about the vocabulary gap, then generate"
- Parser extracts and logs reasoning at DEBUG level
- Tests updated for new prompt structure

### 3. Per-Model End-to-End Benchmarks
Each model generates its own expansions AND uses them for reranking:

#### 35B E2E (v5 prompt, 20% sample)
| Fixture Set | Quality (F2) | Pos Quality | Pos Recall | Noise | Neg Silence |
|-------------|-------------|-------------|------------|-------|-------------|
| basic | 0.780 | 0.670 | 0.680 | 0.189 | 0.962 |
| workflow | 0.692 | 0.580 | 0.621 | 0.328 | 1.000 |

#### 9B E2E — running
#### 4B E2E — pending

### 4. v4/v5 Expansion Corpora
- v4: reasoning-principles prompt (no reasoning field) — generated with 9B
- v5: reasoning-field prompt — generated with 9B
- Quality comparison showed v5 improves noise and neg_silence over v4

### 5. Code Review (2 agents, converged)
Both reviewers found same issues, all fixed:
- Triple blank line → removed
- `mean_quality` mixes distributions → added `positive_quality`
- Missing test assertions → added
- Per-tier quality not displayed → added F2 column to tier table

## Gap Analysis: Path to F2 ≥ 0.85

### Current: 35B basic=0.780, workflow=0.692

### Stage-by-Stage Gaps

**Stage 1: Embedding Retrieval**
- Basic positive recall (embedding only): 0.785 — this is the CEILING for the LLM stage
- Workflow embedding recall: 0.515 — much lower, code model struggles with natural language
- **Gap:** Embedding stage limits total recall. Better expansions could raise this ceiling.
- **Actions:** More targeted expansions, possibly snowflake-arctic for workflow queries

**Stage 2: LLM Reranker (35B)**
- Drops basic recall from 0.785→0.680 (loses ~10% of candidates)
- Workflow noise at 0.328 — too much irrelevant retrieved
- **Gap:** Reranker is too aggressive filtering (false negatives) AND not aggressive enough on noise
- **Actions:** Prompt tuning — more positive examples for edge cases, better negative examples

**Stage 3: Expansion Quality**
- 0-2 empty rules per corpus after generation
- Cross-domain contamination minimal (3 borderline cases)
- But 7.7 avg expansions may not be enough for abstract rules
- **Gap:** Some rules have a vocabulary gap that 7-8 expansions don't bridge
- **Actions:** Analyze which rules have lowest recall, generate targeted expansions

### What Would Move the Needle Most
1. **Raise embedding ceiling** (+5-10% quality): Better expansions for the hardest rules
2. **Reduce reranker false negatives** (+3-5% quality): Prompt tuning with false-negative examples
3. **Reduce workflow noise** (+5% workflow quality): Better cross-domain discrimination

### Target Breakdown
- Need: 0.85 quality
- Negative contribution (38% of basic, score=0.962): 0.38 * 0.962 = 0.366
- Remaining positive contribution needed: 0.85 - 0.366 = 0.484 from 62% of fixtures
- Required positive quality: 0.484 / 0.62 = 0.781
- Current positive quality: 0.670
- Gap: +0.111 positive quality needed

## Key Decisions
1. F2 (β=2) chosen over F1 — recall matters more than precision for rule injection
2. Correct-abstention convention — negatives contribute 1.0/0.0, not excluded
3. `positive_quality` added alongside `mean_quality` for clean comparison
4. Each model generates its own expansions for E2E benchmarks

## What's Next (Priority Order)

### Immediate (for autonomous iteration)
1. Complete 9B and 4B E2E benchmarks
2. Analyze false negatives — which rules/queries fail at each stage
3. Prompt iteration on reranker — target false-negative examples
4. Expansion iteration — regenerate for rules with lowest recall
5. Iterate until F2 ≥ 0.85 on basic, ≥ 0.80 on workflow

### Medium Term
6. Full dataset validation of best configuration
7. `cuecard serve` daemon — avoid model load/unload per hook call
8. Phase 4: PyPI publish

### Future Considerations (from user)
- Index rules.md files, CLAUDE.md, and skills (not just rules.txt/rules.json)
- Prompts must use reasoning guidelines, not hardcoded rules — generalize to new rule sources
- Don't retry empty expansions — some rules naturally don't expand well

## Files Changed This Session

### Source Code
- `src/cuecard/eval.py` — quality_score (F2), positive_recall, positive_quality, _make_tier_summary, format updates
- `src/cuecard/expander.py` — reasoning-field prompt, examples with reasoning, parser extracts reasoning

### Tests
- `tests/test_eval.py` — TestQualityScore (8 tests), assertions for new fields, format label checks
- `tests/test_expander.py` — tests for reasoning field, examples, prompt format

### Eval Corpora
- `eval/corpora/enriched_basic_v4/` — v4 expansions (reasoning principles, no reasoning field)
- `eval/corpora/enriched_basic_v5/` — v5 expansions (reasoning field, generated by 9B)
- `eval/corpora/enriched_workflow_v4/` — v4 workflow expansions
- `eval/corpora/enriched_workflow_v5/` — v5 workflow expansions

### Benchmark Results
- `eval/results/v4-embedding-sample-2026-04-02.json`
- `eval/results/v4-9b-llm-sample-2026-04-02.json`
- `eval/results/v5-embedding-sample-2026-04-02.json`
- `eval/results/v5-9b-llm-sample-2026-04-02.json`
- `eval/results/v5-4b-llm-sample-2026-04-02.json`
- `eval/results/v5-35b-llm-sample-2026-04-02.json`
- `eval/results/e2e-35b-v5-sample-2026-04-02.json`

## Git State
- Branch: master
- 843 tests, 100% coverage, ruff clean, mypy strict
