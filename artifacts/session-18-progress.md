# Session 18 Progress State

> Date: 2026-04-02
> For resumption in next session

## What Was Accomplished

### 1. Eval Metrics Overhaul
- **Bug found:** `recall_at_k` returned 0.0 for negative fixtures (empty should_match), diluting mean_recall
- **Quality Score (F2):** New per-fixture composite metric using F-beta (β=2, recall-weighted)
  - Positive fixtures: `F2 = 5*P*R / (4P + R)` — rewards recall 4x over precision
  - Negative fixtures: `1.0 if silent, 0.0 otherwise` — correct-abstention convention
- **positive_recall / positive_quality:** Averaged only over positive fixtures
- Refactored `_compute_tier_summaries` → `_make_tier_summary` helper
- Updated `format_eval_report` with Quality, Positive Quality, Positive Recall, per-tier F2
- 860 tests, 100% coverage, ruff clean, mypy strict

### 2. Parse Failure Investigation & Fix (BIGGEST WIN: +4.9 pts)
- **Root cause:** Model outputting reasoning prose without JSON, hitting EOS early
- NOT truncation (responses were only ~60 tokens) — model just stopped before JSON
- **Prompt fix:** Changed "Write reasoning BEFORE listing rules" to "ALWAYS return a SINGLE JSON object — nothing else. Put ALL analysis inside the reasoning field."
- **Result:** 0 parse failures (was 1-3/run), basic quality 0.750→0.799
- Hard tier jumped +15.5 pts (0.574→0.729) — parse failures disproportionately hit hard queries
- **Lesson:** Silent fallbacks are insidious. The WARNING was logged but buried in tqdm output.

### 3. Robust JSON Extraction
- `_try_json_parse`: searches all `{` positions last-to-first, brace-depth tracking
- `_extract_rule_refs_from_prose`: regex fallback for "Rule N applies" patterns
- Safety net for any remaining parse issues

### 4. Expansion Prompt v5 (Reasoning Field)
- Added `"reasoning"` field — model analyzes vocabulary gap before generating expansions
- Examples include reasoning analysis in system prompt
- Regenerated with both 9B and 35B

### 5. Fixture Audit & Corrections
- Found 2 clearly wrong expectations:
  - `ts-write-react-component`: Python "type hints" rule on TSX file
  - `docstring-missing`: "type hints" expected for already-typed function
- Identified several debatable expectations (subprocess.shell=True vs eval, git tag vs commit)
- Fixture auditor agent running comprehensive audit of all 587 fixtures

### 6. End-to-End Model Benchmarks
Each model generates its own expansions AND reranks:

| Model | Basic Quality | Workflow Quality |
|-------|-------------|-----------------|
| 35B | 0.803 | 0.748 |
| 9B | 0.727 | 0.725 |
| 4B | pending | pending |

### 7. Code Reviews (2 agents, converged)
All MEDIUM findings addressed: per-tier quality display, positive_quality field, test assertions, triple blank line.

## Key Findings

### Three Categories of Improvement
1. **Bug fixes** (parse failure, metric calculation) — real issues, big impact
2. **Ground truth corrections** (wrong fixture expectations) — honest improvement
3. **Prompt engineering** (JSON-first, reasoning principles) — legitimate tuning

The biggest wins came from fixing our own measurement, not from tuning parameters.

### Saturation Analysis
- Easy tier at 0.882 — near ceiling
- Hard tier at 0.613 — remaining gap, but many are genuinely ambiguous
- 35B expansions are broader than 9B (more cross-domain noise)
- Embedding recall ceiling: ~0.85 positive recall
- 20% sample has ~5% noise band — need full dataset to separate signal from noise

### Architecture Insights
- **Parse failures are silent quality killers.** Each falls back to noisy embedding results.
- **JSON-first prompts > "write reasoning first".** Models interpret "write before" as "output prose".
- **Wrong fixtures are indistinguishable from model failures.** Audit ground truth first.
- **Manual expansion supplements are overfitting.** The expansion system must generate good expansions itself.
- **Negative silence and noise are coupled.** Broader expansions improve recall but increase noise.

## Current Best (35B, v5 reasoning-field expansions, 20% sample, corrected fixtures)

| Fixture | Quality (F2) | Pos Recall | Noise | Neg Silence |
|---------|-------------|------------|-------|-------------|
| Basic | 0.803 | 0.758 | 0.200 | 0.923 |
| Workflow | 0.748 | 0.697 | 0.272 | 1.000 |

Per-tier (Basic):
| Tier | Quality | Recall | Noise |
|------|---------|--------|-------|
| easy | 0.882 | 0.917 | 0.167 |
| medium | 0.694 | 0.765 | 0.315 |
| hard | 0.613 | 0.528 | 0.317 |
| negative | 0.923 | 0.000 | 0.077 |

### 8. Comprehensive Fixture Audit (agent)
- Audited all 587 fixtures against 76 rules
- Found 23 wrong expectations, applied 19 corrections
- Key systemic patterns:
  - Compliant code triggering violation rules (using uv fires "use uv" rule)
  - Running tests triggering TDD rule (running tests IS the check)
  - Read-only operations over-matching (chown, git tag, docker run)
  - Fixes triggering violation rules (adding env vars fires env var rule)
- Proposed 20 new fixtures for coverage gaps
- Suggested new reranker principle: "compliance vs violation"

## What's Next (Priority Order)

### From Fixture Auditor
1. ~~Apply corrections from comprehensive fixture audit~~ DONE
2. Add new fixtures for underrepresented scenarios
3. Re-benchmark with corrected ground truth

### Quality Improvements
4. Full dataset validation (sample_ratio=1.0) to confirm scores
5. Investigate whether snowflake-arctic-m helps workflow (better NL matching)
6. Run 9B and 4B E2E with 35B-quality expansions (decouple expansion model from reranker model)

### Infrastructure
7. `cuecard serve` daemon — avoid model load/unload per hook call
8. Phase 4: PyPI publish
9. Phase 5.1: Markdown parser for CLAUDE.md ingestion

## Files Changed This Session

### Source Code
- `src/cuecard/eval.py` — quality_score (F2), positive_recall, positive_quality, _make_tier_summary
- `src/cuecard/expander.py` — reasoning-field prompt, examples with reasoning
- `src/cuecard/llm_reranker.py` — JSON-first prompt, _try_json_parse, _extract_rule_refs_from_prose, refined reasoning principles

### Tests
- `tests/test_eval.py` — TestQualityScore (8 tests), format assertions, new field assertions
- `tests/test_expander.py` — reasoning field tests
- `tests/test_llm_reranker.py` — TestTryJsonParse, TestExtractRuleRefsFromProse, prose-then-json test

### Fixtures
- `eval/fixtures/basic.json` — 2 wrong expectations corrected

### Eval Corpora
- `eval/corpora/enriched_basic_v5/` — regenerated with 35B (pure LLM)
- `eval/corpora/enriched_workflow_v5/` — regenerated with 35B (pure LLM)

## Git State
- Branch: master, pushed to origin
- 4 commits: metrics, prompt refinement, parse fix, fixture corrections
- 860 tests, 100% coverage, ruff clean, mypy strict
