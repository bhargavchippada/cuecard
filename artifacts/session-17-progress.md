# Session 17 Progress State

> Date: 2026-04-02
> For resumption in next session

## What Was Accomplished

### 1. Qwen3.5 Dense Model Benchmarks
Downloaded and benchmarked 3 dense Qwen3.5 models as reranker replacements for 35B-MoE:
- **Qwen3.5-9B** (5.3GB Q4_K_M) — closest to 35B quality
- **Qwen3.5-4B** (2.6GB Q4_K_M) — best for CPU/laptop
- **Qwen3.5-2B** (1.2GB Q4_K_M) — not viable (too noisy)

### 2. Stop Sequence Fix (Regression)
Discovered `"stop": ["\n\n"]` in `call_local()` (added session 16 for 0.6B stability) was breaking JSON responses on dense models. The stop sequence cuts off responses before the JSON part.

**Fix:** Made stop sequences configurable via `stop` parameter on `call_local()`. Reranker passes `stop=None`, expander keeps default `("\n\n",)`.

### 3. Eval Infrastructure Improvements
- **tqdm progress bars** on `run_eval()` — shows per-fixture progress
- **`sample_ratio` parameter** — stratified sampling for rapid experimentation
  - 20% sampling: ~2-3 min per model (vs 15 min full)
  - Preserves tier distribution (easy/medium/hard/negative)
  - CLI: `--sample-ratio 0.2`
- **Parser fix:** Off-by-one in expansion limit warning (>= → >)

### 4. Prompt Engineering (v1 → v3)
Expanded reranker system prompt from 7 to 13 few-shot examples:
- Example 8: Style rules on code edits (type hints + resource cleanup)
- Example 9: Indirect match (docker build → dependency review)
- Example 10: System command negative (nvidia-smi)
- Example 11: Git local ops negative (stash)
- Example 12: Tmux send-keys (obvious match)
- Example 13: Debug print (obvious match)

New guidelines (principle-based, not hardcoded command lists):
- "When in doubt, include" — false negatives worse than false positives
- "Scan ALL violations" in embedded code
- Principle: "does this change code/state?" for read-only detection
- Principle: "does this affect commit history or remote?" for git ops

### 5. Loss Pattern Analysis
Agent-produced analysis in `artifacts/llm-reranker-loss-analysis-2026-04-02.md`:
- 40% of false negatives: "type hints" and "close resources" rules
- Secondary violations missed in multi-issue code
- Indirect rule mappings fail (docker build → dependencies)
- 9% negative leaks from system commands and git local ops
- Mined sessions worst at 86% silence

### 6. Benchmark Script
`tools/bench_models.py` — automated model benchmarking:
- Starts/stops llama-server automatically
- Runs all fixture sets (basic, workflow, mined)
- Supports `--sample-ratio` for fast iteration
- Saves results to `eval/results/`

## Key Benchmark Results

### Fair Comparison (v3 corpora, 20% sample, same seed)

| Model | Basic Recall | Basic Noise | Basic NegSil | p50ms | GGUF |
|-------|-------------|-------------|--------------|-------|------|
| 35B-MoE | 0.413 | 0.256 | 0.923 | 1083ms | 21GB |
| 9B v3 prompt | 0.407 | 0.173 | 0.962 | 1608ms | 5.3GB |
| 4B | 0.383 | 0.248 | 0.962 | 856ms | 2.6GB |
| 2B | 0.383 | 0.404 | 0.808 | 521ms | 1.2GB |

### Prompt Engineering Progress (9B, basic, 20% sample)

| Prompt | Recall | Noise | NegSil |
|--------|--------|-------|--------|
| v1 (7 examples) | 0.400 | 0.190 | 0.917 |
| v2 (13 examples) | 0.417 | 0.232 | 0.923 |
| v3 (principle guidelines) | 0.407 | 0.173 | 0.962 |

### Verdict
- **9B matches 35B on basic** — within 0.6 pts recall, better noise (-8 pts), better silence (+4 pts)
- **Workflow gap remains** — 9B at 0.389 vs 35B at 0.489 (-10 pts)
- **4B viable for CPU** — 856ms p50, 96.2% neg silence, but -3 pts recall
- **2B not viable** — 40% noise, 50% workflow silence

## Key Architectural Decisions

1. **Stop sequences optional per caller** — reranker needs full response, expander needs degeneration protection
2. **Principle-based guidelines** over hardcoded command lists — generalizes to unseen queries
3. **Stratified sampling** for eval — proportional representation across tiers
4. **Same model for expansion + reranking** — must be generative (rules out cross-encoder-only models)

## What's Next (Priority Order)

### Immediate
1. **Full 9B benchmark running** — validating on all 518 fixtures
2. **Workflow prompt engineering** — close the 10pt gap on UserPromptSubmit
3. **Run 35B with v3 prompt** — verify improvement transfers to larger model

### Medium Term
4. **Expansion prompt engineering** — targeted at loss patterns
5. **Full 4B benchmark** — validate CPU candidate on full dataset
6. **`cuecard serve` daemon** — avoid load/unload per hook call

### Longer Term
7. Phase 4: PyPI publish
8. Phase 5.1: Markdown parser for CLAUDE.md ingestion

## Files Changed This Session

### Source Code
- `src/cuecard/llm_utils.py` — configurable stop sequences
- `src/cuecard/llm_reranker.py` — v3 prompt (13 examples), stop=None
- `src/cuecard/eval.py` — tqdm progress, sample_ratio, stratified sampling
- `src/cuecard/cli_eval.py` — --sample-ratio CLI option
- `src/cuecard/parser.py` — off-by-one fix in expansion limit

### Tests
- `tests/test_eval.py` — sampling tests, tqdm fallback test

### Tools
- `tools/bench_models.py` — automated model benchmarking script

### Artifacts
- `artifacts/session-17-progress.md` — this file
- `artifacts/llm-reranker-loss-analysis-2026-04-02.md` — loss analysis

### Benchmark Results
- `eval/results/qwen35-9b-llm-local-2026-04-02.json` — 9B full (v1 prompt)
- `eval/results/qwen35-9b-v2prompt-sample-llm-local-2026-04-02.json` — 9B sampled (v2 prompt)
- `eval/results/qwen35-9b-v3prompt-sample-llm-local-2026-04-02.json` — 9B sampled (v3 prompt)
- `eval/results/qwen35-4b-sample-llm-local-2026-04-02.json` — 4B sampled
- `eval/results/qwen35-2b-sample-llm-local-2026-04-02.json` — 2B sampled
- `eval/results/qwen35-35b-v3-sample-llm-local-2026-04-02.json` — 35B baseline (v3 corpora)

### Models Downloaded
- `~/models/Qwen3.5-9B-Q4_K_M.gguf` (5.3GB)
- `~/models/Qwen3.5-4B-Q4_K_M.gguf` (2.6GB)
- `~/models/Qwen3.5-2B-Q4_K_M.gguf` (1.2GB)

## Git State
- Branch: master
- 831 tests, 100% coverage, ruff clean, mypy strict
