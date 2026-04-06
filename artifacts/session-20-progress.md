# Session 20 Progress State

> Date: 2026-04-04 through 2026-04-06
> For resumption in next session

## What Was Accomplished

### 1. Apples-to-Apples 9B vs 35B (Shared Corpus)
- Fresh seed=42 benchmarks on v5 corpora
- Real gap: ~5 pts F2 basic, ~3 pts workflow (stale CLAUDE.md showed ~8 pts)
- Gap is noise/discrimination, not recall — 9B beats 35B on PosRecall
- 9B wins hard-tier (+14 F2), 35B wins medium-tier (+11 F2)

### 2. Prompt Tuning (3 Variants on 9B, All Reverted)
- P2-v1 (global compliance): helped NegSil (0.889), crashed easy-tier (-10 F2)
- P2-scoped (Edit/Write only): fixed easy, crashed workflow (-7.3 F2)
- P2+exception (enforcement carve-out): fixed easy, crashed workflow (-10 F2)
- Root cause: 9B confuses violation/enforcement/compliance (reasoning-depth limit)
- Forensic trace analysis: "Rule 3 (type hints) is the target of the check but does not constrain the act of running the tool itself" — 9B silences type-hints when mypy runs

### 3. E2E Methodology Fix (THE BIG FINDING)
- Built `tools/bench_e2e.py` — each model generates own expansions AND reranks
- Deprecated `bench_models.py` (shared-corpus comparisons are unfair)
- **Methodology flip**: shared-corpus showed 35B winning (F2=0.808 vs 0.762); E2E showed 9B winning (0.797 vs 0.785)
- Expansion stats: 4B avg 6.1/rule, 9B avg 6.9, 35B avg 7.7 — diminishing returns past ~7

### 4. Gemma 4 E4B Evaluation
- Downloaded Q8_0 (8.19GB) from unsloth/gemma-4-E4B-it-GGUF
- Upgraded llama.cpp: build 8235 → 8672 (gemma4 architecture support)
- E2E benchmark results: workflow F2=0.820 (best), PosRecall=0.778 (+39% vs 9B), perfect workflow NegSil
- Prompt tuning (2 variants on Gemma, both reverted): Gemma hypersensitive to negative examples — single few-shot crashes PosRecall by 30+ pts

### 5. Paper Design
- Literature search: partially novel — per-rule rationale as DV is unstudied
- Designed minimal test harness (~80 token system prompt)
- 4-condition factorial: Command / Framed / Verbose / Consequence
- Pilot spec: 3 agents × 5 rules × 4 phrasings × 20 trials = 1200 trials
- Artifact: `artifacts/paper-design-rule-rationale.md`

### 6. llama.cpp Jinja Bug Discovery
- llama.cpp b8235's Jinja parser fails on Qwen3.5 multimodal templates
- Workaround: `--jinja --reasoning-budget 0` or `--chat-template chatml`
- bench_e2e.py uses the workaround

## Key Findings

### E2E Methodology Matters
Shared-corpus benchmarks bias toward the expansion-generator model. E2E flips rankings.

### Gemma 4 E4B is the New Default
- Best PosRecall (0.778 vs 9B 0.560) — finds 39% more relevant rules
- Best workflow (F2=0.820, NegSil=1.000)
- Acceptable NegSil tradeoff (0.852 vs 9B 0.962 — ~1 false positive per 10 negatives)
- 1204ms p50 (faster than 9B despite larger GGUF)

### Prompt Tuning Hit a Wall
- 5 variants across 2 models, all reverted
- Both models overcorrect on negative examples
- 9B has compliance/violation/enforcement confusion (reasoning-depth limit)
- Gemma is hypersensitive to negative few-shots (one example = -30 pts PosRecall)
- The reranker prompt is at a local optimum

### Full E2E Comparison

| Model | Basic F2 | PosRecall | NegSil | Workflow F2 | p50ms | GGUF |
|-------|---------|-----------|--------|------------|-------|------|
| **Gemma E4B** | 0.774 | **0.778** | 0.852 | **0.820** | 1204ms | 8.2GB |
| Qwen 9B | **0.797** | 0.560 | **0.962** | 0.725 | 1259ms | 5.3GB |
| Qwen 35B | 0.785 | — | 0.926 | 0.680 | 1294ms | 21GB |
| Qwen 4B | 0.736 | — | 0.852 | 0.577 | 917ms | 2.6GB |

## Current State
- Branch: master
- llama.cpp: build 8672 (upgraded from 8235)
- 9B server running on port 8081 (user's current default)
- Gemma E4B downloaded at ~/models/gemma-4-E4B-it-Q8_0.gguf
- Reranker prompt: original 13 examples + 8 principles (all experiments reverted)
- 946 tests, 100% coverage, ruff clean

## Files Changed This Session

### Source Code
- `src/cuecard/llm_reranker.py` — prompt experiments (all reverted, net zero change)

### Tools
- `tools/bench_e2e.py` — NEW: E2E benchmark script (model-specific expansions)
- `tools/bench_models.py` — deprecated (added notice)

### Eval
- `eval/corpora/enriched_{basic,workflow}_qwen35-{4b,9b,35b}/` — model-specific expansion corpora
- `eval/corpora/enriched_{basic,workflow}_gemma-e4b/` — Gemma expansion corpus
- `eval/results/qwen35-{4b,9b,35b}-e2e-seed42.json` — E2E results
- `eval/results/gemma-e4b-e2e-seed42.json` — Gemma E2E baseline
- `eval/results/gemma-e4b-{fixed,surgical}-e2e-seed42.json` — prompt tuning experiments

### Docs & Artifacts
- `CLAUDE.md` — updated model recommendations (Gemma default), E2E methodology, benchmarks
- `artifacts/session-20-progress.md` — this file
- `artifacts/paper-design-rule-rationale.md` — paper design for compliance study
- `artifacts/e2e-benchmark-2026-04-04.md` — full 3-model E2E report
- `artifacts/gemma-e4b-negsilence-forensics.md` — Gemma NegSil forensic analysis

## What's Next (Priority Order)

### Near-term
1. Commit session 20 changes
2. Switch production cuecard to Gemma E4B (update serve daemon, hook config)
3. Run full-dataset (100%) E2E benchmark for Gemma (confirm 20% sample holds)

### Medium-term
4. Paper pilot: build multi-agent harness for compliance study (1200 trials)
5. Phase 4: PyPI publish
6. Phase 5.1: Markdown parser for CLAUDE.md ingestion

### Future
7. Dual-model architecture (9B for PreToolUse, Gemma for UserPromptSubmit)
8. Fine-tune Gemma on cuecard's abstention distribution (fix NegSil without prompt)
9. Gemma expansion prompt tuning (currently 4.4 avg vs 9B's 6.9 — room to grow)
