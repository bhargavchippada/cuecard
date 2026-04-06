# E2E Model Benchmark — 2026-04-04

True end-to-end comparison: each model generates its own expansions AND reranks.
20% stratified sample, seed=42, jina-code embeddings, llm-local mode.

## Environment

- llama.cpp build: b8235 (ff52ee964)
- GPU: NVIDIA RTX 5090
- Embedding: jinaai/jina-embeddings-v2-base-code
- Server flags: `--jinja --reasoning-budget 0 -ngl 99 -c 16384`
- Eval: `tools/bench_e2e.py` with sample_ratio=0.2, seed=42

### llama.cpp b8235 Jinja workaround

The embedded Qwen3.5 multimodal Jinja templates fail with
`Failed to parse input at pos 20` whenever the request passes
`chat_template_kwargs: {enable_thinking: false}`. Adding
`--reasoning-budget 0` disables thinking server-side and bypasses the
buggy template code path. This is now the canonical flag for Qwen3.5
GGUFs in `tools/bench_e2e.py`.

## Models Tested

| Label | Path | Size | Status |
|-------|------|-----:|--------|
| qwen35-4b | Qwen3.5-4B-Q4_K_M.gguf | 2.6GB | OK |
| qwen35-9b | Qwen3.5-9B-Q4_K_M.gguf | 5.3GB | OK |
| qwen35-35b | Qwen3.5-35B-A3B-Q4_K_M.gguf | 21GB | OK |
| gemma-e4b | gemma-4-E4B-it-Q8_0.gguf | 8.19GB | **BLOCKED** |

**Gemma blocker (verbatim)**:
```
llama_model_load: error loading model: error loading model architecture:
unknown model architecture: 'gemma4'
```
llama.cpp b8235 does not support the `gemma4` architecture. Requires
llama.cpp upgrade/rebuild.

## Results — Basic (PreToolUse, 354 fixtures, 20% sample = 70)

| Metric | 4B | 9B | 35B |
|--------|---:|---:|----:|
| Quality (F2) | 0.736 | **0.797** | 0.785 |
| Positive Recall | 0.703 | **0.834** | 0.752 |
| Noise Ratio | 0.243 | 0.259 | **0.217** |
| Negative Silence | 0.852 | 0.889 | **0.926** |
| Easy F2 | 0.808 | 0.818 | **0.888** |
| Medium F2 | 0.702 | **0.763** | 0.695 |
| Hard F2 | 0.377 | **0.578** | 0.445 |
| p50 latency | **917ms** | 1259ms | 1294ms |
| p95 latency | 1201ms | 2061ms | 1821ms |

## Results — Workflow (UserPromptSubmit, 84 fixtures, 20% sample = 15)

| Metric | 4B | 9B | 35B |
|--------|---:|---:|----:|
| Quality (F2) | 0.577 | **0.725** | 0.680 |
| Positive Recall | 0.545 | **0.667** | **0.667** |
| Noise Ratio | 0.472 | 0.334 | **0.294** |
| Negative Silence | 0.750 | **1.000** | 0.750 |
| Easy F2 | 0.708 | **0.958** | 0.847 |
| Medium F2 | 0.352 | 0.442 | **0.451** |
| Hard F2 | 0.472 | 0.423 | **0.667** |
| p50 latency | **989ms** | 1287ms | 1441ms |
| p95 latency | 1200ms | 1628ms | 1608ms |

## Expansion Generation Stats

| Model | Basic (rules/exp/avg/s) | Workflow (rules/exp/avg/s) | Total time |
|-------|------------------------|----------------------------|-----------:|
| 4B | 30 / 182 / 6.1 / 30.4s | 46 / 233 / 5.1 / 50.9s | 81.3s |
| 9B | 30 / 206 / 6.9 / 43.2s | 46 / 347 / 7.5 / 71.2s | 114.4s |
| 35B | 30 / 231 / 7.7 / 46.7s | 46 / 386 / 8.4 / 70.8s | 117.5s |

Average expansion length: 45-56 chars (tight cluster across models).

## Analysis

1. **9B wins E2E overall.** With its own expansions, 9B achieves the best basic F2 (0.797) and workflow F2 (0.725), beating both 4B (4B lacks capacity) AND 35B (which under-generates expansions for a lower F2 despite better absolute noise). This is a meaningful shift from shared-corpus benchmarks: when 35B's expansions were used to evaluate 9B in prior runs, 35B looked dominant — but when each model uses its own corpus, 9B's reranker + 9B-authored expansions compound favorably.

2. **Expansion count scales with model size, but quality doesn't linearly follow.** 4B produces 5-6 expansions per rule; 9B produces 7 on basic and 7.5 on workflow; 35B produces 7.7 and 8.4. Average expansion length is nearly identical (45-56 chars). More expansions improve recall (9B/35B beat 4B by ~10-13 F2 points on basic) but 35B's extra 80+ expansions don't translate to better F2 over 9B — suggesting diminishing returns past ~7 expansions/rule.

3. **35B has the lowest noise but worse recall.** 35B achieves best noise ratio (0.217 basic, 0.294 workflow) and best negative silence on basic (0.926) but loses on positive recall (0.752 vs 9B's 0.834). 35B is the most selective reranker; 9B is the best balanced.

4. **Tier-level surprises**: on basic-hard, 9B's F2 (0.578) is +13 points over 35B (0.445). On workflow-hard, 35B reclaims dominance (0.667 vs 9B's 0.423). This suggests: 35B handles rare/ambiguous workflow queries better (more training data for multi-step reasoning), but 9B generalizes tool-call rules better (likely from denser training on code).

5. **Surprise: E2E methodology changes rankings.** Prior shared-corpus results (from CLAUDE.md "Qwen3.5 Dense Model Comparison"): 35B basic recall 0.413, 9B basic recall 0.407 — near-parity. But E2E: 9B basic positive_recall 0.834 vs 35B 0.752 — a 10-point spread in 9B's favor. The E2E methodology reveals that pairing a model's expansions with its own reranker outperforms cross-model pairings that shared-corpus benchmarks imply.

6. **Latency: 4B still fastest, 9B moderate, 35B slowest.** 4B p50=917ms (basic) / 989ms (workflow) is ~30% faster than 9B's 1259ms. 35B's MoE architecture (21B active params for 4.7B inference) matches 9B latency (1294ms basic). All three are under 1.5s p50 on the RTX 5090.

## Recommendation

**9B is the new default** for the cuecard LLM reranker:
- Best E2E F2 on both basic (0.797) and workflow (0.725)
- 4x smaller than 35B (5.3GB vs 21GB) — fits 8GB VRAM with headroom
- Best recall on hard basic fixtures (+13 pts over 35B)
- Latency parity with 35B (1259ms vs 1294ms)
- Best negative silence on workflow (1.000)

**Keep 35B as "max quality" option** for workflow-hard queries where its
+24 point F2 edge matters (0.667 vs 9B's 0.423). Use 35B if 21GB VRAM is
available AND workflow accuracy is paramount.

**4B is the CPU/laptop candidate** at 2.6GB with 917ms p50 basic latency,
but the F2 gap to 9B (-6 pts basic, -15 pts workflow) is significant.
Use only when VRAM is tight.

**Gemma-E4B is blocked** pending llama.cpp upgrade supporting `gemma4`
architecture. Re-benchmark once a compatible llama.cpp build is available.

## Files

- Results: `eval/results/qwen35-{4b,9b,35b}-e2e-seed42.json`
- Corpora: `eval/corpora/enriched_{basic,workflow}_{qwen35-4b,qwen35-9b,qwen35-35b}/rules.json`
- Logs: `/tmp/bench-{4b,9b,35b,gemma}.log` (ephemeral)
