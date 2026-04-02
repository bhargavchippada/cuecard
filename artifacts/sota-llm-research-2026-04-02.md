# SOTA LLM Research: Reranker Replacement for Qwen3.5-35B-A3B

> Research date: 2026-04-02
> Goal: Find models (1B-14B) that can match Qwen3.5-35B-A3B reranker quality at lower cost
> Current baseline: Qwen3.5-35B-A3B (MoE, 3B active) → 91% neg silence, 21% noise, ~1.1s GPU latency

## Key Insight

Qwen3.5-35B-A3B is MoE with only **3B active parameters** per token. We're effectively competing against 3B active — not 35B. Dense models in the 4-8B range may match it.

**Critical discovery:** The Qwen3.5 family includes dense models (0.8B, 2B, 4B, 9B, 27B) with a new **Gated DeltaNet** architecture that significantly outperforms Qwen3 at the same size. **Qwen3.5-9B achieves IFEval 91.5 and MMLU-Pro 82.5** — dramatically better than Qwen3-8B. This changes the recommendation.

Also discovered: **Nemotron-3-Nano-30B-A3B** (NVIDIA, 3.5B active) is purpose-built for tool calling/structured output with 3.3x throughput vs Qwen3-30B-A3B.

---

## Top 10 Candidate Models (Ranked)

### Tier 1: Highest Confidence — Direct Replacements

#### 1. Qwen3.5-9B ⭐⭐⭐ (NEW Top Pick — best quality/size ratio)

| Property | Value |
|----------|-------|
| Total params | 9B |
| Active params | 9B (dense) |
| Architecture | Gated DeltaNet + standard attention hybrid (8 groups of 3 DeltaNet + 1 attn) |
| Context window | 262K native, 1.01M with YaRN |
| GGUF Q4_K_M size | ~5.5GB (estimated) |
| VRAM required | ~7GB |
| MMLU-Pro | **82.5** |
| MMLU-Redux | 91.1 |
| IFEval | **91.5** |
| GPQA Diamond | 81.7 |
| LiveCodeBench v6 | 65.6 |
| BFCL-V4 (tool use) | **66.1** |
| Thinking/non-thinking | ✅ `enable_thinking` param (hard switch only, no `/think` `/no_think` soft switch) |
| JSON compliance | Strong — 201-language training, tool use focus |
| License | Apache 2.0 |

**Why #1:** IFEval 91.5 is exceptional — matches our current Qwen3.5-35B-A3B's neg silence numbers. MMLU-Pro 82.5 is 21 points above Qwen3-8B (61 base). BFCL-V4 66.1 means it's trained specifically for tool use. The Gated DeltaNet architecture is more efficient than standard transformers.

**Risk:** Requires latest llama.cpp with `qwen3_5` architecture support. DeltaNet layers may have ~35% slowdown in current builds. Dense 9B = more compute per token than 3B-active MoE.

**Verdict:** Primary benchmark candidate. If IFEval 91.5 translates to our reranking task, this is the winner.

---

#### 2. Qwen3.5-4B ⭐⭐⭐ (Smallest Qwen3.5 — potential dream candidate)

| Property | Value |
|----------|-------|
| Total params | 4B |
| Active params | 4B (dense) |
| Architecture | Gated DeltaNet hybrid |
| Context window | 262K native |
| GGUF Q4_K_M size | ~2.5GB |
| VRAM required | ~3.5GB |
| Thinking/non-thinking | ✅ Hard switch |
| License | Apache 2.0 |

**Why #2:** Same Gated DeltaNet architecture as Qwen3.5-9B. If the architectural improvements scale down, this 4B could match Qwen3-8B quality in a 2.5GB package. Qwen3.5 training on 201 languages with MTP (Multi-Token Prediction) for speculative decoding.

**Risk:** No published benchmarks found for Qwen3.5-4B specifically. Need to test empirically.

**Verdict:** Must-benchmark alongside 9B. If 4B is "good enough," it's the sweet spot — 2.5GB, GPU or CPU viable.

---

#### 3. Qwen3-8B ⭐⭐⭐ (Proven dense candidate)

| Property | Value |
|----------|-------|
| Total params | 8.2B |
| Active params | 8.2B (dense) |
| Architecture | Dense, 36 layers, GQA 32/8 |
| Context window | 128K (32K native) |
| GGUF Q4_K_M size | ~5.0GB |
| VRAM required | ~6GB |
| CPU RAM required | ~6GB |
| MMLU-Pro (base) | 56.73 |
| EvalPlus | 67.65 |
| Thinking/non-thinking | ✅ Seamless toggle (hard + soft `/think` `/no_think`) |
| JSON compliance | Strong — Qwen3 agent training |
| License | Apache 2.0 |

**Why #3:** Well-understood model with extensive community GGUF support. Outperforms Qwen2.5-14B on most benchmarks (MMLU-Pro +5.57, GPQA +11.61, EvalPlus +6.95). Distil labs ranked Qwen3 family #1 across 8 fine-tuning tasks. More mature llama.cpp support than Qwen3.5.

**Expected GPU latency:** ~150 tok/s on RTX 4090 → ~1.3s for 200 tokens.

**Expected CPU latency:** ~12-20 tok/s → ~10-17s for 200 tokens. Too slow for CPU unless output is <50 tokens.

**Verdict:** Reliable fallback if Qwen3.5 has llama.cpp compatibility issues.

---

#### 4. Nemotron-3-Nano-30B-A3B ⭐⭐⭐ (Best tool-calling MoE)

| Property | Value |
|----------|-------|
| Total params | 30B |
| Active params | 3.5B (128 experts, top-6) |
| Architecture | Mamba-2 + MoE + GQA hybrid |
| Context window | 128K |
| GGUF Q4_K_M size | ~18GB |
| VRAM required | ~18GB |
| Throughput | 3.3x Qwen3-30B-A3B |
| Thinking/non-thinking | ✅ |
| JSON compliance | **Excellent** — explicitly fine-tuned on structured outputs + tool calling |
| License | NVIDIA open model license |

**Why notable:** NVIDIA purpose-built this for tool calling and structured output. 3.3x throughput over Qwen3-30B-A3B is massive. Fine-tuned specifically on tool calling data — the closest thing to a "reranker-trained" generative model in the 3B-active class.

**Risk:** Same VRAM as current setup (no size win). NVIDIA license may have restrictions. Mamba-2 layers may have different failure modes than transformer.

**Verdict:** If we're keeping GPU anyway and want maximum quality, this is the tool-calling champion.

---

### Tier 2: Strong Alternatives

#### 5. Qwen3-4B ⭐⭐ (Smallest Qwen3 dense — proven)

| Property | Value |
|----------|-------|
| Total params | 4.0B |
| Active params | 4.0B (dense) |
| Architecture | Dense, 36 layers, GQA 32/8 |
| Context window | 32K (131K with YaRN) |
| GGUF Q4_K_M size | ~2.5GB |
| VRAM required | ~3.5GB |
| MMLU-Pro (base) | 50.58 |
| EvalPlus | 63.53 |
| Thinking/non-thinking | ✅ Seamless toggle (hard + soft) |
| License | Apache 2.0 |

**Why notable:** Qwen blog claims "Qwen3-4B can rival Qwen2.5-72B-Instruct." Distil labs ranked Qwen3-4B-Instruct-2507 as #1 across 8 fine-tuning tasks. Base MMLU-Pro 50.58 outperforms Qwen2.5-7B (45.0) and Gemma-3-4B (29.23). EvalPlus 63.53 matches Qwen2.5-7B.

**Risk:** With Qwen3.5-4B available (same size, better architecture), Qwen3-4B is mainly useful as a "known working" fallback if Qwen3.5 has llama.cpp issues.

**Verdict:** Benchmark as comparison to Qwen3.5-4B to quantify the DeltaNet architecture advantage.

---

#### 6. Gemma-3n-E4B ⭐⭐ (Google's efficiency champion)

| Property | Value |
|----------|-------|
| Total params | 8B |
| Effective params | 4B (MatFormer architecture) |
| Architecture | Matryoshka Transformer, elastic inference |
| Context window | Long context supported |
| Memory footprint | ~3GB |
| GGUF availability | ✅ Via LM Studio, llama.cpp |
| Multimodal | Text, image, audio, video input |
| Language support | 140 languages |
| License | Google open model license |

**Why notable:** MatFormer architecture allows "nested" inference — the full 8B model contains functional 4B and 2B sub-models. Runs with only 3GB memory despite 8B total params. Released June 2025, very recent training.

**Risk:** Optimized for on-device / mobile. May not have the instruction-following precision needed for structured JSON reranking. Less community GGUF tooling than Qwen3.

**Verdict:** Worth benchmarking if Qwen3-4B disappoints — different architecture may have different failure modes.

---

#### 7. Phi-4-mini (3.8B) ⭐⭐

| Property | Value |
|----------|-------|
| Total params | 3.8B |
| Active params | 3.8B (dense) |
| Architecture | Dense decoder-only Transformer, GQA, shared embedding |
| Context window | 128K |
| GGUF Q4_K_M size | ~2.5GB |
| VRAM required | ~3.5GB |
| Structured output | ✅ Trained with `<think>` tag separation |
| IFEval | Weak on base — improved 22pts with reasoning variant |
| License | MIT |

**Why notable:** Microsoft's strong reasoning focus. 128K context at 3.8B is impressive. MIT license. Phi-4-reasoning-plus shows significant IFEval improvement — structured output training helps.

**Risk:** Base Phi-4 has known IFEval weakness — "has trouble strictly following instructions." Our task requires precise JSON compliance. May need the reasoning variant, which uses CoT (adds latency).

**Verdict:** Benchmark the instruct variant. If JSON compliance is poor, skip.

---

#### 8. Qwen3-14B ⭐⭐

| Property | Value |
|----------|-------|
| Total params | 14.8B |
| Active params | 14.8B (dense) |
| Architecture | Dense, 40 layers, GQA 40/8 |
| Context window | 128K |
| GGUF Q4_K_M size | ~9GB |
| VRAM required | ~10GB |
| MMLU-Pro (base) | 61.03 |
| License | Apache 2.0 |

**Why notable:** If 8B doesn't match quality, 14B is the next step. Still fits in 24GB VRAM easily. MMLU-Pro score (61.03) nearly matches 30B-A3B (61.49) — suggesting dense 14B ≈ MoE 30B-A3B quality.

**Risk:** 14.8B dense is ~5x more compute per token than 3B MoE active. GPU latency will be ~2-3x higher. Only justified if quality gain is meaningful.

**Verdict:** Backup candidate if 4B and 8B fall short.

---

#### 9. Gemma-3-4B ⭐⭐

| Property | Value |
|----------|-------|
| Total params | 4B |
| Active params | 4B (dense) |
| Architecture | Dense Transformer |
| Context window | 128K |
| GGUF Q4_K_M size | ~2.5GB |
| MMLU 5-shot | ~67% (12B variant) |
| GGUF availability | ✅ Official Google QAT models |
| License | Gemma license (permissive) |

**Why notable:** Google's official QAT (Quantization-Aware Training) models — quantized during training, not after. Q4_0 12B matches bf16 on MMLU (67.07% vs 67.15%). Very high-quality quantization.

**Risk:** Trails Qwen3-4B in fine-tuning benchmarks. Less instruction-following strength than Qwen3 family.

**Verdict:** Google's QAT approach is interesting but Qwen3-4B likely wins on our task.

---

### Tier 3: Niche / Specialized

#### 10. Granite-4.0-H-Tiny ⭐ (IBM's task-specific MoE)

| Property | Value |
|----------|-------|
| Total params | 7B |
| Active params | 1B |
| Architecture | Mamba-2/Transformer hybrid + MoE |
| GGUF Q4_K_M size | ~4GB |
| Tool calling | ✅ OpenAI-compatible function schema |
| License | Apache 2.0 |

**Why notable:** IBM designed this specifically for "high-volume, low-complexity tasks" — which describes cuecard reranking exactly. 1B active with tool calling built in. Apache 2.0.

**Risk:** 1B active may be too small for our 5K-token few-shot prompt with complex reasoning. IBM models generally trail Qwen/Meta on coding benchmarks. Mamba-2 llama.cpp support may be immature.

**Verdict:** Interesting for the embedded/zero-GPU goal. Test only if Qwen3.5-4B fails.

#### 11. SmolLM3-3B ⭐ (HuggingFace's latest small model)

| Property | Value |
|----------|-------|
| Total params | 3B |
| Active params | 3B (dense) |
| Architecture | Dense Transformer |
| Context window | 128K |
| GGUF Q4_K_M size | ~1.8GB |
| Tool calling | ✅ Native support |
| Dual-mode reasoning | ✅ Can think or not |
| Training data | 11.2T tokens |
| License | Apache 2.0 |

**Why notable:** HuggingFace's flagship small model with native tool calling. Dual-mode reasoning means it can think when needed, skip when not. 11.2T training tokens is massive for a 3B model.

**Risk:** 3B may lack depth for hard classification. Newer/less tested than Qwen3.

**Verdict:** Alternative to Qwen3-4B if we need the absolute smallest viable model.

#### 12. LFM2-8B-A1B ⭐ (Liquid AI's on-device MoE)

| Property | Value |
|----------|-------|
| Total params | 8.3B |
| Active params | 1.5B |
| Architecture | GQA + sparse MoE |
| GGUF size | 4.7GB (Q4_0) to 16.7GB (F16) |
| License | Liquid AI license |

**Why notable:** Only 1.5B active params, fastest MoE in its class, optimized for on-device. Quality comparable to 3-4B dense models.

**Risk:** 1.5B active likely too weak for our use case. Liquid AI license may restrict use. Less community support.

**Verdict:** Only for extreme size-constrained deployments.

---

## Models Considered and Rejected

| Model | Why Rejected |
|-------|-------------|
| Llama-4-Scout (109B, 17B active) | 17B active params — 5.7x our current. GGUF ~55-65GB. Way too large. |
| Llama-4-Maverick (400B, 17B active) | Even larger. Not practical for local. |
| Mistral-Small-3.1 (24B) | Dense 24B — way too large for reranker use case. |
| Qwen3-30B-A3B | Superseded by Qwen3.5-35B-A3B (what we already use). Same active params, older training. |
| DeepSeek-R1-Distill-7B/14B | Based on Qwen2.5 — older training. R1 distillation optimizes for CoT, not classification. |
| Llama-3.2-3B | Significantly weaker than Qwen3-4B on classification (distil labs). IFEval 77.4. |
| OLMoE-1B-7B | Older training (2024), 1.3B active too small for 5K-token prompt. |
| SmolLM-2 (1.7B) | Too small for complex few-shot classification. |
| Qwen3-0.6B / 1.7B / Qwen3.5-0.8B | Too small — below threshold for reliable structured JSON output. |
| Gemma-3-12B / 27B | 12B doesn't beat Qwen3.5-9B; 27B too large. |
| Yi-2 / InternLM-3 | Limited GGUF ecosystem, no clear advantage over Qwen3.5. |
| GLM-4.7-Flash (30B/3B MoE) | Same class as current model, no size reduction. MIT license is nice but not enough reason. |

---

## Qwen3 vs Qwen3.5 Architecture Comparison

| Feature | Qwen3 | Qwen3.5 |
|---------|-------|---------|
| Architecture | Standard Transformer | Gated DeltaNet + Transformer hybrid |
| Context (native) | 32K (dense), 128K (MoE/32B) | **262K** |
| Context (extended) | 131K with YaRN | **1.01M** with YaRN |
| Languages | 119 | **201** |
| Think mode | Hard + soft (`/think`, `/no_think`) | Hard only (`enable_thinking`) |
| MTP (speculative decode) | No | **Yes** |
| Multimodal | Text only | **Vision + Text** |
| llama.cpp arch | `qwen3` (mature) | `qwen3_5` (newer, ~35% DeltaNet overhead in current builds) |

**Key tradeoff:** Qwen3.5 is dramatically better on benchmarks but may have llama.cpp performance penalties from DeltaNet layers. If the ~35% slowdown applies, Qwen3.5-9B at ~100 tok/s is still comparable to Qwen3-8B at ~150 tok/s — but Qwen3.5-9B should produce better quality output.

---

## Qwen3 Dense Model Benchmarks (Base, from Technical Report)

### Qwen3-4B vs Competitors

| Benchmark | Gemma-3-4B | Qwen2.5-3B | Qwen2.5-7B | **Qwen3-4B** |
|-----------|-----------|-----------|-----------|-----------|
| MMLU | 59.51 | 65.62 | 74.16 | **72.99** |
| MMLU-Pro | 29.23 | 34.61 | 45.00 | **50.58** |
| BBH | 51.70 | 56.30 | 70.40 | **72.59** |
| GSM8K | 43.97 | 79.08 | 85.36 | **87.79** |
| EvalPlus | 43.23 | 46.28 | 62.18 | **63.53** |
| MBPP | 46.40 | 54.60 | 63.40 | **67.00** |

### Qwen3-8B vs Competitors

| Benchmark | Llama-3-8B | Qwen2.5-7B | Qwen2.5-14B | **Qwen3-8B** |
|-----------|-----------|-----------|------------|-----------|
| MMLU | 66.60 | 74.16 | 79.66 | **76.89** |
| MMLU-Pro | 35.36 | 45.00 | 51.16 | **56.73** |
| BBH | 57.70 | 70.40 | 78.18 | **78.40** |
| GPQA | 25.80 | 36.36 | 32.83 | **44.44** |
| EvalPlus | 44.13 | 62.18 | 60.70 | **67.65** |
| MBPP | 48.40 | 63.40 | 69.00 | **69.80** |

**Pattern:** Each Qwen3 dense model ≈ Qwen2.5 model at 2x size. Qwen3-4B ≈ Qwen2.5-7B. Qwen3-8B ≈ Qwen2.5-14B.

---

## Inference Performance Estimates

### GPU Performance (Q4_K_M, llama-server, measured benchmarks)

**RTX 5090 (32GB, 1792 GB/s)** — actual benchmarks:

| Model | Params | tg @ 4K ctx | tg @ 8K ctx | PP @ 4K |
|-------|--------|-------------|-------------|---------|
| Qwen3-MoE-30B-A3B | 30B (3B active) | **234 tok/s** | 170 | 6,630 |
| Qwen3.5-35B-A3B | 35B (3B active) | **205 tok/s** | — | 793 |
| Qwen3-8B | 8B dense | 186 tok/s | 170 | 10,406 |
| Qwen3-14B | 14B dense | 124 tok/s | 115 | 6,498 |

**RTX 4090 (24GB, 1008 GB/s)** — actual + estimated:

| Model | GGUF Size | VRAM | Est. tok/s (tg) | Est. 200-token latency |
|-------|-----------|------|------------------|----------------------|
| Qwen3.5-4B | 2.5GB | 3.5GB | 150-220* | 0.9-1.3s |
| Qwen3-4B | 2.5GB | 3.5GB | 200-300 | 0.7-1.0s |
| Qwen3.5-9B | 5.5GB | 6.8GB | 80-120* | 1.7-2.5s |
| Qwen3-8B | 5.0GB | 6GB | ~128 (measured for 8B) | 1.3-1.6s |
| **Current: Qwen3.5-35B-A3B** | **~20GB** | **~21.7GB** | **~140-170** | **~1.1s** |

*Qwen3.5 DeltaNet layers may add ~35% overhead in current llama.cpp builds

**RTX 3070 (8GB)** — actual benchmark:

| Model | tok/s @ 4K | tok/s @ 32K | Note |
|-------|-----------|-------------|------|
| Qwen3.5-9B | **57.9** | 54.9 | Full GPU offload, fits in 8GB |
| LLaMA-3-8B | ~73 | ~62 | Standard transformer, no DeltaNet overhead |

**Critical finding:** Qwen3.5-9B at 6.8GB fits in 8GB GPUs. 58 tok/s = ~3.4s for 200 tokens. With non-thinking mode producing ~50-100 tokens for classification, actual latency would be ~1-2s on a 3070.

### CPU Performance (Q4_K_M)

**Apple Silicon** — actual benchmarks:

| CPU | Model | tok/s | 200-token latency |
|-----|-------|-------|-------------------|
| M4 Max | ~3B | ~100-150 (estimated) | 1.3-2.0s |
| M3 Max | 8B | 51 | 3.9s |
| M2 Ultra | 8B | 76 | 2.6s |

**x86 CPU:**

| CPU | Model | tok/s | 200-token latency |
|-----|-------|-------|-------------------|
| Ryzen AI 9 | Llama-3.2-1B | 50.7 | 3.9s |
| ARM mobile | Llama-3.2-3B | 19.9 | 10s |
| i7-8700K | 8B (CPU spill) | 6.6-10.9 | 18-30s |

**CPU viability verdict:** Only Apple M-series can hit <2s on models >3B. x86 CPU reranking is not viable for generative models >1B. For CPU-only deployment, use cross-encoders (see `small-model-research-2026-04-02.md`).

### Memory Requirements (Q4_K_M, 32K context)

| Model Size | Total VRAM/RAM |
|-----------|---------------|
| 0.6B | ~1.2 GB |
| 3B | ~2.7 GB |
| 4B | ~3.2 GB |
| 9B | **6.8 GB** |
| 14B | ~9.7 GB |
| 35B MoE | ~21.7 GB |

### llama.cpp Optimization Tips

1. **Flash Attention** (`-fa 1`) — reduces KV cache memory
2. **KV cache quantization** (`--cache-type-k q8_0`) — halves context VRAM
3. **100% GPU offload is critical** — even 4 layers to CPU = 70% speed collapse
4. **Only 5 CPU threads** needed to saturate DDR5 bandwidth
5. **ik_llama.cpp fork** — 1.9x faster MoE inference (relevant for Qwen3.5-35B-A3B)

**Key insight:** For <2s CPU latency, we need either:
- A model <3B with short output (<50 tokens) — see `small-model-research-2026-04-02.md`
- Or accept GPU as a requirement for the LLM reranker stage

### Optimization: Speculative Decoding

With a 1B draft model + 8B validator: up to 180+ tok/s on GPU, nearly doubling throughput. This could bring Qwen3-8B into the <1s range.

---

## Critical Comparison: Qwen3 Non-Thinking Mode

All Qwen3 models support seamless thinking/non-thinking toggle:

```python
# Non-thinking mode (our use case)
messages = [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>\n\n</think>\n"}  # Force non-thinking
]
```

Or via `enable_thinking: false` in chat_template_kwargs (same as our current Qwen3.5-35B-A3B setup).

This is critical — non-thinking mode means:
- No internal CoT overhead
- Shorter outputs (no `<think>` blocks)
- ~2-3x faster than thinking mode
- Our current prompt already uses reasoning-in-response (explicit `"reasoning"` field in JSON)

---

## Benchmark Strategy

### Phase 1: Qwen3.5 Head-to-Head (3 models, same hardware)

**Primary candidates:**

1. **Qwen3.5-9B** — Does IFEval 91.5 translate to our reranking task? Establishes new quality ceiling.
2. **Qwen3.5-4B** — Can 4B DeltaNet match our current 3B-active MoE?
3. **Qwen3-8B** — Control/fallback if Qwen3.5 has llama.cpp issues.

**Protocol:**
- Same eval framework (587 fixtures)
- Same prompt (7 few-shot, reasoning-in-response)
- Non-thinking mode for all (`enable_thinking: false`)
- Measure: recall per tier, noise ratio, negative silence, parse failure rate, latency
- Run on same hardware as current Qwen3.5-35B-A3B benchmarks

**Prerequisite:** Verify Qwen3.5 GGUF models work in llama-server with `--jinja` flag. DeltaNet architecture support in llama.cpp may be recent — test basic inference first.

### Phase 2: Explore Alternatives (if Phase 1 disappoints)

4. **Nemotron-3-Nano-30B-A3B** — If we're keeping GPU, this is the tool-calling champion
5. **Qwen3-4B** — Proven architecture, mature llama.cpp support
6. **Gemma-3n-E4B** — Different architecture, different failure modes

### Phase 3: Size Optimization (if Phase 1 identifies a winner)

- Run the smallest viable winner on CPU with Q4_K_M
- Test with shorter output format (reduce from ~200 to ~50 tokens if possible)
- Evaluate speculative decoding (Qwen3.5-2B draft + Qwen3.5-9B validator)

---

## Recommendation

**Benchmark Qwen3.5-9B and Qwen3.5-4B first.** Here's why:

1. **Qwen3.5-9B IFEval 91.5** is the highest instruction-following score in the small model class — this directly predicts JSON compliance and structured output quality
2. **BFCL-V4 66.1** means it's trained on tool use — exactly our domain
3. **Same `enable_thinking` interface** as our current Qwen3.5-35B-A3B setup
4. **Qwen3.5-4B at 2.5GB** could be the dream: same architecture benefits, laptop-viable
5. **If Qwen3.5-9B matches quality at ~5.5GB:** 4x smaller than current, runs on any GPU

**Fallback chain:** Qwen3.5-9B → Qwen3.5-4B → Qwen3-8B → Qwen3-4B → Nemotron-3-Nano

**Don't bother with:** Llama-4 (too large), DeepSeek-R1-Distill (older base, CoT-optimized), OLMoE (too weak), cross-encoders (covered in separate research — see `small-model-research-2026-04-02.md`).

---

## Qwen3.5 Family Reference (Complete)

### Dense Models

| Model | Params | Released | Context |
|-------|--------|---------|---------|
| Qwen3.5-0.8B | 0.8B | 2026-02-28 | 262K |
| Qwen3.5-2B | 2B | 2026-02-28 | 262K |
| **Qwen3.5-4B** | **4B** | **2026-02-27** | **262K** |
| **Qwen3.5-9B** | **9B** | **2026-02-27** | **262K** |
| Qwen3.5-27B | 27B | 2026-02-24 | 262K |

### MoE Models

| Model | Total/Active | Released | Context |
|-------|-------------|---------|---------|
| **Qwen3.5-35B-A3B** (current) | 35B/3B | 2026-02-24 | 262K |
| Qwen3.5-122B-A10B | 122B/10B | 2026-02-24 | 262K |
| Qwen3.5-397B-A17B | 397B/17B | 2026-02-16 | 262K |

---

## Sources

- [Qwen3 Blog: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3 Technical Report (arXiv)](https://arxiv.org/html/2505.09388v1)
- [Qwen3.5-9B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Distil Labs: 12 SLM Benchmark](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/)
- [Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1)
- [Gemma 3n Overview (Google)](https://ai.google.dev/gemma/docs/gemma-3n)
- [Gemma 3n Developer Guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- [Phi-4-mini Model Card (HuggingFace)](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [Phi-4 Technical Report](https://arxiv.org/html/2412.08905v1)
- [Nemotron-3-Nano Model Card (NVIDIA)](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard)
- [Liquid AI LFM2-8B-A1B](https://www.liquid.ai/blog/lfm2-8b-a1b-an-efficient-on-device-mixture-of-experts)
- [IBM Granite 4.0 Docs](https://www.ibm.com/granite/docs/models/granite)
- [SmolLM3-3B (HuggingFace)](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [OLMoE Paper (arXiv)](https://arxiv.org/abs/2409.02060)
- [Llama 4 Scout/Maverick (Meta)](https://www.llama.com/models/llama-4/)
- [NVIDIA llama.cpp Acceleration Blog](https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/)
- [LocalLLM VRAM Guide 2026](https://localllm.in/blog/ollama-vram-requirements-for-local-llms)
- [BentoML: Best Open-Source SLMs 2026](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
- [StructEval Benchmark](https://tiger-ai-lab.github.io/StructEval/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
