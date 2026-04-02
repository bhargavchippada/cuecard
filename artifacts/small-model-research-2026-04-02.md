# Small Model Research: CPU Rerankers for cuecard

> Research date: 2026-04-02
> Goal: Find models <3B params that can rerank 5-20 candidates on CPU in <500ms

## Executive Summary

**There are two fundamentally different approaches, and the best answer uses both:**

1. **Cross-encoder rerankers** (33M-568M params): Score query-document pairs directly via a single forward pass. ~5-50ms per pair on CPU. No text generation, no JSON parsing, no prompt engineering. But they can't follow custom criteria — they score generic "relevance."

2. **Generative rerankers** (0.6B-3.8B params): Use an LLM to reason about relevance with custom instructions. ~300ms-5s per query on CPU. Can follow few-shot prompts, return structured output. But they're slower and may fail at classification with small models.

**Recommendation: Try the Qwen3-Reranker-0.6B-seq-cls conversion first** (cross-encoder mode, no generation). If that fails, try GTE-Reranker-ModernBERT-Base as a fast cross-encoder. Only go generative if cross-encoders can't handle cuecard's domain-specific relevance criteria.

---

## Tier 1: Purpose-Built Cross-Encoder Rerankers

These models score (query, document) pairs directly. One forward pass → one relevance score. No text generation.

### 1. GTE-Reranker-ModernBERT-Base ⭐⭐⭐

| Property | Value |
|----------|-------|
| Parameters | 149M |
| Architecture | ModernBERT (sequence classification) |
| Size on disk | ~600MB (fp32), ~300MB (fp16) |
| Max tokens | 8,192 |
| BEIR avg | 56.73 |
| CoIR (code) | **79.99** |
| MTEB Reranking | 59.24 |
| Hit@1 | 83.00% (matches nemotron-1.2B!) |
| CPU latency | ~5-15ms per pair (estimated from architecture size) |
| Framework | sentence-transformers CrossEncoder, Transformers, ONNX |
| fastembed | ❌ Not natively supported (but ONNX available) |
| License | Apache 2.0 |

**Why it matters:** 149M params, 8x smaller than most competitors, matches 1.2B models on Hit@1. The CoIR (code retrieval) score of 79.99 is critical — this model understands code. ModernBERT architecture is optimized for efficiency.

**Pros:**
- Extremely small and fast
- Strong code retrieval performance (CoIR 79.99)
- ModernBERT architecture is newer/faster than BERT
- ONNX export available for CPU optimization
- Deterministic scoring (no generation variance)

**Cons:**
- Not in fastembed's model list (would need `add_custom_model()` or use sentence-transformers)
- Can't follow custom relevance instructions — scores generic similarity
- May not discriminate "this rule is about git force-push, not git rebase" without custom training
- No GGUF version (not LLM-based)

**How to test with cuecard:**
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("Alibaba-NLP/gte-reranker-modernbert-base")
pairs = [[query, rule.text] for rule in candidates]
scores = model.predict(pairs)  # Returns float scores in [0, 1]
```

---

### 2. Jina Reranker v1-Tiny-en ⭐⭐

| Property | Value |
|----------|-------|
| Parameters | 33M |
| Layers | 4 |
| Size on disk | ~130MB (ONNX) |
| Max tokens | 8,192 |
| BEIR NDCG@10 | 48.54 (92.5% of base) |
| CPU latency | ~2-6ms per pair (estimated, 5x faster than 137M base) |
| Framework | fastembed ✅, sentence-transformers, Transformers |
| fastembed model | `jinaai/jina-reranker-v1-tiny-en` |
| License | Apache 2.0 |

**Why it matters:** Already in fastembed — zero integration work. 33M params is microscopic. 5x faster than the base model on CPU.

**Pros:**
- **Already in fastembed** — `TextCrossEncoder("jinaai/jina-reranker-v1-tiny-en")`
- Only 130MB, loads in <1s
- 5x faster than base model
- Deterministic scoring
- We already have fastembed as a dependency

**Cons:**
- BEIR 48.54 is lower than GTE-ModernBERT's 56.73
- Trained on ms-marco (web search) — same domain mismatch as the MiniLM we already rejected
- No code-specific training
- English only

**How to test with cuecard:**
```python
from fastembed.rerank.cross_encoder import TextCrossEncoder
encoder = TextCrossEncoder("jinaai/jina-reranker-v1-tiny-en")
scores = list(encoder.rerank(query, [r.text for r in candidates]))
```

---

### 3. Jina Reranker v1-Turbo-en

| Property | Value |
|----------|-------|
| Parameters | 37.8M |
| Layers | 6 |
| Size on disk | ~150MB (ONNX) |
| Max tokens | 8,192 |
| BEIR NDCG@10 | 49.60 (95% of base) |
| CPU latency | ~3-10ms per pair (estimated, 3x faster than base) |
| Framework | fastembed ✅ |
| fastembed model | `jinaai/jina-reranker-v1-turbo-en` |
| License | Apache 2.0 |

Similar to tiny but slightly better quality. Same ms-marco domain mismatch concern.

---

### 4. BAAI/bge-reranker-base

| Property | Value |
|----------|-------|
| Parameters | 278M |
| Size on disk | ~1.04GB (ONNX) |
| BEIR avg | ~53 |
| CPU latency | ~88s for 100 docs (per benchmark), so ~15-20ms per pair |
| Framework | fastembed ✅ |
| fastembed model | `BAAI/bge-reranker-base` |
| License | MIT |

Decent but heavier than GTE-ModernBERT for worse performance. The bge-reranker-v2-m3 (568M) is even heavier at ~257s for 100 docs on CPU.

---

### 5. mxbai-rerank-base-v2 (mixedbread)

| Property | Value |
|----------|-------|
| Parameters | ~500M (Qwen-2.5 architecture) |
| BEIR avg | 55.57 |
| Code search | 31.73 |
| GPU latency (A100) | 0.67s |
| CPU latency | Not benchmarked (likely ~1-3s for 20 docs) |
| ONNX | Available via export |
| fastembed | ❌ |
| License | Apache 2.0 |

**Note:** Code search score of 31.73 is much lower than GTE-ModernBERT's 79.99. Trained with GRPO which is innovative, but the code performance is weak.

---

### 6. FlashRank (ms-marco-TinyBERT-L-2-v2)

| Property | Value |
|----------|-------|
| Parameters | ~4.5M |
| Size on disk | **~4MB** |
| Max tokens | 512 |
| Framework | FlashRank (ONNX, no PyTorch/Transformers needed) |
| CPU latency | Ultra-fast (~1-3ms per pair) |
| License | Apache 2.0 |

**The smallest possible reranker.** No PyTorch or Transformers dependency. But only 512 tokens max and quality is significantly lower than larger models. Could be useful as a pre-filter.

---

## Tier 2: Generative Rerankers (LLM-based)

These use an LLM to read the query+document and generate "yes"/"no" or a relevance score. Can follow custom instructions.

### 7. Qwen3-Reranker-0.6B ⭐⭐⭐ (Best generative option)

| Property | Value |
|----------|-------|
| Parameters | 0.6B |
| Size (Q4_K_M) | **395MB** |
| Size (Q8_0) | 639MB |
| Architecture | Causal LM (generative) |
| Max tokens | 32K |
| MTEB-R | 65.80 |
| MTEB-Code | **73.42** |
| MMTEB-R | 66.36 |
| GGUF | ✅ Multiple quantizations available |
| License | Apache 2.0 |

**Critical discovery: `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`** — A community conversion that turns this generative model into a **sequence classification model**. This means:
- Single forward pass (no token generation)
- Returns logits directly → sigmoid → score
- Deterministic (no sampling variance)
- Much faster on CPU than generative inference
- Can use with sentence-transformers `CrossEncoder`

**This is potentially the best of both worlds:** a model trained specifically for reranking with custom instructions (MTEB-Code 73.42!) but runnable as a fast cross-encoder.

**Pros:**
- Purpose-trained for reranking (not generic chat)
- Excellent code benchmark (73.42)
- Supports custom instructions ("Given coding rules and a tool action...")
- Seq-cls conversion available for fast CPU inference
- GGUF available for generative mode via llama-server
- Apache 2.0

**Cons:**
- 0.6B is larger than cross-encoders (395MB-639MB vs 130-600MB)
- Seq-cls conversion is community-maintained (tomaarsen), not official
- The seq-cls version needs the special prompt format (instruction + query + document)
- Unclear if seq-cls mode preserves the instruction-following benefit
- We already tested Qwen3-0.6B (chat model) and it failed — but this is the reranker variant

**How to test with cuecard (seq-cls mode):**
```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls")

# Format with cuecard-specific instruction
instruction = "Given a coding rule and an AI agent tool action, determine if the rule is relevant"
prefix = '<|im_start|>system\nJudge whether the Document meets the requirements...<|im_end|>\n<|im_start|>user\n'
query_formatted = f"{prefix}<Instruct>: {instruction}\n<Query>: {tool_action}\n"
doc_formatted = f"<Document>: {rule_text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

pairs = [[query_formatted, doc_formatted] for rule_text in candidate_texts]
scores = model.predict(pairs)
```

**How to test with cuecard (GGUF generative mode):**
```bash
llama-server -m Qwen3-Reranker-0.6B-Q4_K_M.gguf --port 8082 -ngl 0 -c 4096
```
Then use existing `llm_reranker.py` with adapted prompt.

---

### 8. InRanker-small (T5, 60M) / InRanker-base (220M)

| Property | Value |
|----------|-------|
| Parameters | 60M / 220M |
| Architecture | T5 seq2seq |
| BEIR NDCG@10 | 48.07 (60M) / 50.08 (220M) |
| Framework | rerankers library, Transformers |
| GGUF | ❌ (T5, not causal LM) |

Distilled from monoT5-3B. Generates "true"/"false" tokens. Lower quality than cross-encoders. T5 architecture is not GGUF-compatible. Available through the `rerankers` library.

---

### 9. Qwen2.5-1.5B-Instruct (General LLM for reranking)

| Property | Value |
|----------|-------|
| Parameters | 1.5B |
| Size (Q4_K_M) | ~1GB |
| Max tokens | 128K |
| GGUF | ✅ (official Qwen GGUF repo) |
| CPU speed | ~20-30 tok/s on modern x86 (estimated) |
| JSON output | Good (Qwen2.5 improved structured output) |

Could run our existing few-shot reranker prompt. 1.5B is 2.5x larger than Qwen3-Reranker-0.6B but has general instruction following. Expected latency: ~2-4s for our 5K token prompt on CPU (too slow for 500ms target).

---

### 10. Gemma-2-2B-it

| Property | Value |
|----------|-------|
| Parameters | 2.6B |
| Size (Q4) | ~1.2-1.5GB |
| GGUF | ✅ (bartowski, others) |
| CPU speed | ~10-20 tok/s (estimated) |
| JSON compliance | Good |

Too large for <500ms on CPU. Strong reasoning but the latency doesn't fit.

---

## Tier 3: Hybrid/Novel Approaches

### 11. ColBERT v2 (Late Interaction)

Not suitable for our use case — ColBERT is designed for retrieval, not reranking. Requires pre-computed token embeddings. Overkill for 5-20 candidates.

### 12. Fine-tuned cross-encoder

Could fine-tune GTE-ModernBERT or bge-reranker-base on cuecard's specific task. This would teach a cross-encoder our domain-specific relevance criteria. Requires training data (our 587 fixtures could work).

---

## Latency Estimates Summary

| Model | Params | Type | Est. CPU latency (20 candidates) | Size |
|-------|--------|------|----------------------------------|------|
| FlashRank TinyBERT | 4.5M | Cross-encoder | **~20-60ms** | 4MB |
| Jina-tiny-en | 33M | Cross-encoder | **~40-120ms** | 130MB |
| Jina-turbo-en | 38M | Cross-encoder | **~60-200ms** | 150MB |
| GTE-ModernBERT-Base | 149M | Cross-encoder | **~100-300ms** | 600MB |
| bge-reranker-base | 278M | Cross-encoder | ~300-600ms | 1GB |
| Qwen3-Reranker-0.6B (seq-cls) | 600M | Cross-encoder* | ~200-500ms | 1.2GB |
| Qwen3-Reranker-0.6B (GGUF gen) | 600M | Generative | ~1-3s | 395MB |
| mxbai-rerank-base-v2 | 500M | Cross-encoder | ~500ms-1.5s | 1GB |
| bge-reranker-v2-m3 | 568M | Cross-encoder | ~1-2s | 1.1GB |
| Qwen2.5-1.5B (GGUF gen) | 1.5B | Generative | ~2-4s | 1GB |
| Gemma-2-2B (GGUF gen) | 2.6B | Generative | ~3-6s | 1.5GB |

*seq-cls mode: single forward pass, but 600M params means ~3x slower than 149M for the same approach

---

## Ranked Recommendations

### Rank 1: Qwen3-Reranker-0.6B-seq-cls (Cross-encoder mode)
**Why:** Purpose-trained for reranking, MTEB-Code 73.42 (best code score of any small model), supports custom instructions via prompt formatting, and the seq-cls conversion makes it usable as a fast cross-encoder. This model was literally designed for our task.
**Risk:** The seq-cls conversion may not preserve instruction-following quality. Needs benchmarking.
**Test first:** Run on our 354 basic fixtures in seq-cls mode, compare to embedding-only baseline.

### Rank 2: GTE-Reranker-ModernBERT-Base
**Why:** 149M params, CoIR 79.99 (understands code), ModernBERT is fast. Proven architecture with strong benchmarks.
**Risk:** Can't follow custom relevance criteria. May suffer same domain mismatch as MiniLM (trained on general search, not coding rules). But CoIR score suggests it handles code better.
**Test:** Use sentence-transformers CrossEncoder, score candidates, compare threshold-based filtering.

### Rank 3: Jina-reranker-v1-tiny-en (via fastembed)
**Why:** Already in fastembed (zero integration work), 33M params, ~2-6ms per pair.
**Risk:** Trained on ms-marco (web search). We already know MiniLM (also ms-marco) regressed on code. But it's so cheap to test that it's worth trying.
**Test:** `TextCrossEncoder("jinaai/jina-reranker-v1-tiny-en")` — 5 minutes to benchmark.

### Rank 4: Qwen3-Reranker-0.6B (GGUF generative mode)
**Why:** If seq-cls doesn't work, try the full generative approach with a simplified prompt. 395MB Q4_K_M. Custom instructions possible.
**Risk:** ~1-3s latency on CPU exceeds 500ms target. May need prompt simplification (our 5K prompt won't fit well in 4K context).
**Test:** `llama-server -m Qwen3-Reranker-0.6B-Q4_K_M.gguf --port 8082 -ngl 0 -c 4096`

### Rank 5: FlashRank TinyBERT (ultra-fast fallback)
**Why:** 4MB, no PyTorch dependency, ~1-3ms per pair. Could serve as a fast pre-filter before LLM reranking.
**Risk:** Quality is the lowest of all options. 512 token limit may truncate rules.
**Test:** `pip install flashrank` — trivial to try.

---

## Overall Strategy Recommendation

**Two-phase approach:**

### Phase A: Fast cross-encoder benchmark (1-2 hours)
Test models #1, #2, #3 on our 354 basic fixtures:
1. Qwen3-Reranker-0.6B-seq-cls
2. GTE-Reranker-ModernBERT-Base  
3. Jina-reranker-v1-tiny-en (already in fastembed)

Metric: Can any cross-encoder achieve >80% negative silence with a simple score threshold?

### Phase B: Generative fallback (if Phase A fails)
If no cross-encoder handles the domain-specific discrimination:
1. Test Qwen3-Reranker-0.6B in GGUF generative mode with simplified prompt
2. Accept ~1-2s latency if quality is there

### Phase C: Hybrid (best of both)
Use a fast cross-encoder (Jina-tiny, ~50ms) as Stage 2 filter, then Qwen3-Reranker-0.6B-seq-cls (~300ms) as Stage 3. Total: ~350ms.

---

## Key Insight: Cross-Encoder vs Our Current Approach

Our current LLM reranker works because it can follow **custom criteria**:
- "Is this git rule relevant to a Bash: git commit action?"
- "The user is running Read: README.md — no coding rules apply"

A generic cross-encoder just scores "how semantically related are these two texts?" It can't discriminate between "git force-push" and "git commit" the way our few-shot prompt can.

**The Qwen3-Reranker-0.6B-seq-cls is the only model that might bridge this gap** — it accepts an instruction parameter that can encode our custom criteria. This is why it's Rank 1 despite being larger.

If cross-encoders can't handle this discrimination, we're stuck with generative approaches (slower) or fine-tuning a cross-encoder on our data (requires training infrastructure).

---

## Sources

- [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) — HuggingFace model card
- [Qwen3-Reranker-0.6B-seq-cls](https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls) — Seq classification conversion
- [Qwen3-Reranker-0.6B GGUF](https://huggingface.co/Mungert/Qwen3-Reranker-0.6B-GGUF) — Quantized versions
- [GTE-Reranker-ModernBERT-Base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) — 149M reranker
- [Jina Rerankers Turbo and Tiny](https://jina.ai/news/smaller-faster-cheaper-jina-rerankers-turbo-and-tiny/) — 33M/38M models
- [fastembed Supported Models](https://qdrant.github.io/fastembed/examples/Supported_Models/) — Cross-encoder list
- [mxbai-rerank-v2 blog](https://www.mixedbread.com/blog/mxbai-rerank-v2) — GRPO-trained reranker
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) — Ultra-lightweight ONNX rerankers
- [rerankers library](https://github.com/AnswerDotAI/rerankers) — Unified reranking API
- [InRanker paper](https://arxiv.org/abs/2401.06910) — Distilled T5 rerankers
- [Reranker Speed Showdown](https://medium.com/@xiweizhou/speed-showdown-reranker-1f7987400077) — CPU benchmarks
- [ZeroEntropy Reranking Guide](https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025) — Cross-encoder vs LLM comparison
- [Reranker Benchmark Top 8](https://research.aimultiple.com/rerankers/) — Model comparison
- [Sentence Transformers Efficiency](https://sbert.net/docs/cross_encoder/usage/efficiency.html) — ONNX/OpenVINO backends
