# Small Model Investigation: CPU-Only Reranking + Expansion

> Goal: Replace Qwen3.5-35B (21GB, GPU required) with a 0.6B model (~400MB, CPU-only) for both expansion generation and reranking, making cuecard fully self-contained without GPU dependency.

## Motivation

The current quality stack requires:
- fastembed embedding model (~150MB, CPU) — automatic download
- Qwen3.5-35B via llama-server (21GB, GPU) — manual setup

This is a barrier to adoption. If a 0.6B model can achieve 80%+ of the reranker's quality, the entire pipeline becomes:
```
pip install cuecard && cuecard setup → done
```

No GPU. No llama-server. No manual model download.

## The Two Tasks

### 1. Expansion Generation (offline, index time)
- **Current:** Qwen3.5-35B generates 3-10 trigger phrases per rule (~1s/rule, GPU)
- **Target:** 0.6B model generates same quality expansions (~2-5s/rule, CPU)
- **Quality bar:** Expansions must bridge vocabulary gap (cosine delta > +0.20 on hard fixtures)
- **Latency tolerance:** High — runs once per `cuecard setup` or `cuecard rules expand`

### 2. Reranking (online, query time)
- **Current:** Qwen3.5-35B classifies 5-20 candidates as relevant/irrelevant (~1.5s, GPU)
- **Target:** 0.6B model does same classification (~300-500ms, CPU)
- **Quality bar:** Negative silence >80%, noise <30% (vs 91%/21% with 35B)
- **Latency tolerance:** Medium — must be <1s total pipeline for PreToolUse hooks

## Models to Benchmark

| Model | Size (Q4/Q8) | Type | Expected Strength |
|-------|-------------|------|-------------------|
| **Qwen3-0.6B** | 379MB (Q4) | General chat | Promptable, can follow few-shot examples |
| **Qwen3-Reranker-0.6B** | 639MB (Q8) | Purpose-built reranker | MTEB-Code 73.42, trained for relevance scoring |
| Phi-4-mini (3.8B) | ~2.2GB (Q4) | Strong reasoning | If 0.6B isn't enough quality |
| Llama-3.2-1B | ~700MB (Q4) | General | Fallback comparison |

Priority: Qwen3-Reranker-0.6B > Qwen3-0.6B > Phi-4-mini

## Benchmark Plan

### Phase 1: Reranker Quality (via llama-server, CPU-only)

For each model:
1. Start llama-server on CPU: `llama-server -m ~/models/<model>.gguf --port 8082 -ngl 0 -c 4096 --jinja`
2. Run enriched+LLM benchmark on basic.json (354 fixtures) using port 8082
3. Capture: noise, negative silence, recall, latency per query
4. Per-tier breakdown (easy/medium/hard/negative)

**Baseline comparison:** Qwen3.5-35B on GPU (noise=21.4%, neg silence=91.0%, recall=44.8%, p50=1.6s)

**Pass criteria:**
- Negative silence > 80%
- Noise < 30%
- Recall within 5% of 35B baseline
- p50 latency < 1s on CPU

### Phase 2: Expansion Generation Quality

For each model that passes Phase 1:
1. Generate expansions for 10 sample rules (5 coding, 5 workflow)
2. Compare expansion quality to Qwen3.5-35B output:
   - Vocabulary diversity (unique tokens)
   - Cosine delta on hard fixtures
   - Number of near-duplicates (cosine > 0.85)
   - Trigger specificity (are they concrete tool commands or vague paraphrases?)
3. Run mini-benchmark: build index with small-model expansions, eval on 50 hard fixtures

**Pass criteria:**
- Avg cosine delta > +0.15 on hard fixtures (vs +0.276 with 35B v2 prompt)
- < 20% near-duplicate rate
- > 60% of expansions rated as "concrete trigger phrase" (manual spot-check)

### Phase 3: Integrated Benchmark

Best small model from Phases 1-2:
1. Generate all expansions with the small model
2. Build enriched index
3. Run reranking with the same small model
4. Full 587-fixture benchmark
5. Compare to Qwen3.5-35B end-to-end

### Phase 4: Embedding the Model (no server)

If quality is good enough:
1. Evaluate `llama-cpp-python` for in-process inference
2. Design `cuecard serve` daemon:
   - Auto-starts on first hook call
   - Keeps model loaded in memory
   - Shuts down after idle timeout (5 min default)
   - Unix socket or localhost HTTP for IPC
3. Alternative: fastembed-style integration if model is small enough to load per-call (<2s)

## Architecture: One Model, Two Modes

```python
# Config
[pipeline]
mode = "llm-local"

[pipeline.llm]
model = "Qwen3-0.6B-Q4_K_M.gguf"  # auto-downloaded by cuecard setup
local_endpoint = "auto"             # cuecard manages the server
```

When `local_endpoint = "auto"`:
1. `cuecard setup` downloads the GGUF to `~/.cuecard/models/`
2. At query time, cuecard checks if the daemon is running
3. If not, starts it (loads model, ~2-5s cold start)
4. Subsequent calls are fast (~300ms)
5. Daemon auto-exits after 5 min idle

This means:
- First hook call after boot: ~5s (cold start)
- Subsequent calls: ~300ms
- No manual server management
- Model serves both expansion and reranking

## Quality Expectations

| Config | Neg Silence | Noise | Recall | p50 | GPU? |
|--------|-------------|-------|--------|-----|------|
| Embedding only | 0% | 84% | 47% | 15ms | No |
| + 35B reranker (current) | 91% | 21% | 45% | 1.6s | Yes |
| + 0.6B reranker (target) | >80% | <30% | >40% | <500ms | **No** |

If the 0.6B model hits these targets, we can make `llm-local` the default mode with zero GPU requirement. That's the product goal.

## Risk Factors

1. **Prompt length:** Our reranker prompt is ~5K tokens (7 few-shot examples). A 0.6B model with 4K context may need a simplified prompt.
2. **Classification accuracy:** Binary classification is simple, but cross-domain discrimination ("git commit" vs "Update README") requires nuance that small models may lack.
3. **Expansion quality:** Small models may produce more paraphrases and fewer concrete trigger phrases.
4. **CPU latency variance:** Depends heavily on CPU (x86 vs ARM, AVX2 vs AVX-512).

## Decision Points

- If 0.6B model achieves >80% neg silence: **proceed with embedded model**
- If 0.6B fails but 3.8B (Phi-4-mini) succeeds: **still viable, ~2GB download**
- If all small models fail to reach 80% neg silence: **keep server-based architecture, optimize LLM prompt for smaller models**

## References

- Session 16 benchmarks: `eval/results/enriched-llm-local-2026-04-02.json`
- Qwen3-Reranker: MTEB-Code 73.42 (mentioned in CLAUDE.md model recommendations)
- Current reranker prompt: `src/cuecard/llm_reranker.py`
- Current expansion prompt: `src/cuecard/expander.py`
