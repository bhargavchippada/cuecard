# Small Model Inference Benchmarks: llama.cpp Performance Research

> Compiled 2026-04-02 from web searches of benchmark repos, blog posts, GitHub discussions, and hardware review sites.

## 1. GPU Token Generation Speed (Decode, tok/s)

All numbers are Q4_K_M quantization unless noted. "tg" = text generation (decode), measured in tokens/second.

### RTX 5090 (32GB, 1792 GB/s bandwidth)

| Model | Params | tg @ 4K ctx | tg @ 8K ctx | tg @ 32K ctx | PP @ 4K ctx |
|-------|--------|-------------|-------------|--------------|-------------|
| Qwen3-8B | 8B | 186 | 170 | 112 | 10,406 |
| Qwen3-14B | 14B | 124 | 115 | 82 | 6,498 |
| Qwen3-MoE-30B-A3B | 30B (3B active) | **234** | 170 | 111 | 6,630 |
| Qwen3-32B | 32B | 61 | 56 | 44 | 2,931 |
| Qwen3.5-35B-A3B (from VRAM guide) | 35B (3B active) | **205** | — | — | 793 |

Source: [Hardware Corner RTX 5090 Benchmarks](https://www.hardware-corner.net/rtx-5090-llm-benchmarks/), [LocalLLM.in VRAM Guide](https://localllm.in/blog/llamacpp-vram-requirements-for-local-llms)

**Key insight:** MoE models (30B-A3B) are FASTER than dense 8B models on GPU because only ~3B params are active per token. The 5090 achieves 234 tok/s on the MoE vs 186 tok/s on dense 8B.

### RTX 4090 (24GB, 1008 GB/s bandwidth)

| Model | Params | tg @ 512 | tg @ 1024 | tg @ 4096 | tg @ 8192 | PP @ 1024 |
|-------|--------|----------|-----------|-----------|-----------|-----------|
| LLaMA-3-8B Q4_K_M | 8B | 131 | 128 | 119 | 111 | 6,899 |
| LLaMA-3-8B F16 | 8B | 55 | 54 | 53 | 51 | 9,056 |

Source: [GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)

### RTX 3090 (24GB, 936 GB/s bandwidth)

| Model | Params | tg @ 512 | tg @ 1024 | tg @ 4096 | tg @ 8192 | PP @ 1024 |
|-------|--------|----------|-----------|-----------|-----------|-----------|
| LLaMA-3-8B Q4_K_M | 8B | 115 | 112 | 97 | 87 | 3,865 |
| LLaMA-3-8B F16 | 8B | 47 | 47 | 45 | 43 | 4,240 |

Source: [GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)

### RTX 3070 (8GB, 386 GB/s bandwidth)

| Model | Params | tg @ 4K ctx | tg @ 8K ctx | tg @ 16K ctx | tg @ 32K ctx | PP @ 4K ctx |
|-------|--------|-------------|-------------|--------------|--------------|-------------|
| Qwen3.5-9B Q4_K_M | 9B | 57.9 | 57.7 | 56.6 | 54.9 | 1,932 |
| GLM-4.6V-Flash Q4_K_M | ~9B | 58.2 | 57.4 | 54.9 | 17.4* | 2,376 |
| Nemotron-12B Q4_K_M | 12B | 10.9 | 10.5 | 8.7 | 6.6 | — |
| Gemma-3-12B Q4_K_M | 12B | 8.6 | 7.0 | 5.5 | 4.3 | — |
| Phi-4-14B Q4_K_M | 14B | 8.2 | 6.6 | 4.0 | 1.8 | — |
| LLaMA-3-8B Q4_K_M | 8B | ~73 | ~71 | ~67 | ~62 | 2,284 |

*GLM-4.6V spills 4 layers to CPU at 32K, causing 70% speed collapse.
Source: [LocalLLM.in 8GB VRAM Guide](https://localllm.in/blog/best-local-llms-8gb-vram-2025), [GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)

### Estimated GPU Performance for Small Models (3B-4B)

No direct RTX benchmarks found for 3B/4B models specifically. Extrapolating from the data:

| Model | Params | Estimated tg (RTX 4090) | Estimated tg (RTX 3090) | Notes |
|-------|--------|-------------------------|-------------------------|-------|
| Qwen3-4B Q4_K_M | 4B | ~200-250 | ~170-200 | Proportional to bandwidth/param ratio |
| Llama-3.2-3B Q4_K_M | 3B | ~250-300 | ~200-250 | Smaller model = higher tok/s |
| Gemma-3-4B Q4_K_M | 4B | ~200-250 | ~170-200 | Similar architecture size |
| Phi-4-mini (3.8B) Q4_K_M | 3.8B | ~200-250 | ~170-200 | Dense model |

**Rationale:** The RTX 4090 achieves ~128 tok/s on 8B Q4_K_M. A 4B model has roughly half the weights, so decode should be roughly 2x faster (memory-bandwidth bound). The RTX 5090 achieves 234 tok/s on a 30B MoE with 3B active params, confirming that 3B active params can sustain 200+ tok/s on high-bandwidth GPUs.

**Gemma-3-4B on Intel Iris Xe (Vulkan):** 14 tok/s decode, 254 tok/s prefill. (From llama.cpp Vulkan discussion)


## 2. CPU Performance

### Apple Silicon

| Hardware | Model | tg (tok/s) | PP (tok/s) | Notes |
|----------|-------|------------|------------|-------|
| M1 7-core 8GB | LLaMA-3-8B Q4_K_M | 10 | 87 | Very slow |
| M1 Max 32-core 64GB | LLaMA-3-8B Q4_K_M | 35 | 355 | |
| M2 Ultra 76-core 192GB | LLaMA-3-8B Q4_K_M | 76 | 1,024 | |
| M3 Max 40-core 64GB | LLaMA-3-8B Q4_K_M | 51 | 678 | |
| M3/M4 (general) | 7-8B Q4_K_M | 60-120 | — | Community reports |
| M4 Max (estimated) | 3B Q4_K_M | ~150+ | ~1,100+ | Based on Llama-3.2-3B reports |

Source: [GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference), [llama.cpp Apple Silicon Discussion](https://github.com/ggml-org/llama.cpp/discussions/4167)

**Key data point:** Llama 3.2 3B achieves over 1,100 tok/s prefill on M4 Max, saturating memory bandwidth. Text generation for 3B models on M4 Max is estimated at ~100-150 tok/s.

### x86 CPU (AVX-512 / AVX2)

| Hardware | Model | tg (tok/s) | Notes |
|----------|-------|------------|-------|
| AMD Ryzen AI 9 HX 375 | Llama-3.2-1B Q4 | 50.7 | CPU with NPU assist |
| Arm mobile (Cortex-X) | Llama-3.2-3B | 19.9 | Mobile CPU, Arm-optimized kernel |
| Intel Xeon 4th Gen (1 socket) | 6B-20B models | 12.5-50 | Server CPU |
| i7-8700K + DDR4 (CPU only) | 8B Q4_K_M (spilled from GPU) | 6.6-10.9 | Partial CPU offload, PCIe bottleneck |

Source: Various (AMD blog, Arm newsroom, academic papers)

### CPU Inference: Can Sub-4B Models Hit <2s for 200 Tokens?

**200 tokens in <2s = need 100+ tok/s decode.**

| Platform | Model Size | Estimated tok/s | 200 tokens in... | Feasible? |
|----------|-----------|-----------------|-------------------|-----------|
| M4 Max (546 GB/s) | 0.6B Q4 (~400MB) | ~300-500 | 0.4-0.7s | YES |
| M4 Max | 1.7B Q4 (~1GB) | ~200-300 | 0.7-1.0s | YES |
| M4 Max | 3B Q4 (~1.8GB) | ~100-150 | 1.3-2.0s | BORDERLINE |
| M4 Max | 4B Q4 (~2.2GB) | ~80-120 | 1.7-2.5s | LIKELY NO |
| M3 Max (400 GB/s) | 0.6B Q4 | ~200-350 | 0.6-1.0s | YES |
| M3 Max | 3B Q4 | ~80-100 | 2.0-2.5s | NO |
| i7-13700K + DDR5 (~77 GB/s) | 0.6B Q4 | ~80-120 | 1.7-2.5s | BORDERLINE |
| i7-13700K + DDR5 | 1.7B Q4 | ~40-60 | 3.3-5.0s | NO |
| i7-13700K + DDR5 | 3B Q4 | ~20-35 | 5.7-10.0s | NO |
| Ryzen 9 + DDR5 (~89 GB/s) | 0.6B Q4 | ~90-140 | 1.4-2.2s | BORDERLINE |
| Ryzen 9 + DDR5 | 1.7B Q4 | ~45-70 | 2.9-4.4s | NO |
| Server Xeon AVX-512 (~200 GB/s) | 0.6B Q4 | ~200-300 | 0.7-1.0s | YES |
| Server Xeon AVX-512 | 3B Q4 | ~50-80 | 2.5-4.0s | NO |

**Formula basis:** Token generation is memory-bandwidth bound. Approximate: `tok/s ~ bandwidth_GB_s / model_size_GB * efficiency_factor`. Efficiency factor is ~0.6-0.8 for well-optimized llama.cpp on the platform.

### Key Finding: GPU Becomes Mandatory At...

For **<1s latency on 200 output tokens** (200+ tok/s decode):
- **0.6B models:** CPU-feasible on Apple Silicon (M3+) and high-bandwidth server CPUs
- **1.7B models:** CPU-feasible only on M4 Max or server-class hardware
- **3B+ models:** GPU mandatory for <1s latency on consumer hardware
- **4B+ models:** GPU mandatory even on Apple Silicon for <1s

For **<2s latency on 200 output tokens** (100+ tok/s):
- **0.6B-1.7B:** Feasible on M3 Max+, borderline on fast x86 with DDR5
- **3B:** Feasible only on M4 Max or dedicated GPU
- **4B+:** GPU mandatory


## 3. Memory Requirements

### VRAM for GPU Inference (Q4_K_M, 32K context)

| Model Size | Weights (VRAM) | KV Cache (32K) | Total VRAM | Fits On |
|-----------|----------------|----------------|------------|---------|
| ~0.6B | ~0.4 GB | ~0.05 GB | ~1.2 GB | Any GPU |
| ~1.7B | ~1.0 GB | ~0.1 GB | ~1.9 GB | Any GPU |
| ~3B | ~1.8 GB | ~0.2 GB | ~2.7 GB | Any GPU (4GB+) |
| ~4B | ~2.2 GB | ~0.2 GB | ~3.2 GB | Any GPU (4GB+) |
| ~8B | ~4.6 GB | ~0.5 GB | ~5.8 GB | 8GB+ |
| ~9B (Qwen3.5) | 5.1 GB | 1.0 GB | **6.8 GB** | 8GB |
| ~12B | ~6.5 GB | ~0.8 GB | ~8.1 GB | 12GB+ |
| ~14B | ~7.8 GB | ~1.2 GB | ~9.7 GB | 12GB+ |
| ~27B (Qwen3.5) | 16.1 GB | 2.0 GB | **18.1 GB** | 24GB |
| ~35B MoE (Qwen3.5-A3B) | 21.1 GB | 0.6 GB | **21.7 GB** | 24GB |

Source: [LocalLLM.in VRAM Guide](https://localllm.in/blog/llamacpp-vram-requirements-for-local-llms)

### RAM for CPU Inference

Rule of thumb: model file size + 1-2 GB overhead + KV cache. For Q4_K_M:
- 0.6B: ~1.5 GB RAM
- 1.7B: ~2.5 GB RAM
- 3B: ~3.5 GB RAM
- 4B: ~4.0 GB RAM
- 8B: ~6.5 GB RAM

### Quantization Levels and Quality

From perplexity measurements on LLaMA-3-70B (applies proportionally to smaller models):

| Quant | Size vs FP16 | Perplexity Delta | Quality Impact |
|-------|-------------|-----------------|----------------|
| Q4_K_M | ~30% | +4.83% | **Recommended sweet spot** |
| Q4_K_S | ~28% | +6.15% | Minimal degradation |
| Q5_K_M | ~35% | +1.23% | Near-lossless |
| Q6_K | ~41% | +0.47% | Negligible loss |
| Q8_0 | ~53% | +0.03% | Essentially lossless |

Source: [GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)


## 4. Answers to Key Questions

### Q: At what model size does GPU become mandatory for <1s latency on ~200 tokens output?

**Answer: ~3B parameters on consumer hardware, ~1.7B on typical x86 CPUs.**

- 200 tokens in <1s requires 200+ tok/s decode
- On consumer x86 CPU (DDR5, ~77-89 GB/s bandwidth): only 0.6B models can potentially reach this
- On Apple M4 Max (546 GB/s): 0.6B-1.7B models can hit this; 3B is borderline at ~1.3-2.0s
- On any discrete GPU (even RTX 3070): 3B-4B models easily exceed 200 tok/s
- For cuecard's use case (~200 token classification output), a 0.6B model on CPU might work, but 3B+ needs GPU

### Q: Best quantization for classification tasks?

**Answer: Q4_K_M is the recommended default.** For classification/reranking:

- **Q4_K_M**: Best balance of speed and quality. Only 4.83% perplexity increase on 70B; even less on smaller models where the task is simpler
- **Q8_0**: If you have the memory and want maximum quality (essentially lossless). 1.7x larger files
- **For 0.6B models specifically**: Q8_0 is viable since files are tiny (~639MB). The quality preservation matters more at small model sizes where every bit of capacity counts
- **For 3B-4B models**: Q4_K_M is fine; the model has enough capacity that 4-bit compression doesn't materially hurt classification accuracy

### Q: Any llama.cpp optimizations for small models?

**Answer: Yes, several:**

1. **Flash Attention (`--flash-attn on` or `-fa 1`)**: Reduces KV cache memory and improves speed. Especially important for keeping models fully in GPU VRAM.

2. **KV Cache Quantization (`--cache-type-k q8_0 --cache-type-v q8_0`)**: Cuts context VRAM by ~50%. Qwen3.5 series handles this with "near-lossless accuracy." Critical for maximizing context on limited VRAM.

3. **`--fit on`**: Auto-calculates max GPU layers to prevent OOM crashes. Default in Ollama/LM Studio.

4. **Jinja template support (`--jinja`)**: Required for models with `chat_template_kwargs` (like thinking mode control). Already used in cuecard's llama-server setup.

5. **Disable thinking mode**: For classification, `enable_thinking: false` via Jinja templates. Saves massive latency (7s -> 889ms in cuecard's own benchmarks).

6. **Speculative decoding**: Use a 0.5-1B draft model with a larger target model for ~1.5-2.3x speedup. Requires same tokenizer family. Best for code generation, not classification.

7. **Thread count tuning**: For CPU inference, just 5 threads saturate dual-channel DDR5 bandwidth. More threads = diminishing returns.

8. **Batch size for prefill (`-b`)**: Higher batch = faster prompt processing. Default 512 is usually fine for small models.

9. **ik_llama.cpp fork**: Reports ~1.9x faster MoE inference through fused operations, better AVX-512 utilization. Source: [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)

10. **CPU spill is catastrophic**: Even 4 layers spilling from GPU to CPU causes 70% speed collapse (GLM-4.6V example: 55 -> 17.4 tok/s). For small models, ensure 100% GPU offload.


## 5. Implications for Cuecard

### Current Setup (Qwen3.5-35B-A3B, GPU required)
- RTX 5090: 234 tok/s decode, ~0.9s for 200 tokens
- RTX 4090/3090: 128-186 tok/s decode (estimated from 8B scaling + MoE advantage)
- 21.7 GB VRAM at 32K context

### Proposed Small Model Options

| Model | Size (Q4) | Size (Q8) | GPU tok/s (4090 est.) | CPU tok/s (M4 Max est.) | CPU tok/s (x86 DDR5 est.) |
|-------|-----------|-----------|----------------------|------------------------|-----------------------------|
| Qwen3-0.6B | ~400MB | ~639MB | ~400-600 | ~300-500 | ~80-120 |
| Qwen3-Reranker-0.6B | ~400MB | ~639MB | ~400-600 | ~300-500 | ~80-120 |
| Qwen3-1.7B | ~1.0GB | ~1.7GB | ~250-350 | ~200-300 | ~45-70 |
| Phi-4-mini (3.8B) | ~2.2GB | ~3.8GB | ~200-250 | ~80-120 | ~20-35 |
| Llama-3.2-3B | ~1.8GB | ~3.0GB | ~250-300 | ~100-150 | ~20-35 |

### Latency Budget for Cuecard Reranker

Current: ~1.5s with Qwen3.5-35B on GPU (cuecard benchmarks show 1143ms p50)

Target: <1s total pipeline for PreToolUse hooks, <500ms for reranker specifically

| Model | Platform | ~200 token output | Meets <500ms? |
|-------|----------|-------------------|---------------|
| Qwen3-0.6B Q4 | RTX 4090 | ~0.3-0.5s | YES |
| Qwen3-0.6B Q8 | RTX 4090 | ~0.4-0.6s | YES |
| Qwen3-0.6B Q4 | M4 Max (CPU/Metal) | ~0.4-0.7s | BORDERLINE |
| Qwen3-0.6B Q4 | x86 CPU DDR5 | ~1.7-2.5s | NO |
| Qwen3-0.6B Q4 | x86 CPU DDR4 | ~2.5-4.0s | NO |
| Qwen3-1.7B Q4 | RTX 4090 | ~0.6-0.8s | BORDERLINE |
| Qwen3-1.7B Q4 | x86 CPU DDR5 | ~2.9-4.4s | NO |
| Qwen3-Reranker-0.6B (cross-encoder seq-cls) | Any CPU | ~200-500ms | YES (different arch) |

### Critical Insight: Cross-Encoder vs Generative Reranker

Qwen3-Reranker-0.6B as a **cross-encoder** (seq-cls mode) is fundamentally different from generative inference:
- No token generation loop needed (single forward pass per candidate)
- 5-20 candidates = 5-20 forward passes (or batched)
- Expected latency: 50-200ms total on CPU for 20 candidates
- This bypasses the memory-bandwidth bottleneck entirely

For **generative** reranking (current cuecard approach):
- 0.6B on x86 CPU: too slow (1.7-2.5s for 200 tokens)
- 0.6B on GPU: fast enough (~0.3-0.5s) but defeats the "no GPU" goal
- Cross-encoder approach is the only viable path for CPU-only reranking at <500ms


## Sources

- [Hardware Corner: RTX 5090 LLM Benchmarks](https://www.hardware-corner.net/rtx-5090-llm-benchmarks/)
- [LocalLLM.in: llama.cpp VRAM Requirements 2026](https://localllm.in/blog/llamacpp-vram-requirements-for-local-llms)
- [LocalLLM.in: Best LLMs for 8GB VRAM 2026](https://localllm.in/blog/best-local-llms-8gb-vram-2025)
- [GPU-Benchmarks-on-LLM-Inference (GitHub)](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)
- [llama.cpp NVIDIA CUDA Performance Discussion](https://github.com/ggml-org/llama.cpp/discussions/15013)
- [llama.cpp Apple Silicon Discussion](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [llama.cpp Vulkan Performance Discussion](https://github.com/ggml-org/llama.cpp/discussions/10879)
- [NVIDIA: Accelerating LLMs with llama.cpp on RTX](https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/)
- [Puget Systems: CPU Speed Effects on GPU Inference](https://www.pugetsystems.com/labs/articles/effects-of-cpu-speed-on-gpu-inference-in-llama-cpp/)
- [AMD: Llama.cpp on Ryzen AI 300](https://www.amd.com/en/blogs/2024/accelerating-llama-cpp-performance-in-consumer-llm.html)
- [ik_llama.cpp fork (faster MoE)](https://github.com/ikawrakow/ik_llama.cpp)
- [Speculative Decoding in llama.cpp (DeepWiki)](https://deepwiki.com/ggml-org/llama.cpp/8.3-flash-attention-and-optimizations)
- [Qwen3 Official Blog](https://qwenlm.github.io/blog/qwen3/)
- [Phoronix: RTX 5090 llama.cpp Review](https://www.phoronix.com/review/nvidia-rtx5090-llama-cpp/2)
- [OpenBenchmarking: llama.cpp Benchmarks](https://openbenchmarking.org/test/pts/llama-cpp)
