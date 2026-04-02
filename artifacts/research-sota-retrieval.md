# SOTA Retrieval Research for Cuecard

> Research date: 2026-04-01
> Current baseline: BAAI/bge-small-en-v1.5 (67MB, 384-dim) + jinaai/jina-embeddings-v2-base-code (768-dim)
> Runtime: fastembed (ONNX)

---

## A. Code-Specific Embedding Models (2024-2025)

### Top Candidates

#### 1. nomic-ai/CodeRankEmbed (137M params) -- RECOMMENDED

- **Dimensions:** 768
- **Size:** ~521MB (F32 safetensors); ONNX TBD
- **Context:** 8,192 tokens
- **License:** Apache-2.0
- **Architecture:** Bi-encoder based on Snowflake/snowflake-arctic-embed-m-long
- **Training:** CoRNStack dataset (21M high-quality text-code pairs, dual-consistency filtering, progressive hard negative mining)
- **Benchmarks:**
  - CodeSearchNet MRR: **77.9** (vs Jina-Code-v2: 67.2, CodeSage-Small: 64.9)
  - CoIR NDCG@10: **60.1** (vs Jina-Code-v2: 58.4, CodeSage-Large 1.3B: 59.4)
- **Query prefix:** "Represent this query for searching relevant code: {query}"
- **fastembed:** Not natively supported yet. Would need ONNX conversion or sentence-transformers fallback.
- **Why it matters:** At 137M params, it beats CodeSage-Large (1.3B) on CoIR while being 10x smaller. Beats jina-code-v2 by +10.7 MRR and +1.7 CoIR NDCG. Same architecture family as models already in fastembed (Arctic).
- **Expected improvement over bge-small:** Significant for code-related queries. BGE-small is a general-purpose model with no code-specific training. CodeRankEmbed should improve recall on code/tool queries by 15-25%.

#### 2. google/embeddinggemma-300m (308M params) -- STRONG CONTENDER

- **Dimensions:** 768 (Matryoshka: truncatable to 512, 256, 128)
- **Size:** ~1.2GB (F32); QAT variants available (Q4_0, Q8_0)
- **Context:** 2,048 tokens
- **License:** Gemma Terms of Use (restrictive for some commercial uses)
- **Architecture:** Gemma 3 encoder with T5Gemma initialization
- **Benchmarks:**
  - MTEB Code v1: **68.76** (768d), 68.48 (512d), 66.74 (256d)
  - MTEB English v2: 69.67 (768d)
  - MTEB Multilingual v2: 61.15 (768d)
  - Highest-ranking model under 500M on MTEB code, English, AND multilingual
- **Prompt system:** Task-specific prompts:
  - Retrieval query: `task: search result | query: {content}`
  - Code retrieval: `task: code retrieval | query: {content}`
  - Document: `title: none | text: {content}`
- **ONNX:** Available at `onnx-community/embeddinggemma-300m-ONNX`
- **fastembed:** Not natively supported, but ONNX model exists for manual integration.
- **Why it matters:** Best all-around small model. Code retrieval score of 68.76 is excellent for its size. Task-specific prompts are perfect for cuecard's use case (code retrieval prompt for queries, document prompt for rules). Matryoshka means we can use 256d for speed with minimal quality loss.
- **Expected improvement over bge-small:** +15-20% on code retrieval tasks. Task-specific prompts alone should add 5-10% for our mixed NL+code queries.
- **Caveat:** Gemma license is more restrictive than Apache-2.0. Does NOT support float16 (must use float32 or bfloat16).

#### 3. nomic-ai/nomic-embed-code (7B params) -- REFERENCE ONLY

- **Dimensions:** 768 (assumed from CodeRankEmbed family)
- **Size:** ~26GB (F32)
- **Architecture:** Qwen2.5-Coder-7B-Instruct base, 7B params
- **Benchmarks:** Outperforms Voyage Code 3 and OpenAI Embed 3 Large on CodeSearchNet
  - Python MRR: 81.7, Java: 80.5, Go: 93.8
- **ONNX/fastembed:** Not feasible at 7B -- too large for CPU inference
- **Verdict:** State-of-the-art but way too large for cuecard's use case. CodeRankEmbed (137M) from the same family is the practical choice.

#### 4. voyage-code-3 -- API ONLY

- **Dimensions:** 2048/1024/512/256 (Matryoshka)
- **Context:** 32K tokens
- **CoIR benchmark:** ~77.33 NDCG (highest among all models)
- **Availability:** API only (Voyage AI) -- no local model, no ONNX
- **Verdict:** Best absolute quality but requires API calls. Not viable for cuecard's latency-sensitive local-first approach.

#### 5. Snowflake/snowflake-arctic-embed-m-v2.0 (305M params)

- **Dimensions:** 768 (Matryoshka support)
- **Size:** ONNX fp32: 1,169MB, int8: 296MB
- **License:** Apache-2.0
- **Architecture:** Built on GTE-multilingual, 113M non-embedding params
- **Benchmarks:** Strong on MTEB retrieval (multilingual focus), no specific code scores published
- **fastembed:** Open issue (#426) requesting support -- not yet integrated
- **Verdict:** Good general retrieval model but no code-specific training. Not compelling over EmbeddingGemma for our use case.

#### 6. jina-embeddings-v3 / v4

- **v3:** 570M params, 1024-dim, code retrieval CoIR: 55.07
- **v4:** 2048-dim default, code retrieval CoIR: 71.59, code.query/code.passage adapters
- **License:** cc-by-nc-4.0 (non-commercial)
- **fastembed:** v3 is listed in fastembed but under non-commercial license
- **Verdict:** v4 is excellent (71.59 CoIR) but large and NC-licensed. v3 is worse than CodeRankEmbed on code tasks.

### Summary Table: Code Embedding Models

| Model | Params | Dims | Size | CSN MRR | CoIR NDCG | MTEB Code | ONNX | fastembed | License |
|-------|--------|------|------|---------|-----------|-----------|------|----------|---------|
| **bge-small-en-v1.5** (baseline) | 33M | 384 | 67MB | -- | -- | -- | Yes | Yes | MIT |
| **jina-code-v2** (current) | 161M | 768 | ~650MB | 67.2 | 58.4 | -- | Yes | Yes | Apache |
| **CodeRankEmbed** | 137M | 768 | ~521MB | **77.9** | **60.1** | -- | No* | No* | Apache |
| **EmbeddingGemma-300m** | 308M | 768 | ~1.2GB | -- | -- | **68.76** | Yes | No* | Gemma |
| nomic-embed-code | 7B | 768 | ~26GB | 81.7 | -- | -- | No | No | Apache |
| voyage-code-3 | ? | 1024 | API | 80.8 | 77.3 | -- | No | No | Proprietary |
| jina-embeddings-v4 | ? | 2048 | ~2GB | -- | 71.6 | -- | ? | No | CC-BY-NC |

\* = ONNX conversion possible but not pre-built; fastembed support could be added

### Recommendation

**Short-term (drop-in):** Stay with jina-embeddings-v2-base-code. It's already in fastembed and decent.

**Medium-term (high impact):** Convert CodeRankEmbed to ONNX and integrate. It beats jina-code-v2 by +10.7 MRR on CodeSearchNet and +1.7 on CoIR, with similar model size. Same Arctic architecture family means ONNX conversion should be straightforward.

**If license permits:** EmbeddingGemma-300m with its code retrieval task prompt is the best all-around option, with ONNX already available. The task-prefix system is ideal for cuecard.

---

## B. Hybrid Retrieval (BM25 + Semantic)

### Why Hybrid Matters for Cuecard

Cuecard queries mix exact identifiers (`git commit`, `src/utils.py`, `def foo`) with semantic intent ("fix auth bug"). Pure semantic search may miss exact tool names and file paths. Pure BM25 misses semantic similarity. Hybrid catches both.

### Quantitative Evidence

- **General domain:** Hybrid BM25+dense retrieval shows **15-30% better recall** than either method alone (multiple sources, 2024-2025 consensus)
- **E-commerce (lexical-heavy):** Only +1.7% NDCG over dense-only (WANDS benchmark) -- suggesting hybrid helps MOST when queries have mixed keyword+semantic content
- **Code search:** Exact identifiers (function names, file paths, CLI flags) are where BM25 adds the most value over semantic-only

### Fusion Approaches

#### Reciprocal Rank Fusion (RRF) -- RECOMMENDED

- Score-agnostic: `RRF(d) = sum(1/(k + rank_i))` across all result lists, k=60
- No score normalization needed (BM25 scores and cosine similarity are on different scales)
- Robust, tuning-free, industry standard (Cormack et al., SIGIR 2009)
- Used by Elasticsearch, Weaviate, Qdrant natively

#### Weighted Linear Combination

- `final_score = alpha * semantic_score + (1 - alpha) * bm25_score`
- Requires score normalization (min-max or z-score)
- Needs alpha tuning per domain (typical: alpha=0.7 for semantic-heavy, 0.5 for balanced)
- More tunable but less robust than RRF

#### SPLADE++ (learned sparse)

- Neural sparse model that learns term importance
- Available in fastembed: `prithivida/Splade_PP_en_v1` (532MB)
- Better than BM25 for semantic matching but slower
- Can replace BM25 in hybrid pipeline

### Implementation Options

#### rank-bm25 (Python, pure)

- `pip install rank-bm25` (rank-bm25==0.2.2)
- Lightweight, no dependencies
- `BM25Okapi(tokenized_corpus)` then `.get_scores(tokenized_query)`
- No persistence -- must rebuild on each load (fast for small corpora like cuecard rules)

#### fastembed SparseTextEmbedding (BM25)

- `Qdrant/bm25` model in fastembed (10KB!)
- Produces sparse vectors compatible with Qdrant hybrid search
- Already in our dependency tree
- Requires IDF computation from corpus

#### Tantivy (Rust, via tantivy-py)

- Full-text search engine (like Lucene but Rust)
- Overkill for cuecard's small corpus (<1000 rules)
- Better for larger rule sets or if persistent index is needed

### Recommended Approach for Cuecard

1. **Use fastembed's built-in BM25** (`Qdrant/bm25`, 10KB) for sparse embeddings at index time
2. At query time, run both dense (bge-small or CodeRankEmbed) and sparse (BM25) in parallel
3. Fuse with RRF (k=60, no tuning needed)
4. This adds minimal latency (<5ms for BM25 on small corpus) and no new dependencies

**Expected improvement:** +10-20% recall on queries containing exact tool names, file paths, or CLI commands. Minimal improvement on pure semantic queries.

### Architecture

```
Query: "Bash: git commit -m 'fix auth bug'"
    |
    +---> Dense embed (bge-small) ---> top-20 by cosine sim --+
    |                                                          |
    +---> Sparse embed (BM25) -----> top-20 by BM25 score ----+
                                                               |
                                                          RRF fusion
                                                               |
                                                          top-k results
```

---

## C. Query Expansion for Tool Calls

### The Problem

Cuecard queries look like:
- `Bash: git commit -m 'fix auth bug'`
- `Edit: src/utils.py old_string='def foo' new_string='def bar'`
- `Write: /path/to/config.json content='{"key": "value"}'`

These are structured tool calls, not natural language. Embedding models trained on NL queries will poorly match these against rules written as NL instructions ("Always run tests before committing").

### Expansion Strategies (No LLM Required)

#### 1. Template-Based Query Decomposition -- RECOMMENDED

Extract structured fields from tool calls and expand into searchable NL:

```python
# Input: "Bash: git commit -m 'fix auth bug'"
# Expanded queries:
[
    "git commit",                    # command
    "committing code changes",       # semantic expansion
    "fix auth bug",                  # content/message
    "running bash commands",         # tool type
    "git version control commit",    # domain expansion
]
```

Each expanded query is embedded separately, results are fused with RRF. This is deterministic, fast, and requires no LLM.

#### 2. Tool-Specific Prefix Mapping

Map tool names to semantic categories:

```python
TOOL_EXPANSIONS = {
    "Bash": ["shell command", "terminal", "CLI"],
    "Edit": ["code editing", "file modification", "refactoring"],
    "Write": ["file creation", "writing files"],
    "Read": ["reading files", "file inspection"],
    "Grep": ["code search", "pattern matching"],
    "Glob": ["file search", "finding files"],
}
```

Prepend these as additional query terms for BM25 or as separate embedding queries.

#### 3. Multi-Query Embedding with Max Pooling

Embed multiple expanded queries, take element-wise max across embedding dimensions:

```python
queries = expand_tool_call(raw_query)  # returns 3-5 expanded queries
embeddings = model.query_embed(queries)  # embed each
fused = np.max(np.stack(embeddings), axis=0)  # max pool
```

This captures the union of semantic signals. Published approach from "Query2doc" (EMNLP 2023) and subsequent work.

#### 4. Pseudo-Relevance Feedback (PRF)

1. Run initial retrieval with raw query
2. Take top-3 results
3. Extract key terms from those results
4. Re-query with original + extracted terms

Effective but adds latency (two retrieval rounds). Better suited as an optional enhancement.

#### 5. ExCS-Style Code Expansion (Research, 2024)

ExCS (Nature Scientific Reports, 2024) expands code documents offline by predicting potential queries for each code snippet. For cuecard, this means expanding RULES at index time:

```
Rule: "Always run tests before committing"
Expanded: ["git commit", "pre-commit", "testing", "pytest", "CI/CD",
           "code quality", "test suite", "commit hook"]
```

Index both original and expanded text. This is a one-time cost at index build.

### Recommended Approach for Cuecard

**Phase 1 (deterministic, no LLM):**
1. Template-based decomposition of tool calls (strategy 1)
2. Tool-specific prefix mapping (strategy 2)
3. Rule expansion at index time (strategy 5, simplified)

**Phase 2 (if needed):**
4. Multi-query max pooling (strategy 3)
5. PRF for complex queries (strategy 4)

**Expected improvement:** +15-25% recall on structured tool-call queries. The gap between "Bash: git commit" and "Always run tests before committing" is primarily a vocabulary mismatch that expansion directly addresses.

---

## D. Instruction-Tuned Embedding Models

### Overview

Instruction-tuned models accept a task description prefix that steers the embedding space toward the specific retrieval task. This is highly relevant for cuecard: we can instruct the model to "retrieve coding rules relevant to this tool action."

### Top Models

#### 1. google/embeddinggemma-300m -- BEST FIT (see Section A)

- **Task prompts:** `task: code retrieval | query: {content}` for queries, `title: none | text: {content}` for rules
- **Why ideal:** Purpose-built task prefix system with a dedicated "code retrieval" task type
- **ONNX:** Available (`onnx-community/embeddinggemma-300m-ONNX`)
- **fastembed:** Not supported yet, but ONNX model exists
- **Size:** ~1.2GB (F32), smaller with quantization

#### 2. intfloat/multilingual-e5-large-instruct (560M params)

- **Dimensions:** 1024
- **Size:** ~2.2GB
- **Instruction format:** Prepend task description to query
  - `Instruct: Retrieve coding rules relevant to this developer action\nQuery: git commit -m 'fix bug'`
- **MTEB:** Strong general performance, instruction following improves retrieval by 2-4% over non-instruct E5
- **ONNX:** Available via Qdrant conversion
- **fastembed:** Supported (`intfloat/multilingual-e5-large`)
- **Verdict:** Good but large (2.2GB). The instruction mechanism is powerful but model size may be prohibitive for cuecard's "lightweight" goal.

#### 3. intfloat/multilingual-e5-small (118M params)

- **Dimensions:** 384
- **Size:** ~470MB
- **fastembed:** Supported
- **ONNX:** Supported
- **Instruction support:** Uses `query:` and `passage:` prefixes (simpler than full instruction tuning)
- **Verdict:** Lightweight alternative to bge-small with prefix support. Similar quality, slightly better on multilingual. No specific code advantage.

#### 4. hkunlp/instructor-xl (1.5B params) / instructor-large (335M)

- **Dimensions:** 768
- **Size:** instructor-large ~1.3GB, instructor-xl ~6GB
- **Instruction format:** `Represent the coding rule for retrieval: {rule_text}`
- **ONNX:** Available via Spark NLP
- **fastembed:** Not natively supported
- **Verdict:** Flexible instruction mechanism but older (2023). Outperformed by EmbeddingGemma and E5-instruct on MTEB. Not recommended.

#### 5. Alibaba-NLP/gte-Qwen2-1.5B-instruct

- **Dimensions:** Elastic (user-configurable)
- **Size:** ~6GB
- **MTEB:** #1 on MTEB at time of release (June 2024)
- **fastembed:** Not supported
- **ONNX:** Not practical at 1.5B
- **Verdict:** Excellent quality but too large for CPU inference. Reference only.

### Instruction Tuning for Cuecard's Use Case

The key insight: cuecard has TWO distinct embedding tasks:

1. **Index time (rules):** "Represent this coding convention for retrieval: {rule_text}"
2. **Query time (tool calls):** "Retrieve coding rules relevant to this developer tool action: {tool_call}"

Instruction-tuned models can optimize for BOTH sides of the asymmetric retrieval independently.

### fastembed Compatibility Summary

| Model | Params | Dims | Size | fastembed | ONNX | Instruction | License |
|-------|--------|------|------|----------|------|-------------|---------|
| bge-small-en-v1.5 (baseline) | 33M | 384 | 67MB | Yes | Yes | No | MIT |
| multilingual-e5-small | 118M | 384 | ~470MB | Yes | Yes | Prefix only | MIT |
| multilingual-e5-large | 560M | 1024 | ~2.2GB | Yes | Yes | Prefix only | MIT |
| nomic-embed-text-v1.5 | 137M | 768 | ~540MB | Yes | Yes | Prefix (`search_query:`) | Apache |
| EmbeddingGemma-300m | 308M | 768 | ~1.2GB | No | Yes | Full task prompt | Gemma |
| instructor-large | 335M | 768 | ~1.3GB | No | Partial | Full instruction | Apache |

---

## Overall Recommendations

### Tier 1: Quick Wins (no model change)

1. **Add BM25 hybrid retrieval** using fastembed's built-in `Qdrant/bm25` (10KB). Fuse with RRF. Expected: +10-20% recall on keyword-heavy queries. Zero new dependencies.

2. **Add template-based query expansion** for tool calls. Decompose `Bash: git commit -m 'fix'` into multiple sub-queries. Expected: +15-25% recall on structured queries.

3. **Add rule expansion at index time**. Augment rules with related keywords/tool names. Expected: +5-10% recall.

### Tier 2: Model Upgrade (medium effort)

4. **Switch to nomic-ai/CodeRankEmbed** (137M, 768d, Apache-2.0). Convert to ONNX, integrate as fastembed custom model or via sentence-transformers. Expected: +10-15% recall on code queries over jina-code-v2.

5. **Or switch to EmbeddingGemma-300m** (308M, 768d, ONNX available). Use `task: code retrieval | query:` prefix. Expected: +15-20% recall. Concern: Gemma license, larger model size.

### Tier 3: Architecture Enhancement (higher effort)

6. **Multi-query expansion with max pooling**. Embed multiple expanded queries, fuse embeddings. Expected: +5-10% additional recall on complex queries.

7. **SPLADE++ sparse retrieval** instead of BM25 for neural-aware keyword matching. Available in fastembed (532MB). Expected: +3-5% over BM25 hybrid.

### Priority Order

For maximum impact with minimum effort: **1 > 2 > 4 > 3 > 5 > 6 > 7**

The combination of BM25 hybrid (item 1) + query expansion (item 2) + CodeRankEmbed (item 4) should yield a **25-40% recall improvement** over the current bge-small baseline, with the bulk of gains coming from hybrid retrieval and query expansion (no model change needed).

---

## Sources

- [fastembed supported models (DeepWiki)](https://deepwiki.com/qdrant/fastembed/6-supported-models)
- [fastembed GitHub](https://github.com/qdrant/fastembed)
- [CodeRankEmbed (HuggingFace)](https://huggingface.co/nomic-ai/CodeRankEmbed)
- [nomic-embed-code (HuggingFace)](https://huggingface.co/nomic-ai/nomic-embed-code)
- [EmbeddingGemma-300m (HuggingFace)](https://huggingface.co/google/embeddinggemma-300m)
- [EmbeddingGemma ONNX (HuggingFace)](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)
- [CoRNStack paper (arXiv 2412.01007)](https://arxiv.org/abs/2412.01007)
- [EmbeddingGemma paper (arXiv 2509.20354)](https://arxiv.org/abs/2509.20354)
- [voyage-code-3 blog post](https://blog.voyageai.com/2024/12/04/voyage-code-3/)
- [CoIR benchmark (ACL 2025)](https://github.com/CoIR-team/coir)
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Hybrid search with RRF (Elasticsearch)](https://www.elastic.co/what-is/hybrid-search)
- [Query expansion survey (arXiv 2509.07794)](https://arxiv.org/pdf/2509.07794)
- [ExCS: code expansion for code search (Nature, 2024)](https://www.nature.com/articles/s41598-024-73907-6)
- [Query2doc (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.585.pdf)
- [Advanced RAG query expansion (Haystack)](https://haystack.deepset.ai/blog/query-expansion)
- [Snowflake Arctic Embed v2.0 (HuggingFace)](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0)
- [jina-embeddings-v4 (Jina AI)](https://jina.ai/models/jina-embeddings-v4/)
- [Best embedding models 2025 (BentoML)](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)
