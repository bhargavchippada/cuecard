# cuecard

> The right rule, at the right moment.

Contextual rule enforcement for AI coding agents. cuecard retrieves your most relevant guidelines and injects them before every tool call and user message — so the agent always follows your rules.

## How It Works

1. **You write rules** in plain text files (coding standards, workflow guidelines, security policies)
2. **cuecard indexes** them using semantic embeddings (jina-code, ~15ms)
3. **Optionally, expand rules** with LLM-generated paraphrases for better matching on hard cases
4. **On every agent action**, cuecard retrieves the most relevant rules via dense + BM25 hybrid retrieval
5. **An LLM reranker** (optional, Qwen3.5-35B) filters noise and selects truly relevant rules (~1s)

## Supported Events

- **PreToolUse** — fires before each tool call (`Bash`, `Read`, `Edit`, etc.)
- **UserPromptSubmit** — fires when the user sends a message

Both use a unified index. The LLM reranker distinguishes coding rules from workflow rules based on event context.

## Quick Start

```bash
# Install
uv pip install cuecard

# Setup (downloads embedding model, builds index)
cuecard setup

# Add rules
cuecard rules add "Use uv for all Python package operations, never pip"
cuecard rules add "Run quality checks before every commit"

# Generate expansions for better hard-case matching (optional, needs local LLM)
cuecard rules expand --backend local --endpoint http://localhost:8081/v1

# Install as Claude Code hook
cuecard install claude-code

# Test it
cuecard retrieve "Bash: pip install requests"
```

## Pipeline

```
Query → Multi-retriever (dense + BM25 sparse, ~20ms)
      → RRF fusion
      → LLM re-ranking (Qwen3.5-35B, reasoning-in-response, ~1s)
      → Inject relevant rules into agent context
```

**Three modes:**
- `embedding` — fast, CPU-only, dense + BM25 hybrid, ~20ms
- `llm-local` — best quality, needs local LLM server, ~1s
- `llm-haiku` — best quality, needs Anthropic API, ~500ms

## Rule Expansions

Rules are short and abstract ("Review dependencies for vulnerabilities"). Queries are concrete and code-like ("docker build -t myapp ."). Expansions bridge this vocabulary gap:

```bash
# Generate paraphrases for each rule using a local LLM
cuecard rules expand --backend local

# Only expand rules that don't have expansions yet
cuecard rules expand --missing-only

# Preview without calling LLM
cuecard rules expand --dry-run
```

Each rule gets 5-10 trigger phrases like "docker build with untrusted base image" or "pip install new package check CVEs". These are embedded alongside the canonical rule, and the best match across all expansions is used (max-score parent collapse).

**Validated improvement:** On hard fixtures, expansions raise average cosine similarity from 0.187 to 0.558 (+0.371), turning 10/12 misses into hits.

## Quality (587 fixtures)

**Enriched retrieval (jina-code + expansions + BM25, embedding mode):**

| Tier | Recall | Noise |
|------|--------|-------|
| Easy | 95.3% | 83.1% |
| Medium | 76.8% | — |
| Hard | 60.0% | — |

**With LLM reranker (Qwen3.5-35B, llm-local mode):**

| Event Type | Recall | Noise | Neg Silence | Latency |
|-----------|--------|-------|-------------|---------|
| PreToolUse (354 fixtures) | 42.2% | 21.4% | 92.9% | 1.1s |
| UserPromptSubmit (84 fixtures) | 46.2% | 24.5% | 95.8% | 3.0s |

The LLM reranker is essential for noise filtering and negative silence. Embedding mode alone achieves high recall but cannot reject irrelevant queries.

6 embedding models benchmarked. See CLAUDE.md for full comparison.

Eval dataset: 438 curated fixtures + 149 mined from real developer sessions across 7 projects.

## Local LLM Server

For best quality, run a local Qwen3.5-35B instance:

```bash
llama-server -m ~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --port 8081 -ngl 99 -c 16384 --jinja
```

## Configuration

Two config files, layered (project wins):
- **Global:** `~/.cuecard/config.toml`
- **Project:** `./cuecard.toml`

```toml
[embedding]
model = "jinaai/jina-embeddings-v2-base-code"

[retrieval]
top_k = 5
threshold = 0.30
fusion_k = 60              # RRF parameter
sparse_enabled = true       # Enable BM25 hybrid retrieval

[pipeline]
mode = "llm-local"

[pipeline.llm]
local_endpoint = "http://localhost:8081/v1"
thinking = false

[expansion]
max_per_rule = 10
max_expansion_length = 200
```

## Development

```bash
uv sync
uv run pytest --cov=cuecard --cov-fail-under=100
uv run ruff check src/ tests/
uv run mypy src/
uv run mutmut run                    # Mutation testing
```

826+ tests, 100% coverage, ruff clean, mypy strict.

## License

MIT
