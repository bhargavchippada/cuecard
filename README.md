# cuecard

> The right rule, at the right moment.

Contextual rule enforcement for AI coding agents. cuecard retrieves your most relevant guidelines and injects them before every tool call and user message — so the agent always follows your rules.

## How It Works

1. **You write rules** in plain text files (coding standards, workflow guidelines, security policies)
2. **cuecard indexes** them using semantic embeddings (jina-code, ~15ms)
3. **On every agent action**, cuecard retrieves the most relevant rules and injects them into context
4. **An LLM reranker** (optional, Qwen3.5-35B) filters noise and selects truly relevant rules (~1s)

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

# Install as Claude Code hook
cuecard install claude-code

# Test it
cuecard retrieve "Bash: pip install requests"
```

## Pipeline

```
Query → Embedding retrieval (jina-code, top-20, ~15ms)
      → LLM re-ranking (Qwen3.5-35B, reasoning-in-response, ~1s)
      → Inject relevant rules into agent context
```

**Three modes:**
- `embedding` — fast, CPU-only, ~15ms, moderate quality
- `llm-local` — best quality, needs local LLM server, ~1s
- `llm-haiku` — best quality, needs Anthropic API, ~500ms

## Quality (438 fixtures)

| Event Type | Recall | Noise | Neg Silence | Latency |
|-----------|--------|-------|-------------|---------|
| PreToolUse (354 fixtures) | 42.2% | 21.4% | 92.9% | 1.1s |
| UserPromptSubmit (84 fixtures) | 46.2% | 24.5% | 95.8% | 3.0s |

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

[pipeline]
mode = "llm-local"

[pipeline.llm]
local_endpoint = "http://localhost:8081/v1"
thinking = false
```

## Development

```bash
uv sync
uv run pytest --cov=cuecard --cov-fail-under=100
uv run ruff check src/ tests/
uv run mypy src/
```

541 tests, 100% coverage, ruff clean, mypy strict.

## License

MIT
