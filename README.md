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
# 1. Install globally (recommended — makes `cuecard` available everywhere)
uv tool install .

# 2. Initial setup (creates ~/.cuecard/, downloads embedding model, builds index)
cuecard setup

# 3. Add your rules
cuecard rules add --global "Use uv for all Python package operations, never pip"
cuecard rules add --global "Run quality checks before every commit"
cuecard rules add --global "Never commit secrets to git"

# 4. Configure pipeline mode (interactive — choose embedding, llm-local, or llm-haiku)
cuecard configure

# 5. Rebuild index with your rules
cuecard index

# 6. Generate expansions for better matching (optional, needs local LLM)
cuecard rules expand

# 7. Install as Claude Code hook (registers PreToolUse + UserPromptSubmit)
cuecard install claude-code

# 8. Test it
cuecard retrieve "Bash: pip install requests"
```

You can re-run `cuecard configure` at any time to change settings.

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

**With LLM reranker (Qwen3.5-35B, llm-local mode, full dataset):**

| Event Type | Quality (F2) | Pos Recall | Noise | Neg Silence | Latency |
|-----------|-------------|-----------|-------|-------------|---------|
| PreToolUse (354 fixtures) | 0.782 | 76.4% | 20.0% | 87.6% | 1.5s |
| UserPromptSubmit (84 fixtures) | 0.776 | 74.4% | 24.8% | 95.8% | 1.5s |

The LLM reranker is essential for noise filtering and negative silence. Embedding mode alone achieves high recall but cannot reject irrelevant queries.

6 embedding models benchmarked. See CLAUDE.md for full comparison.

Eval dataset: 438 curated fixtures + 149 mined from real developer sessions across 7 projects.

## Writing Effective Rules

Rules that explain **why** achieve compliance. Rules that just say **what** don't.

```
# BAD — agent ignores this (0/3 compliance in live testing)
Send Enter after every tmux send-keys command

# GOOD — agent follows this (2/2 compliance in live testing)
Always append Enter after tmux send-keys text — without it the text
is pasted into the prompt but never submitted, so the target session
never receives or executes the message
```

**Guidelines for writing rules:**
- Explain the **consequence of violation**, not just the action
- Include the **failure mode** — what goes wrong if the rule is ignored
- Keep rules under 500 characters (the enforced limit)
- One rule per line in your rules file
- Rules are retrieved semantically — exact wording matters less than meaning

This mirrors a finding from prompt engineering: reasoning principles ("without Enter, text is pasted but never submitted") outperform command lists ("always send Enter"). The agent follows rules it understands, not rules it's told to obey.

## Local LLM Server

For best quality, run a local Qwen3.5 instance:

```bash
# Recommended: 9B (matches 35B quality, 4x smaller)
llama-server -m ~/models/Qwen3.5-9B-Q4_K_M.gguf \
  --port 8081 -ngl 99 -c 16384 --jinja

# CPU/laptop: 4B (2.6GB, ~850ms per query)
llama-server -m ~/models/Qwen3.5-4B-Q4_K_M.gguf \
  --port 8081 -ngl 0 -c 16384 --jinja

# Max quality: 35B MoE (needs 32GB GPU)
llama-server -m ~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --port 8081 -ngl 99 -c 16384 --jinja
```

## Configuration

Run `cuecard configure` for interactive setup, or edit config files directly.

Two config files, layered (project overrides global):
- **Global:** `~/.cuecard/config.toml` — applies to all projects
- **Project:** `./cuecard.toml` — project-specific overrides

```toml
[sources]
rules = ["rules/global.txt"]     # Rule files (relative to config dir)

[embedding]
model = "BAAI/bge-small-en-v1.5"  # Embedding model (see model table below)

[retrieval]
top_k = 5                  # Max results (embedding mode only; LLM modes use 20 internally)
threshold = 0.30            # Min similarity score
fusion_k = 60              # RRF parameter
sparse_enabled = true       # Enable BM25 hybrid retrieval

[pipeline]
mode = "embedding"          # embedding | llm-local | llm-haiku

[pipeline.llm]
local_endpoint = "http://localhost:8081/v1"  # For llm-local mode
thinking = false            # Enable LLM thinking mode (slower, not recommended)

[expansion]
max_per_rule = 10           # Max expansions per rule
max_expansion_length = 200  # Max chars per expansion
```

**Pipeline modes:**
- `embedding` — fast, CPU-only, dense + BM25 hybrid (~20ms). No noise filtering.
- `llm-local` — best quality. Needs a local LLM server (Qwen3.5-9B recommended). ~1s latency.
- `llm-haiku` — best quality via API. Needs `ANTHROPIC_API_KEY`. ~500ms latency.

## CLI Reference

### Setup and Status

```bash
cuecard setup                       # Download model, create config, build index
cuecard setup --check               # Verify setup (config + index exist)
cuecard setup --allow-custom-model   # Use a model not on the allowlist
cuecard configure                    # Interactive config (pipeline mode, endpoint, thresholds)
cuecard config                       # Show resolved config (global + project merged)
cuecard status                       # Show hook status, index health, model info
```

### Rules Management

```bash
cuecard rules                        # List all rules with numbers and source files
cuecard rules --global               # List only global rules
cuecard rules --project              # List only project rules
cuecard rules add "rule text"        # Add rule to project rules.txt
cuecard rules add "rule text" --global  # Add rule to ~/.cuecard/rules/global.txt
cuecard rules remove 3               # Remove rule #3 (number from `cuecard rules`)
cuecard rules search "commit"        # Search rules by substring
cuecard rules sources                # Show configured rule file paths and status
```

### Expansions

```bash
cuecard rules expand                 # Generate LLM expansions for all rules
cuecard rules expand --backend local # Use local LLM (default)
cuecard rules expand --backend haiku # Use Anthropic Haiku API
cuecard rules expand --endpoint URL  # Custom LLM endpoint (default: localhost:8081)
cuecard rules expand --missing-only  # Only expand rules without expansions
cuecard rules expand --dry-run       # Preview what would be expanded
```

### Index

```bash
cuecard index                        # Rebuild index from scratch (preserves expansions)
cuecard index --status               # Show index status (model, rules, dim, sources)
```

### Retrieval and Inspection

```bash
cuecard retrieve "Bash: git push --force"  # Retrieve matching rules for a query
cuecard retrieve "query" --mode llm-local  # Override pipeline mode
cuecard retrieve "query" --top-k 10        # Override max results
cuecard retrieve "query" --threshold 0.40  # Override similarity threshold
cuecard format "Bash: git push --force"    # Show final injectable text for a query
cuecard parse                              # Show all parsed rules with provenance
cuecard embed                              # Show embedding stats (model, dim, count)
```

### Hook Management

```bash
cuecard install claude-code          # Register hooks in ~/.claude/settings.json
cuecard uninstall claude-code        # Remove hooks from settings.json
cuecard status                       # Check if hooks are installed
cuecard log                          # Show recent hook log entries (last 20)
cuecard log --limit 50               # Show more entries
cuecard log --stats                  # Show aggregate statistics (latency, coverage, top rules)
```

The `cuecard hook` command exists but is hidden — it is the internal entry point called by Claude Code hooks via stdin/stdout.

### Daemon Server

```bash
cuecard serve                        # Start persistent daemon on port 8452
cuecard serve --port 9000            # Start on custom port
cuecard serve --daemon               # Fork to background
cuecard serve --stop                 # Stop running daemon
```

The daemon keeps the embedding model loaded in memory, eliminating cold-start latency. The hook adapter automatically uses the daemon when running (with 500ms timeout fallback to direct mode).

### Evaluation

```bash
cuecard eval fixtures.json                  # Run eval against fixture file
cuecard eval fixtures.json --mode llm-local # Use LLM reranker
cuecard eval fixtures.json --model MODEL    # Override embedding model
cuecard eval fixtures.json --corpus-dir DIR # Custom corpus directory
cuecard eval fixtures.json --corpus-override "a.json,b.json"  # Specific corpus files
cuecard eval fixtures.json --sample-ratio 0.2  # Stratified 20% sample (fast iteration)
cuecard eval fixtures.json --top-k 10 --threshold 0.25 --dedup-threshold 0.90
```

## Development

```bash
uv sync
uv run pytest --cov=cuecard --cov-fail-under=100
uv run ruff check src/ tests/
uv run mypy src/
uv run mutmut run                    # Mutation testing
```

946 tests, 100% coverage, ruff clean, mypy strict.

## License

MIT
