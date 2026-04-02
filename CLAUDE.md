# cuecard

> The right rule, at the right moment.

Contextual rule enforcement for AI coding agents. Retrieves relevant user-defined rules via semantic similarity and injects them before every tool call.

## Project Structure

```
cuecard/
├── CLAUDE.md               # This file
├── README.md               # Public-facing readme
├── .gitignore
├── artifacts/              # Design docs
│   ├── prd-v1.md           # PRD v1.2 — core pipeline (converged)
│   └── multi-stage-retrieval-prd.md  # Multi-stage PRD v1.1 (converged)
├── src/cuecard/            # Core library (agent-agnostic)
│   ├── __init__.py         # Public API: load_config, retrieve, format_rules
│   ├── loader.py           # Unified index loading with freshness + scope composition
│   ├── models.py           # Frozen dataclasses (Rule, Provenance, Index, etc.)
│   ├── config.py           # Load/merge/validate cuecard.toml configs
│   ├── security.py         # Path validation, secrets scrubbing, permissions
│   ├── parser.py           # Parse rule files → list[Rule] (dispatch by .txt/.json)
│   ├── indexer.py          # Embed rules via fastembed, build index, rules.json I/O
│   ├── freshness.py        # mtime + hash checking, full rebuild on change
│   ├── retriever.py        # query_embed → parent collapse → top-k → dedup
│   ├── formatter.py        # Format results for injection
│   ├── pipeline.py         # Multi-stage pipeline orchestrator (dense + sparse + LLM)
│   ├── reranker.py         # Cross-encoder re-ranking (Stage 2, opt-in)
│   ├── llm_reranker.py     # LLM re-ranking (Stage 3, opt-in)
│   ├── llm_utils.py        # Shared LLM helpers (validate_endpoint, call_local/haiku)
│   ├── expander.py         # LLM-based rule expansion generation
│   ├── eval.py             # Evaluation framework (precision, recall, MRR, nDCG)
│   ├── logger.py           # Structured JSONL logging with secrets scrubbing
│   ├── cli.py              # Typer CLI (setup, config, retrieve, rules, eval, etc.)
│   ├── cli_rules.py        # Rules subcommands (add, remove, search, expand)
│   ├── cli_hooks.py        # Install/uninstall/status/log commands
│   ├── cli_eval.py         # Eval command
│   ├── py.typed            # PEP 561 marker
│   ├── retrievers/         # Pluggable retriever adapters
│   │   ├── __init__.py     # Retriever protocol, ScoredCandidate, fuse()
│   │   ├── dense.py        # DenseRetriever (cosine + parent collapse)
│   │   └── sparse.py       # SparseRetriever (BM25Okapi + parent collapse)
│   └── adapters/           # Agent-specific wrappers
│       ├── __init__.py
│       └── claude_code.py  # Claude Code PreToolUse hook
├── tools/
│   └── notebook.ipynb      # Step-by-step visualization (Phase 3)
├── eval/                   # Evaluation framework (Phase 3)
│   ├── corpora/
│   ├── fixtures/
│   └── results/
├── examples/
│   ├── cuecard.toml        # Example config
│   └── rules.txt           # Example rules
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_config.py
│   ├── test_security.py
│   ├── test_parser.py
│   ├── test_indexer.py
│   ├── test_freshness.py
│   ├── test_retriever.py
│   ├── test_formatter.py
│   ├── test_pipeline.py
│   ├── test_reranker.py
│   ├── test_llm_reranker.py
│   ├── test_llm_utils.py
│   ├── test_expander.py
│   ├── test_fusion.py
│   ├── test_retrievers_dense.py
│   ├── test_retrievers_sparse.py
│   ├── test_eval.py
│   └── test_cli.py
├── pyproject.toml
└── LICENSE
```

## Tech Stack

- **Language:** Python 3.11+
- **Package manager:** uv (NEVER pip)
- **CLI:** Typer + Rich
- **Embeddings:** fastembed (ONNX, ~50MB vs 2GB PyTorch)
- **Default model:** BAAI/bge-small-en-v1.5 (67MB, 384-dim)
- **Testing:** pytest + pytest-cov
- **Mutation testing:** mutmut 3.x
- **Linting:** ruff
- **Type checking:** mypy

## Key Commands

```bash
uv sync                                        # Install dependencies
uv run pytest                                  # Run tests (fast, mocked)
uv run pytest --cov=cuecard --cov-fail-under=100  # With 100% coverage
uv run pytest -m slow                          # Run integration tests (real model)
uv run ruff check src/ tests/                  # Lint
uv run mypy src/                               # Type check

# Mutation testing (mutmut)
uv run mutmut run                              # Run on entire codebase (slow)
uv run mutmut run "cuecard.retriever*"         # Run on specific module(s)
uv run mutmut results                          # List surviving mutants
uv run mutmut show <mutant-name>               # Show diff for a specific mutant
uv run mutmut browse                           # Interactive TUI for results
```

### Mutation Testing

mutmut verifies that tests actually detect code changes (mutations). 100% line coverage does not mean the logic is tested -- mutmut finds cases where a line is covered but the test suite would not catch a bug in that line.

- **Surviving mutant** = a code change that tests fail to detect (weak test coverage for that logic)
- **Killed mutant** = tests correctly detect the change
- Run on specific modules to keep runtime reasonable: `uv run mutmut run "cuecard.retriever*"`
- The `results` command only shows surviving mutants (killed ones are omitted)
- Use `mutmut show <name>` to see what mutation survived and decide if a test is needed
- Config is in `[tool.mutmut]` in `pyproject.toml`
- Note: `tests/test_cli.py::TestAdapterMainGuard::test_main_guard` is deselected from mutmut runs due to subprocess JSON parsing flakiness under fork

## Conventions

### Code Style
- Immutable data — frozen dataclasses everywhere, never mutate
- Small files (<400 lines), small functions (<50 lines)
- Type hints everywhere, no `Any` types
- All configurable values in `config.py` with validation

### Testing
- 100% coverage required
- Unit tests: mock fastembed (return deterministic vectors)
- Integration tests: `@pytest.mark.slow`, use real BGE-small model
- `tmp_path` for all file system tests — never touch real `~/.cuecard/`

### Architecture
- Core library is agent-agnostic — no Claude Code imports in core modules
- Adapters are thin wrappers in `src/cuecard/adapters/`
- Multi-stage pipeline: multi-retriever (dense + sparse) → cross-encoder (opt-in) → LLM (opt-in)
- pipeline.py orchestrates all stages, CLI/adapter delegate to it
- Retriever adapter pattern: `Retriever` protocol in `retrievers/__init__.py`, pluggable dense/sparse/future
- RRF (Reciprocal Rank Fusion) merges results from multiple retrievers
- Parent-child collapse: rules have expansions (paraphrases), each embedded separately, max-score collapse via `rule_map`
- JSON intermediate: `rules.json` is the canonical format. Parser→JSON→Indexer. Expansions survive rebuilds via merge.
- Scoped caches: global index in `~/.cuecard/index/`, project index in `.cuecard/index/`. Both loaded and composed at retrieval time via `loader.py`. No cross-project rule leakage.
- Every pipeline step independently callable via CLI
- Provenance on every data object — trace back to source file + line
- Per-retriever observability: `RetrieverTrace` tracks candidate count, latency, unique rules per retriever

### Security
- Path validation: canonicalize via `Path.resolve()`, enforce allowed dirs
- Secrets scrubbing: regex-based redaction before logging
- File permissions: 0o700 dirs, 0o600 files, set at creation time
- Rule length limit: 500 chars max per rule
- Cache paths: fixed locations, not user-configurable

### Asymmetric Encoding (CRITICAL)
- **Index time:** `model.passage_embed(rule_texts)` for rules
- **Query time:** `model.query_embed([query_text])` for tool context
- NEVER use a single generic `embed()` for both
- fastembed returns generators — materialize with `list()`

### Index Design
- Full rebuild on any file change (no incremental splice)
- Atomic writes: NamedTempFile + os.replace
- File locking: fcntl.flock during writes
- Integrity check: checksum + rule_map length + bm25_corpus length validation on load
- Freshness checking via `load_or_build()` auto-rebuilds stale indexes on source file changes
- Metadata v2: stores `rule_map`, `bm25_corpus`, per-rule `expansions` alongside embeddings
- Backwards compat: v1 metadata auto-synthesizes identity rule_map, bm25_corpus=None

### Expansion Design
- Rules can have 0-10 expansions (paraphrases, trigger phrases, code patterns)
- Generated via `cuecard rules expand` using local LLM or Haiku
- Stored in `rules.json` intermediate format, preserved across index rebuilds
- Each expansion is embedded separately; max-score collapse selects the best match
- Constants: `MAX_EXPANSION_LENGTH=200`, `MAX_EXPANSIONS_PER_RULE=10` (in models.py)
- BM25 sparse retrieval uses expansion texts for keyword matching

## Key Design Decisions

See `artifacts/prd-v1.md` Section 17 (Decisions) and `artifacts/enriched-retrieval-prd.md` Section 10 (Decisions D1-D24).

Critical ones:
- D1: Semantic embeddings from day one (not keyword-based)
- D4: Max-score parent collapse (not mean, not weighted)
- D5: RRF over learned fusion (simple, no training data needed)
- D6: Top-k=5, threshold=0.30 (was 0.35 — that dropped 34% of relevant rules)
- D7a: Asymmetric encoding (query_embed vs passage_embed)
- D10: Full rebuild on change (no incremental splice corruption)
- D12: rule_map always present (identity for old indexes)
- D14: Inline BM25Okapi (~35 lines, no dependency)
- D17: Merge preserves expansions for unchanged rules
- S1: Path validation with allowlist (prevents traversal)

## Model Recommendations (from benchmarks)

| Stage | Model | Why |
|-------|-------|-----|
| Embedding | jina-embeddings-v2-base-code | Best recall (82.5%), code-specific |
| Cross-encoder (opt-in) | Xenova/ms-marco-MiniLM-L-6-v2 | 13ms, but regressions on code |
| LLM (lightweight) | Qwen3-Reranker-0.6B via llama-server | MTEB-Code 73.42, promptable, 370MB |
| LLM (full) | Qwen3.5-35B-A3B via llama-server | Best quality, MoE, free |
| LLM (API) | Haiku via claude-agent-sdk | Fast, Max subscription |

## Config

Two locations, layered:
- **Global:** `~/.cuecard/config.toml` — rules for all projects
- **Project:** `./cuecard.toml` — project-specific rules

Merge: scalars = project wins, sources = union, model = project wins (must match dim).

New config fields (with defaults):
```toml
[retrieval]
fusion_k = 60              # RRF parameter
sparse_enabled = true       # Enable BM25 retriever

[expansion]
max_per_rule = 10           # Max expansions per rule
max_expansion_length = 200  # Max chars per expansion
```

## Implementation Status

- **Phase 1:** Core pipeline — COMPLETE
- **Phase 2:** Adapter + logging — COMPLETE
- **Phase 3:** Eval framework + notebook — COMPLETE
- **Multi-stage:** Pipeline + reranker + LLM reranker — COMPLETE
- **Eval metrics:** noise ratio, context waste, per-tier breakdown, negative silence — COMPLETE
- **Quality iteration:** 354 PreToolUse + 84 UserPromptSubmit fixtures — COMPLETE
- **Unified events:** UserPromptSubmit support, reasoning-in-response prompt — COMPLETE
- **Enriched retrieval:** JSON intermediate, expansion-aware indexing, parent collapse, BM25+RRF, expansion CLI — COMPLETE (735+ tests, 100% coverage)
- **Eval dataset:** 587 fixtures (438 original + 149 mined from 7 real projects)
- **Phase 4:** Publish — pending

## Event Types

cuecard supports two Claude Code hook events:

- **PreToolUse**: Triggered before each tool call (Bash, Read, Edit, etc.). Retrieves coding rules.
- **UserPromptSubmit**: Triggered when the user sends a message. Retrieves workflow/process rules.

Both event types use the same pipeline. The adapter prefixes queries with the event type (`Bash: git commit` or `UserPromptSubmit: add auth to the API`). The LLM reranker uses event context to discriminate between coding and workflow rules.

## LLM Reranker Setup

For llm-local mode (best quality):
```bash
# Start Qwen3.5-35B-A3B with Jinja template support
llama-server -m ~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --port 8081 -ngl 99 -c 16384 --jinja
```

Key requirements:
- `--jinja` flag (NOT `--chat-template chatml`) — needed for `chat_template_kwargs`
- `-c 16384` — 7 few-shot system prompt needs ~5K tokens
- Thinking disabled by default via `enable_thinking: false` for ~1s latency
- `_MAX_TOKENS=1024` — reasoning-in-response needs room for 3-5 sentence analysis
- Reasoning captured in `LLMParseResult.reasoning` for debugging

## Quality Benchmarks

### PreToolUse (354 fixtures, jina-code + Qwen3.5-35B, reasoning prompt)

| Mode | Noise | NegSilence | Recall | p50 Latency |
|------|-------|------------|--------|-------------|
| embedding (t=0.30) | 64.4% | 23.6% | 40.6% | 23ms |
| **llm-local (reasoning)** | **21.4%** | **92.9%** | **42.2%** | 1143ms |

Per-tier (LLM-local):
| Tier | Recall | Noise | Silence |
|------|--------|-------|---------|
| easy | 85.2% | 25.4% | — |
| medium | 66.0% | 30.3% | — |
| hard | 40.4% | 32.7% | — |
| negative | — | 7.1% | 92.9% |

### UserPromptSubmit (84 fixtures, jina-code + Qwen3.5-35B, reasoning prompt)

| Mode | Noise | NegSilence | Recall | p50 Latency |
|------|-------|------------|--------|-------------|
| embedding (t=0.30) | 89.8% | 4.2% | 31.5% | 10ms |
| **llm-local (reasoning)** | **24.5%** | **95.8%** | **46.2%** | 3008ms |

Per-tier (LLM-local):
| Tier | Recall | Noise | Silence |
|------|--------|-------|---------|
| easy | 93.2% | 25.0% | — |
| medium | 47.1% | 44.6% | — |
| hard | 50.0% | 25.3% | — |
| negative | — | 4.2% | 95.8% |

Cross-encoder (MiniLM) is a regression on code — skip it. Use llm-local or llm-haiku.

**Note:** These benchmarks are PRE-enrichment (no expansions, no BM25). Post-enrichment benchmarks pending.

### Expansion Validation (hand-crafted, 8 hard fixtures)

| Metric | Raw | With Expansions | Delta |
|--------|-----|-----------------|-------|
| Avg cosine similarity | 0.187 | 0.558 | +0.371 |
| Threshold crossings (N→Y) | — | 10/12 pairs | — |

LLM-generated expansions (v2 prompt): 8/12 threshold crossings, avg delta +0.276.

## When in Doubt

1. Read the PRD — `artifacts/prd-v1.md` is the source of truth
2. Check the Decisions table (Section 17)
3. If not covered, make a decision, document it, continue
