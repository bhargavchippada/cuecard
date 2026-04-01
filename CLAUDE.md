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
│   ├── __init__.py         # Public API: load_or_build, retrieve, format_rules
│   ├── models.py           # Frozen dataclasses (Rule, Provenance, Index, etc.)
│   ├── config.py           # Load/merge/validate cuecard.toml configs
│   ├── security.py         # Path validation, secrets scrubbing, permissions
│   ├── parser.py           # Parse rule files → list[Rule] (dispatch by suffix)
│   ├── indexer.py          # Embed rules via fastembed, build index, atomic writes
│   ├── freshness.py        # mtime + hash checking, full rebuild on change
│   ├── retriever.py        # query_embed → dot product → top-k → dedup
│   ├── formatter.py        # Format results for injection
│   ├── pipeline.py         # Multi-stage pipeline orchestrator
│   ├── reranker.py         # Cross-encoder re-ranking (Stage 2, opt-in)
│   ├── llm_reranker.py     # LLM re-ranking (Stage 3, opt-in)
│   ├── eval.py             # Evaluation framework (precision, recall, MRR, nDCG)
│   ├── logger.py           # Structured JSONL logging with secrets scrubbing
│   ├── cli.py              # Typer CLI (setup, config, retrieve, rules, eval, etc.)
│   ├── py.typed            # PEP 561 marker
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
```

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
- Multi-stage pipeline: embedding → cross-encoder (opt-in) → LLM (opt-in)
- pipeline.py orchestrates all stages, CLI/adapter delegate to it
- Every pipeline step independently callable via CLI
- Provenance on every data object — trace back to source file + line

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
- Integrity check: checksum + rule count validation on load

## Key Design Decisions

See `artifacts/prd-v1.md` Section 17 (Decisions) for the full table (27 decisions).

Critical ones:
- D1: Semantic embeddings from day one (not keyword-based)
- D6: Top-k=5, threshold=0.30 (was 0.35 — that dropped 34% of relevant rules)
- D7a: Asymmetric encoding (query_embed vs passage_embed)
- D10: Full rebuild on change (no incremental splice corruption)
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

## Implementation Status

- **Phase 1:** Core pipeline — COMPLETE
- **Phase 2:** Adapter + logging — COMPLETE
- **Phase 3:** Eval framework + notebook — COMPLETE
- **Multi-stage:** Pipeline orchestrator + reranker + LLM reranker — COMPLETE (494 tests, 100% coverage)
- **Eval metrics:** Precision/context efficiency metrics — COMPLETE (noise ratio, context waste, per-tier breakdown, negative silence rate)
- **Phase 4:** Publish — pending
- **Phase 4:** Publish (PyPI, GitHub, CI)

## When in Doubt

1. Read the PRD — `artifacts/prd-v1.md` is the source of truth
2. Check the Decisions table (Section 17)
3. If not covered, make a decision, document it, continue
