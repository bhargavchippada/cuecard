# cuecard

> The right rule, at the right moment.

Contextual rule enforcement for AI coding agents. Retrieves relevant user-defined rules via semantic similarity and injects them at the stages that matter — before actions (PreToolUse), on user messages (UserPromptSubmit), when spawning subagents (SubagentStart), and at turn end (Stop). PostToolUse was removed to avoid over-triggering; it may return later with tighter post-action matching.

## Project Structure

```
cuecard/
├── CLAUDE.md               # This file
├── README.md               # Public-facing readme
├── .gitignore
├── artifacts/              # Design docs & session state
│   ├── README.md                     # Index of consolidated history docs
│   ├── project-history.md            # Phase timeline, decisions, status
│   ├── design-history.md             # PRD evolution, architecture choices
│   ├── evaluation-history.md         # Benchmark history, baselines, deltas
│   └── research-notes.md             # Rule-rewrite research, prompt iterations
├── src/cuecard/            # Core library (agent-agnostic)
│   ├── __init__.py         # Public API: load_config, retrieve, format_rules
│   ├── models.py           # Frozen dataclasses, safety caps, ResolvedConfig (single source of truth)
│   ├── config.py           # Load/merge/validate cuecard.toml configs
│   ├── security.py         # Path validation, secrets scrubbing, permissions
│   ├── logger.py           # Structured JSONL logging with secrets scrubbing
│   ├── serve.py            # Persistent daemon server (HTTP, PID management)
│   ├── _entry.py           # Fast CLI entry point (hook fast-path, then Typer)
│   ├── _math.py            # L2 normalization helper
│   ├── py.typed            # PEP 561 marker
│   ├── indexing/           # Parsing, embedding, expansion, freshness, loading
│   │   ├── parser.py       # Parse rule files → list[Rule] (.txt/.json/.toml)
│   │   ├── indexer.py      # Embed rules via fastembed, build index, rules.json I/O
│   │   ├── freshness.py    # mtime + hash checking, full rebuild on change
│   │   ├── loader.py       # Unified index loading with freshness + scope composition
│   │   └── expander.py     # LLM-based rule expansion generation
│   ├── retrieval/          # Pipeline, retrievers, rerankers, affinity, formatting
│   │   ├── pipeline.py     # Multi-stage pipeline orchestrator (dense + sparse + LLM)
│   │   ├── retriever.py    # query_embed → parent collapse → top-k → dedup
│   │   ├── dense.py        # DenseRetriever (cosine + parent collapse)
│   │   ├── sparse.py       # SparseRetriever (BM25Okapi + parent collapse)
│   │   ├── fusion.py       # Retriever protocol, ScoredCandidate, RRF fuse()
│   │   ├── reranker.py     # Cross-encoder re-ranking (Stage 2, opt-in)
│   │   ├── llm_reranker.py # LLM re-ranking (Stage 3, opt-in)
│   │   ├── llm_utils.py    # Shared LLM helpers (validate_endpoint, call_local/haiku)
│   │   ├── affinity.py     # Event/tool affinity inference, storage, event mask
│   │   └── formatter.py    # Format results for injection
│   ├── eval/               # Evaluation harness, metrics, reporting
│   │   ├── harness.py      # load_fixtures, run_eval, dataclasses
│   │   ├── metrics.py      # IR metrics (precision, recall, MRR, nDCG, noise, quality)
│   │   └── report.py       # Tier summaries, per-event metrics, report formatting
│   ├── cli/                # Typer CLI commands
│   │   ├── main.py         # Core commands (config, retrieve, index, serve, migrate)
│   │   ├── setup.py        # Interactive setup + configure commands
│   │   ├── rules.py        # Rules subcommands (add, remove, search, expand)
│   │   ├── hooks.py        # Install/uninstall/status/log commands
│   │   └── eval_cmd.py     # Eval command
│   └── adapters/           # Agent-specific wrappers
│       ├── __init__.py
│       └── claude_code.py  # Claude Code hook adapter (4 active events)
├── tools/
│   ├── bench_e2e.py        # E2E benchmark (default — model generates own expansions + reranks)
│   └── notebook.ipynb      # 35-cell debugging tool (per-stage viz, loss analysis, prompts, batch eval)
├── eval/                   # Evaluation framework (Phase 3)
│   ├── corpora/
│   ├── fixtures/
│   └── results/
├── examples/
│   ├── cuecard.toml        # Example config
│   └── rules.txt           # Example rules
├── tests/                  # Mirrors src/ structure
│   ├── conftest.py         # Shared fixtures + network guard + 1s timeout
│   ├── test_models.py, test_config.py, test_security.py, test_logger.py, test_math.py, test_e2e.py
│   ├── indexing/           # 9 files: parser, indexer (3), freshness, loader, expander (3)
│   ├── retrieval/          # 15 files: pipeline (2), retriever, dense, sparse, fusion,
│   │                       #   reranker, llm_reranker (4), llm_utils, affinity, event_mask, formatter
│   ├── eval/               # 3 files: metrics, per_event, pipeline
│   ├── cli/                # 8 files: config, eval, hooks, index, migrate, retrieve, rules, setup
│   ├── adapters/           # 1 file: adapter
│   └── serve/              # 6 files: cli, client, handler, http, lifecycle, pid
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
uv run pytest                                  # Run tests (fast, ~4.3s, no real endpoints)
uv run pytest -m slow                          # Run integration tests (real model)
uv run pytest -m '' --cov=cuecard --cov-fail-under=100  # Coverage (all tests incl. slow)
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
- Note: `tests/cli/test_cli_hooks.py::TestAdapterMainGuard::test_main_guard` is deselected from mutmut runs due to subprocess JSON parsing flakiness under fork

## Conventions

### Code Style
- Immutable data — frozen dataclasses everywhere, never mutate
- Small files (<400 lines), small functions (<50 lines)
- Type hints everywhere, no `Any` types
- All configurable values in `ResolvedConfig` (models.py) — single source of truth for defaults + validation metadata. `config.py` derives validators and defaults from field metadata. Adding a new config field: add to ResolvedConfig with default + metadata, add TOML key extraction in `_extract_flat()`

### Testing
- 100% coverage required (run with `-m ''` to include slow tests)
- Unit tests: mock fastembed (return deterministic vectors)
- Integration tests: `@pytest.mark.slow`, use real BGE-small model, excluded by default via `addopts`
- `tmp_path` for all file system tests — never touch real `~/.cuecard/`
- **Network guard**: autouse conftest fixture blocks `socket.create_connection` and `socket.getaddrinfo` for non-localhost hosts — tests that need external DNS must mock at `cuecard.*.socket.getaddrinfo`
- **1s timeout**: pytest-timeout enforces 1s per test (thread method). Slow tests are exempt via marker exclusion
- Tests mirror src/ structure: `tests/indexing/`, `tests/retrieval/`, `tests/eval/`, `tests/cli/`, `tests/adapters/`, `tests/serve/`
- **Monkeypatch rule**: always patch the *defining* module, not re-exports (e.g., `cuecard.cli.main._home_dir`, not `cuecard.cli._home_dir`)

### Architecture
- Core library is agent-agnostic — no Claude Code imports in core modules
- Adapters are thin wrappers in `src/cuecard/adapters/`
- Multi-stage pipeline: multi-retriever (dense + sparse) → cross-encoder (opt-in) → LLM (opt-in)
- pipeline.py orchestrates all stages; ALL retrieval paths (CLI, adapter, eval) route through `run_pipeline()` — even `embedding` mode uses the full dense+sparse+RRF stage
- Retriever adapter pattern: `Retriever` protocol in `retrieval/fusion.py`, pluggable dense/sparse/future
- RRF (Reciprocal Rank Fusion) merges results from multiple retrievers
- Parent-child collapse: rules have expansions (paraphrases), each embedded separately, max-score collapse via `rule_map`
- JSON intermediate: `rules.json` is the canonical format. Parser→JSON→Indexer. Expansions survive rebuilds via merge. Both `cuecard index` and `load_or_build()` use the same merge lifecycle.
- Scoped caches: global index in `~/.cuecard/index/`, project index in `.cuecard/index/`. Both loaded and composed at retrieval time via `loader.py`. No cross-project rule leakage.
- Every pipeline step independently callable via CLI
- Provenance on every data object — trace back to source file + line
- Per-retriever observability: `RetrieverTrace` tracks candidate count, latency, unique rules per retriever

### Pipeline Diagram (llm-local mode — current production default)

```
                         ┌─────────────────────────────┐
                         │   Claude Code hook event    │
                         │ (PreToolUse│UserPromptSubmit│
                         │   SubagentStart│Stop)       │
                         └──────────────┬──────────────┘
                                        │ stdin JSON
                                        ▼
                         ┌─────────────────────────────┐
                         │  adapter/claude_code.py     │
                         │  • detect event             │
                         │  • dispatch to handler      │
                         │  • scrub_secrets()          │
                         │  • build query string       │
                         └──────────────┬──────────────┘
                                        │ (query, event, tool_name)
                                        ▼
                         ┌─────────────────────────────┐
                         │  daemon fast-path (:8452)   │──── hit ────┐
                         │  else cold-load model       │             │
                         └──────────────┬──────────────┘             │
                                        │ miss                       │
                                        ▼                            │
                         ┌─────────────────────────────┐             │
                         │  load_or_build(Index)       │             │
                         │  + AffinityIndex            │             │
                         │  rules_global.txt  (107)    │             │
                         └──────────────┬──────────────┘             │
                                        ▼                            │
          ┌───────────────────────── run_pipeline ────────────────┐  │
          │                                                       │  │
          │ ╔═════════════════════════════════════════════════╗   │  │
          │ ║ Stage 0  Event mask (affinity)                  ║   │  │
          │ ║   filter rules to this event's affinity         ║   │  │
          │ ║   (PreToolUse → tool_use + both)                ║   │  │
          │ ║   (workflow events → workflow + both)           ║   │  │
          │ ║   Optional: query expansion (OFF by default)    ║   │  │
          │ ╚═════════════════════════════════════════════════╝   │  │
          │                        │                              │  │
          │                        ▼                              │  │
          │ ╔═════════════════════════════════════════════════╗   │  │
          │ ║ Stage 1  Retrieval  (always runs)               ║   │  │
          │ ║                                                 ║   │  │
          │ ║   ┌──────────────┐      ┌──────────────┐        ║   │  │
          │ ║   │ Dense        │      │ Sparse BM25  │        ║   │  │
          │ ║   │ jina-code-v2 │      │ (opt-in;     │        ║   │  │
          │ ║   │ 768d cosine  │      │  OFF default)│        ║   │  │
          │ ║   │ parent-max   │      │ parent-max   │        ║   │  │
          │ ║   └──────┬───────┘      └──────┬───────┘        ║   │  │
          │ ║          │                     │                ║   │  │
          │ ║          └──────── RRF ────────┘                ║   │  │
          │ ║                fusion_k=10                      ║   │  │
          │ ║        top_k=12 (llm_candidates)                ║   │  │
          │ ║        threshold=0.25 (llm_recall_threshold)    ║   │  │
          │ ╚══════════════════════╤══════════════════════════╝   │  │
          │                        │                              │  │
          │                        ▼                              │  │
          │ ╔═════════════════════════════════════════════════╗   │  │
          │ ║ Stage 2  Cross-encoder rerank  (SKIPPED in      ║   │  │
          │ ║   llm-local mode — regresses on code)           ║   │  │
          │ ╚═════════════════════════════════════════════════╝   │  │
          │                        │                              │  │
          │                        ▼                              │  │
          │ ╔═════════════════════════════════════════════════╗   │  │
          │ ║ Stage 3  LLM reranker                           ║   │  │
          │ ║   Gemma-4-E4B via llama-server :8081            ║   │  │
          │ ║   system prompt: 13 few-shot examples,          ║   │  │
          │ ║     strict trigger semantics,                   ║   │  │
          │ ║     "when in doubt, exclude"                    ║   │  │
          │ ║   input:  ≤12 numbered candidate rules          ║   │  │
          │ ║   output: JSON {reasoning, rules: [ids...]}     ║   │  │
          │ ║   capped at top_k=7 final                       ║   │  │
          │ ╚══════════════════════╤══════════════════════════╝   │  │
          └────────────────────────┼──────────────────────────────┘  │
                                   │ RankedResult[]                  │
                                   ▼                                 ▼
                         ┌─────────────────────────────┐  ◄──── join ┘
                         │  formatter.format_rules()   │
                         │  + per-event label          │
                         │    (PREVENT/PROPAGATE/AUDIT)│
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │  hookSpecificOutput JSON    │
                         │  {additionalContext: ...,   │
                         │   permissionDecision: allow}│ (PreToolUse only)
                         └──────────────┬──────────────┘
                                        │ stdout
                                        ▼
                                 Claude Code
                             (rules injected into
                                agent context)
```

Key knobs (defaults from `ResolvedConfig`):

| Knob | Default | Notes |
|---|---|---|
| Index size | 107 rules | `rules_global.txt` |
| Embedding model | jina-code-v2 (768d) | `config.model_name` |
| Event mask | ON | always when affinity present |
| BM25 sparse | **OFF** | session 31 — no net benefit with code-specific dense |
| Query expansion | **OFF** | session 31 — net F2 regression |
| Stage-1 `top_k` | 12 (`llm_candidates`) | candidates fed to LLM |
| Stage-1 `threshold` | 0.25 (`llm_recall_threshold`) | wider recall for LLM |
| RRF `fusion_k` | 10 | |
| Final `top_k` | 7 | after LLM |
| LLM reranker | Gemma-4-E4B (local) | llama-server :8081 |
| Daemon | localhost:8452 | fast-path, 500ms timeout |

Graceful degradation: every stage catches its own exceptions and falls back to the previous stage's results, logging the error in `StageTrace.error`. The pipeline never hard-fails — worst case, you get dense-only results.

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

### Affinity Design
- Rules have event/tool affinity — which hook events and tools they apply to
- Two modes: `infer` (LLM classifies at index time, default) and `strict` (explicit annotations only)
- **Minimum retrieval scope**: LLM classifies rules as `tool_use` (→ PreToolUse), `workflow` (→ UserPromptSubmit+SubagentStart+Stop), or `both` (→ all 4 events). Prompt asks "when would the agent misbehave without this rule?" — tool_use if at tool time, workflow if at planning time, both only if agent can plan AROUND the tool event entirely. PostToolUse was removed from the active event set to avoid over-triggering.
- `AffinityIndex` uses O(1) dict lookup (not frozen dataclass — follows `Index` pattern)
- `LoadedIndex` composite return type wraps `(Index, AffinityIndex | None)`
- `load_or_build()` returns `LoadedIndex | None` — all call sites destructure
- Event mask: boolean numpy array applied post-scoring in both dense and sparse retrievers
- BM25 IDF stats preserved (mask is post-scoring, not pre-filtering)
- `affinity.json` sidecar with SHA-256 checksum integrity validation
- `KNOWN_HOOK_EVENTS` canonical constant in `models.py` — single source of truth
- TOML rule format: `[[rules]]` with optional `events` and `tools` frozenset fields
- `cuecard migrate` converts .txt → .toml (comments NOT preserved)
- `run_eval()` accepts optional `affinity` parameter and passes `event` + `affinity` to `run_pipeline()`

### Expansion Design
- Rules can have 0-5 expansions by default (paraphrases, trigger phrases, code patterns)
- Generated via `cuecard rules expand` using local LLM or Haiku
- Stored in `rules.json` intermediate format, preserved across index rebuilds
- Each expansion is embedded separately; max-score collapse selects the best match
- Safety caps in models.py: `MAX_EXPANSION_LENGTH=MAX_RULE_LENGTH` (500), `MAX_EXPANSIONS_PER_RULE=10`, `MAX_RULES_PER_FILE=500`, `MAX_REQUEST_BYTES=1M`, `MAX_TOOL_NAME_LENGTH=200`
- Operational defaults in ResolvedConfig: `expansion_max_per_rule=5`, `expansion_max_length=500`, `expansion_dedup_threshold=0.80`
- BM25 sparse retrieval uses expansion texts for keyword matching

## Key Design Decisions

See `artifacts/prd-v1.md` Section 17 (Decisions) and `artifacts/enriched-retrieval-prd.md` Section 10 (Decisions D1-D24).

Critical ones:
- D1: Semantic embeddings from day one (not keyword-based)
- D4: Max-score parent collapse (not mean, not weighted)
- D5: RRF over learned fusion (simple, no training data needed), fusion_k=10 (was 60 — full-sample sweep showed k=10 wins on all metrics for 107-rule corpus)
- D6: Top-k=7, threshold=0.30 (was 5/0.35 — top_k=5 caused crowding drops on multi-rule events like git commit)
- D25: LLM receives `llm_candidates` rules (default 12) — configurable in TOML `[retrieval] llm_candidates = 12`
- D7a: Asymmetric encoding (query_embed vs passage_embed)
- D10: Full rebuild on change (no incremental splice corruption)
- D12: rule_map always present (identity for old indexes)
- D14: Inline BM25Okapi (~35 lines, no dependency)
- D17: Merge preserves expansions for unchanged rules
- S1: Path validation with allowlist (prevents traversal)

## Model Recommendations

| Stage | Model | Why |
|-------|-------|-----|
| Embedding | jina-embeddings-v2-base-code | Best PreToolUse hard recall, code-specific, 22ms |
| Cross-encoder (opt-in) | Xenova/ms-marco-MiniLM-L-6-v2 | 13ms, but regresses on code — skip |
| **LLM (GPU, default)** | **Gemma-4-E4B via llama-server** | Best PosRecall, best workflow metrics, ~1200ms p50 (8.2GB Q8_0) |
| LLM (GPU, alt) | Qwen3.5-9B / Qwen3.5-35B-A3B | Historical comparisons kept in `artifacts/evaluation-history.md` |
| LLM (CPU/laptop) | Qwen3.5-4B via llama-server | 2.6GB, ~900ms p50, viable for laptop |
| LLM (API) | Haiku via claude-agent-sdk | Fast, Max subscription |

Historical multi-model comparisons (sessions 19–29) are captured in `artifacts/evaluation-history.md` and are no longer maintained here — Gemma-4-E4B is the current production default.

### Gemma 4 E4B Setup

```bash
# Recommended: Gemma 4 E4B Q8_0 (default, best all-rounder)
llama-server -m /home/turiya/models/gemma-4-E4B-it-Q8_0.gguf \
  --port 8081 -ngl 99 -c 98304 --jinja -np 5 --reasoning off
```

Requires llama.cpp build ≥8672 (gemma4 architecture support added after b8235).

**Key flags:**
- `--reasoning off` — disables thinking server-side; no need for `chat_template_kwargs` in client code
- `-np 5` — 5 parallel slots (context split: 98304/5 = 19660 tokens per slot)
- `-c 98304` — total context (~20GB VRAM with model, fits on 32GB GPU)
- `--jinja` — required for Gemma's chat template

The same model must handle both expansion generation and reranking — rules out cross-encoder-only models.

### Embedding Model Comparison (archived — embedding mode, 354 PreToolUse fixtures)

| Model | Dim | Easy | Medium | Hard | Recall | Noise | p50ms |
|-------|-----|------|--------|------|--------|-------|-------|
| jina-code-v2 | 768 | 95.3% | 76.8% | 60.0% | 49.0% | 83.1% | 22ms |
| mxbai-embed-large | 1024 | 96.0% | 77.7% | 57.7% | 49.1% | 83.6% | 45ms |
| snowflake-arctic-m | 768 | 94.2% | 73.7% | 57.5% | 47.5% | 83.7% | 14ms |
| bge-small (default) | 384 | 94.2% | 74.1% | 51.1% | 46.8% | 84.0% | 4ms |

Key insight: model choice gives ~4% recall spread. The LLM reranker gives ~70% improvement on noise/silence. jina-code-v2 is the current embedding default.

## Config

Two locations, layered:
- **Global:** `~/.cuecard/config.toml` — rules for all projects
- **Project:** `./cuecard.toml` — project-specific rules

Merge: scalars = project wins, sources = union, model = project wins (must match dim).

Config fields (all defaults from ResolvedConfig):
```toml
[retrieval]
fusion_k = 10               # RRF parameter (full-sample sweep → 10)
sparse_enabled = true       # BM25 retriever — ON in ResolvedConfig; eval harness defaults to False (session 31: no net benefit with code-specific dense)
llm_candidates = 12         # Rules sent to LLM reranker
llm_recall_threshold = 0.25 # Embedding threshold for LLM modes (wider recall)
reranker_model = "Xenova/ms-marco-MiniLM-L-6-v2"  # Cross-encoder model (opt-in only)

[serve]
port = 8452                 # Daemon port

[expansion]
max_per_rule = 5            # Max expansions per rule (default 5, cap 10)
max_length = 500            # Max chars per expansion (= MAX_RULE_LENGTH)
dedup_threshold = 0.80      # Cosine similarity for semantic dedup

[pipeline.llm]
max_tokens = 1024           # Max tokens for LLM response
timeout = 60.0              # HTTP timeout for LLM calls (seconds)
```

## Implementation Status

**Complete:**
- Core pipeline (dense + sparse + RRF + LLM reranker, graceful degradation, per-stage traces)
- Adapter + JSONL logging with secrets scrubbing
- Eval framework: F2 / PosRecall / Noise / NegSil + per-event breakdown, stratified sampling, tqdm progress
- Enriched retrieval: JSON intermediate, expansion-aware indexing, parent-max collapse, expansion CLI
- Closed-loop hooks: TOML rule format, event/tool affinity inference, event mask retrieval, 4 hook adapters (PreToolUse, UserPromptSubmit, SubagentStart, Stop), per-event eval metrics
- E2E benchmarking (`tools/bench_e2e.py`) — each model generates its own affinities + expansions + reranks. Corpora cached at `eval/corpora/enriched_{label}/`.
- Expansion prompt v5: reasoning-field prompt for expansions (structured CoT before generating)
- Prompt engineering: few-shot examples with strict trigger semantics, "when in doubt, exclude"
- Eval corpus: 107 rules in `eval/corpora/rules_global.txt`, one fixture file per event (`pre_tool_use.json`, `workflow.json`, `stop.json`, `subagent_start.json`)
- Tagged corpus: `eval/corpora/rules_global_tagged.json` — 60 tool_use, 44 workflow, 3 both (ground truth for affinity accuracy measurement)
- CLI UX: `cuecard configure`, `cuecard serve` daemon, expand progress bar
- Hook format: correct `hookEventName` + `permissionDecision`, `hook_event_name` input field detection
- Global install (`uv tool install`) + `cuecard hook` CLI entry point
- Live validation in real Claude Code sessions
- **Session 31 cleanup:** PostToolUse removed from active events (over-triggers; PreToolUse already covers prevention); basic.json → pre_tool_use.json; stop_mined.json merged into stop.json; one file per event

**Pending / In flight:**
- Phase 6 Completion Gate: PRD converged (v1.2). Stop hook reads `last_assistant_message` + `transcript_path`, always blocks up to `max_stop_blocks`. Implementation pending.
- Phase 7 Rule Quality: trigger-aware rule rewriting ("When X: do Y — because Z"). 63/107 done (PreToolUse subset), 44 workflow rules remaining.
- Publish to PyPI
- Multi-source parsing (markdown, YAML, CLAUDE.md): DRAFT PRD (`artifacts/phase5-multi-source-prd-draft.md`)

## Event Types

cuecard supports four Claude Code hook events (closed-loop enforcement). PostToolUse was intentionally removed — it over-triggered with the current retrieval pipeline, and PreToolUse already covers the prevention use case. It may return later with tighter post-action matching.

- **PreToolUse**: Before each tool call. PREVENT — inject action constraints.
- **UserPromptSubmit**: When user sends a message. GUIDE — process/methodology rules.
- **SubagentStart**: When a subagent spawns. PROPAGATE — rules for delegated work.
- **Stop**: When a turn ends. AUDIT — verify rules were followed.

All event types use the same pipeline. The adapter dispatches to per-event handlers that construct queries:
- PreToolUse: `"{tool_name}: {tool_input[:500]}"`
- UserPromptSubmit: `"UserPromptSubmit: {prompt[:500]}"`
- SubagentStart: `"SubagentStart:{agent_type}: {prompt[:500]}"`
- Stop (current): `"Stop: {stop_reason[:500]}"` (fallback: "turn completed") — minimal context
- Stop (Phase 6): `"Stop: User asked: {user_prompt[:300]} | Agent said: {last_assistant_msg[:200]}"` — reads transcript

**Stop hook input fields (from Claude Code):**
- `last_assistant_message` — agent's final response text
- `transcript_path` — full session JSONL (extractable user prompts)
- `stop_hook_active` — boolean loop guard (unverified, counter is primary guard)
- `session_id` — session identifier
- Stop output uses flat `{"decision": "block", "reason": "..."}` NOT `hookSpecificOutput` wrapper

SubagentStart and Stop both apply `scrub_secrets()` before query construction.

### Hook Output Format (CRITICAL)

Claude Code hooks require specific JSON output formats per event type:

**PreToolUse** — must include `permissionDecision`:
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow", "additionalContext": "..."}}
```

**UserPromptSubmit** — NO `permissionDecision` (omit to allow):
```json
{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "..."}}
```

**Input field**: Claude Code sends `hook_event_name` (not `event`). The adapter checks both for backwards compat.

**Stderr**: Any stderr output causes "hook error" display. Hook command uses `2>/dev/null`.

### Serve Daemon

`cuecard serve` runs a persistent HTTP server on localhost:8452 that keeps the embedding model loaded. The hook adapter (`claude_code.py`) tries the daemon first (500ms timeout), falls back to direct model loading if unavailable. Saves ~200ms cold-start per hook call.

## LLM Reranker Setup

Default: Gemma-4-E4B — see **Gemma 4 E4B Setup** above for the canonical launch command.

Alternative models (historical — keep for reference, not benchmarked in session 31+):
```bash
# Qwen3.5-9B — lower PosRecall than Gemma, similar F2
llama-server -m ~/models/Qwen3.5-9B-Q4_K_M.gguf --port 8081 -ngl 99 -c 16384 --jinja

# Qwen3.5-4B — CPU/laptop viable, 2.6GB
llama-server -m ~/models/Qwen3.5-4B-Q4_K_M.gguf --port 8081 -ngl 0 -c 16384 --jinja
```

Key requirements (apply to all models):
- `--jinja` flag — required for Gemma's chat template
- `--reasoning off` — disables thinking server-side, eliminates `chat_template_kwargs` overhead
- `-np 5` — 5 parallel slots (19660 tokens/slot with `-c 98304`)
- `-c 98304` — total context (~20GB VRAM with model)
- `LLM_MAX_TOKENS=1024` — reasoning-in-response needs room for 3-5 sentence analysis
- Reasoning captured in `LLMParseResult.reasoning` for debugging
- Reranker passes `stop=None` to `call_local()` — no stop sequence for JSON responses
- Expander keeps default `stop=("\n\n",)` to prevent repetition degeneration

### Prompt Engineering (session 31)
- System prompt has 13 few-shot examples with strict trigger semantics
- Principles (in order): what-could-go-wrong → think-one-step-ahead → read-only-needs-no-rules → match-actions-not-keywords → inspect-code-content → **don't-fire-rules-already-followed** → **trigger-conditions-are-strict** → **when-in-doubt-exclude**
- "When in doubt, exclude" replaced the older "when in doubt, include" (session 31) — precision over recall for the tool-use pipeline
- Rules are preventive, not congratulatory (e.g. `uv run mypy` does NOT fire the "run mypy before commit" rule)
- Trigger verbs are matched strictly: "When running git commit" does NOT fire on `git rebase`, `git diff`, `git merge`, `git tag`

### Expansion Prompt (v5 + v4 format)
- Reasoning-field prompt: model reasons about vocabulary gap before generating expansions
- v4 format: balanced abstract concept tags + specific tool-name examples (`ceil(max/2)` abstract + `floor(max/2)` specific)
- Cross-domain boundary enforcement (PreToolUse vs UserPromptSubmit)
- Semantic dedup (cosine > 0.85) removes near-duplicate expansions

## Eval Metrics

### Quality Score (F2) — Primary Metric
Single composite metric per fixture using F-beta with beta=2 (recall-weighted):

- **Positive fixtures** (has should_match): `F2 = 5*P*R / (4P + R)`
  - Rewards finding correct rules (recall) while penalizing noise (precision)
  - Weights recall 4x over precision — missed rules are worse than extra rules
- **Negative fixtures** (no should_match): `1.0 if silent, 0.0 otherwise`
  - Correct abstention convention — staying silent is a perfect outcome

### Key Metrics
- **positive_recall**: Recall averaged only over positive fixtures (excludes negatives)
- **positive_quality**: F2 averaged only over positive fixtures (comparable across datasets)
- **mean_quality**: F2 averaged over ALL fixtures (includes negatives as 1.0/0.0)
- **negative_silence_rate**: Fraction of negative fixtures with 0 retrieved
- **noise_ratio**: Fraction of retrieved rules that are irrelevant

Note: `mean_recall` still includes negatives as 0.0 for backwards compatibility. Use `positive_recall` for the correct positive-only metric.

## Quality Benchmarks

### Current Baseline (session 33 — Gemma-4-E4B, all 4 tiers traced, 20% sample, seed=42)

**Corpus:** `eval/corpora/rules_global.txt` (107 rules) · **Cache:** `enriched_gemma-e4b-s32/`
**Pipeline:** dense (jina-code-v2) → event mask → LLM reranker (Gemma-4-E4B, 13 few-shot, "when in doubt, exclude")

| Tier | N | F2 | PosRecall | Noise | NegSil | p50ms |
|------|--:|----:|---------:|------:|-------:|------:|
| **pre_tool_use** (PreToolUse) | 87 | **0.662** | 0.764 | 0.391 | 0.634 | 3025 |
| **workflow** (UserPromptSubmit) | 41 | **0.689** | 0.449 | 0.317 | 0.913 | 2756 |
| **stop** (Stop) | 33 | **0.556** | 0.474 | 0.460 | 0.714 | 2833 |
| **subagent_start** (SubagentStart) | 33 | **0.398** | 0.857 | 0.704 | 0.211 | 2892 |

**Event mask (rules visible after inferred affinity):**
- PreToolUse: 64/107 (43 masked)
- UserPromptSubmit, Stop, SubagentStart: 45/107 each (62 masked) — binary affinity lumps all three workflow events together, so they see an identical rule set.

**Stage latency (median):** retrieval ~40ms · LLM reranker ~2800ms. LLM dominates end-to-end.

**Known gaps (from traces at `eval/results/gemma-e4b-s32-traced-seed42/`):**
- SubagentStart is the worst tier — Noise 0.704, NegSil 0.211. Planner/architect/tdd-guide rules fire on every subagent, including pure-observation ones.
- Stop has embedding recall failures — rules phrased imperatively don't retrieve on past-tense queries (`stop-sensitive-file-created` missed rotate-secrets rule at stage 1).
- Reranker drops correct rules it saw in stage 1 (`mined-git-add-commit-no-verify` dropped 2 of 4 matches) — "when in doubt, exclude" slightly over-tuned for multi-rule scenarios.
- Rule-keyword drift on test runners: `vitest`, `jest`, `npx jest --coverage` all trigger the pytest-specific "5s suite" rule.

### Traced Benchmark Artifacts

`tools/bench_e2e.py` produces a full debug bundle per run at `eval/results/{label}-traced-seed{seed}/`:

- `report.md` — metrics tables, stage latencies, top FNs/FPs/noisy TPs with stage-1 candidates and LLM reasoning
- `summary.json` — machine metrics per tier
- `config.json` — run config (label, seed, corpus path, endpoint)
- `traces/{tier}/{fixture_id}.json` — per-fixture: pipeline stages, event mask stats (`embeddings_masked` vs rule-level `rules_masked`), stage-1 candidates with scores, LLM prompt + raw response + reasoning + selected indices, final metrics, classification (`tp_exact`/`tp_noisy`/`partial`/`fn`/`fp`/`tn`)
- `llm_calls.jsonl` — flat per-call log of every LLM interaction

Every run is reproducibly debuggable without re-invoking the LLM. Tracing is achieved by monkey-patching `llm_reranker.rerank_llm` with thread-local fixture-ID correlation — core pipeline/harness code is untouched.

### Archived Benchmarks

Older multi-model and per-session benchmark tables are preserved in `artifacts/evaluation-history.md` and `artifacts/project-history.md`. They compared Qwen 4B/9B/35B and earlier prompt iterations and are NOT directly comparable to the current baseline.

### Quality Gap Analysis (still valid, sessions 25–31)

**Affinity is not the bottleneck.** Inferred affinity matches ground truth at ~90% accuracy; the remaining mismatches are genuinely borderline. Affinity mask gives PreToolUse ~28% rule reduction (tool_use + both).

**The bottleneck is embedding/expansion quality.** Session 26 `top_k=30` diagnostic showed only +0.8% recall improvement over `top_k=5` — correct rules simply never reach the reranker because they aren't in the embedding neighborhood of the query.

**Root cause: rule text is the only retrieval signal.** Rules that describe WHAT but not WHEN match every topically-similar query. "Run convergence reviews" matches everything about reviews/quality. "After completing an implementation phase: run convergence reviews" matches specifically. This drives Phase 7 (trigger-aware rule rewriting).

**Session 31 discoveries:**
- Baseline F2 can be inflated by *coordinated wrongness* (fixtures + LLM wrong in the same direction). Strict trigger-condition audits revealed true F2.
- Few-shot examples override principles — principles said "strict" but examples showed loose matching → LLM followed examples. Tune both.
- BM25 sparse adds almost nothing when dense is code-specific (jina-code-v2) — disabled in eval by default.
- Query expansion (tag-to-tag matching) works in isolation (+0.019 F2) but loses net by displacing specific tool-shaped expansions — disabled by default.

## Benchmarking Guidelines

**CRITICAL: Follow these rules for correct, comparable benchmarks.**

### Running E2E Benchmarks
```bash
# Standard benchmark (20% sample, reuses cached expansions if present)
uv run python tools/bench_e2e.py \
  --model-path ~/models/gemma-4-E4B-it-Q8_0.gguf \
  --label gemma-e4b --no-server --sample-ratio 0.20 --seed 42

# Single tier only (faster iteration)
uv run python tools/bench_e2e.py \
  --model-path ~/models/gemma-4-E4B-it-Q8_0.gguf \
  --label gemma-e4b --no-server --sample-ratio 0.20 --seed 42 \
  --tiers pre_tool_use

# Fresh benchmark (regenerates affinity + expansions — slow, ~2 min for 107 rules)
uv run python tools/bench_e2e.py \
  --model-path ~/models/gemma-4-E4B-it-Q8_0.gguf \
  --label gemma-e4b-fresh --no-server --force-expand --sample-ratio 0.20 --seed 42
```

Tiers: `pre_tool_use`, `workflow`, `stop`, `subagent_start`

### Rules for Valid Benchmarks
1. **Each model generates its own expansions + affinity.** Never share corpora between models — shared-corpus comparisons unfairly bias toward the expansion-generator. Use `--force-expand` with a unique `--label` per model.
2. **Always use inferred affinity, never ground truth.** The benchmark infers affinity via LLM at runtime. Ground truth labels (`rules_global_tagged.json`) are for accuracy measurement only — using them for event masking is data leakage.
3. **Use `--no-server` with an existing llama-server.** The benchmark doesn't start/stop servers. Start `llama-server` manually first (see Gemma 4 E4B Setup above).
4. **Same seed for comparison.** Always `--seed 42` when comparing models/configs. Different seeds produce different fixture samples.
5. **`--sample-ratio 0.20` for iteration, `1.0` for final numbers.** 20% gives n=87 for PreToolUse (~50s per tier). Full dataset for publication-grade numbers.
6. **Pipeline stages run in order:** dense retrieval (+optional sparse) → RRF fusion → event mask → LLM reranker. To isolate stages, run with `mode="embedding"` (stage 1 only) vs `mode="llm-local"` (all stages).
7. **Results saved to** `eval/results/{label}-e2e-seed{seed}.json`
8. **Phase 0/1 regenerate on every run.** Affinity inference always runs fresh. Expansions are cached at `eval/corpora/enriched_{label}/rules.json` — reused unless `--force-expand`.

The single authoritative Latest Baseline lives in the **Quality Benchmarks → Current Baseline** section above.

## When in Doubt

1. Read the PRD — `artifacts/prd-v1.md` is the source of truth
2. Check the Decisions table (Section 17)
3. If not covered, make a decision, document it, continue
