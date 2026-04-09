# cuecard

> The right rule, at the right moment.

Contextual rule enforcement for AI coding agents. Retrieves relevant user-defined rules via semantic similarity and injects them at every stage — before actions (PreToolUse), after actions (PostToolUse), on user messages (UserPromptSubmit), when spawning subagents (SubagentStart), and at turn end (Stop).

## Project Structure

```
cuecard/
├── CLAUDE.md               # This file
├── README.md               # Public-facing readme
├── .gitignore
├── artifacts/              # Design docs & session state
│   ├── prd-v1.md           # PRD v1.2 — core pipeline (converged)
│   ├── enriched-retrieval-prd.md     # Enriched retrieval design
│   ├── multi-stage-retrieval-prd.md  # Multi-stage PRD v1.1 (converged)
│   ├── phase4-closed-loop-hooks-prd.md # Closed-loop hooks PRD v1.2 (converged)
│   ├── paper-design-rule-rationale.md # Compliance paper design
│   ├── phase6-completion-gate-prd.md  # Completion gate Stop hook PRD v1.2 (converged)
│   ├── phase7-rule-quality-prd.md     # Trigger-aware rule rewriting PRD v1.0
│   └── session-25-progress.md         # Current session state
├── src/cuecard/            # Core library (agent-agnostic)
│   ├── __init__.py         # Public API: load_config, retrieve, format_rules
│   ├── loader.py           # Unified index loading with freshness + scope composition
│   ├── models.py           # Frozen dataclasses (Rule, Provenance, Index, etc.)
│   ├── config.py           # Load/merge/validate cuecard.toml configs
│   ├── security.py         # Path validation, secrets scrubbing, permissions
│   ├── affinity.py         # Event/tool affinity inference, storage, event mask
│   ├── parser.py           # Parse rule files → list[Rule] (dispatch by .txt/.json/.toml)
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
│   ├── cli.py              # Typer CLI (setup, config, configure, retrieve, rules, eval, serve, etc.)
│   ├── cli_rules.py        # Rules subcommands (add, remove, search, expand)
│   ├── cli_hooks.py        # Install/uninstall/status/log commands
│   ├── cli_eval.py         # Eval command
│   ├── serve.py            # Persistent daemon server (HTTP, PID management)
│   ├── py.typed            # PEP 561 marker
│   ├── retrievers/         # Pluggable retriever adapters
│   │   ├── __init__.py     # Retriever protocol, ScoredCandidate, fuse()
│   │   ├── dense.py        # DenseRetriever (cosine + parent collapse)
│   │   └── sparse.py       # SparseRetriever (BM25Okapi + parent collapse)
│   └── adapters/           # Agent-specific wrappers
│       ├── __init__.py
│       └── claude_code.py  # Claude Code PreToolUse hook
├── tools/
│   ├── bench_e2e.py        # E2E benchmark (default — model generates own expansions + reranks)
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
- pipeline.py orchestrates all stages; ALL retrieval paths (CLI, adapter, eval) route through `run_pipeline()` — even `embedding` mode uses the full dense+sparse+RRF stage
- Retriever adapter pattern: `Retriever` protocol in `retrievers/__init__.py`, pluggable dense/sparse/future
- RRF (Reciprocal Rank Fusion) merges results from multiple retrievers
- Parent-child collapse: rules have expansions (paraphrases), each embedded separately, max-score collapse via `rule_map`
- JSON intermediate: `rules.json` is the canonical format. Parser→JSON→Indexer. Expansions survive rebuilds via merge. Both `cuecard index` and `load_or_build()` use the same merge lifecycle.
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

### Affinity Design
- Rules have event/tool affinity — which hook events and tools they apply to
- Two modes: `infer` (LLM classifies at index time, default) and `strict` (explicit annotations only)
- **Minimum retrieval scope**: LLM classifies rules as `tool_use` (→ PreToolUse+PostToolUse), `workflow` (→ UserPromptSubmit+SubagentStart+Stop), or `both` (→ all 5 events). Prompt asks "when would the agent misbehave without this rule?" — tool_use if at tool time, workflow if at planning time, both only if agent can plan AROUND the tool event entirely. 93.5% accuracy. Ground truth: 60 tool_use, 44 workflow, 3 both.
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
- D5: RRF over learned fusion (simple, no training data needed), fusion_k=10 (was 60 — full-sample sweep showed k=10 wins on all metrics for 107-rule corpus)
- D6: Top-k=7, threshold=0.30 (was 5/0.35 — top_k=5 caused crowding drops on multi-rule events like git commit)
- D25: LLM receives `llm_candidates` rules (default 12) — configurable in TOML `[retrieval] llm_candidates = 12`
- D7a: Asymmetric encoding (query_embed vs passage_embed)
- D10: Full rebuild on change (no incremental splice corruption)
- D12: rule_map always present (identity for old indexes)
- D14: Inline BM25Okapi (~35 lines, no dependency)
- D17: Merge preserves expansions for unchanged rules
- S1: Path validation with allowlist (prevents traversal)

## Model Recommendations (updated session 20)

| Stage | Model | Why |
|-------|-------|-----|
| Embedding | jina-embeddings-v2-base-code | Best PreToolUse hard recall (60%), code-specific, 22ms |
| Embedding (workflow alt) | snowflake-arctic-embed-m | Best UserPromptSubmit hard recall (73.3%), 14ms |
| Cross-encoder (opt-in) | Xenova/ms-marco-MiniLM-L-6-v2 | 13ms, but regressions on code — skip |
| **LLM (GPU, default)** | **Gemma-4-E4B via llama-server** | **Best PosRecall (+39% vs 9B), best workflow F2=0.820, perfect workflow NegSil, 1204ms (8.2GB Q8_0)** |
| LLM (GPU, best basic F2) | Qwen3.5-9B via llama-server | Best basic F2=0.797, best basic NegSil=0.962, but lower PosRecall (5.3GB) |
| LLM (GPU, low noise) | Qwen3.5-35B-A3B via llama-server | Lowest noise (0.217), highest basic NegSil (0.926), MoE (21GB) |
| LLM (CPU/laptop) | Qwen3.5-4B via llama-server | 2.6GB, 917ms p50, F2=0.736, viable for laptop |
| LLM (API) | Haiku via claude-agent-sdk | Fast, Max subscription |

### E2E Model Comparison (each model generates own expansions + reranks, 20% sample, seed=42)

**IMPORTANT**: Always use `tools/bench_e2e.py` for model comparison. Each model must
generate its own expansions — shared-corpus benchmarks unfairly bias toward the
expansion-generator model. `bench_models.py` is deprecated.

| Model | Basic F2 | Basic PosRecall | Basic Noise | Basic NegSil | Workflow F2 | Workflow NegSil | p50ms | GGUF |
|-------|---------|----------------|-------------|--------------|------------|-----------------|-------|------|
| **Gemma-4-E4B** | 0.774 | **0.778** | 0.266 | 0.852 | **0.820** | **1.000** | **1204ms** | 8.2GB |
| Qwen3.5-9B | **0.797** | 0.560 | 0.259 | **0.962** | 0.725 | 0.750 | 1259ms | 5.3GB |
| Qwen3.5-35B | 0.785 | — | **0.217** | 0.926 | 0.680 | 0.750 | 1294ms | 21GB |
| Qwen3.5-4B | 0.736 | — | 0.243 | 0.852 | 0.577 | 0.750 | 917ms | 2.6GB |

**Key insights**:
- **Gemma E4B wins on recall and workflow.** 39% more relevant rules found (PosRecall 0.778
  vs 0.560), workflow F2 +9.5 pts, perfect workflow silence. Best all-rounder for production.
- **Qwen 9B wins basic F2 and NegSil.** Best for use cases where false positives are expensive.
- **E2E methodology matters.** Shared-corpus benchmarks showed Qwen 35B winning. E2E (each
  model generates own expansions) flips rankings — Gemma and 9B both beat 35B.

**Prompt tuning on both models (5 variants total) was reverted** — every variant overcorrected.
Gemma is hypersensitive to negative few-shot examples (even one crashes PosRecall by 30+ pts).
The prompt is at a local optimum for both models.

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

### Embedding Model Comparison (enriched, embedding mode, 354 basic fixtures)

| Model | Dim | Easy | Medium | Hard | Recall | Noise | p50ms |
|-------|-----|------|--------|------|--------|-------|-------|
| jina-code-v2 | 768 | 95.3% | 76.8% | 60.0% | 49.0% | 83.1% | 22ms |
| mxbai-embed-large | 1024 | 96.0% | 77.7% | 57.7% | 49.1% | 83.6% | 45ms |
| snowflake-arctic-m | 768 | 94.2% | 73.7% | 57.5% | 47.5% | 83.7% | 14ms |
| bge-small (default) | 384 | 94.2% | 74.1% | 51.1% | 46.8% | 84.0% | 4ms |

Key insight: model choice gives ~4% recall spread. The LLM reranker gives ~70% improvement on noise/silence. Embedding mode alone cannot achieve negative silence >0% (except jina-v3 at 6%, but 457ms).

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
- **Eval metrics:** noise ratio, context waste, per-tier breakdown, negative silence, quality (F2), positive recall — COMPLETE
- **Quality iteration:** 354 PreToolUse + 84 UserPromptSubmit fixtures — COMPLETE
- **Unified events:** UserPromptSubmit support, reasoning-in-response prompt — COMPLETE
- **Enriched retrieval:** JSON intermediate, expansion-aware indexing, parent collapse, BM25+RRF, expansion CLI — COMPLETE (826 tests, 100% coverage)
- **Benchmarking:** 6 embedding models, raw vs enriched vs LLM, v1 vs v3 expansion prompts — COMPLETE
- **Small model investigation:** Qwen3-0.6B not viable; Qwen3.5-4B/9B benchmarked — COMPLETE
- **LLM robustness:** Configurable stop sequences, JSON extraction, retry on parse failure — COMPLETE
- **Prompt engineering:** 16 few-shot examples (13 original + 3 category-aware negatives), category-aware guidance, principle-based guidelines — COMPLETE
- **Model benchmarking:** Qwen3.5-9B/4B/2B vs 35B on v3 corpora with sampling — COMPLETE
- **Eval infrastructure:** tqdm progress, stratified sampling, bench_models.py script — COMPLETE
- **E2E benchmarking:** `tools/bench_e2e.py` — each model generates its own expansions AND reranks. Default mode going forward. `bench_models.py` deprecated (shared-corpus comparisons are unfair). Corpora cached at `eval/corpora/enriched_{tier}_{label}/` per model.
- **Expansion prompt v5:** Reasoning-field prompt for expansions (structured CoT before generating) — COMPLETE
- **Eval corpus:** 109 rules in `eval/corpora/rules_global.txt` (unified global + basic + workflow). All fixture files reference `rules_global.txt`.
- **Eval dataset:** 1096+ fixtures across 5 event types (441 PreToolUse, 219 UserPromptSubmit, 131 PostToolUse, 135 Stop, 170 SubagentStart). All events balanced 44-49% positive. Cross-event consistency enforced (tool_use rules only in PreToolUse/PostToolUse fixtures). 5 verification rounds converged. +41 mined Stop fixtures in Phase 6 format (`eval/fixtures/stop_mined.json`).
- **Tagged corpus:** `eval/corpora/rules_global_tagged.json` — each rule tagged as tool_use/workflow/both. Ground truth uses minimum-retrieval-scope principle (session 26): 60 tool_use, 44 workflow, 3 both. Affinity prompt accuracy: 93.5%.
- **Fixture realism (session 25):** Stop fixtures use realistic `"Stop: end_turn"` queries. PostToolUse: 3 compliance-as-violation errors fixed, 5 output format fixes. SubagentStart: trimmed 2.6→1.5 rules/pos. All events rebalanced to 46-53% positive.
- **Phase 6 Completion Gate:** PRD converged (v1.2, 3 review rounds). Stop hook reads `last_assistant_message` + `transcript_path`, always blocks up to `max_stop_blocks` (Option A), shared `execute_stop_gate()` function. Pending implementation.
- **Phase 7 Rule Quality:** PRD written. Rewrite all 109 rules with trigger conditions ("When X: do Y — because Z"). 63/109 rewritten (PreToolUse subset). Remaining 46 workflow rules in progress.
- **CLI UX:** `cuecard configure` interactive setup, `cuecard serve` daemon, expand progress bar — COMPLETE
- **Hook format:** Correct `hookEventName` + `permissionDecision` for PreToolUse, `hook_event_name` input field detection — COMPLETE
- **Global install:** `uv tool install` support, `cuecard hook` CLI entry point — COMPLETE
- **Live validation:** Verified agent compliance in real Claude Code sessions — COMPLETE
- **Closed-loop hooks (Phase 4):** TOML rule format, event/tool affinity inference, event mask retrieval, 5 hook adapters (PreToolUse/PostToolUse/UserPromptSubmit/SubagentStart/Stop), per-event eval metrics — COMPLETE (1148 tests, 100% coverage)
  - Phase 1: TOML parser + Rule.events/tools + migrate CLI + KNOWN_HOOK_EVENTS
  - Phase 2: Affinity inference + storage (AffinityIndex O(1) lookup, LLM fallback to strict)
  - Phase 3: Event mask in retrieval + LoadedIndex return type + numpy advanced indexing
  - Phase 4a: Multi-event adapter with scrub_secrets on PostToolUse/SubagentStart/Stop
  - Phase 4b: Hook registration for all 5 events
  - Phase 5: Eval corpus migration + new fixtures (PostToolUse/Stop/SubagentStart)
  - Phase 6: Benchmark with Gemma — pending
- **Publish:** pending
- **Multi-source parsing (markdown, YAML, CLAUDE.md):** DRAFT PRD (`artifacts/phase5-multi-source-prd-draft.md`)

## Event Types

cuecard supports five Claude Code hook events (closed-loop enforcement):

- **PreToolUse**: Before each tool call. PREVENT — inject action constraints.
- **PostToolUse**: After each tool call. VERIFY — check compliance against output.
- **UserPromptSubmit**: When user sends a message. GUIDE — process/methodology rules.
- **SubagentStart**: When a subagent spawns. PROPAGATE — rules for delegated work.
- **Stop**: When a turn ends. AUDIT — verify rules were followed.

All event types use the same pipeline. The adapter dispatches to per-event handlers that construct queries:
- PreToolUse: `"{tool_name}: {tool_input[:500]}"`
- PostToolUse: `"PostToolUse:{tool_name}: {tool_input[:200]} → {scrub_secrets(tool_output[:500])}"`
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

PostToolUse, SubagentStart, and Stop all apply `scrub_secrets()` before query construction.

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

For llm-local mode (best quality):
```bash
# Recommended: Qwen3.5-9B (matches 35B on basic, 4x smaller)
llama-server -m ~/models/Qwen3.5-9B-Q4_K_M.gguf \
  --port 8081 -ngl 99 -c 16384 --jinja

# Alternative: Qwen3.5-35B-A3B (best workflow recall, needs 32GB GPU)
llama-server -m ~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --port 8081 -ngl 99 -c 16384 --jinja

# CPU/laptop: Qwen3.5-4B (2.6GB, 856ms p50)
llama-server -m ~/models/Qwen3.5-4B-Q4_K_M.gguf \
  --port 8081 -ngl 0 -c 16384 --jinja
```

Key requirements:
- `--jinja` flag — required for Gemma's chat template
- `--reasoning off` — disables thinking server-side, eliminates `chat_template_kwargs` overhead
- `-np 5` — 5 parallel slots (19660 tokens/slot with `-c 98304`)
- `-c 98304` — total context (~20GB VRAM with model)
- `_MAX_TOKENS=1024` — reasoning-in-response needs room for 3-5 sentence analysis
- Reasoning captured in `LLMParseResult.reasoning` for debugging
- Reranker passes `stop=None` to `call_local()` — no stop sequence for JSON responses
- Expander keeps default `stop=("\n\n",)` to prevent repetition degeneration

### Prompt Engineering
- System prompt has 13 few-shot examples (7 original + 6 from loss analysis)
- Guidelines use reasoning principles ("does this change code/state?") not hardcoded command lists
- "When in doubt, include" — false negatives worse than false positives
- Loss analysis: `artifacts/llm-reranker-loss-analysis-2026-04-02.md`

### Expansion Prompt
- Reasoning-field prompt: model reasons about vocabulary gap before generating expansions
- Variable 3-10 expansions based on rule complexity
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

### With v5 Expansions (reasoning-field prompt, 20% sample, seed=42, jina-code + LLM, 2026-04-04)

| Model | Basic F2 | Basic PosRecall | Basic Noise | Basic NegSil | p50ms | p95ms |
|-------|---------|-----------------|-------------|--------------|-------|-------|
| **35B** | **0.808** | 0.808 | **0.205** | **0.889** | 1274ms | **1718ms** |
| 9B | 0.762 | **0.832** | 0.316 | 0.815 | 1350ms | 3440ms |

| Model | Workflow F2 | Workflow PosRecall | Workflow Noise | Workflow NegSil |
|-------|-------------|-------------------|----------------|-----------------|
| **35B** | **0.632** | **0.621** | **0.406** | 0.750 |
| 9B | 0.604 | 0.576 | 0.439 | 0.750 |

**Per-tier basic (seed=42):**
| Tier | 9B F2 | 35B F2 | Gap |
|------|------:|-------:|----:|
| Easy | 0.880 | 0.888 | tie |
| Medium | 0.707 | **0.817** | 35B +11 |
| Hard | **0.581** | 0.438 | **9B +14** |
| Negative | 0.815 | **0.889** | 35B +7 |

**Key findings (apples-to-apples, same corpus + seed)**:
- Real gap is ~5 pts F2 on basic, ~3 pts on workflow — much smaller than older cached numbers suggested
- **The gap is noise, not recall.** 9B actually beats 35B on PosRecall; 35B wins via discrimination
- **9B wins hard-tier (+14 pts)**, 35B wins medium-tier (+11 pts). 35B is conservative; that helps medium, hurts hard
- 9B p95 latency is 2x worse (tail risk). 35B more predictable

Cross-encoder (MiniLM) is a regression on code — skip it. Use llm-local or llm-haiku.

**Historical (stale — different corpus, pre-v5 expansions)**: 35B basic 0.750, 9B basic 0.682, etc. See session 19 notes.

### Expansion Validation (hand-crafted, 8 hard fixtures)

| Metric | Raw | With Expansions | Delta |
|--------|-----|-----------------|-------|
| Avg cosine similarity | 0.187 | 0.558 | +0.371 |
| Threshold crossings (N→Y) | — | 10/12 pairs | — |

LLM-generated expansions (v2 prompt): 8/12 threshold crossings, avg delta +0.276.

### With 109-Rule Corpus (session 24, 40% sample, seed=42, Gemma E4B, jina-code + LLM + event mask)

**Corpus:** `eval/corpora/rules_global.txt` (109 rules — global + basic + workflow unified)
**Fixtures:** 831 total (486 pos, 345 neg), verified in 4 rounds
**Pipeline:** dense + sparse + RRF (top_k=12, threshold=0.25) → LLM reranker (16 examples, category-aware) → event mask (binary affinity)

| Event | F2 | PosRecall | Noise | NegSil | p50ms |
|-------|------|-----------|-------|--------|-------|
| PreToolUse | 0.566 | 0.319 | 0.408 | 0.523 | 1415 |
| UserPromptSubmit | 0.618 | 0.444 | 0.384 | 0.833 | 1402 |
| PostToolUse | 0.528 | 0.415 | 0.493 | 0.316 | 1472 |
| Stop | 0.426 | 0.414 | 0.595 | 0.100 | 1457 |
| SubagentStart | 0.248 | 0.296 | 0.814 | 0.000 | 1522 |

**Key findings (109 rules vs 32 rules):**
- 3.4x corpus expansion caused ~30% F2 drop — expected, more rules = harder discrimination
- Event mask helps PreToolUse +3% F2 and PostToolUse +3% F2 (noise reduction)
- Prompt tuning (category awareness, 3 new negative examples) recovered +5-8% on PreToolUse/Workflow
- Compliance-as-violation was the #1 fixture error — 25 false positives removed in R3
- Binary affinity classification (tool_use/workflow with golden examples) >> 5-event classification

**Reranker prompt changes (session 24):**
- Replaced "when in doubt, include" with category-aware guidance (concrete=include, process=exclude on tool events)
- Added 3 negative examples: process-rules-excluded-from-pytest, LLM-rules-excluded-from-edit, high-candidate-discrimination-on-git-diff
- Added RULE CATEGORIES section mapping rule types to event types
- Reduced LLM candidate input from top_k=20/threshold=0.20 to top_k=12/threshold=0.25

### Quality Gap Analysis (session 25)

**Affinity accuracy:** 86.2% (after ground truth correction — 20 rules re-tagged workflow→both). Remaining 15 mismatches are genuinely borderline. Affinity is NOT the bottleneck.

**Affinity mask impact:** Applied post-scoring (`scores[~mask] = -inf`) in both dense and sparse retrievers. PreToolUse gets 78/109 rules (28% reduction). Workflow events get ALL 109 rules (no reduction). The binary classification helps PreToolUse but doesn't help workflow events.

**Pipeline bottleneck (verified — session 26 top_k=30 diagnostic):**
- top_k=5 vs top_k=30: only +0.8% recall improvement
- **Bottleneck is embedding/expansion quality, NOT the LLM reranker**
- Correct rules never reach the reranker — they're not in the embedding neighborhood
- Path forward: better expansions, richer queries (Phase 6), or better embedding model

**Root cause: rule text is the only retrieval signal.** Rules that describe WHAT but not WHEN match every topically-similar query. "Run convergence reviews" matches everything about reviews/quality. "After completing an implementation phase: run convergence reviews" matches specifically.

**Planned fix (Phase 7):** Rewrite all 109 rules with trigger conditions ("When X: do Y — because Z"). Adds tool names and action context as vocabulary for embeddings. Target: PreToolUse F2 ≥0.666 (+10 pts). 63/109 rules rewritten (PreToolUse subset), remaining 46 in progress.

**Fixture realism (session 25):** 831→984 fixtures. Stop fixtures use realistic `"Stop: end_turn"`. PostToolUse: 3 compliance-as-violation fixes. SubagentStart: trimmed over-specification. All events rebalanced to 46-53% positive. +41 mined Stop fixtures in Phase 6 format.

## When in Doubt

1. Read the PRD — `artifacts/prd-v1.md` is the source of truth
2. Check the Decisions table (Section 17)
3. If not covered, make a decision, document it, continue
