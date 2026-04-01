# cuecard PRD v1.2

> The right rule, at the right moment.

**Review history:** v1.0 → Round 1 (3 reviews). v1.1 → Round 2 (3 reviews). v1.2 addresses Round 2 findings (1 HIGH, 5 MEDIUM).

## 1. Objective

Build an agent-agnostic Python library that retrieves the most relevant user-defined rules and injects them into AI coding agent tool calls via semantic similarity. cuecard ensures agents remember what you told them — without dumping everything into context.

### Success Criteria

- [ ] `cuecard setup` downloads embedding model and builds initial index
- [ ] `cuecard retrieve "Bash: git commit -m 'fix'"` returns relevant rules with scores and provenance
- [ ] Claude Code PreToolUse hook injects retrieved rules into `additionalContext`
- [ ] Index auto-refreshes when rule files change (full rebuild on any change)
- [ ] Retrieval latency < 500ms per tool call (including cold start)
- [ ] CLI exposes every pipeline step independently for inspection
- [ ] Jupyter notebook visualizes each step's output for experimentation
- [ ] Golden test fixtures enable offline model comparison
- [ ] 100% test coverage on all code
- [ ] Installable via `uv pip install cuecard`
- [ ] Path validation on all config source paths (no traversal)
- [ ] Secrets scrubbing on all logged tool inputs
- [ ] File permissions 0o700/0o600 on all cuecard directories/files

## 2. Non-Goals (v0.1)

- Markdown file chunking and summarization (v0.2)
- CDT behavioral prediction / delulu integration (v0.3+)
- Daemon mode for persistent model loading (v0.2)
- Structured tool-specific query extraction (v0.2)
- CI-integrated evaluation (v0.2)
- LLM-assisted annotation of golden dataset (v0.2)
- Incremental index updates (v0.3 — full rebuild is fast enough for <500 rules)

## 3. Architecture

### 3.1 Core Library (Agent-Agnostic)

```
┌─────────────────────────────────────────────┐
│              cuecard (core library)          │
│                                             │
│  config → parser → indexer → retriever      │
│           → formatter → logger              │
│  (Python package, no agent dependencies)    │
└──────────┬──────────┬──────────┬────────────┘
           │          │          │
     ┌─────┴──┐  ┌────┴───┐  ┌──┴──────┐
     │ Claude │  │ Codex  │  │ Gemini  │
     │ Code   │  │ CLI    │  │ CLI     │
     │ adapter│  │ adapter│  │ adapter │
     └────────┘  └────────┘  └─────────┘
```

The core library exposes three functions:

```python
import cuecard

index = cuecard.load_or_build(reindex: bool = True)  # handles config, freshness, rebuild
results = cuecard.retrieve(index, query)               # asymmetric query embed + dedup
context = cuecard.format_rules(results)                # plain text for injection
```

**`load_or_build(reindex=True)`**: When `reindex=True` (default), checks freshness and triggers full rebuild if any source file changed. Worst-case latency: ~1-2s (model load + embed 500 rules). Typical latency with no changes: <10ms (mtime check only). When `reindex=False`, skips freshness checks entirely and loads the cached index as-is — useful for latency-sensitive paths that defer freshness to a background `cuecard index` call.

**Note:** fastembed's `passage_embed()` and `query_embed()` return generators, not arrays. The implementation must materialize them: `np.array(list(model.passage_embed(texts)))`.

Adapters live inside the package at `src/cuecard/adapters/` and are thin wrappers (~15 lines) that handle agent-specific I/O protocols.

### 3.2 Pipeline Steps

```
Source Files              cuecard.toml / ~/.cuecard/config.toml
    ↓                              ↓
┌──────────┐
│ 1. CONFIG│  Load + merge global and project configs
└────┬─────┘  Output: ResolvedConfig
     ↓
┌──────────┐
│ 2. PARSE │  Read rule files, dispatch by suffix (.txt/.md)
└────┬─────┘  Output: list[Rule] with provenance
     ↓
┌──────────────────┐
│ 3. SUMMARIZE     │  (v0.2) Distill long chunks for embedding
└────┬─────────────┘  v0.1: pass-through (rules are already concise)
     ↓
┌──────────┐
│ 4. EMBED │  Encode rule text via passage_embed() → dense vectors
└────┬─────┘  Output: numpy array (N, dim), L2-normalized
     ↓
┌──────────┐
│ 5. INDEX │  Persist embeddings + metadata to disk
└────┬─────┘  Output: .npz + metadata.json in cache dir
     ↓
┌──────────────┐
│ 6. FRESHNESS │  Check mtime → content hash → full rebuild if changed
└────┬─────────┘  Output: up-to-date index
     ↓
┌────────────────┐
│ 7. MODEL LOAD  │  Load embedding model (cold start ~300ms, cached after)
└────┬───────────┘  Implicit in EMBED and RETRIEVE, explicit in docs
     ↓
┌────────────┐
│ 8. RETRIEVE│  query_embed() → dot product → top-k + threshold → dedup
└────┬───────┘  Output: list[RankedResult]
     ↓
┌──────────┐
│ 9. FORMAT │  Assemble results into injectable plain text
└──────────┘  Output: string for additionalContext
```

Each step is independently callable via CLI and inspectable in the notebook.

### 3.3 Multi-Stage Retrieval Pipeline

The retrieval pipeline has up to 3 stages. Each stage is independently valuable — any subsequent stage can be disabled without breaking the system.

```
All rules (N)
     ↓
Stage 1: EMBEDDING RETRIEVAL (fast, high recall)
  query_embed → dot product → top-20 candidates
  Deterministic, CPU, <5ms, always runs
     ↓
Stage 2: CROSS-ENCODER RE-RANKING (lightweight, high precision)
  Cross-encoder scores each (query, rule) pair
  Deterministic, CPU, ONNX, <100ms for 20 items
  Optional — improves precision without LLM cost
     ↓
Stage 3: LLM RE-RANKING (semantic understanding)
  LLM reads candidates + tool context, picks relevant ones
  Non-deterministic, GPU or API
  Optional — highest quality, highest cost
     ↓
Final: top-k results → format → inject
```

**Cost pyramid:** embeddings (free, 5ms) → cross-encoder (free, 50ms) → local LLM (free, 500ms) → API LLM ($0.001, 300ms).

**Quality target:** 90%+ recall at the end of the pipeline. Each stage should be independently tuned to maximize its quality before relying on the next.

**Provenance:** Every result at every stage logs which rules survived and which were dropped, enabling per-stage quality analysis in the notebook.

**Configuration:**
```toml
[retrieval]
mode = "embedding"          # "embedding" | "rerank" | "rerank-llm-local" | "rerank-llm-haiku"
top_k = 5                   # final output
recall_top_k = 20           # stage 1 candidates for re-ranking
recall_threshold = 0.2      # looser threshold for stage 1

[reranker]
model = "..."               # cross-encoder model (TBD based on research)

[llm]
local_endpoint = "http://localhost:8081/v1"
local_ctx_size = 8192
haiku_model = "claude-haiku-4-5-20251001"
thinking = false
```

### 3.4 Asymmetric Encoding (CRITICAL)

Many embedding models (BGE, Nomic, Snowflake) use different prefixes for documents vs queries to optimize retrieval:

- **Documents (rules):** encoded via `passage_embed()` — no prefix, or model-specific document prefix
- **Queries (tool context):** encoded via `query_embed()` — model-specific query prefix (e.g., BGE uses `"Represent this sentence: "`, Nomic uses `"search_query: "`)

fastembed handles this automatically via its `query_embed()` and `passage_embed()` methods. **The implementation MUST use these separate methods, never a single generic `embed()` for both.** Using the wrong method produces systematically lower similarity scores and degraded retrieval quality.

**At index time:** `model.passage_embed(rule_texts)` for all rules
**At query time:** `model.query_embed([query_text])` for the tool context

### 3.4 Data Model

```python
@dataclass(frozen=True)
class Provenance:
    file: str                          # resolved absolute path
    line_start: int                    # 1-indexed
    line_end: int                      # inclusive
    section_path: tuple[str, ...] = () # for markdown chunks (v0.2)
    chunk_type: str = "rule"           # "rule" | "paragraph" | "list_item" | "code_block"

@dataclass(frozen=True)
class Rule:
    text: str                          # max 500 chars enforced at parse time
    provenance: Provenance
    summary: str | None = None         # for long chunks (v0.2), None means use text

@dataclass(frozen=True)
class RankedResult:
    rule: Rule
    score: float                       # dot product (L2-normalized = cosine similarity)

@dataclass(frozen=True)
class SourceMeta:
    mtime: float
    content_hash: str                  # "sha256:..."
    rule_count: int

@dataclass(frozen=True)
class Index:
    embeddings: numpy.ndarray          # (N, dim), L2-normalized
    rules: tuple[Rule, ...]            # parallel to embedding rows
    model_name: str
    dim: int
    sources: dict[str, SourceMeta]     # resolved path → metadata
```

### 3.5 Index Structure

Persisted to disk as two files:

**`embeddings.npz`** — numpy compressed array of shape `(N, dim)`, L2-normalized.

**`metadata.json`**:
```json
{
    "version": 1,
    "model": "BAAI/bge-small-en-v1.5",
    "dim": 384,
    "checksum": "sha256:...",
    "created": "2026-04-01T12:00:00Z",
    "sources": {
        "/home/user/rules.txt": {
            "mtime": 1711929600.0,
            "content_hash": "sha256:abc123...",
            "rule_count": 4
        }
    },
    "rules": [
        {"text": "Never commit secrets to git", "file": "/home/user/rules.txt", "line_start": 1, "line_end": 1}
    ]
}
```

**Integrity check:** On load, verify `checksum` matches the sha256 of `embeddings.npz` and `len(rules)` matches `embeddings.shape[0]`. On mismatch, log warning and trigger full rebuild.

**No positional `chunk_indices`.** Rules are stored as a flat parallel array — rule[i] corresponds to embedding[i]. On any file change, the entire index is rebuilt from scratch (Section 5). This eliminates the splice corruption class of bugs.

## 4. Config

### 4.1 Locations

- **Global:** `~/.cuecard/config.toml` — rules that apply everywhere
- **Project:** `./cuecard.toml` — project-specific rules and overrides

### 4.2 Config Resolution (Merge Semantics)

When both global and project configs exist, they are merged as follows:

**Scalar values** (top_k, threshold, verbose, etc.): project overrides global. Unspecified project keys inherit from global. Unspecified global keys use built-in defaults.

**`sources.rules`**: union by resolved absolute path. Duplicates (same resolved path) are deduplicated.

**`embedding.model`**: project wins. If project specifies a different model than global, the global index is re-embedded with the project model. Both indexes MUST use the same model and dimension — validated at merge time. Mismatched model/dim raises `ConfigError`.

**`logging.log_file`**: project cannot override (always `~/.cuecard/log.jsonl`). Prevents writing logs to arbitrary paths.

**`logging.redact`**: project cannot override (global-only, default true). Prevents a project config from silently disabling secrets scrubbing system-wide.

**Worked example:**
```toml
# Global: ~/.cuecard/config.toml
[sources]
rules = ["~/.cuecard/rules/global.txt"]
[retrieval]
top_k = 5
threshold = 0.35
[embedding]
model = "BAAI/bge-small-en-v1.5"

# Project: ./cuecard.toml
[sources]
rules = ["rules.txt"]
[retrieval]
top_k = 10

# Resolved:
# sources.rules = ["~/.cuecard/rules/global.txt", "./rules.txt"]  (union)
# retrieval.top_k = 10  (project overrides)
# retrieval.threshold = 0.35  (inherited from global)
# embedding.model = "BAAI/bge-small-en-v1.5"  (inherited from global)
```

### 4.3 Path Resolution

Source paths are resolved in this order:
1. **Tilde expansion:** `~` → user home directory
2. **Relative resolution:** relative paths resolved against the config file's parent directory
3. **Glob expansion:** `*.txt`, `**/*.md` patterns expanded
4. **Symlinks:** resolved to real target (via `Path.resolve()`). Real target must pass allow-list check
5. **Zero matches:** warning logged, not an error (files may not exist yet)

### 4.4 Path Validation (Security)

All resolved source paths are validated:
- Canonicalized via `Path.resolve()` — removes `..` and resolves symlinks to real targets
- Paths containing `..` in the raw config are rejected before resolution (defense in depth)
- For project config: paths must resolve under the **directory containing `cuecard.toml`** (not `cwd`) OR under explicitly declared `allowed_dirs`
- For global config: paths must resolve under `~/.cuecard/` OR under explicitly declared `allowed_dirs`
- Symlinks are resolved to their real target by `Path.resolve()` — the real target must be within allowed dirs
- Violation raises `ConfigError` with a clear message

```toml
[sources]
rules = ["rules.txt"]
# allowed_dirs = ["~/.claude/rules/"]  # explicitly whitelist external dirs
```

### 4.5 Config Validation

All config values are validated against constraints after parsing:

| Field | Type | Range | Default |
|-------|------|-------|---------|
| `retrieval.top_k` | int | 1–50 | 5 |
| `retrieval.threshold` | float | 0.0–1.0 | 0.35 |
| `retrieval.dedup_threshold` | float | 0.0–1.0 | 0.95 |
| `hooks.query_max_length` | int | 50–2000 | 500 |
| `logging.max_log_size_mb` | int | 1–1000 | 10 |
| `logging.verbose` | bool | — | false |
| `logging.redact` | bool | — | true (global-only) |
| `sources.allowed_dirs` | list[str] | valid paths | [] |

Invalid values raise `ConfigError`.

### 4.6 Format

```toml
# ~/.cuecard/config.toml or ./cuecard.toml

[sources]
rules = [
    "rules.txt",                          # one rule per line
    # "~/.claude/rules/common/*.md",      # markdown files (v0.2)
]
# allowed_dirs = []                       # whitelist dirs outside project/home

[embedding]
model = "BAAI/bge-small-en-v1.5"          # fastembed model name

[retrieval]
top_k = 5                                 # max rules to inject (1-50)
threshold = 0.35                          # minimum cosine similarity (0.0-1.0)
dedup_threshold = 0.95                    # semantic dedup on results (0.0-1.0)

[hooks]
events = ["PreToolUse"]                   # which hook events to fire on
query_max_length = 500                    # truncate tool input to this length

[logging]
verbose = false                           # print to stderr
# log_file is always ~/.cuecard/log.jsonl (not configurable for security)
max_log_size_mb = 10                      # rotate at this size (1-1000)

[index]
# Cache locations are fixed (not user-configurable) for security:
#   global: ~/.cuecard/index/
#   project: .cuecard/index/
# This prevents index injection via attacker-controlled cache paths.
```

### 4.7 Rules File Format (v0.1)

Plain text, one rule per line. Empty lines and `#` comments are ignored. Max 500 characters per rule — longer lines are truncated with a warning.

```
# rules.txt — my coding rules

Never commit secrets (API keys, tokens, passwords) to git
Use uv for all Python package operations, never pip
Send Enter after every tmux send-keys command
Always validate user input at system boundaries
Run quality checks before every commit
Use frozen dataclasses for immutable data
100% test coverage on all new code
```

## 5. Index Freshness Protocol

On every `load_or_build()` call:

```
1. Load existing index from disk (if exists)
2. Validate index integrity: checksum matches, rule count matches embedding rows
   - On mismatch: log warning, trigger full rebuild (step 5)
3. For each source file in resolved config:
   a. os.stat() → get current mtime
   b. If mtime matches stored mtime → skip (fast path, ~0.1ms)
   c. If mtime differs → read file, compute SHA-256 hash
   d. If hash matches stored hash → update stored mtime in memory, continue
   e. If hash differs → mark index as stale
4. Check for removed sources: files in index but NOT in current config → mark stale
5. If stale OR no index exists:
   a. Re-parse ALL source files → list[Rule]
   b. Embed ALL rules via passage_embed() → numpy array
   c. Write new embeddings.npz + metadata.json atomically (write to temp, rename)
   d. Acquire file lock (fcntl.flock) during write to prevent concurrent corruption
6. Update mtime cache in memory
7. Return loaded Index
```

**Key design choice:** Any file change triggers a FULL rebuild, not incremental splicing. For <500 rules, full rebuild takes <1s. This eliminates positional index corruption bugs entirely. Incremental updates deferred to v0.3.

**Atomic writes:** Use `NamedTemporaryFile` + `os.replace()` for both files. If the process crashes mid-write, the old index remains valid.

**File locking:** `fcntl.flock()` on a `.cuecard/index/.lock` file during writes. Reads do not acquire locks — a reader that detects inconsistency (integrity check fails) triggers a rebuild.

## 6. Retrieval Algorithm

```
Input: query string (e.g., "Bash: git commit -m 'fix bug'")
Config: top_k, threshold, dedup_threshold, model

1. Load model (cached after first load in process)
2. Encode query via model.query_embed([query_text]) → query_vec (1, dim)
   NOTE: query_embed() applies the model-specific query prefix automatically
3. Load global index + project index (if exists)
4. Validate model name + dim match between global and project index
   - On mismatch: raise ConfigError (user must align models)
5. Concatenate embeddings + rules from both indexes
   - Exact-text dedup at merge time (set-based, O(N))
6. Compute dot product: scores = combined_embeddings @ query_vec.T → (N,)
   NOTE: dot product = cosine similarity because vectors are L2-normalized
7. Filter: keep only scores >= threshold (default 0.35)
8. Sort by score descending
9. Semantic dedup: for each pair in candidates, if dot(rule_i_emb, rule_j_emb) > dedup_threshold, drop the lower-scored one
10. Take top_k results
11. Return list[RankedResult] with provenance
```

## 7. Injection Format

The formatted output injected into `additionalContext`:

```
[cuecard — user-defined guidelines relevant to this action]
- Never commit secrets (API keys, tokens, passwords) to git
- Run quality checks before every commit
- Use frozen dataclasses for immutable data
```

The framing label `[cuecard — user-defined guidelines...]` signals to the model that these are user-authored constraints, not system instructions. This provides structural isolation from potential prompt injection in rule content.

No scores, no provenance in the injection — clean and directive. Full provenance available in CLI output, log.jsonl, and notebook.

**Rule length limit:** Each rule is capped at 500 characters at parse time. This limits the blast radius of any single rule as an injection vector.

## 8. Security

### 8.1 Trust Model

cuecard injects user-authored text into an AI agent's context. The trust boundary is: **rule files must come from trusted sources.** A malicious rule file can influence agent behavior.

Mitigations:
- **Path validation** (Section 4.4): prevents config from referencing arbitrary files
- **Length limits** (500 chars/rule): limits injection payload size
- **Labeled injection** (Section 7): framed as "user-defined guidelines" not "system instructions"
- **File permissions** (Section 8.4): prevents unauthorized modification of rule files

### 8.2 Secrets Scrubbing in Logs

The `query` field in log entries may contain sensitive data from tool inputs (e.g., `.env` file contents in a Write call). Before logging:

1. Truncate query to `query_max_length` (default 500 chars)
2. Apply secrets scrubbing: regex-based detection and redaction of:
   - API keys (`sk-live-...`, `AKIA...`, `ghp_...`, etc.)
   - Connection strings (`postgresql://user:pass@...`)
   - Bearer tokens (`Bearer eyJ...`)
   - Generic `KEY=value` patterns where KEY contains `SECRET`, `TOKEN`, `PASSWORD`, `API_KEY`
3. Redacted values replaced with `[REDACTED]`
4. Scrubbing applied to both `query` field and `results[].text` field

Users who need full unredacted logs can opt in via `logging.redact = false` in config (default: true).

### 8.3 Config Security

- `logging.log_file` is NOT configurable — always `~/.cuecard/log.jsonl`. Prevents log writes to arbitrary paths.
- All config values validated against type and range constraints (Section 4.5).
- Source paths validated and canonicalized (Section 4.4).

### 8.4 File Permissions

All cuecard directories and files are created with restrictive permissions:

| Path | Permission | Rationale |
|------|-----------|-----------|
| `~/.cuecard/` | 0o700 | Contains rules, index, logs |
| `~/.cuecard/log.jsonl` | 0o600 | May contain tool input fragments |
| `~/.cuecard/index/` | 0o700 | Contains embedded rules |
| `.cuecard/` | 0o700 | Project-local index |
| All files under above | 0o600 | Owner read/write only |

Permissions are set at creation time via `os.open()` + `os.fdopen()`, not post-hoc `chmod`, to avoid TOCTOU race.

`cuecard setup` verifies permissions and warns if existing files are too permissive.

### 8.5 Model Download

- `cuecard setup` downloads from HuggingFace and requires network access — documented in README
- v0.1: `--model` flag restricted to a known-good allowlist of models in the fastembed catalog
- `--allow-custom-model` flag required for models outside the allowlist
- fastembed's built-in hash verification applies to its catalog models

## 9. Observability

### 9.1 Structured Log

Every retrieval appends to `~/.cuecard/log.jsonl` (after secrets scrubbing):

```json
{
    "timestamp": "2026-04-01T12:34:56.789Z",
    "event": "PreToolUse",
    "tool_name": "Bash",
    "query": "Bash: git commit -m [REDACTED]",
    "query_truncated": false,
    "results": [
        {"text": "Never commit secrets...", "score": 0.87, "file": "rules.txt", "line": 1},
        {"text": "Run quality checks...", "score": 0.72, "file": "rules.txt", "line": 5}
    ],
    "injected_count": 2,
    "total_rules": 50,
    "index_rebuilt": false,
    "latency_ms": 42,
    "model": "BAAI/bge-small-en-v1.5"
}
```

### 9.2 Stderr (Verbose Mode)

When `verbose = true`:
```
[cuecard] PreToolUse:Bash → 2 rules injected (42ms)
```

### 9.3 Log Rotation

When `log.jsonl` exceeds `max_log_size_mb`, rename to `log.jsonl.1` and start fresh. Keep at most 2 rotated files.

## 10. CLI

```bash
# Setup
cuecard setup                          # Download model, build initial index, verify permissions
cuecard setup --model nomic-ai/nomic-embed-text-v1.5  # Use specific model
cuecard setup --check                  # Verify setup (permissions, model, index integrity)

# Rule management (txt files only — md files are user-managed)
cuecard rules                          # List ALL rules (txt + md) with numbers + source
cuecard rules --global                 # List only global rules
cuecard rules --project                # List only project rules
cuecard rules add "Send Enter after tmux send-keys"    # Add to project rules.txt
cuecard rules add --global "Never commit secrets"       # Add to global rules.txt
cuecard rules remove 3                 # Remove rule #3 (txt only; refuses for md-sourced rules)
cuecard rules search "tmux"            # Search across all rules
cuecard rules sources                  # Show rule file paths + status

# Inspect pipeline steps
cuecard config                         # Show resolved config (global + project merged)
cuecard parse                          # Show all parsed rules with provenance
cuecard embed                          # Show embedding stats (model, dim, count, size)
cuecard index                          # Rebuild index from scratch
cuecard index --status                 # Show index status (freshness, file hashes, integrity)

# Retrieval
cuecard retrieve "Bash: git commit"    # Retrieve top-k rules with scores + provenance
cuecard retrieve "Bash: git commit" --top-k 10 --threshold 0.1  # Override defaults
cuecard format "Bash: git commit"      # Show final injectable text

# Evaluation
cuecard eval fixtures.json             # Run evaluation on golden test fixtures
cuecard eval fixtures.json --model BAAI/bge-small-en-v1.5  # Compare models

# Logging
cuecard log                            # Show recent log entries
cuecard log --stats                    # Aggregate stats (top rules, coverage, latency p50/p95/p99)
```

**Rule management scope:** `add` and `remove` only operate on `.txt` rule files. Markdown files are indexed automatically but managed by the user in their editor. Attempting to `remove` a rule sourced from a `.md` file prints an error with the file path and line number for manual editing.

## 11. Adapters

### 11.1 Claude Code Adapter

Lives at `src/cuecard/adapters/claude_code.py` (inside the package, installable via `python -m cuecard.adapters.claude_code`).

**Plugin structure** (for Claude Code plugin installation):
```
src/cuecard/adapters/claude_code/
├── .claude-plugin/
│   └── plugin.json
└── hooks/
    ├── hooks.json
    └── pretooluse.py  → imports from cuecard core
```

**Hook registration** (`hooks.json`):
```json
{
    "hooks": [
        {
            "event": "PreToolUse",
            "hooks": [
                {
                    "type": "command",
                    "command": "python -m cuecard.adapters.claude_code"
                }
            ]
        }
    ]
}
```

**Hook implementation** (~20 lines):
```python
#!/usr/bin/env python3
"""Claude Code PreToolUse hook for cuecard."""
import json
import sys

from cuecard import load_or_build, retrieve, format_rules
from cuecard.logger import log_retrieval

def main():
    data = json.loads(sys.stdin.read())
    tool_name = data.get("tool_name", "")
    tool_input = str(data.get("tool_input", ""))[:500]
    query = f"{tool_name}: {tool_input}"

    try:
        index = load_or_build()
        results = retrieve(index, query)
        if results:
            context = format_rules(results)
            data.setdefault("hookSpecificOutput", {})
            data["hookSpecificOutput"]["additionalContext"] = context
        log_retrieval(tool_name, query, results)
    except Exception as e:
        print(f"[cuecard] Error: {e}", file=sys.stderr)

    print(json.dumps(data))

if __name__ == "__main__":
    main()
```

### 11.2 Other Adapters (v0.2+)

Codex and Gemini CLI adapters follow the same pattern — read their hook format, call core library, write their output format.

## 12. Evaluation Framework

### 12.1 Golden Test Fixtures

```json
[
    {
        "id": "git-commit-secrets",
        "query": "Bash: git commit -m 'fix auth bug'",
        "corpus": "rules_basic.txt",
        "should_match": ["Never commit secrets (API keys, tokens, passwords) to git"],
        "should_not_match": ["Send Enter after every tmux send-keys command"],
        "difficulty": "medium"
    }
]
```

Difficulty tiers: `easy` (keyword overlap), `medium` (semantic gap), `hard` (deep reasoning), `negative` (no rules should match).

### 12.2 Evaluation Metrics

**Recall metrics** (did we find the right rules?):
- **Recall@k**: of the `should_match` rules, how many are in the top k?
- **MRR** (Mean Reciprocal Rank): is the most relevant rule ranked first?
- **nDCG@k**: relevance-weighted ranking quality

**Precision metrics** (are we injecting irrelevant rules?):
- **Precision@k**: of the k rules returned, how many are in `should_match`?
- **Anti-precision**: rules in `should_not_match` that appeared (should be 0)
- **Noise ratio**: `irrelevant_count / retrieved_count` — what fraction of injected rules are noise?

**Context efficiency metrics** (what's the cost of false positives?):
- **Retrieved count**: how many rules actually returned (may be < top_k due to threshold filtering)
- **Context waste ratio**: `chars_of_irrelevant_rules / total_chars_retrieved` — 0.0 = perfect, 1.0 = all waste
- **Negative silence rate**: for negative fixtures (empty `should_match`), fraction where we correctly returned 0 results. Target: 100%.

**Operational metrics**:
- **Latency**: p50/p95/p99 retrieval time

**Per-difficulty breakdown**: all metrics reported per difficulty tier (easy/medium/hard/negative) to identify where the pipeline struggles. Negative fixtures are the precision stress test — any non-zero retrieval is a context waste.

**Design rationale**: High recall with low precision is worse than moderate recall with high precision. An irrelevant rule injected into agent context is not neutral — it competes for attention with the relevant rules, can confuse the agent, and wastes context tokens. The goal is 90%+ recall AND low noise ratio (< 0.3).

### 12.3 Model Comparison

```bash
cuecard eval fixtures.json --model BAAI/bge-small-en-v1.5
cuecard eval fixtures.json --model jinaai/jina-embeddings-v2-base-code
cuecard eval fixtures.json --model nomic-ai/nomic-embed-text-v1.5
cuecard eval fixtures.json --model snowflake/snowflake-arctic-embed-m
cuecard eval fixtures.json --model mixedbread-ai/mxbai-embed-large-v1
```

### 12.4 Known Limitation: Query-Document Domain Mismatch

Tool call queries (e.g., `"Bash: tmux send-keys -t cody 'hello'"`) are structurally different from rule text (e.g., `"Send Enter after every tmux send-keys command"`). Queries contain code syntax; rules are natural language directives. General-purpose embedding models may not bridge this gap well for all cases.

Mitigation: evaluate code-specific models (jina-embeddings-v2-base-code) early in Phase 2. The evaluation framework will surface cases where domain mismatch causes poor retrieval.

### 12.5 Dataset Roadmap

| Phase | Size | Queries | Corpora | Annotations |
|-------|------|---------|---------|-------------|
| v0.1 | ~100 | Hand-crafted tool calls | rules_basic.txt | Hand-annotated |
| v0.2 | ~500 | Real tool calls from delulu sessions | txt + md files | LLM-annotated, spot-checked |
| v0.3 | 1000+ | Real + synthetic edge cases | txt + md + yaml skills | LLM-annotated + reviewed |

## 13. Package Structure

```
cuecard/
├── src/cuecard/                    # Core library (agent-agnostic)
│   ├── __init__.py                 # Public API: load_or_build, retrieve, format_rules
│   ├── config.py                   # Load/merge/validate cuecard.toml configs
│   ├── parser.py                   # Parse rule files → list[Rule] (dispatch by suffix)
│   ├── indexer.py                  # Embed rules, build index, atomic writes
│   ├── retriever.py                # Query index, rank, dedup
│   ├── formatter.py                # Format results for injection
│   ├── freshness.py                # mtime + hash freshness checking
│   ├── logger.py                   # Structured logging with secrets scrubbing
│   ├── security.py                 # Path validation, secrets patterns, permissions
│   ├── models.py                   # Frozen dataclasses (Rule, Provenance, Index, etc.)
│   ├── cli.py                      # Typer CLI entrypoint
│   ├── py.typed                    # PEP 561 type stub marker
│   └── adapters/                   # Agent-specific wrappers (inside package)
│       ├── __init__.py
│       └── claude_code.py          # Claude Code PreToolUse hook
├── tools/
│   └── notebook.ipynb              # Step-by-step visualization notebook
├── eval/                           # Evaluation framework
│   ├── corpora/                    # Test rule sets
│   │   └── rules_basic.txt
│   ├── fixtures/                   # Golden test fixtures
│   │   └── basic.json
│   └── results/                    # Model comparison outputs
├── examples/
│   ├── cuecard.toml                # Example project config
│   └── rules.txt                   # Example rules file
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_parser.py
│   ├── test_indexer.py
│   ├── test_retriever.py
│   ├── test_formatter.py
│   ├── test_freshness.py
│   ├── test_logger.py
│   ├── test_security.py
│   └── test_cli.py
├── artifacts/
│   └── prd-v1.md                   # This document
├── pyproject.toml
├── README.md
├── LICENSE                         # MIT
└── .gitignore
```

## 14. Dependencies

| Package | Version | Why | Size | License |
|---------|---------|-----|------|---------|
| fastembed | >=0.4 | ONNX embedding inference | ~50MB + model | Apache 2.0 |
| numpy | >=1.24 | Dot product, array ops | (comes with fastembed) | BSD |
| typer | >=0.9 | CLI framework | ~50KB | MIT |
| rich | >=13 | CLI output formatting | ~1MB | MIT |

**Python:** 3.11+ (for `tomllib` in stdlib).

**Dev dependencies:** pytest, pytest-cov, ruff, mypy.

**Default model:** `BAAI/bge-small-en-v1.5` (67MB download on `cuecard setup`).

**Models to evaluate:**

| Model | Dim | Size | MTEB Retrieval | Notes |
|-------|-----|------|----------------|-------|
| BAAI/bge-small-en-v1.5 | 384 | 67MB | ~46 | Default. Best small model in fastembed. |
| jinaai/jina-embeddings-v2-base-code | 768 | ~320MB | — | Code-specific. Trained on code + technical docs. |
| nomic-ai/nomic-embed-text-v1.5 | 768 | 520MB | ~49 | 8K context, MRL support. Strong general model. |
| snowflake/snowflake-arctic-embed-m | 768 | 430MB | ~52 | Good quality/size ratio. |
| mixedbread-ai/mxbai-embed-large-v1 | 1024 | 640MB | ~54 | Best quality in fastembed catalog. |

Default will be updated based on evaluation results from golden fixtures. Model comparison happens in Phase 2 (not deferred to Phase 3).

## 15. Implementation Phases

### Phase 1: Core Pipeline (MVP)

**Deliverable:** `cuecard setup`, `cuecard retrieve`, working index with freshness and security.

- `models.py` — Rule, Provenance, RankedResult, Index, SourceMeta, ResolvedConfig
- `security.py` — path validation, secrets scrubbing patterns, permission helpers
- `config.py` — load global + project TOML, merge with defined semantics, validate
- `parser.py` — dispatch by suffix (.txt → plaintext, .md → NotImplementedError), produce Rules with provenance, enforce 500-char limit
- `indexer.py` — passage_embed() via fastembed, save .npz + metadata.json atomically with file locking
- `freshness.py` — mtime + hash checking, full rebuild on any change, integrity check
- `retriever.py` — query_embed(), dot product, threshold filter, exact + semantic dedup, top-k
- `formatter.py` — format with labeled boundary prefix
- `cli.py` — setup (with permission check), config, parse, embed, index, retrieve, format
- Tests: 100% coverage on all modules

### Phase 2: Adapter + Logging + Model Comparison

**Deliverable:** Working Claude Code plugin with install/uninstall. Model comparison results.

- `logger.py` — structured logging with secrets scrubbing, rotation, 0o600 permissions
- `adapters/claude_code.py` — PreToolUse hook
- `cli.py` — add `install claude-code`, `uninstall claude-code`, `status`, `log`, `log --stats` commands
- Model comparison: run eval across all 5 candidate models, pick best default
- Tests: adapter I/O, logger, scrubbing

### Phase 3: Evaluation Framework + Notebook

**Deliverable:** Golden fixtures, eval CLI, notebook.

- `eval/corpora/rules_basic.txt` — 50-100 hand-crafted rules
- `eval/fixtures/basic.json` — ~100 hand-annotated fixtures
- `cli.py` — add `eval` command with precision/recall/MRR/nDCG/latency
- `tools/notebook.ipynb` — step visualization, model comparison, score distributions
- Tests: eval metric calculation

### Phase 4: Publish

**Deliverable:** Installable package on PyPI, public GitHub repo.

- `pyproject.toml` — metadata, entry points, dependencies
- `README.md` — quickstart, config reference, CLI reference, trust model docs
- `LICENSE` — MIT
- GitHub repo, CI (ruff + mypy + pytest + eval regression)
- `.gitignore` template that includes `.cuecard/`

## 16. Future Enhancements (TODO)

Listed in priority order:

1. **Markdown chunking** — parse .md files by section structure, chunk at semantic boundaries (v0.2)
2. **Summarization step** — LLM or extractive summary for long chunks, embed summary, inject full text (v0.2). Pluggable modes. Post-PARSE, pre-EMBED transform.
3. **Daemon mode** — long-running process keeps model loaded, eliminates cold start (v0.2)
4. **Structured query extraction** — tool-specific query construction (Bash: command+flags, Edit: filepath+context) (v0.2)
5. **UserPromptSubmit event** — broader rule matching on user intent (v0.2)
6. **Codex/Gemini adapters** — thin wrappers for other agents (v0.2)
7. **CI evaluation** — run eval on every PR, fail if recall drops (v0.2)
8. **LLM-annotated golden dataset** — scale to 500+ fixtures using Haiku (v0.2)
9. **Real query extraction from delulu** — pull actual tool calls from sessions (v0.2)
10. **YAML skill parsing** — parse Claude Code skill files as rule sources (v0.3)
11. **CDT behavioral weighting** — weight retrieval by predicted user behavior (v0.3+)
12. **Rule priority/override** — project rules can override or suppress global rules (v0.3)
13. **In-process caching** — keep index in memory for daemon/server mode (v0.3)
14. **Binary quantization** — 32x smaller index storage for large rule sets (v0.3)
15. **Incremental index updates** — append-only with compaction, replacing full rebuild (v0.3)
16. **Approximate nearest neighbors** — FAISS/hnswlib for >100K chunks (v0.3)
17. **Rule signing** — cryptographic verification of rule file integrity (v0.3)

## 17. Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | Retrieval approach | Semantic embeddings | Keyword matching misses semantic relationships |
| D2 | Index storage | Pre-computed .npz + metadata JSON | Avoid rebuilding on every call |
| D3 | Rule format (v0.1) | Plain text, one per line, 500 char max | Simplest. Markdown v0.2 |
| D3a | Summarization | Skip for v0.1 | Rules are concise. Pluggable modes later |
| D4 | Config | `~/.cuecard/config.toml` + `./cuecard.toml` | Global + project layering |
| D4a | Config merge | Scalars: project wins. Sources: union. Model: project wins (must match) | Explicit, no ambiguity |
| D5 | Core architecture | Agent-agnostic library + in-package adapters | Same core for all agents |
| D6 | Injection volume | Top-k=5 with threshold=0.35 | 0.35 is meaningful filter for BGE-small |
| D7 | Query construction | `tool_name: tool_input[:500]` | Good signal, truncation prevents bloat |
| D7a | Asymmetric encoding | query_embed() for queries, passage_embed() for rules | Required by BGE/Nomic/Snowflake models |
| D8 | Model loading | Accept ~300ms cold start | Imperceptible next to tool execution |
| D9 | Package structure | `src/cuecard/` with adapters inside package | Resolves import path issues |
| D10 | Index freshness | mtime fast path → hash confirm → full rebuild | No splice corruption, fast enough for <500 rules |
| D10a | Atomic writes | NamedTempFile + os.replace + fcntl.flock | Prevents partial writes and concurrent corruption |
| D11 | Dependencies | fastembed, numpy, typer, rich | Lightweight (~50MB vs 2GB PyTorch) |
| D11a | Default model | BGE-small-en-v1.5 (67MB) | Best in size class. Compare early in Phase 2 |
| A1 | Injection format | Labeled boundary, no scores/provenance | Structural isolation from injection, clean for model |
| A2 | Hook events | PreToolUse default, UserPromptSubmit opt-in | Most targeted |
| A3 | Index location | `~/.cuecard/index/` global, `.cuecard/index/` project | Mirrors config layering |
| A4 | Deduplication | Exact text at merge + semantic (>0.95) on results | Two-layer: cheap then precise |
| A5 | Observability | log.jsonl (scrubbed) + stderr verbose | Persistent + quick debug |
| A6 | First run | `cuecard setup` required | Explicit model download, permission check |
| A7 | Evaluation | Golden fixtures → 1k dataset with latency benchmarks | Real queries, offline comparison |
| S1 | Path validation | Canonicalize + allowlist dirs | Prevents traversal attacks |
| S2 | Log security | Secrets scrubbing, 0o600, fixed path | Prevents credential leakage |
| S3 | File permissions | 0o700 dirs, 0o600 files, set at creation | Multi-user system safety |
| S4 | Model allowlist | Known-good models only, --allow-custom-model escape | Supply chain risk mitigation |

## 18. Review History

### Round 1 (v1.0 → v1.1)

**Architecture review:** 0 CRITICAL, 3 HIGH, 7 MEDIUM, 7 LOW
**Security review:** 0 CRITICAL, 3 HIGH, 4 MEDIUM, 1 LOW
**ML methodology review:** 1 CRITICAL, 2 HIGH, 2 MEDIUM, 4 LOW

All CRITICAL and HIGH findings addressed in v1.1:
- ML-1 (CRIT): Added Section 3.3 asymmetric encoding, D7a
- ML-2 (HIGH): Threshold raised from 0.15 to 0.35
- ML-3 (HIGH): Model comparison moved to Phase 2, Section 12.4 documents limitation
- SEC-1 (HIGH): Added Section 4.4 path validation, S1
- SEC-2 (HIGH): Added Section 7 labeled boundary, 500-char rule limit, Section 8.1 trust model
- SEC-3 (HIGH): Added Section 8.2 secrets scrubbing, S2
- ARCH-5.2 (HIGH): Replaced incremental splice with full rebuild, D10
- ARCH-2.1 (HIGH): Added exact-text dedup at merge time in retrieval step 5
- ARCH-6.1 (HIGH): Added Section 4.2 config resolution with worked example

MEDIUM findings addressed: adapter location, Index type, parser dispatch, config validation, file permissions, file locking, model validation at merge, path resolution order.

### Round 2 (v1.1 → v1.2)

**Architecture review:** 0 CRITICAL, 0 HIGH, 1 MEDIUM, 4 LOW
**Security review:** 0 CRITICAL, 1 HIGH, 3 MEDIUM, 3 LOW
**ML methodology review:** 0 CRITICAL, 0 HIGH, 1 MEDIUM, 3 LOW

All HIGH and MEDIUM findings addressed in v1.2:
- SEC-NEW-4 (HIGH): Cache paths made non-configurable — fixed locations only
- ARCH-N-1 (MED): `reindex=False` parameter documented in API
- ML-F4 (MED): fastembed generator return type documented
- SEC-1-gap (MED): Project directory anchored to cuecard.toml parent dir, symlink contradiction resolved
- SEC-3/NEW-2 (MED): `logging.redact` restricted to global config only
- Validation table updated with `logging.redact` and `sources.allowed_dirs`

**Convergence status:** Two consecutive rounds with 0 CRITICAL, 0 HIGH (after fixes). PRD ready for implementation.

## 19. Implementation Notes

### First-Time Experience

`cuecard setup` with no existing config:
1. Creates `~/.cuecard/` (0o700) with `config.toml` template and `rules/global.txt` with commented-out example rules
2. Prints: "Created config. Add rules to ~/.cuecard/rules/global.txt, then run `cuecard setup` again."
3. Second `cuecard setup`: downloads model, builds index, verifies permissions

`cuecard retrieve` with no setup: prints `[cuecard] Not set up. Run 'cuecard setup'.` to stderr, returns empty results, exit 0. Never blocks work.

### Testing Strategy

- **Unit tests:** Mock fastembed. Each module tested in isolation with deterministic vectors. Fast, no model download.
- **Integration test:** One `@pytest.mark.slow` test uses real BGE-small model, runs full pipeline (parse → embed → index → retrieve → format). Verifies real embeddings produce meaningful similarity.
- **CI:** Caches model in `~/.cache/fastembed/` between runs. Runs slow tests.
- **Coverage:** 100% on all modules.

### Hook Lifecycle (Phase 2)

```bash
cuecard install claude-code      # symlink/copy adapter to ~/.claude/plugins/
cuecard uninstall claude-code    # remove adapter, clean up
cuecard status                   # show installed hooks, index health, model info
```

## 20. Open Questions

None remaining — all ambiguities resolved.
