# Phase 5: Multi-Source Parsing & Smart Chunking — DRAFT PRD

> Status: DRAFT — initial direction from session 16 discussions. Needs refinement before implementation.

## Objective

Enable cuecard to ingest rules from any text source — markdown files, YAML skills, CLAUDE.md, project docs — without requiring special formatting. The system intelligently chunks documents at semantic boundaries and indexes them alongside plain-text rules in a unified pipeline.

## Motivation

Today cuecard only parses `.txt` (one rule per line) and `.json` (structured intermediate). Users want to:
1. Point cuecard at their existing `CLAUDE.md` and have it "just work"
2. Index `~/.claude/rules/**/*.md` rule files with heading structure
3. Index Claude Code skill files (YAML frontmatter + markdown body)
4. Add project-specific guidelines from any `.md` doc without reformatting
5. NOT maintain separate files for coding vs workflow rules — one CLAUDE.md has both

## Design Principles

1. **No special formatting required.** Users point at files; cuecard figures out the chunks.
2. **Intelligent chunking, not line splitting.** Headings, bullets, and code blocks are semantic boundaries.
3. **Context preservation.** A bullet under `## Security` carries "Security" as context — the heading is part of the rule's meaning.
4. **Read-only sources.** `.md` and `.yaml` files are indexed but managed by the user in their editor. `cuecard rules add/remove` only operates on `.txt` files. Attempting to remove a rule from a `.md` source prints the file path and line number for manual editing.
5. **LLM reranker is the quality gate.** We don't require source-side tagging of rules by event type. The embedding stage does its best; the LLM reranker discriminates using event context.
6. **Backward compatible.** Existing `.txt` and `.json` sources continue to work unchanged.

## Architecture

### Current Pipeline
```
.txt / .json  →  parse_rules()  →  rules.json  →  build_index()  →  Index
                                                                        ↓
                                                              run_pipeline()  →  results
```

### Phase 5 Pipeline
```
.txt           →  parse_txt()      ─┐
.json          →  parse_json()      │
.md            →  parse_markdown()  ├→  rules.json  →  build_index()  →  Index
.yaml/.skill   →  parse_yaml()     │                                       ↓
CLAUDE.md      →  parse_markdown() ─┘                           run_pipeline()  →  results
```

`parse_rules()` dispatches by file extension. Each parser returns `list[Rule]` with provenance.

### Markdown Chunking Strategy

**Heading-based primary chunks:**
- Each `##` or `###` section becomes a candidate chunk
- The heading text becomes part of the rule context (stored in `section_path`)
- Nested headings inherit parent context: `## Security > ### Input Validation`

**Bullet-list extraction within sections:**
- Standalone bullet lists (not under a heading with prose) → each bullet = one rule
- Bullets within a prose section → section is the chunk, bullets are context
- Nested bullets collapse into their parent bullet

**Code blocks:**
- Fenced code blocks within a section stay with that section's chunk
- Standalone code blocks (not under a relevant heading) are skipped

**Heuristics for what IS a rule vs what ISN'T:**
- Short imperative statements (< 500 chars) → likely rules
- Long prose paragraphs → likely documentation, skip or summarize
- Config examples, file trees, command output → skip (not actionable rules)
- Tables → extract row-level rules if rows are short imperatives

**Chunk size target:** 20-500 chars. Chunks > 500 get summarized (v5.2).

### Data Model Extensions

Already designed in prd-v1.md (lines 187-194):

```python
@dataclass(frozen=True)
class Rule:
    text: str
    provenance: Provenance
    expansions: tuple[str, ...] = ()
    # Phase 5 additions:
    section_path: tuple[str, ...] = ()    # ("Security", "Input Validation")
    chunk_type: str = "rule"              # "rule" | "paragraph" | "list_item" | "code_block"
    summary: str | None = None            # for long chunks — embed summary, inject full text
    source_type: str = "txt"              # "txt" | "json" | "md" | "yaml"
```

`section_path` is critical for retrieval quality — it provides hierarchical context that embeddings can use. A rule "Validate all inputs" under `## Security > ### API Endpoints` is very different from the same text under `## Testing`.

### Source Configuration

```toml
[sources]
rules = [
    "rules.txt",                          # plain text, one per line (current)
    "CLAUDE.md",                          # markdown, smart chunking
    "docs/guidelines.md",                 # any markdown
    "~/.claude/rules/common/*.md",        # glob patterns
    # "skills/*.yaml",                    # future: skill files
]
```

The config already supports multiple paths. Phase 5 just adds parsers for new extensions.

### CLAUDE.md Specific Handling

CLAUDE.md files have a known structure across projects:
- `## Project Structure` — file tree (skip, not rules)
- `## Tech Stack` — metadata (skip)
- `## Key Commands` — commands (index as rules)
- `## Conventions` — rules (index, each bullet = rule)
- `## Key Design Decisions` — context (skip or summarize)

We could ship a CLAUDE.md-aware parser that knows common section patterns. But per principle #1, we shouldn't require this structure — the generic markdown parser should handle any layout.

### YAML / Skill File Parsing (v5.2)

Claude Code skills have:
```yaml
---
name: commit
description: Create a git commit
event: PreToolUse
tools: [Bash, Read]
---

# Instructions
When creating commits...
```

The YAML frontmatter carries metadata (event type, tools). The markdown body is the rule content. This is the one source type where event_type IS available from the source — use it as a retrieval hint when present.

## Quality Architecture (from session 16 benchmarks)

### The Three-Layer Quality Stack

| Layer | What It Fixes | Latency | Required? |
|-------|--------------|---------|-----------|
| Embedding (jina-code + expansions + BM25) | Vocabulary gap, keyword matching | ~15ms | Yes |
| Threshold | Weak matches | ~0ms | Yes |
| LLM reranker (Qwen3.5-35B) | Cross-domain noise, negative silence | ~1s | For quality target |

### Key Findings (session 16)

1. **Embedding mode alone cannot achieve 90%+ quality.** Negative silence is 0% for all models except jina-v3 (6%, but 457ms). The LLM reranker is essential.
2. **Model choice gives ~4% spread on recall.** The LLM reranker gives ~70% improvement on noise/silence. Invest in the reranker.
3. **jina-embeddings-v2-base-code** is the best embedding model for PreToolUse (60% hard recall). snowflake-arctic-embed-m is best for UserPromptSubmit (73% hard recall).
4. **Cross-domain noise is the main problem.** Generic workflow rules match everything in embedding space. The LLM reranker discriminates using event context — no source-side tagging needed.
5. **Expansion v3 prompt** with event-type awareness and cross-domain DON'T rules improved workflow medium recall by +26 pts.
6. **Sparse retrieval (BM25)** is a tradeoff: +2% recall, -1.7% negative silence on unified corpus. Not a clear default win.

### Recommended Defaults

```toml
[embedding]
model = "jinaai/jina-embeddings-v2-base-code"   # best code recall

[retrieval]
top_k = 5
threshold = 0.30
sparse_enabled = true    # BM25 helps recall, LLM reranker handles noise

[pipeline]
mode = "llm-local"       # production quality requires LLM reranker

[pipeline.llm]
local_endpoint = "http://localhost:8081/v1"
```

Fallback: `mode = "embedding"` when no LLM is available. Users accept lower quality (0% negative silence, ~85% noise).

## Implementation Phases

### Phase 5.1: Markdown Parser (core)
- `parse_markdown()` function in `parser.py`
- Heading-based chunking with `section_path`
- Bullet extraction
- Code block handling
- Provenance with line numbers
- Config: `.md` files in `sources.rules`
- Tests: parse real CLAUDE.md files, verify chunk quality

### Phase 5.2: Summarization + YAML
- Long chunks (> 500 chars) → LLM summarization before embedding
- `summary` field: embed summary, inject full text at format time
- YAML frontmatter parser for skill files
- Event type hint from YAML metadata

### Phase 5.3: Smart Expansion Generation
- Auto-expand rules on first index build when LLM is available
- `cuecard setup` generates expansions if `pipeline.mode` includes `llm`
- `--no-expand` flag to opt out
- Expansion quality validation: cosine diversity check

### Phase 5.4: Default Model Migration
- Switch default from bge-small to jina-code
- Migration path: auto-rebuild index on model change
- Document the tradeoff (2x latency, better quality)

## Open Questions

1. **How aggressive should markdown chunking be?** Every bullet = a rule, or only bullets that look like imperatives?
2. **Should we summarize or skip long prose sections?** Summarization adds LLM cost at index time.
3. **CLAUDE.md structure detection?** Generic parser vs CLAUDE.md-aware parser vs both?
4. **Expansion auto-generation:** On every index build, or only on `cuecard setup`?
5. **Multi-model support:** Should users be able to specify different models per scope (global vs project)?

## Success Criteria

- [ ] `cuecard setup` with `sources.rules = ["CLAUDE.md"]` produces a working index
- [ ] Rules from markdown sections have correct `section_path` and line provenance
- [ ] Chunk quality: >80% of extracted chunks are actionable rules (manual audit on 5 real CLAUDE.md files)
- [ ] No regression on existing `.txt` and `.json` parsing
- [ ] Enriched retrieval quality maintained or improved with markdown-sourced rules
- [ ] `cuecard rules remove` on markdown-sourced rule prints file:line for manual editing

## References

- prd-v1.md Section 18 (Future): lines 856-871
- prd-v1.md Data model: lines 187-194
- unified-events-prd.md line 51: "single CLAUDE.md will contain both coding and workflow rules"
- Session 16 benchmarks: `eval/results/` (raw, enriched, model-comparison, v3)
- Expansion quality review: `artifacts/expansion-quality-review-2026-04-02.md`
