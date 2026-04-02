# Session 15 Progress (2026-04-02)

**Status:** IN PROGRESS — implementation complete, review converged, mutation testing in progress

## What Was Done

### Expansion Validation (Pre-Implementation)
- Hand-crafted expansion validation: 12/12 hard fixtures improved, avg delta +0.371
- LLM expansion validation (v1 prompt): 11/12 improved, but only 4/12 cross threshold
- LLM expansion validation (v2 prompt): 12/12 improved, 8/12 cross threshold, avg delta +0.276
- Key insight: v2 prompt with DO/DON'T guidelines and 3 golden examples nearly doubles effectiveness
- Validated expansion approach before implementation

### Implementation (Phases 1-5)
All 5 phases of enriched-retrieval-prd.md implemented:

1. **Phase 1: JSON Intermediate Format** (team-lead, direct)
   - `Rule.expansions: tuple[str, ...]` field added to frozen dataclass
   - `_parse_json()` in parser.py with validation (length, count, version)
   - `save_rules_json()` / `load_rules_json()` / `merge_rules_json()` in indexer.py
   - `loader.py` integration: preserves expansions across rebuilds
   - Constants centralized in `models.py`: `MAX_EXPANSION_LENGTH`, `MAX_EXPANSIONS_PER_RULE`

2. **Phase 2: Expansion-Aware Indexing** (builder-core agent)
   - `Index.rule_map` and `Index.bm25_corpus` fields with backwards-compatible defaults
   - `build_index()` embeds canonical + expansions, builds rule_map
   - Metadata v2 serialization with backwards compat for v1
   - `ResolvedConfig` gains: `fusion_k`, `sparse_enabled`, `expansion_max_per_rule`, `expansion_max_length`

3. **Phase 3: Parent-Child Collapse** (builder-core agent)
   - `retrieve()` in retriever.py: `np.maximum.at` parent collapse via rule_map
   - Precomputed `canonical_row` dict for O(1) dedup lookups
   - `merge_indexes()` returns `Index` (not tuple), merges rule_maps, handles bm25_corpus

4. **Phase 4: Retriever Adapter + BM25 + Fusion** (builder-retrieval agent)
   - `src/cuecard/retrievers/` package: Protocol, ScoredCandidate, fuse()
   - `DenseRetriever` wrapping cosine similarity with parent collapse
   - `SparseRetriever` with inline BM25Okapi (~35 lines), cached per instance
   - RRF fusion with configurable k parameter
   - `RetrieverTrace` + `RetrievalStageTrace` for per-retriever observability
   - Pipeline updated: `_run_retrieval_stage()` with sparse error isolation

5. **Phase 5: Expansion CLI + LLM Utils** (builder-expand agent)
   - `src/cuecard/llm_utils.py` extracted from llm_reranker.py
   - `src/cuecard/expander.py` with v2 prompt, nonce-delimiter defense, scrub_secrets
   - `cuecard rules expand` CLI subcommand (--backend, --endpoint, --dry-run, --missing-only)

### Eval Dataset Expansion
- **Fixture audit** (auditor agent): 25 unrealistic expectations removed, 6 fixtures reclassified, 2 corpus rules reworded
- **Session mining v1** (miner agent): 69 new fixtures from cuecard/delulu/canopy-ai
- **Session mining v2** (deep-miner agent): 80 new fixtures from 7 projects (delulu, canopy-ai, cuecard, antigravity, faster-whisper x2, forceatlas2)
- **Grand total**: 587 fixtures (438 original + 149 mined)

### Code Review (Convergence-Based)
**Round 1**: 5 parallel agents (code, security, python, deep logic, PRD adherence)
- 1 CRITICAL: `retrieve()` using embedding indices as rule indices (IndexError with expansions)
- 8 HIGH: merge bm25 misalignment, BM25 rebuilt per query, O(n) canonical_row, duplicated constants, untrusted source.file, path traversal in remove, etc.
- 5 MEDIUM: getattr on typed fields, bare except without details, scrub_secrets missing
- All fixed, re-verified: 731 tests, 100% coverage

**Round 2**: 2 parallel agents (code+logic, security)
- 0 CRITICAL, 0 HIGH, 2 MEDIUM (logging detail), 4 LOW
- Fixed the MEDIUMs, converged

### Mutation Testing (mutmut)
- Installed mutmut 3.5.0, configured in pyproject.toml
- Initial run: 106 surviving mutants across critical files
- Mutant-killer agent working on: BM25 formula (49), parent collapse (13), RRF fusion (2), dedup logic (1)
- In progress — agent iterating to convergence

## Files Changed (New)
- `src/cuecard/retrievers/__init__.py` — Retriever protocol, ScoredCandidate, fuse()
- `src/cuecard/retrievers/dense.py` — DenseRetriever
- `src/cuecard/retrievers/sparse.py` — BM25Okapi, SparseRetriever
- `src/cuecard/llm_utils.py` — Shared LLM helpers
- `src/cuecard/expander.py` — LLM expansion generation
- `tests/test_fusion.py` — RRF fusion tests
- `tests/test_retrievers_dense.py` — Dense retriever tests
- `tests/test_retrievers_sparse.py` — Sparse retriever tests
- `tests/test_expander.py` — Expansion generation tests
- `tests/test_llm_utils.py` — LLM utils tests
- `eval/fixtures/mined-sessions.json` — 69 mined fixtures
- `eval/fixtures/mined-sessions-v2.json` — 80 mined fixtures
- `artifacts/fixture-audit-2026-04-02.md` — Audit summary
- `artifacts/expansion-validation.py` — Hand-crafted expansion experiment
- `artifacts/expansion-llm-validation.py` — LLM expansion experiment
- `artifacts/expansion-prompt-iteration.py` — Prompt v1 vs v2 comparison

## Files Changed (Modified)
- `src/cuecard/models.py` — Rule.expansions, Index.rule_map/bm25_corpus, RetrieverTrace, RetrievalStageTrace, centralized constants
- `src/cuecard/parser.py` — _parse_json dispatch
- `src/cuecard/indexer.py` — save/load rules.json, merge_rules_json, expansion-aware build_index, metadata v2
- `src/cuecard/loader.py` — rules.json integration
- `src/cuecard/retriever.py` — parent collapse, merge_indexes returns Index
- `src/cuecard/pipeline.py` — multi-retriever stage, sparse error isolation
- `src/cuecard/llm_reranker.py` — imports from llm_utils
- `src/cuecard/config.py` — new fields, imports from llm_utils
- `src/cuecard/cli_rules.py` — expand subcommand, path validation in remove
- `eval/fixtures/basic.json` — audited expectations
- `eval/fixtures/combined.json` — rebuilt from audited basic + workflow
- `eval/corpora/rules_basic.txt` — 2 rules reworded

## Pending Next Steps
1. Mutant-killer agent to converge (kill all dangerous mutants)
2. Run full eval benchmark with enriched index (Phase 6 from PRD)
3. Phase 7: Update notebook
4. Phase 4 (publish): PyPI, GitHub CI
5. Dogfood as Claude Code hook with real expansions
