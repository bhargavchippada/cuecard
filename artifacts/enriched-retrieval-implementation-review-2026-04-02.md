# Enriched Retrieval Implementation Review

Date: 2026-04-02

Scope:
- PRD reviewed: `artifacts/enriched-retrieval-prd.md`
- Docs reviewed: `README.md`, `CLAUDE.md`
- Runtime reviewed: enriched retrieval, loader, CLI, adapter, eval, retrievers, indexer, parser, config
- Validation run:
  - `uv run pytest -q` -> `731 passed, 2 warnings in 2.17s`
  - `uv run ruff check src tests` -> passed
  - `uv run mypy src` -> passed

## Executive Summary

The implementation is directionally strong. The data model, expansion-aware indexing, parent-child collapse, sparse retriever, and test coverage are all materially better than the previous baseline. The codebase now has the right primitives for enriched retrieval.

The main issue is that the new architecture is not wired through consistently. Two of the most important PRD promises are currently broken in live paths:

1. `cuecard index` does not consume persisted `rules.json` expansions, so the expansion lifecycle described in the PRD does not actually hold for the main CLI rebuild path.
2. The dense+sparse fused retrieval stage is bypassed in default `embedding` mode across the CLI, adapter, and eval harness, so enriched retrieval is not actually active in the default runtime path.

Because of those two issues, benchmark improvements from enriched retrieval are likely understated or absent in the paths users hit most often.

## Findings

### 1. High: `cuecard rules expand` -> `cuecard index` lifecycle is broken

Files:
- `artifacts/enriched-retrieval-prd.md:146`
- `artifacts/enriched-retrieval-prd.md:148`
- `src/cuecard/cli.py:319`
- `src/cuecard/cli.py:330`
- `src/cuecard/loader.py:73`
- `src/cuecard/loader.py:79`

PRD contract:
- Step 1: `cuecard index` or `load_or_build()` creates or merges `rules.json`
- Step 2: `cuecard rules expand` writes expansions into `rules.json`
- Step 3: `cuecard index` reads `rules.json` and embeds canonical text plus expansions

Actual behavior:
- `load_or_build()` follows that contract. It parses source files, merges cached `rules.json`, saves the canonical JSON intermediate, and builds the index from merged rules.
- `cuecard index` does not. Its rebuild path directly parses raw source files and calls `build_index(...)`, bypassing `rules.json` entirely.

Why this matters:
- A user can successfully run `cuecard rules expand`, observe populated `rules.json`, then rebuild with `cuecard index` and silently lose all expansion value in the generated embeddings and BM25 corpus.
- This breaks the core enriched retrieval lifecycle and makes manual indexing behave differently from loader-driven runtime indexing.

Concrete reproduction:

```python
parsed_rules 1
embedding_rows 1
bm25_corpus ('Never commit secrets',)
cached_expansions ('AKIA in source', 'hardcoded API key')
```

That result shows persisted expansions exist in `rules.json`, but the direct `cuecard index` path still builds a one-row index from raw source text only.

Recommended fix:
- Make `cuecard index` reuse the same scope rebuild path as `load_or_build()`, or extract a shared helper that:
  - parses sources
  - merges cached `rules.json`
  - writes refreshed `rules.json`
  - builds the index from merged rules
- Treat `load_or_build()` as the canonical lifecycle and stop duplicating rebuild logic in `cli.py`.

Risk if left unfixed:
- Users will assume expansion quality gains are active when they are not.
- Benchmark runs that depend on `cuecard index` can materially underreport retrieval quality.

### 2. High: enriched retrieval is bypassed in default `embedding` mode

Files:
- `src/cuecard/pipeline.py:108`
- `src/cuecard/pipeline.py:115`
- `src/cuecard/pipeline.py:117`
- `src/cuecard/pipeline.py:177`
- `src/cuecard/cli.py:387`
- `src/cuecard/cli.py:394`
- `src/cuecard/cli.py:432`
- `src/cuecard/adapters/claude_code.py:90`
- `src/cuecard/adapters/claude_code.py:98`
- `src/cuecard/eval.py:356`
- `src/cuecard/retriever.py:39`

The new retrieval stage lives in `pipeline._run_retrieval_stage()`. That is where:
- dense retrieval runs
- sparse/BM25 retrieval runs
- RRF fusion happens
- per-retriever traces are produced

But the default runtime path still avoids that stage:
- CLI `retrieve` calls legacy `retriever.retrieve()` when mode is `embedding`
- CLI `format` always calls legacy `retriever.retrieve()`
- Claude Code adapter calls legacy `retrieve()` when config mode is `embedding`
- Eval uses legacy `retrieve()` for `embedding` mode

Impact:
- In the default mode, enriched retrieval is effectively off.
- Expansion-aware BM25 and RRF only run when the user opts into a non-default pipeline mode.
- This contradicts the architecture described in `CLAUDE.md`, which says pipeline orchestration is central and the retrieval stage always runs.

Concrete reproduction:

```python
retrieve []
pipeline ['Never commit secrets']
```

In that setup, the query only matched an expansion phrase. Legacy `retrieve()` returned nothing; `run_pipeline(..., mode="embedding")` returned the correct rule via the enriched stage.

Recommended fix:
- Route all retrieval entry points through `run_pipeline()`, including `embedding` mode.
- Keep `retriever.retrieve()` as an internal dense retriever primitive, not as a top-level runtime path.
- Ensure `cuecard format` uses the same retrieval path as the adapter so “what you see” matches actual hook behavior.

Risk if left unfixed:
- Default-quality benchmarks will not reflect the enriched retrieval work.
- Users will need to discover a non-obvious mode dependency before seeing the intended gains.

### 3. High: eval non-embedding modes crash against the real pipeline

Files:
- `src/cuecard/eval.py:278`
- `src/cuecard/eval.py:359`
- `src/cuecard/pipeline.py:153`
- `src/cuecard/pipeline.py:174`
- `tests/test_eval.py:555`

`run_eval(..., mode="rerank")` now passes `_EvalConfig` into `run_pipeline()`. `_EvalConfig` only defines:
- `top_k`
- `threshold`
- `dedup_threshold`
- `query_max_length`

The real pipeline now also reads:
- `config.sparse_enabled`
- `config.fusion_k`

Result:
- non-embedding eval modes crash with `AttributeError` when they hit the real pipeline

Concrete reproduction:

```python
AttributeError: '_EvalConfig' object has no attribute 'sparse_enabled'
```

Why tests missed it:
- `tests/test_eval.py:555` patches `cuecard.pipeline.run_pipeline`, so the integration path is never exercised.

Recommended fix:
- Expand `_EvalConfig` to include the new retrieval fields with the same defaults as `ResolvedConfig`
- Add at least one non-mocked integration test that executes the real pipeline in `rerank` or `llm-*` mode with a mock embedding model and a real `Index`

Risk if left unfixed:
- Benchmarking in rerank or LLM modes is unreliable because the harness crashes when used for real measurements.

### 4. Medium: `cuecard rules expand` can operate on stale cached `rules.json`

Files:
- `src/cuecard/cli_rules.py:245`
- `src/cuecard/cli_rules.py:247`
- `src/cuecard/cli_rules.py:249`
- `src/cuecard/loader.py:68`
- `src/cuecard/loader.py:74`
- `src/cuecard/loader.py:76`

`cuecard rules expand` currently does this:
- load `rules.json` from cache if present
- only parse source files if `rules.json` is missing

That means cached JSON wins over current source files, even after source edits.

Concrete reproduction:

```python
parsed_from_source ['Original rule']
rules_expand_would_use ['Stale cached rule']
```

Impact:
- expansions can be generated for obsolete rule text
- the next freshness-aware rebuild may drop those expansions because text equality no longer matches
- users can spend time and tokens expanding stale content

Recommended fix:
- Before expansion, always parse current source files and merge with cached `rules.json` using the same exact-text preservation rules already implemented in `loader.py`
- Save the refreshed merged `rules.json`, then call the LLM expander

Risk if left unfixed:
- intermittent “expansions disappeared” behavior after source edits
- wasted expansion generation cost

### 5. Medium: documentation still describes a different runtime than the one shipped

Files:
- `README.md:9`
- `README.md:41`
- `README.md:76`
- `README.md:101`
- `CLAUDE.md:139`
- `CLAUDE.md:140`
- `CLAUDE.md:219`

Current drift:
- `README.md` still describes the system as embedding retrieval plus optional LLM reranking, without explaining dense+sparse fusion or the `rules.json` intermediate
- config examples do not include `fusion_k`, `sparse_enabled`, or expansion settings
- `README.md` still says `541 tests`, but the suite is now `731 passed`
- `CLAUDE.md` says the pipeline orchestrates all stages and CLI/adapter delegate to it, which is not true in default `embedding` mode today

Impact:
- contributors reading docs will build the wrong mental model
- operators may think enriched retrieval is active by default when current wiring says otherwise

Recommended fix:
- Update README and CLAUDE together, not independently
- Document the real lifecycle explicitly:
  - sources -> parse/merge -> `rules.json`
  - `rules expand` enriches `rules.json`
  - index build embeds canonical plus expansions
  - runtime retrieval uses dense+sparse fusion in all modes
- Keep benchmark/test counts current or remove exact counts from top-level docs

### 6. Low: `cuecard hooks status` only checks the global cache

Files:
- `src/cuecard/cli_hooks.py:175`
- `src/cuecard/cli_hooks.py:181`

The status command loads only `cfg.global_cache_dir`. After the scoped-cache work, that is incomplete and can report “no valid index found” even when a project-scoped index exists and runtime retrieval works.

Recommended fix:
- Either use `load_or_build(..., reindex=False)` for status, or report global and project scope independently.

This is lower severity than the retrieval path issues, but it will confuse debugging once project-scoped enriched retrieval is used more heavily.

## Test Gaps

The suite is large and generally strong, but it missed the highest-severity regressions because the most important boundaries are still under-tested.

### Missing end-to-end coverage for expansion lifecycle

Needed test:
- `rules.txt` -> `rules.json` with expansions -> `cuecard index` -> retrieval hit on an expansion-only query

Current gap:
- there are good unit tests for parser, indexer, loader merge, sparse retriever, and expander
- there is no end-to-end test proving the user-facing CLI lifecycle actually preserves and uses expansions

### Missing integration coverage for `embedding` mode through the pipeline

Needed tests:
- CLI `retrieve` in default mode exercises `run_pipeline(..., mode="embedding")`
- adapter default mode exercises fused retrieval, not legacy dense-only retrieval
- CLI `format` matches the adapter retrieval path

Current gap:
- tests focus on non-default pipeline modes or mock the pipeline boundary
- no test locks in that enriched retrieval is active in the default runtime path

### Missing real eval integration test for non-embedding modes

Needed test:
- `run_eval(..., mode="rerank")` with a real `Index` and the real pipeline, using mocks only for expensive external components

Current gap:
- current eval mode coverage patches `run_pipeline`, so config shape drift is invisible

### Missing status coverage for scoped caches

Needed test:
- status behavior when only a project-scoped index exists

Current gap:
- no test appears to verify post-scope status accuracy

## Suggested Remediation Order

1. Unify rebuild logic so `cuecard index` uses the same `rules.json` merge lifecycle as `load_or_build()`.
2. Route all retrieval entry points through `run_pipeline()`, including `embedding` mode and `cuecard format`.
3. Fix `_EvalConfig` to match current pipeline requirements and add a real integration test.
4. Refresh `rules expand` from source files before expanding.
5. Update README and CLAUDE after the runtime wiring is corrected.
6. Fix `hooks status` to reflect scoped caches accurately.

## Positive Notes

Several parts of the implementation are solid and worth preserving as-is:

- `Rule.expansions` added directly to the frozen dataclass was the right compatibility choice.
- Parent-child collapse via `rule_map` is clean and well tested.
- Sparse retriever design is appropriately lightweight for this corpus size.
- `load_or_build()` now embodies a much cleaner source-of-truth model than the older direct index usage.
- Shared LLM utilities and the expansion parser are materially cleaner than the previous ad hoc split.
- The test suite breadth is high; the remaining issue is integration coverage at the actual runtime seams.

## Bottom Line

The enriched retrieval implementation is close, but it is not fully live in the paths that matter most. The core primitives are in place; the remaining work is primarily wiring and lifecycle consistency. Fixing the three high-severity issues above should unlock the quality gains the PRD intended without requiring a redesign.
