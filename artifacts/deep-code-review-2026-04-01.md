# cuecard Deep Code Review

Date: 2026-04-01

Scope:
- Core runtime under `src/cuecard/`
- Test suite under `tests/`
- Key docs: `README.md`, `CLAUDE.md`, `artifacts/prd-v1.md`

Verification performed:
- `uv run pytest -q` -> 541 passed, 2 warnings
- `uv run ruff check src tests` -> passed
- `uv run mypy src` -> passed

## Executive Summary

The project is in much better shape than a typical early-stage repo: the test suite is large, fast, and green, and static analysis is clean. The main risks are not build-breakage issues. They are architectural correctness and product-contract drift:

1. Project-scoped behavior is effectively broken in the runtime path.
2. Freshness/rebuild behavior described in the design docs is not implemented in the public/runtime path.
3. Retrieval quality is lower than intended because query normalization exists but is not used.
4. The docs promise APIs and behavior that do not exist.
5. The CLI has grown far past the repository's own maintainability constraints.

The first two items are the most important because they affect correctness, isolation between projects, and whether users can trust the injected rules.

## Findings

### 1. Critical: project-specific rules can be ignored or leak across projects

Files:
- [src/cuecard/adapters/claude_code.py](/home/turiya/projects/cuecard/src/cuecard/adapters/claude_code.py#L76)
- [src/cuecard/adapters/claude_code.py](/home/turiya/projects/cuecard/src/cuecard/adapters/claude_code.py#L77)
- [src/cuecard/cli.py](/home/turiya/projects/cuecard/src/cuecard/cli.py#L267)
- [src/cuecard/cli.py](/home/turiya/projects/cuecard/src/cuecard/cli.py#L306)
- [src/cuecard/cli.py](/home/turiya/projects/cuecard/src/cuecard/cli.py#L321)
- [src/cuecard/cli.py](/home/turiya/projects/cuecard/src/cuecard/cli.py#L325)
- [src/cuecard/config.py](/home/turiya/projects/cuecard/src/cuecard/config.py#L395)
- [src/cuecard/config.py](/home/turiya/projects/cuecard/src/cuecard/config.py#L396)
- [CLAUDE.md](/home/turiya/projects/cuecard/CLAUDE.md#L151)

What is wrong:
- The config model exposes both `global_cache_dir` and `project_cache_dir`, and the docs explicitly describe project-specific rules and overrides.
- The Claude Code adapter calls `load_config()` without `project_dir`, so it does not load `./cuecard.toml` at all.
- The adapter then loads only `config.global_cache_dir`.
- The CLI `index`, `retrieve`, and `format` commands do resolve project config from `Path.cwd()`, but they still read and write only `cfg.global_cache_dir`.

Why this matters:
- If a user builds the index inside project A, project A rules can end up stored in the global cache.
- When the hook runs elsewhere, it can load those cached project A rules even though project A config is no longer in scope.
- Conversely, if a user expects project-specific rules to apply in the hook, they may never be loaded because the hook ignores `project_dir`.

Impact:
- Incorrect rule injection.
- Cross-project rule contamination.
- Potential confidentiality issue if sensitive internal rules from one repo influence another repo's session.

Recommended fix:
- Decide on one cache model and implement it consistently.
- Best option: keep separate caches and compose them at retrieval time.
- Adapter path should load config with `project_dir=Path.cwd()` and then load both global and project caches when present.
- CLI `index` should build and persist the correct cache for the current scope instead of always writing to the global cache.
- Add an end-to-end test that builds indexes for two different projects and proves there is no rule leakage between them.

### 2. High: the documented freshness contract is not implemented in the runtime retrieval path

Files:
- [artifacts/prd-v1.md](/home/turiya/projects/cuecard/artifacts/prd-v1.md#L62)
- [artifacts/prd-v1.md](/home/turiya/projects/cuecard/artifacts/prd-v1.md#L67)
- [artifacts/prd-v1.md](/home/turiya/projects/cuecard/artifacts/prd-v1.md#L386)
- [src/cuecard/__init__.py](/home/turiya/projects/cuecard/src/cuecard/__init__.py#L3)
- [src/cuecard/cli.py](/home/turiya/projects/cuecard/src/cuecard/cli.py#L325)
- [src/cuecard/cli.py](/home/turiya/projects/cuecard/src/cuecard/cli.py#L382)
- [src/cuecard/adapters/claude_code.py](/home/turiya/projects/cuecard/src/cuecard/adapters/claude_code.py#L77)

What is wrong:
- The PRD is centered on `load_or_build(reindex=True)` handling integrity and freshness before retrieval.
- There is no `load_or_build` implementation in the package.
- The actual runtime paths for the adapter, `cuecard retrieve`, and `cuecard format` do a plain `load_index(...)` and proceed.
- `check_freshness(...)` is only used in explicit rebuild flows like `setup` and `index`.

Why this matters:
- Rule edits after indexing are invisible until a manual rebuild happens.
- The product promise is "retrieve the right rule at the right moment"; stale indexes undercut that promise directly.
- The docs imply a self-healing runtime path, but the implementation is manual-only.

Impact:
- Silent stale behavior.
- Hard-to-debug user reports because the retrieval call still succeeds.
- Mismatch between product claims and actual semantics.

Recommended fix:
- Implement a single `load_or_build(...)` entry point and make the adapter and CLI retrieval commands use it.
- Keep `reindex=False` as an explicit low-latency mode if needed, but do not make it the undocumented default.
- Add tests that edit a source file after initial indexing and assert that runtime retrieval sees the new rule without a manual `cuecard index`.

### 3. Medium: query normalization is implemented and tested, but retrieval never uses it

Files:
- [src/cuecard/retriever.py](/home/turiya/projects/cuecard/src/cuecard/retriever.py#L28)
- [src/cuecard/retriever.py](/home/turiya/projects/cuecard/src/cuecard/retriever.py#L82)
- [tests/test_retriever.py](/home/turiya/projects/cuecard/tests/test_retriever.py#L349)

What is wrong:
- `normalize_query()` exists specifically to strip tool prefixes like `Bash:` and `Read:` before embedding.
- `retrieve()` embeds `query` directly and never calls `normalize_query(query)`.
- The helper has standalone tests, which makes this look intentional in the test suite even though it is dead in the main path.

Why this matters:
- Tool prefixes become part of the semantic query even though the code comments say the opposite.
- This can reduce recall, especially for short action strings where the prefix is a large share of the input.

Impact:
- Lower retrieval quality than the documented design intends.
- False negatives that are hard to notice because results still look plausible.

Recommended fix:
- Normalize inside `retrieve()` immediately before truncation and embedding.
- Add at least one behavioral test asserting that `retrieve()` produces the same result for `"Bash: git commit"` and `"git commit"` when the same fake embedding model is used.

### 4. Medium: documentation and public API drift is substantial

Files:
- [CLAUDE.md](/home/turiya/projects/cuecard/CLAUDE.md#L18)
- [artifacts/prd-v1.md](/home/turiya/projects/cuecard/artifacts/prd-v1.md#L62)
- [src/cuecard/__init__.py](/home/turiya/projects/cuecard/src/cuecard/__init__.py#L15)

What is wrong:
- `CLAUDE.md` and the PRD describe `load_or_build` as part of the public API.
- The package exports `load_config`, `retrieve`, and `format_rules`, but no `load_or_build`.
- The docs also describe freshness/rebuild behavior as a core capability, but the public surface does not expose that workflow.

Why this matters:
- Internal docs are acting as the source of truth for implementation, but they are not truthful on one of the most important product contracts.
- This increases the risk of future work being built on the wrong assumptions.

Impact:
- Onboarding confusion.
- Misleading design history.
- Incorrect user expectations for library consumers and maintainers.

Recommended fix:
- Either implement the documented API or rewrite the docs to match the actual public surface.
- Keep one source-of-truth document for runtime semantics and link everything else to it.
- Add a lightweight docs-to-code consistency check in CI for exported public symbols mentioned in docs.

### 5. Medium: the CLI has exceeded the repo's own maintainability boundary

Files:
- [CLAUDE.md](/home/turiya/projects/cuecard/CLAUDE.md#L91)
- [src/cuecard/cli.py](/home/turiya/projects/cuecard/src/cuecard/cli.py)

What is wrong:
- The project convention says small files should stay under 400 lines.
- `src/cuecard/cli.py` is 950 lines.

Why this matters:
- This file now contains setup, indexing, retrieval, formatting, rule management, hook installation, export, and eval orchestration.
- Large command modules are where scope-specific bugs like the cache inconsistency above tend to accumulate because behavior is duplicated across commands.

Impact:
- Harder review and refactoring.
- Higher chance of behavior drift between commands.
- Tests remain green while architectural inconsistencies grow.

Recommended fix:
- Split the CLI into command modules by domain: setup/index, retrieval, rules, hook integration, eval/export.
- Extract shared cache-loading and config-resolution logic into a single internal service module.

## Secondary Notes

- The project is testing the current implementation well, but it is not testing the architecture it claims to have.
- I did not find failing static analysis or failing tests in the reviewed environment.
- The two test warnings are worth keeping visible, but they are not the primary problems:
  - `RuntimeWarning` when executing `cuecard.adapters.claude_code` via `runpy`
  - expected warning for permissive file mode in `secure_open(...)`

## Suggested Remediation Order

1. Fix cache scoping and project isolation first.
2. Implement or remove the `load_or_build` freshness contract.
3. Wire `normalize_query()` into real retrieval.
4. Update docs to reflect the corrected runtime model.
5. Refactor `cli.py` after the runtime behavior is settled.

## Bottom Line

The codebase is not suffering from low quality in the usual sense; it is suffering from contract drift. Tests prove the current implementation is internally consistent, but the implementation is not consistent with the documented product model around project scoping and freshness. Those two gaps should be treated as release blockers for a production plugin.
