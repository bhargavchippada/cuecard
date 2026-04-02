# Post-Fix Review And Golden Eval

Date: 2026-04-02

## Scope

Reviewed:
- runtime seams updated from the prior review
- relevant docs (`README.md`, `CLAUDE.md`)
- eval command and harness
- golden-fixture benchmarks on the current implementation

Validation run:
- `uv run ruff check src tests` -> passed
- `uv run mypy src` -> passed
- `uv run pytest -q` -> `763 passed, 2 warnings in 2.07s`

Golden fixture sets evaluated:
- `basic.json` (354)
- `workflow.json` (84)
- `combined.json` (438)
- `mined-sessions.json` (69)
- `mined-sessions-v2.json` (80)
- unified aggregate `golden_587` = `combined + mined + mined_v2` (587 unique fixture IDs)

Models evaluated in `embedding` mode:
- `BAAI/bge-small-en-v1.5`
- `jinaai/jina-embeddings-v2-base-code`

Additional validation:
- real `rerank` run on unified `golden_587` with `BAAI/bge-small-en-v1.5`
- unified-index ablation for `sparse_enabled` on `golden_587`

## Executive Summary

The previously reported correctness issues look fixed. The important lifecycle and routing seams now line up with the enriched-retrieval design:
- `cuecard index` now merges cached `rules.json` before rebuilding ([src/cuecard/cli.py](/home/turiya/projects/cuecard/src/cuecard/cli.py#L341))
- CLI retrieval and formatting now always go through `run_pipeline()` ([src/cuecard/cli.py](/home/turiya/projects/cuecard/src/cuecard/cli.py#L400), [src/cuecard/cli.py](/home/turiya/projects/cuecard/src/cuecard/cli.py#L432))
- the Claude Code adapter now always uses the pipeline path ([src/cuecard/adapters/claude_code.py](/home/turiya/projects/cuecard/src/cuecard/adapters/claude_code.py#L88))
- eval config now includes `sparse_enabled` and `fusion_k` and always exercises the real pipeline ([src/cuecard/eval.py](/home/turiya/projects/cuecard/src/cuecard/eval.py#L277), [src/cuecard/eval.py](/home/turiya/projects/cuecard/src/cuecard/eval.py#L356))
- `rules expand` now refreshes from source files before merging cached expansions ([src/cuecard/cli_rules.py](/home/turiya/projects/cuecard/src/cuecard/cli_rules.py#L245))
- hook status now checks both global and project indexes ([src/cuecard/cli_hooks.py](/home/turiya/projects/cuecard/src/cuecard/cli_hooks.py#L181))

The remaining problems are no longer basic wiring bugs. They are now concentrated in two places:
1. the eval CLI still fails against the repository’s own fixture layout unless `--corpus-dir eval/corpora` is supplied
2. the live unified-index quality is still well below target, especially on hard fixtures and negative silence

## Findings

### 1. High: unified-index quality is still far from acceptable on the full golden set

This is the main remaining product issue.

On the full 587-fixture unified benchmark using both corpora as one index (`rules_basic.txt + rules_workflow.txt`), current `embedding` mode is still dominated by cross-domain noise and poor abstention:

| Model | Recall | Precision | Noise | Neg Silence | Hard Recall | p50 latency |
|---|---:|---:|---:|---:|---:|---:|
| BGE-small | 0.339 | 0.105 | 0.895 | 0.000 | 0.310 | 3.9 ms |
| Jina code | 0.343 | 0.120 | 0.861 | 0.038 | 0.303 | 17.3 ms |

Implications:
- hard recall is still only about 30%
- negative silence is effectively broken in the live mixed-corpus setup
- the system is still injecting mostly irrelevant rules on many negative queries

A few rules dominate negative false positives and they are mostly generic workflow/meta rules rather than code-specific guidance. On the unified negative set, frequent false positives included:
- `Update README when user-facing behavior, setup steps, or dependencies change`
- `Update CLAUDE.md when project structure, architecture, or conventions change`
- `Never implement a feature without reading 2-3 similar files first`
- `Search GitHub for existing implementations and proven patterns before writing from scratch`
- `Use conventional commit format`

That pattern strongly suggests Stage 1 still lacks enough routing or abstention to keep workflow/process rules out of unrelated code/tool queries.

### 2. High: `cuecard eval` defaults to the wrong corpus directory for this repo’s actual layout

Files:
- [src/cuecard/cli_eval.py](/home/turiya/projects/cuecard/src/cuecard/cli_eval.py#L53)
- [tests/test_cli.py](/home/turiya/projects/cuecard/tests/test_cli.py#L1430)

`cuecard eval` resolves `corpus_dir` to `fixture_file.parent` when `--corpus-dir` is omitted. In this repository, fixtures live in `eval/fixtures/` and corpora live in `eval/corpora/`, so the default command path fails on the project’s own benchmark data.

Concrete reproduction:

```python
FileNotFoundError: [Errno 2] No such file or directory: 'eval/fixtures/rules_basic.txt'
```

This is not just a missing convenience feature; the current default test explicitly codifies the wrong behavior for the repo layout.

Impact:
- `cuecard eval eval/fixtures/basic.json` is broken unless the caller already knows to add `--corpus-dir eval/corpora`
- that makes the main benchmark path easy to misuse
- it also explains why eval CLI coverage did not catch real-data issues

### 3. Medium: sparse retrieval is still a tradeoff, not a clear default win, on the unified corpus

On unified `golden_587` with `jinaai/jina-embeddings-v2-base-code`:

| Mode | Recall | Precision | Noise | Neg Silence |
|---|---:|---:|---:|---:|
| Dense only (`sparse_enabled=false`) | 0.323 | 0.123 | 0.847 | 0.055 |
| Hybrid (`sparse_enabled=true`) | 0.343 | 0.120 | 0.861 | 0.038 |

Hybrid buys about +2.0 recall points, but it makes precision, noise, and negative silence worse.

On BGE-small, sparse retrieval changes the picture much less because dense retrieval is already over-firing on negatives, but it still does not solve abstention.

This means `sparse_enabled = true` is not yet supported by strong evidence as a universal default for the unified live index. It is helping recall, but the cost is real and visible in the benchmark.

### 4. Medium: shipped default model and public-facing docs still imply different runtime behavior

Files:
- [src/cuecard/cli.py](/home/turiya/projects/cuecard/src/cuecard/cli.py#L31)
- [src/cuecard/config.py](/home/turiya/projects/cuecard/src/cuecard/config.py#L50)
- [README.md](/home/turiya/projects/cuecard/README.md#L104)
- [CLAUDE.md](/home/turiya/projects/cuecard/CLAUDE.md#L86)

Code default:
- `BAAI/bge-small-en-v1.5`

README config example:
- `jinaai/jina-embeddings-v2-base-code`

This matters because the two models behave differently on the same 587-fixture benchmark:
- BGE-small gives slightly higher recall in the per-corpus run, but very poor abstention and worse noise in the unified runtime scenario
- Jina code gives lower recall in some slices, but better precision/noise and some negative silence

The docs are no longer just describing interchangeable options. They are implying meaningfully different operational behavior.

### 5. Medium: Stage 2 reranking does not fix the unified-index quality problem

Real `rerank` run on unified `golden_587` with BGE-small:

| Mode | Recall | Precision | Noise | Neg Silence | Hard Recall | p50 latency |
|---|---:|---:|---:|---:|---:|---:|
| Embedding | 0.339 | 0.105 | 0.895 | 0.000 | 0.310 | 3.9 ms |
| Rerank | 0.358 | 0.115 | 0.885 | 0.000 | 0.390 | 128.8 ms |

This validates that the rerank path now runs for real, which is good.

But it also shows the actual quality story:
- reranking improves hard recall materially
- it does not restore negative silence
- it still leaves overall precision/noise far from acceptable
- latency increases by roughly 33x at p50

So the current bottleneck remains Stage 1 candidate quality and mixed-corpus routing, not the absence of Stage 2.

### 6. Low: the current tests still miss the most important realistic eval path

The suite is large and healthy overall, but there is still no real integration test that covers:
- actual fixture files in `eval/fixtures/`
- actual corpora in `eval/corpora/`
- the CLI default corpus resolution path
- a quality sanity check on the mixed-corpus golden benchmark

That gap matters more now than raw line coverage because the biggest remaining issues are behavioral and dataset-level.

## Benchmark Notes

### Per-corpus vs unified

The per-fixture-corpus benchmark is noticeably better than the unified-index benchmark.

`golden_587`, per designated corpus:

| Model | Recall | Precision | Noise | Neg Silence | Hard Recall |
|---|---:|---:|---:|---:|---:|
| BGE-small | 0.402 | 0.127 | 0.873 | 0.000 | 0.464 |
| Jina code | 0.385 | 0.164 | 0.767 | 0.145 | 0.337 |

`golden_587`, unified index:

| Model | Recall | Precision | Noise | Neg Silence | Hard Recall |
|---|---:|---:|---:|---:|---:|
| BGE-small | 0.339 | 0.105 | 0.895 | 0.000 | 0.310 |
| Jina code | 0.343 | 0.120 | 0.861 | 0.038 | 0.303 |

That gap is important. The benchmark degradation is not just “model quality”; it is strongly tied to the mixed-corpus retrieval setup used by the real runtime.

### Categories still missing on hard fixtures

Across the hard zero-recall cases, the most common missed rule families were:
- explicit error handling / no silent failure
- dependency review and security checks
- resource cleanup / close connections and sessions
- env vars / secrets handling
- evidence-first workflow rules for experimentation

Those are not fringe cases. They are central policy categories, which is another reason the current hard-tier numbers are still a blocker.

### Query truncation

Queries longer than 500 chars occurred in 14 of 587 fixtures, mostly medium/hard. This is not the main issue, but it is worth tracking because long multi-file or heredoc-style contexts are overrepresented in the hard tier.

## Suggestions

1. Fix `cuecard eval` so the repo’s default golden benchmark works without extra flags. At minimum, it should discover `eval/corpora/` when the fixture file is under `eval/fixtures/`.
2. Add one real CLI integration test that runs `cuecard eval` against the repo fixture/corpus layout instead of patching `run_eval()`.
3. Add a benchmark gate for unified `golden_587` in CI or release review. The minimum useful checks are hard recall, noise ratio, and negative silence.
4. Treat mixed-corpus routing as the next retrieval problem to solve. The current false positives are dominated by generic workflow rules, which means better routing or partitioning is likely higher leverage than more reranking.
5. Re-evaluate the default embedding model explicitly. The current docs and code default do not describe the same operating point, and the benchmark tradeoff is large enough that this should be an intentional product decision.
6. Keep `sparse_enabled` empirical. Right now it is a recall knob with visible quality cost, not an unambiguous improvement.

## Bottom Line

The implementation fixes from the previous review appear to be in place and working. The remaining issues are now mostly about real-world evaluation quality and one still-broken eval CLI default.

The most important fact from this pass is simple: the runtime is now wired correctly enough that the benchmarks are meaningful, and those benchmarks still show that the unified retrieval setup is well short of the target on hard recall, noise, and negative silence.
