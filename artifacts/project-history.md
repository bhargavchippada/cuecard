# Project History

This file consolidates the durable outcomes from sessions 20-32.

## 2026-04-04 to 2026-04-06

### Session 20
- Corrected the benchmark methodology from shared-corpus comparisons to true
  end-to-end evaluation where each model generates its own expansions and does
  its own reranking.
- Found that the methodology change flipped the apparent model ranking: the
  shared-corpus setup overstated 35B's advantage, while end-to-end evaluation
  showed 9B performing better overall.
- Evaluated Gemma 4 E4B, upgraded `llama.cpp` for `gemma4` support, and found
  that Gemma delivered the strongest workflow quality and much better positive
  recall.
- Ran prompt-tuning experiments on Qwen 9B and Gemma, then reverted them after
  quality regressions or instability.
- Designed the rule-rationale paper concept and documented the central finding:
  consequence-bearing rules improved compliance more than stronger authority
  framing alone.

### Session 21
- Switched production serving from Qwen 9B to Gemma E4B.
- Fixed daemon and hook reliability issues, including timeout handling, faster
  hook dispatch, safer shutdown behavior, immediate port reuse, and input/output
  event consistency.
- Completed two rounds of code review and resolved the remaining code and
  security issues for the production daemon path.
- Split oversized tests into smaller files and updated the global rule about
  file-size limits from 400 to 800 lines preferred.

## 2026-04-06 to 2026-04-07

### Session 22
- Cleaned duplicated sections from `SOUL.md`.
- Built the separate `alphaloom` project end to end, including PRD refinement,
  implementation across eight phases, review convergence, and an eight-bug E2E
  cleanup cycle. This session is primarily historical context rather than active
  cuecard product work.

### Session 23
- Designed and implemented the closed-loop hook system for five events:
  `PreToolUse`, `UserPromptSubmit`, `PostToolUse`, `Stop`, and
  `SubagentStart`.
- Added event/tool affinity inference, event masking, hook adapters, expanded
  eval fixtures, and per-event metrics.
- Completed code and security review convergence for the new hook path.
- Established the first event-by-event benchmark baseline for the closed-loop
  architecture.

### Session 24
- Expanded the eval corpus from 32 to 109 rules.
- Expanded fixtures from 500 to 831 across all five events.
- Ran multi-round fixture verification, fixed corpus-text mismatches, restored
- dropped rules, remapped legacy texts, and cleaned unmatched references.
- Benchmarked the expanded corpus after the verification pass.

## 2026-04-08

### Session 25
- Expanded global rules from 87 to 94 with new classification/eval rules.
- Converged the Phase 6 completion-gate PRD.
- Performed a realism overhaul on fixtures, especially `Stop`,
  `SubagentStart`, and workflow fixtures, to better reflect real queries and
  better positive/negative balance.
- Corrected affinity ground truth, which materially improved affinity accuracy.

### Session 26
- Activated the v2 trigger-aware corpus and updated tagged rule metadata.
- Rewrote the affinity prompt around a "minimum retrieval scope" principle and
  materially improved affinity accuracy.
- Rebalanced fixture distributions, removed cross-event leakage, and continued
  rewriting stop fixtures into realistic completion-gate format.
- Confirmed that poor recall was primarily an embedding/expansion problem rather
  than an LLM reranker bottleneck.

## 2026-04-08 to 2026-04-09

### Session 28
- Completed a full code review and cleanup pass over the source tree.
- Centralized defaults, validators, and operational constants into
  `ResolvedConfig` and `models.py`.
- Added several new config fields for expansion and reranker tuning.
- Hardened secret scrubbing, OpenAI key detection, loopback validation, and
  exception logging.

### Session 29
- Reduced test runtime from roughly 11.3s to 4.3s through better mocks and
  shared fixtures.
- Added a network guard to stop accidental real-network access during tests.
- Enforced a one-second timeout per non-slow test.
- Restructured source and test directories into clearer subpackages.
- Removed hidden parameter defaults from retrieval/expansion paths so callers
  always pass explicit config.

## 2026-04-09 to 2026-04-10

### Session 30
- Optimized benchmark infrastructure by sharing affinity inference and adding
  parallel LLM work.
- Made expansion generation affinity-aware, which corrected incorrect expansion
  styles for workflow-only events.
- Analyzed expansion failures and identified two main classes: cross-language
  vocabulary gaps and semantic reasoning gaps.
- Found that tag-to-tag matching was a very strong bridge across vocabulary
  variation.
- Confirmed that the embedding pre-filter is load-bearing for the LLM reranker;
  sending too many candidates to the LLM degraded quality.
- Ran a broad fixture audit and corrected many mislabeled cases.

### Session 31
- Implemented query expansion in the retrieval pipeline.
- Upgraded the rule expansion prompt to produce balanced abstract and specific
  expansions.
- Extended `bench_e2e.py` with query-expansion and `llm_candidates` controls.
- Performed stage-by-stage analysis and found that query expansion, in its
  current form, regressed overall quality rather than improving it.

### Session 32
- Removed `PostToolUse` from the active event set because it was over-triggering
  under the current retrieval architecture and `PreToolUse` already covered the
  highest-value prevention path.
- Updated source, tests, fixtures, corpora, and `CLAUDE.md` to reflect a
  four-event model.
- Renamed `basic.json` to `pre_tool_use.json` and merged mined stop fixtures
  into the canonical `stop.json`.

## Current Throughline

- Retrieval quality improvements came more from better corpus/fixture design,
  event routing, and candidate discipline than from simply increasing model
  size.
- Trigger-aware rules and explicit event scope were major turning points in
  reducing noise.
- Query expansion is implemented but still needs stricter control because the
  measured quality impact was negative in the latest evaluation.
