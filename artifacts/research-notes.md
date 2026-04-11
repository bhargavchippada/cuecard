# Research Notes

This file consolidates benchmark, analysis, and review conclusions that are
still worth keeping as historical context.

## Benchmark and Model Findings

### End-to-end benchmarking
- Shared-corpus model comparisons were misleading; true end-to-end evaluation is
  the correct methodology because expansion generation changes the candidate pool
  before reranking.
- Gemma 4 E4B showed strong workflow performance and high positive recall once
  `llama.cpp` support was added.

### Closed-loop hook baseline
- The first embedding-only benchmark for the five-event hook system established
  a useful baseline but also showed very high noise across events without LLM
  reranking.

### Negative-silence forensics
- Gemma over-fired on some negatives by treating topical relation as sufficient
  applicability.
- A repeated pattern was reasoning from plausible consequence rather than from
  the actual evidence present in the event payload.

## Retrieval and Reranker Analysis

### Retrieval strategy
- The main bottleneck was upstream candidate generation, not raw LLM reasoning
  capacity.
- cuecard behaves like a tiny-corpus, high-ambiguity, high-abstention ranking
  problem, which favors routing, expansion quality, and multi-view retrieval.

### Reranker prompt analysis
- The instruction "when in doubt, include" became harmful as the rule corpus
  grew because it encouraged tangential matches and inflated noise.
- The reranker needed tighter abstention behavior once the candidate set started
  spanning many semantic neighborhoods.

### LLM reranker loss patterns
- Some rule families, especially type hints and resource cleanup, were routinely
  under-retrieved because the prompt underrepresented those classes of examples.
- Embedded code content was often not being used strongly enough to justify
  secondary matches.

### Query expansion stage analysis
- Query expansion was implemented, but the latest per-stage analysis showed it
  regressed overall quality in its current form.
- That made the feature experimental rather than production-ready despite being
  wired into the benchmark path.

## Rule and Rewrite Analysis

### Rule category review
- Early category review concluded that several rules originally marked
  `tool_use` should actually have broader workflow relevance.
- Later trigger-aware work refined this further around a minimum-retrieval-scope
  principle rather than broad topical relevance.

### Rewrite quality review
- The strongest rewrite guidance was to avoid fragile enumerations of specific
  commands and to prefer compact, general triggers.
- Good rewrites improved retrievability without turning rules into long,
  tool-specific prompt engineering.

## Static Code Review on 2026-04-10

- The review found the strongest current risk in query expansion timeout
  handling: the interface implied a two-second cap, but the implementation could
  still block for the full local LLM timeout.
- It also noted stale prompt tests, benchmark overclaim risk, and observability
  drift after moving to expansion-heavy indexes.

## Research / Paper Direction

- The most interesting research direction preserved here is the rule-rationale
  finding: consequence-bearing rules seemed to improve compliance more than
  stronger imperative framing.
- That remains a design insight and possible paper topic, not a shipped product
  feature by itself.
