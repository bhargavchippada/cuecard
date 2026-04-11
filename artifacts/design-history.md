# Design History

This file consolidates the durable design direction from the PRD artifacts.

## Core Product

### `prd-v1`
- Established cuecard as an agent-agnostic rule-retrieval library that injects
  the right rules at the right moment instead of front-loading all guidance into
  prompt context.
- Set the baseline product goals: fast retrieval, hook integration, offline
  evaluation, inspectable pipeline stages, and strong security defaults.

## Retrieval Architecture Evolution

### Multi-stage retrieval
- Added the three-stage retrieval model: embeddings first, optional
  cross-encoder second, optional LLM reranker last.
- Preserved the principle that lightweight stages should do as much routing work
  as possible before expensive LLM reasoning.

### Enriched retrieval
- Introduced structured rule metadata, richer rule representations, and
  multi-retriever fusion to improve candidate recall without paying LLM latency
  on the hot path.
- Framed the retrieval problem as a small-corpus, high-ambiguity ranking task
  where fusion and routing matter more than simply swapping in a larger dense
  model.

### Unified event system
- Generalized the original single-event architecture into a unified event-aware
  system spanning tool actions and workflow turns.
- Established the idea that retrieval quality depends heavily on event context,
  not just semantic similarity over raw text.

## Hook-System Expansion

### Phase 4: Closed-loop hooks
- Extended cuecard from advisory `PreToolUse` injection into a closed-loop
  system that could advise before actions, verify after actions, propagate
  instructions to subagents, and audit end-of-turn behavior.
- Added TOML rule metadata, event affinity inference, and event masking as core
  mechanisms for keeping rule retrieval scoped.

### Phase 6: Completion gate
- Added a Stop-hook design that blocks premature termination, retrieves relevant
  completion criteria from the user request and transcript, and gives the agent
  bounded opportunities to finish the work before allowing stop.
- Kept the hook itself non-LLM to preserve predictable latency.

## Corpus and Rule-Quality Evolution

### Phase 7: Trigger-aware rewriting
- Shifted rule writing from "what to do" to "when this applies, do X because Y."
- Made trigger conditions a first-class retrieval signal because the embedding
  layer only sees the rule text and its expansions.
- Treated event category assignment and trigger phrasing as part of the retrieval
  system, not just documentation hygiene.

### Rule-rewrite review direction
- Preferred concise, language-agnostic triggers over long enumerations of tools
  or libraries.
- Avoided over-specification when examples would narrow retrieval too much or go
  stale quickly.

## Query/Expansion Work

### Phase 8: unified expansion prompt and tag matching
- Developed the idea that vocabulary mismatch, not just model weakness, drives a
  large fraction of misses.
- Identified tag-to-tag matching as a strong bridge between abstract rule text
  and concrete user/tool phrasing.
- Extended the same expansion logic to both the rule side and the query side,
  while keeping the LLM reranker grounded on the raw query.

## Deferred / Historical Direction

### Phase 5: multi-source parsing
- Proposed indexing Markdown, YAML skill files, and project documentation
  without special formatting by chunking them at semantic boundaries.
- This remained a draft direction rather than a converged implementation plan.

## Design Principles That Survived

- Route and filter early; reason late.
- Retrieval must be event-aware.
- Rules need explicit trigger language to be retrievable.
- The LLM reranker is valuable, but only after the candidate set is small and
  clean.
- Evaluation artifacts and fixture quality are part of the product, not
  incidental support work.
