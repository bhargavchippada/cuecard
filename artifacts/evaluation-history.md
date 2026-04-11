# Evaluation History

This file consolidates the durable fixture work and the main lessons from the
verification artifacts.

## Corpus Expansion Fallout

- The move from the original small corpus to the larger global corpus created a
  temporary mismatch between fixture expectations and corpus text.
- The central failure mode was exact-text drift: fixtures referenced old short
  rule text while evaluation compared against new full-form rule text.
- Some older rules had also disappeared during intermediate corpus revisions,
  which made valid positives impossible to retrieve until the corpus was fixed.

## Major Verification Findings

### Cross-file consistency
- All fixture families eventually had to be remapped to the canonical
  `rules_global.txt` wording.
- Workflow fixtures were the most affected because the old workflow corpus had
  been merged into the global corpus with different wording and rationale
  suffixes.

### Compliance vs violation confusion
- A recurring labeling error treated visible compliance as a positive violation.
- Examples included running `ruff`, creating a branch, or showing properly
  structured async code and then expecting those fixtures to match "fix this"
  rules.
- Cleaning these cases materially improved negative-silence accuracy and made
  evaluation better aligned with actual hook behavior.

### Cross-event leakage
- Many generic coding rules had been expected in workflow-only events.
- The later trigger-aware cleanup removed a large number of these cross-event
  expectations so that `UserPromptSubmit`, `Stop`, and `SubagentStart` focused
  on process rules while tool-time checks handled direct code/tool constraints.

### Over-specified positives
- `SubagentStart` positives were originally overloaded with too many expected
  rules, making perfect recall impossible under the reranker's output cap.
- Trimming fixtures to the one or two most task-specific rules made the metric
  far more meaningful.

## Key Cleanup Passes

### Audit and round-1 verification
- Identified wrong expectations, missing expectations, corpus mismatches, and
  dropped-rule references.
- Established that fixture maintenance had to include both corpus-level fixes
  and fixture-level edits.

### Round-2 verification
- Confirmed that the expanded 109-rule corpus restored previously dropped rules.
- Applied many fixture-level additions, removals, and reclassifications.

### Round-3 verification
- Focused on realistic positives and negatives, especially around compliance
  versus violation and event-appropriate rule selection.
- Left `Stop` largely stable and concentrated changes on `basic`,
  `post_tool_use`, `workflow`, and `subagent_start`.

### Realism overhaul
- Rebalanced the positive/negative mix across event types.
- Made `Stop` fixtures look more like actual completion-gate inputs.
- Added many realistic negatives for workflow and subagent events.

## Durable Outcomes

- Fixture quality is a first-order determinant of benchmark credibility.
- Negative fixtures need the same scrutiny as positives; otherwise the system
  gets punished for correct abstention or correct retrieval.
- Event-specific scope matters as much in fixture authoring as it does in rule
  authoring.
- Canonical fixture schemas should stay minimal; experimental keys should not
  leak into long-term datasets.

## Current State

- Historical verification notes were consolidated here because the individual
  round reports were useful during corpus migration but no longer need to live
  as separate files.
- The active source of truth for fixtures now lives under `eval/fixtures/`, not
  in this directory.
