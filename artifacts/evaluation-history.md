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

---

## Sessions 33–35 — prompt & fixture iteration on pre_tool_use

These entries were moved out of `CLAUDE.md` during the 2026-04-18 CLAUDE.md
trim. They are preserved verbatim as the source of truth for how the
session-35 baseline was reached.

### pre_tool_use progression (20% sample, seed=42, n=86)

| Run | F2 | PosRecall | Noise | NegSil |
|---|---:|---:|---:|---:|
| s33 promptv3 | 0.662 | 0.764 | 0.391 | 0.634 |
| s34 promptv3 + 11 fixture fixes | 0.691 | 0.893 | 0.386 | 0.561 |
| s34 promptv4 (new prompt) | 0.780 | 0.785 | 0.230 | 0.829 |
| s34 promptv4 + 10 more fixture fixes | 0.757 | 0.659 | 0.239 | 0.829 |
| s35 promptv5 (tight persona, 7 ex) | 0.663 | 0.567 | 0.277 | 0.732 |
| s35 promptv5b (tight + 11 ex) | 0.707 | 0.651 | 0.266 | 0.732 |
| s35 promptv5d (v4 HOW TO DECIDE + persona + 13 ex) | 0.780 | 0.662 | 0.179 | 0.878 |
| s35 promptv5d + cache flags | 0.794 | 0.685 | 0.200 | 0.902 |
| s35 promptv5wf (+ workflow principles + 4 ex) | 0.795 | 0.696 | 0.221 | 0.854 |

### Session 34 wins (pre_tool_use)
- NegSil +19.5 pts (0.634 → 0.829) — reranker now silences congratulatory
  fires, literal-grep misrouting, read-is-diagnostic, trivial-edit reflex,
  .md-is-not-a-module.
- Noise −15.2 pts (0.391 → 0.239) — same prompt rewrite.
- 21 fixtures hand-audited and corrected across two rounds.

### Session 35 wins (pre_tool_use)
- F2 +0.037 (0.757 → 0.794) — "Core philosophy" persona paragraph added on
  top of v4's full HOW TO DECIDE + 13 examples.
- NegSil +7.3 pts (0.829 → 0.902) — persona calls out "preventive, not
  congratulatory", "reads are diagnostic", "trivial edits are not new APIs",
  "docs ≠ code modules", "localhost ≠ production".
- Noise −3.9 pts (0.239 → 0.200); PosRecall +2.6 pts (0.659 → 0.685).
- Prompt caching on llama-server (`--cache-reuse 256 --ctx-checkpoints 64`
  + client `cache_prompt: true`) — `prompt_ms` drops ~4x on warm prefix.
- Failed experiment (v5/v5b): tight persona + short HOW TO DECIDE + fewer
  examples = -0.094 F2. Long-form principles and anchored examples are
  complementary, not redundant, at this model scale.

### Remaining gaps from session 35
- `hardcoded config → use env vars` rule has a fuzzy trigger — fires on
  YAML secrets files (correct) but also on timeout constants and
  `.env.example` (incorrect). Rule text needs sharpening, not prompt tuning.
- `read_similar` on new code modules is LLM-inconsistent — prompt rule 6
  ("trivial edit") may be over-applied to test-file writes.
- `multi-tool-*` fixtures miss "review deps for vulnerabilities" — the LLM
  chains rules poorly across `&&`-joined commands.

---

## 2026-04-11 — full-set fixture audits

Three sequential 100% audit passes, executed after moving to traced
artifacts. Artifact directories under `eval/results/`:

1. `gemma-e4b-s32-full100-initial-traced-seed42/` — initial baseline.
2. `gemma-e4b-s32-full100-fixed-traced-seed42/` — after bad fixtures
   removed.
3. `gemma-e4b-s32-full100-postprompt-fixed2-traced-seed42/` — after
   workflow prompt tuning + round-2 fixture cleanup.

### Methodology

- Run all 4 tiers on 100% with traced artifacts first.
- Audit disagreements from `traces/{tier}/{fixture_id}.json`, prioritizing
  `unknown`, contradictory goldens, and positives with empty `should_match`.
- Remove fixtures that were intrinsically bad baselines rather than real
  model checks.
- Add missing expectations only when the user prompt plainly implied the
  rule.
- Keep the baseline artifact intact; rerun with a distinct `--suffix` so
  before/after remain comparable.

### Round 1 — full-set audit (full100-fixed)

Fixture changes applied:
- `pre_tool_use`: removed 9 bad positives with no defensible rule trigger;
  converted `mined-whisper-edit-test` into a mocking expectation.
- `user_prompt_submit`: fixed contradictory `prompt-rule-neg-has-consequences`
  (same rule in both SM and SNM); replaced bad coverage expectation with
  test-plan/review-gate expectations.
- `stop`: removed observational/bad-stop fixtures; filled missing
  expectations for commit/push, compact/save-state, docs updates,
  dependency adoption, PRD review, hook-debugging, mutmut convergence,
  and CLAUDE.md updates.
- `subagent_start`: widened under-specified goldens to include task
  classification, agent-selection, PRD-first, conventional-commit, and
  dependency-audit expectations; converted `sub-credential-manager` from
  bad negative to positive.

| Tier | N init | F2 init | N fixed | F2 fixed | PosRecall | Noise | NegSil |
|---|---:|---:|---:|---:|---:|---:|---:|
| pre_tool_use | 441 | 0.696 | 432 | 0.705 | 0.638 | 0.233 | 0.817 |
| user_prompt_submit | 219 | 0.658 | 219 | 0.615 | 0.456 | 0.329 | 0.772 |
| stop | 176 | 0.490 | 164 | 0.603 | 0.656 | 0.438 | 0.627 |
| subagent_start | 170 | 0.406 | 170 | 0.485 | 0.686 | 0.561 | 0.394 |

Interpretation:
- `user_prompt_submit` got stricter and dropped from 0.658 to 0.615 — not a
  regression but the new goldens exposed genuine retrieval/reranker weakness.
  Treat `0.615` as the more honest baseline.
- `stop` improved materially once empty/underspecified goldens were
  cleaned up.
- `subagent_start` improved but still has substantial overfire; fixture
  fixes alone did not solve the planner/general-purpose confusion.

### Workflow prompt tuning follow-up

Targeted 100% traced runs after the round-1 audit. Baseline:
`full100-fixed`. Three prompt-change passes (`workflowfix1..3`).

Prompt changes tested:
- Adapter-side `UserPromptSubmit` hinting for terse workflow prompts
  (compaction, delegation, full-dataset benchmarking, schema-change,
  quality-debugging).
- Reranker principles: keep complementary safeguards together on workflow
  events; if the user is trying to skip a safeguard, return the safeguard.
- Broad workflow-oriented expansion examples — tested then partially
  reverted after they widened retrieval neighborhoods too much.

| Tier | Baseline F2 | wf1 | wf2 | wf3 | Baseline PosRecall | wf3 PosRecall |
|---|---:|---:|---:|---:|---:|---:|
| user_prompt_submit | 0.615 | 0.649 | 0.657 | 0.640 | 0.456 | 0.532 |
| stop | 0.603 | 0.577 | 0.592 | 0.565 | 0.656 | 0.643 |
| subagent_start | 0.485 | 0.446 | 0.363 | 0.430 | 0.686 | 0.680 |

What the experiments showed:
- `UserPromptSubmit` benefits from query-side help. `workflowfix3` improved
  F2 0.615 → 0.640 and PosRecall 0.456 → 0.532 on the audited 100% set.
- Broad workflow expansion prompt changes are not safe globally. They
  improved `UserPromptSubmit` retrieval but caused cross-tier overfire,
  especially on `SubagentStart`.
- `Stop` is sensitive to retrospective/summary broadening. Adapter-side
  `Stop` hint inflation raised recall but also noise.
- Reranker guardrails alone are not sufficient to offset retrieval
  broadening on `Stop` and `SubagentStart`.

Working conclusion:
- Good direction: narrow `UserPromptSubmit`-only retrieval help for terse
  workflow prompts.
- Bad direction: globally teaching the shared expansion index to associate
  generic workflow phrases with delegation/compaction/schema/debugging
  rules.
- `SubagentStart` remains the main blocker and needs either finer event
  affinity or a dedicated prompt/query strategy.

### Round 2 — targeted golden cleanup (postprompt-fixed2)

Fixed only remaining disagreements that were genuine golden-quality issues:
- `pre_tool_use`: removed already-compliant positives (`docker-run-env-file`,
  `edit-python-file`); corrected `mark-slow-neg-already-marked` to expect the
  slow/e2e-marking rule.
- `user_prompt_submit`: removed over-strong expectations
  (`wf-medium-save-state`, `wf-easy-write-tests-first`); added missing TDD
  expectation to `three-layer-quality-gate-user`.
- `subagent_start`: removed already-compliant positives
  (`sub-neg-search-agent`, `sub-neg-architect-caching-strategy`); added
  real-data validation rule to `sub-builder-mocked-tests-done`;
  added per-milestone convergence-review rule to
  `sub-ralph-convergence-review-phase3`.

| Tier | Postprompt F2 | +Fixture F2 | PosRecall | Noise | NegSil |
|---|---:|---:|---:|---:|---:|
| pre_tool_use | 0.717 | 0.719 | 0.626 | 0.203 | 0.852 |
| user_prompt_submit | 0.632 | 0.686 | 0.579 | 0.306 | 0.816 |
| stop | 0.601 | 0.577 | 0.728 | 0.485 | 0.533 |
| subagent_start | 0.412 | 0.405 | 0.640 | 0.649 | 0.298 |

Net result versus the earlier audited baseline (`full100-fixed`):
- `pre_tool_use`: 0.705 → 0.719
- `user_prompt_submit`: 0.615 → 0.686
- `stop`: 0.603 → 0.577
- `subagent_start`: 0.485 → 0.405

Operational takeaway:
- Current `master` is a net win for `PreToolUse` and especially
  `UserPromptSubmit`.
- `Stop` and `SubagentStart` still need separate event-specific work; more
  shared workflow prompt broadening is unlikely to solve them cleanly.
