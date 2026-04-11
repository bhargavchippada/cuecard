# Session 35 Progress

**Date:** 2026-04-10
**Branch:** master
**Head at compact:** will be the commit immediately after this write

## Headline wins

1. **promptv5d + cache flags**: F2 on pre_tool_use 0.757 → **0.794** (+0.037), NegSil +4.9 pts. Added the "Core philosophy" persona paragraph on top of v4's full HOW TO DECIDE + 13 examples. Committed as `c3d25c4`.
2. **llama-server prompt caching**: pinned `cache_prompt: true` in `call_local`, documented `--cache-reuse 256 --ctx-checkpoints 64` in CLAUDE.md and README. Empirically verified: prompt_ms drops 137→35ms (~4x) on warm prefix.
3. **promptv5wf (workflow reranker additions)**: F2 on user_prompt_submit 0.661 → **0.701** (+0.040), PosRecall 0.347 → 0.611 (+0.264 — MAJOR). Added principles #10 (semantic intent for workflow events) and #11 (avoid topic overlap), plus 4 few-shot examples (14-17). PreToolUse F2 held at 0.795 (no regression). Committed as `2c49acd`.
4. **user_prompt_submit fixture rename**: `workflow.json` → `user_prompt_submit.json`; tier key `"workflow"` → `"user_prompt_submit"` in `bench_e2e.py`. The affinity category "workflow" (tool_use vs workflow rule classification) is untouched — different concept.
5. **UPS fixture audit (iterative)**: Fixed 11 fixtures total across 2 rounds — 5 trimmed ground truth (round 1), 6 flipped between positive/negative based on whether the query text supported the expected rule (round 2).
6. **🎯 Affinity classifier accuracy: 89.7% → 93.5%** (100/107 vs GT tagged corpus). Added 3 generalizable principles to the affinity prompt:
   - Tool mentions in parentheses are hints about *where* the problem manifests, not the trigger
   - Code patterns for LLM/prompt/retry/pipeline handling are tool_use, not workflow
   - "When running X: require/ensure Y" disciplines are usually `both`
   - `both` recall: 0/3 → 3/3 after these changes.

## Current baseline (not yet measured on fresh expansions)

| Tier | N | F2 | PosRecall | Noise | NegSil | p50 |
|---|---:|---:|---:|---:|---:|---:|
| **pre_tool_use** | 86 (20%) | 0.795 | 0.696 | 0.221 | 0.854 | 3429ms |
| **user_prompt_submit** (20%) | 41 | 0.701 | 0.611 | 0.291 | 0.826 | 2954ms |
| **user_prompt_submit** (100%, before fixture flips) | 219 | 0.629 | 0.482 | 0.325 | 0.773 | 3124ms |
| **user_prompt_submit** (100%, after 6 flips) | 219 | 0.624 | 0.479 | 0.324 | 0.779 | 3098ms |
| stop | 33 (20%) | 0.556 (s33) | 0.474 | 0.460 | 0.714 | 2833 |
| subagent_start | 33 (20%) | 0.398 (s33) | 0.857 | 0.704 | 0.211 | 2892 |

## Uncommitted work at compact time (being committed now)

1. `src/cuecard/indexing/expander.py` — workflow branch rewrite:
   - New `specific_guidance` with 6 user framings (imperative, assessment, problem report, help request, opinion, status update)
   - Domain disambiguation principle (ML pipeline vs CI/CD, AI agents vs HTTP user agents)
   - Replaced 3 workflow few-shot examples with framing-diverse versions
   - New examples: "Classify every task" (5 framings), "Before building on unvalidated dependency" (3 framings, with direct imperatives), "When building multi-stage ML pipelines" (grounds domain to ML/retrieval, not CI/CD), "Update README" (mostly unchanged)
2. `src/cuecard/retrieval/affinity.py` — new affinity prompt with 3 critical patterns + 9 worked examples (was 7). The improvement that drove 89.7→93.5%.
3. `tests/indexing/test_expander_prompt.py` — updated two tests to match new workflow examples + framings principle
4. `eval/fixtures/user_prompt_submit.json` — 6 fixtures flipped based on audit (wf-easy-search-existing, wf-easy-save-artifacts, wf-medium-retry-failed, wf-medium-slow-pipeline, prompt-avoid-mutation, prompt-rule-neg-has-consequences changed from negative → positive with appropriate ground-truth rules)
5. `CLAUDE.md` — session 35 affinity result documented

## Not yet done / unfinished

1. **Regenerate workflow expansions** with the new expander prompt (`--force-expand` on labels). This is the next step — the current benchmark baselines still use promptv4-fixfict2's old expansions. Once regenerated, UPS should gain Stage1 recall (56 of 64 positive misses were Stage1 failures).
2. **Run full UPS benchmark with new affinity + new expansions** to measure the end-to-end impact.
3. **Run PreToolUse regression check** with the new affinity (affinity changes may shift some rules from tool_use to workflow, removing them from PreToolUse candidates).
4. **Stop and subagent_start tiers** still on session 33 numbers.
5. **Remaining 7 affinity errors** after 93.5% — edge cases, 1-2 per pattern. Could hit 95%+ with more targeted examples but diminishing returns.
6. **Remaining UPS fixture audit**: the 100% run found 91 problem fixtures (38 FN + 26 partial + 27 FP); I triaged 27 FPs and flipped 6. ~80 more unfixed (most are Stage1 retrieval failures that need expansion regen, not fixture fixes).

## Next session kickoff steps

1. Start llama-server with the current flags (already running, PID 3504247):
   ```
   llama-server -m /home/turiya/models/gemma-4-E4B-it-Q8_0.gguf \
     --port 8081 -ngl 99 -c 98304 --jinja -np 5 --reasoning off \
     --cache-reuse 256 --ctx-checkpoints 64
   ```
2. Regenerate expansions with new prompt:
   ```
   uv run python tools/bench_e2e.py --model-path ... --label gemma-e4b-s32 \
     --suffix promptv5wf-fresh --no-server --force-expand \
     --sample-ratio 1.0 --seed 42 --tiers user_prompt_submit
   ```
3. Expected impact: some portion of the 56 Stage1 misses should resolve (rules now have framing-diverse expansions). If UPS F2 breaks 0.70 on 100%, that's a clear win. If it drops below 0.60, regression — investigate.
4. Then run pre_tool_use with `--force-expand` to verify no regression on that tier.
5. Measure affinity accuracy again on the fresh run (should be 93.5% or better).

## Key files touched this session

- `src/cuecard/retrieval/llm_reranker.py` (session-wide — promptv5d → promptv5wf, 17 examples)
- `src/cuecard/retrieval/llm_utils.py` (cache_prompt pin, seed pin)
- `src/cuecard/indexing/expander.py` (workflow branch rewrite)
- `src/cuecard/retrieval/affinity.py` (3 new principles + examples, drove 89.7→93.5%)
- `eval/fixtures/user_prompt_submit.json` (renamed from workflow.json; 11 fixtures fixed across 2 rounds)
- `tools/bench_e2e.py` (tier rename workflow → user_prompt_submit)
- `tools/notebook.ipynb` (subagent sync, MODEL_NAME match, unused imports cleanup)
- `CLAUDE.md` (session 35 baselines, cache flags doc, affinity result)
- `README.md` (cache flags doc)
- `tests/retrieval/test_llm_utils.py` (cache_prompt test)
- `tests/retrieval/test_llm_reranker_prompt.py` (example assertion updates)
- `tests/indexing/test_expander_prompt.py` (workflow framings + new examples)

## Benchmarks stored in eval/results/

Fresh runs from session 35:
- `gemma-e4b-s32-promptv5-traced-seed42` (failed tight prompt experiment)
- `gemma-e4b-s32-promptv5b-traced-seed42` (failed tight + restored examples)
- `gemma-e4b-s32-promptv5c-traced-seed42` (v5b with v4 exp cache)
- `gemma-e4b-s32-promptv5d-traced-seed42` (v5d with v4 exp cache — +0.023 F2 win)
- `gemma-e4b-s32-promptv5d-rerun-traced-seed42` (v5d fresh exp — stable)
- `gemma-e4b-s32-promptv5d-cached-traced-seed42` (v5d + server cache flags)
- `gemma-e4b-s32-promptv5wf-ups-traced-seed42` (workflow additions, 20% UPS)
- `gemma-e4b-s32-promptv5wf-ptu-traced-seed42` (workflow additions, PreToolUse regression check)
- `gemma-e4b-s32-promptv5wf-ups2-traced-seed42` (20% UPS after first round of fixture flips)
- `gemma-e4b-s32-promptv5wf-ups-full-traced-seed42` (100% UPS baseline)
- `gemma-e4b-s32-promptv5wf-ups-full2-traced-seed42` (100% UPS after 6 fixture flips)
- `gemma-e4b-s32-promptv5wf-ups-full2b-traced-seed42` (in progress when compact started, may not exist)

## Lessons learned (for SOUL.md / memory)

1. **CoT reasoning is load-bearing at Gemma-E4B scale** — trimming reasoning from 145 → 38 tokens saved 644ms but cost −0.081 F2. Model needs the thinking space.
2. **max_workers parallelism masks per-request latency** — benchmark p50 is worst-case under 5-way contention, not real serving latency. A single hook call sees ~900ms, benchmark reports ~3s. Always measure both.
3. **Expansion prompt is the Stage1 ceiling** — 87% of UPS positive failures are "rule not even in top-12 candidates". The LLM reranker works; the embedding layer doesn't. Fix expansions first, reranker second.
4. **Workflow expansions need framing diversity** — "add auth to the API" + "should we document that?" isn't enough. Real user messages come in 6+ framings (imperative, question, problem report, help request, opinion, status update). Cover them all or the embedding misses matches.
5. **Affinity `both` classification was broken** — classifier collapsed everything to tool_use. Adding one strong principle ("When running X: require Y is usually both") fixed all 3 cases. Generalizable principles > memorized examples.
6. **Fixture difficulty changes break stratified sampling** — at 20% sample ratio with seed fixed, changing a fixture's difficulty shifts which bucket it lands in, changing the sample set. Run 100% for authoritative comparisons.
7. **Tool mentions in parens are hints not triggers** — "(Bash: git log, Read): rotate secrets" doesn't mean the rule fires on every git log. It means "when credentials are visible in the output of these tools". Affinity classifier must learn to read past the parenthetical.
