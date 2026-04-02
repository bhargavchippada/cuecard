# Session 12 Progress — Embedding Quality Research

**Date:** 2026-04-01
**Status:** ALL 4 TASKS COMPLETE

## Tasks Completed

### Task 1: SOTA Research
- **Artifact:** `artifacts/research-sota-retrieval.md`
- CodeRankEmbed (nomic-ai, 137M, 768-dim, Apache-2.0): +10.7 MRR over jina-code on CodeSearchNet
- EmbeddingGemma-300m (Google, 308M, ONNX): 68.76 MTEB Code, Gemma license
- BM25 hybrid with RRF k=60 documented (but see Task 4 — regresses)
- 5 query expansion strategies documented
- Instruction-tuned model comparison table

### Task 2: Fixture Expansion
- **File:** `eval/fixtures/basic.json` (226 → 357 fixtures)
- New coverage: Rust(11), Go(10), TypeScript(14), file ops(14), DB(6), Docker(6), CI/CD(4), security(6), multi-tool(5), large edits(3), mined JSONL(16), borderline(14), negative pure-read(16)
- All references validated against rules_basic.txt

### Task 3: Loss Evaluation
- **Artifact:** `artifacts/false-negative-analysis.md`
- 46 fixtures with recall < 1.0, 78 missed rule instances
- REASONABLE_MISS: 23 (29%), FIXABLE_FIXTURE: 22 (28%), FIXABLE_CORPUS: 18 (23%), FIXABLE_EMBEDDING: 15 (19%)
- Most missed rule: "Use type hints on all function signatures" (21 times, mostly wrong fixture expectations)
- Priority: fix 10 wrong fixtures > rewrite 3 corpus rules > better model

### Task 4: Embedding Experiments
- **Artifact:** `artifacts/embedding-experiments.md`
- Model: jina-code-v2, corpus: rules_basic.txt (32 rules), fixtures: basic.json (226)
- BM25 Hybrid (0.7/0.3): Recall=0.459 (-0.051) — **REGRESSION**
- BM25 Hybrid (0.5/0.5): Recall=0.348 (-0.162) — **REGRESSION**
- Rule Augmentation: Recall=0.527 (+0.017), MRR=0.500 (+0.035) — **IMPROVEMENT**
- Query Aug "coding rule:": Recall=0.530 (+0.020) — mixed (hurts MRR)
- Key finding: tool-call queries are hostile to BM25 keyword matching

## Pending Next Steps
1. Fix ~10 wrong fixture expectations (free recall gain)
2. Rewrite 3 narrow corpus rules (file handles, CSRF, magic numbers)
3. Implement rule augmentation (category prefix) in production indexer
4. Evaluate CodeRankEmbed model (convert to ONNX, benchmark)
5. Update CLAUDE.md with findings

## No Commits Made
All changes remain uncommitted per user instruction.
