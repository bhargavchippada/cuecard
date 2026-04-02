# Retrieval Quality Strategy for cuecard

Date: 2026-04-01

Scope:
- Latest local benchmark/session artifacts
- Local eval corpora and fixtures
- Current external retrieval/reranking techniques and models

Primary question:
- If the current bottleneck is lightweight retrieval, how do we raise candidate recall enough that the existing `Qwen3.5-35B-A3B-Q4_K_M.gguf` reranker can actually solve hard cases while keeping latency low?

---

## 1. Executive Summary

The main bottleneck is upstream of the LLM reranker.

Your current 35B reranker is already strong enough to clean up noise and reason over ambiguous candidates. The bigger problem is that hard queries often never surface the right rules in the candidate set. On the latest benchmarks, this is visible in three ways:

1. Hard-task recall is still far below target even when reranking is enabled.
2. Unified retrieval adds a large cross-domain noise penalty.
3. The corpora are tiny, but the query space is broad and structurally diverse.

The most important design insight from the corpora is this:

> cuecard is not a large-corpus search problem. It is a tiny-corpus, high-ambiguity, high-abstention ranking problem.

That changes what “best architecture” looks like.

With only 30 coding rules, 46 workflow rules, and 76 unified rules, the highest-leverage path is not “one better embedding model and the same pipeline.” It is:

1. Better routing before retrieval
2. Better query expansion
3. Better document/rule expansion
4. Multi-view retrieval instead of a single dense dot product
5. Small, cheap intermediate reranking before the 35B model
6. Hard-only expensive reasoning paths

If the goal is truly 90%+ on all dimensions, I do not think a single zero-shot dense retriever over raw rule text will get there, even with a larger model. The route to 90% is a system design change, not just a model swap.

---

## 2. What the Local Artifacts Say

### 2.1 Latest benchmark picture

From [session-13-report.md](/home/turiya/projects/cuecard/artifacts/session-13-report.md):

Separate PreToolUse index:
- easy recall: 85.2%
- medium recall: 66.0%
- hard recall: 40.4%
- negative silence: 92.9%
- noise: 21.4% overall

Separate UserPromptSubmit index:
- easy recall: 93.2%
- medium recall: 47.1%
- hard recall: 50.0%
- negative silence: 95.8%
- noise: 24.5% overall

Unified index:
- easy recall: 80.6%
- medium recall: 65.1%
- hard recall: 33.3%
- negative silence: 88.7%
- noise: 31.6% overall

Key implication:
- unification is currently expensive in precision/noise
- hard tasks remain the weakest point even after reranking
- the 35B reranker is not the main failure point; candidate generation is

### 2.2 The corpus structure matters more than it first appears

From `eval/`:
- `rules_basic.txt`: 30 coding/tool rules
- `rules_workflow.txt`: 46 workflow/process rules
- `combined.json`: 438 fixtures

Fixture distribution:
- easy: 84
- medium: 140
- hard: 63
- negative: 151

Average expected matches:
- `combined.json`: 1.21 rules/query

Zero-match queries:
- `combined.json`: 153 fixtures

Tool/event distribution:
- `Bash`: 175
- `Edit`: 85
- `UserPromptSubmit`: 84
- `Write`: 52
- `Read`: 22
- `Grep`: 11
- `Glob`: 9

Implications:
- abstention is a first-class requirement, not a side metric
- this is not “retrieve as many relevant things as possible”
- this is “retrieve 0-2 highly correct rules with high confidence”

### 2.3 The hard set is not ordinary semantic search

Representative hard examples:
- `Bash: docker build -t myapp:latest .` -> should retrieve dependency-vulnerability review
- `Bash: grep -r 'AKIA' . --include='*.py'` -> should retrieve secret-related guidance and rotation
- `Write: src/components/Button.tsx ...` -> should retrieve XSS/sanitization guidance
- `UserPromptSubmit: pilot scored 95 percent on 10 samples, ship it` -> should retrieve “don’t trust small sample benchmarks”

These are not simple nearest-neighbor phrasing tasks. They require:
- tool understanding
- action abstraction
- domain mapping
- latent risk inference
- strong abstention on negatives

That is why a single dense similarity score is not enough.

---

## 3. Why Stage 1 Is Failing

### 3.1 Single-vector dense retrieval is too lossy for this task

The current failure mode is not just “weak model.” It is “weak representation.”

A single embedding has to compress:
- tool type
- file path/language
- literal code or shell snippets
- action intent
- risk semantics
- domain routing signal

That is too much for hard queries, especially when the rule text is short and abstract.

### 3.2 Query structure and rule structure do not match

Queries are often structured or code-like:
- JSON tool payloads
- shell commands
- file paths
- code fragments

Rules are short natural-language directives:
- “Review all dependencies for known vulnerabilities before adding”
- “Always close file handles, database connections, network sockets, and other resources”
- “Never trust small sample benchmark results”

This is a classic representation mismatch problem.

### 3.3 Unified retrieval is causing domain interference

The local benchmarks already show the penalty:
- separate corpora are materially cleaner
- unified retrieval increases cross-domain noise by about 10 points

That means the system currently lacks enough routing signal before semantic scoring.

### 3.4 The corpus is tiny, which is actually an opportunity

The unified corpus is only 76 rules.

That means you do not need to think like a web-scale ANN retrieval system. You can afford:
- multiple query views
- multiple rule views
- multiple scorers
- exact scoring over all rules
- small reranking over all or most rules

This is a major strategic advantage.

---

## 4. The Core Strategic Shift

Instead of:

`one query -> one dense vector -> top-k -> 35B reranker`

I recommend moving toward:

`one query -> routed multi-view candidate generation -> cheap structured scoring -> small reranker -> 35B reranker only when needed`

In practice, that means expanding in five places:

1. Query expansion
2. Rule/document expansion
3. Retrieval expansion
4. Routing/filter expansion
5. Supervision/distillation expansion

---

## 5. Where to Expand Upstream

### 5.1 Query expansion

This is the highest-ROI near-term change.

#### A. Deterministic tool-aware decomposition

Turn one tool query into several retrieval views.

Example:
- raw query: `Bash: git commit -m 'fix auth bug'`

Expanded views:
- `git commit`
- `committing code changes`
- `pre-commit quality checks`
- `commit message format`
- `security before commit`

Example:
- raw query: `Write: {"file_path":"Dockerfile","content":"RUN pip install flask"}`

Expanded views:
- `writing a Dockerfile`
- `adding a Python dependency`
- `using pip for Python package installation`
- `dependency security review`

Why this should help:
- hard tasks often fail because the literal query surface does not match the rule surface
- the corpus is tiny, so multi-query expansion is cheap

Pros:
- deterministic
- explainable
- low latency
- can be tuned by tool family

Cons:
- requires hand-built templates and parsers
- can add noise if expansions are too broad

#### B. Entity and risk extraction

Extract explicit triggers from queries:
- commands: `git commit`, `pip install`, `docker run`
- libraries/apis: `aiohttp.ClientSession`, `psycopg2.connect`, `socket.socket`
- security tokens: `AKIA`, `sk_live_`, `Bearer`, `.env`
- file/path/language hints: `.py`, `.tsx`, `Dockerfile`, `.github/workflows`

Then use them as:
- sparse retrieval terms
- metadata filters
- boosted query features

Pros:
- especially valuable for medium/hard security and code-pattern queries
- very cheap

Cons:
- requires a maintained extraction vocabulary

#### C. Hard-only LLM query expansion

For hard or low-confidence queries only, generate 2-5 short paraphrase/intention queries before retrieval.

Relevant prior work:
- HyDE: hypothetical document generation for zero-shot dense retrieval
- Query2doc / synthetic expansion style approaches

Pros:
- can lift hard recall significantly
- lets you keep the 35B reranker focused on reasoning rather than discovery

Cons:
- adds latency
- should not run on every query
- needs careful prompting to avoid generic expansions

Recommendation:
- do this only for hard-predicted or low-margin queries

### 5.2 Rule/document expansion

This may be even more important than swapping the encoder.

Because the rule corpus is tiny, you can afford rich offline expansion per rule.

#### A. Per-rule synthetic paraphrases

For each canonical rule, generate:
- 5-20 paraphrases
- tool-trigger phrasings
- code-pattern phrasings
- positive examples
- negative/counterexample cues

Then index those child texts and map them back to the parent rule.

Example:
- canonical: `Always close file handles, database connections, network sockets, and other resources`
- expansions:
  - `close aiohttp ClientSession objects`
  - `do not leak socket connections`
  - `ensure db.connect() resources are closed`
  - `close SMTP and HTTP client sessions`

Pros:
- directly addresses vocabulary mismatch
- cheap at runtime
- highly compatible with current architecture

Cons:
- offline generation quality matters
- can drift if synthetic expansions are too broad

#### B. Structured metadata for every rule

Add tags such as:
- domain: `coding`, `workflow`, `security`, `git`, `testing`, `docs`
- language: `python`, `typescript`, `shell`, `docker`, `agnostic`
- tool family: `bash`, `edit`, `write`, `read`, `userprompt`
- intent type: `abstain`, `commit`, `dependency`, `resource-cleanup`, `security`

Use the tags for:
- payload filtering
- score priors
- route-specific indexes

Pros:
- likely the cleanest fix for unified-index noise
- extremely cheap

Cons:
- requires a tagging policy
- some rules are multi-domain

#### C. Example-rich rule entries

Instead of indexing only the rule sentence, index a richer representation:
- title
- canonical rule
- triggers
- examples
- anti-examples

Pros:
- improves semantic anchoring
- especially good for short abstract rules

Cons:
- more index text to curate
- may require separate parent/child result collapsing

### 5.3 Retrieval expansion

Do not rely on one score source.

#### A. Dense + sparse hybrid retrieval

Recommended baseline change.

Use:
- dense retrieval for semantic abstraction
- sparse/BM25 retrieval for exact identifiers, commands, file names, and literals
- fuse with Reciprocal Rank Fusion (RRF)

Why this fits cuecard:
- many queries contain literal anchors: `pip`, `tmux send-keys`, `AKIA`, `eval()`, `shell=True`, `Dockerfile`
- many rules also contain literal anchors
- the corpus is tiny, so sparse retrieval is basically free

Pros:
- low engineering risk
- low latency
- strong fit for command- and code-heavy queries

Cons:
- pure BM25 will not solve workflow/hard reasoning tasks
- must be fused carefully, not used alone

#### B. Multi-index specialist retrieval

Instead of one unified index, maintain specialist indexes:
- coding/tool rules
- workflow/process rules
- maybe security-only and git-only views

Then either:
- route to one specialist first, or
- retrieve from 2-3 likely specialists and union the candidates

Pros:
- directly addresses current unified noise problem
- easier to tune per event type

Cons:
- routing mistakes can hurt recall
- configuration gets more complex

Recommendation:
- use soft routing, not hard routing
- retrieve from top 2 predicted domains, not exactly 1

#### C. Late interaction retrieval

For hard tasks, single-vector dense retrieval may be the wrong representation class.

Use late interaction / multi-vector methods:
- ColBERT-style MaxSim
- BGE-M3 multi-vector capability
- Qdrant multivector support if you want an off-the-shelf engine

Why it helps:
- preserves more token-level evidence
- better for matching localized evidence like `AKIA`, `shell=True`, `ClientSession`, `docker build`, `95 percent on 10 samples`

Pros:
- much better match for structured short texts and mixed code/NL queries
- especially promising on hard and medium tasks

Cons:
- more implementation work
- index representation is heavier than a single dense vector

This is one of the strongest technical candidates if you want a real hard-task jump.

### 5.4 Routing and filter expansion

This is how you improve negative silence and reduce cross-domain waste.

#### A. Intent/difficulty router

Before retrieval, predict:
- event type
- tool family
- likely domain(s): coding vs workflow vs security vs git vs docs
- likely difficulty
- abstain likelihood

This can be:
- rule-based + regex + file-path heuristics first
- later replaced or supplemented with a small classifier

Pros:
- very cheap
- large upside for unified noise and negative handling

Cons:
- routing errors can suppress good candidates if used too aggressively

#### B. Read-only / chit-chat abstention gate

You have many negative queries:
- `Read: ...`
- `git blame`
- `git diff`
- `what time is it`
- `hello`

These should not hit the full retrieval stack equally.

Pros:
- easiest path to 95%+ negative silence
- reduces wasted latency

Cons:
- must be conservative to avoid false abstention

### 5.5 Supervision/distillation expansion

This is the highest-upside longer-term path.

#### A. Distill the 35B reranker into a smaller upstream scorer

Use the current 35B model as a teacher to label:
- query -> rule relevance pairs
- query -> domain
- query -> abstain

Train or calibrate a smaller model for Stage 1.5 / Stage 2.

Why this matters:
- the 35B model already knows a lot of the semantics you need
- the problem is not accessible to the 35B because candidate recall is weak

Pros:
- strong fit for your exact task
- can preserve task-specific reasoning without 35B latency at every stage

Cons:
- requires data generation and offline training
- must generalize to new rule sets

#### B. Synthetic training data from the rules themselves

For each rule, generate:
- easy/medium/hard positives
- near-miss negatives
- domain-confusion negatives
- abstention negatives

This is especially powerful because your corpus is tiny.

Pros:
- helps the retriever learn the actual task distribution
- can target exactly the failure patterns seen in `false-negative-analysis.md`

Cons:
- synthetic data quality must be checked
- can overfit if generation is narrow

---

## 6. Model and Technique Options to Benchmark

These are the most relevant options for cuecard, given the bottleneck and latency constraint.

### 6.1 Dense encoders

#### Option 1: stay on `jina-embeddings-v2-base-code` for baseline comparisons

Pros:
- already integrated
- good coding-query behavior
- low migration risk

Cons:
- clearly not enough for hard-task recall ceiling

#### Option 2: `nomic-ai/CodeRankEmbed`

Why interesting:
- purpose-built for code retrieval
- much stronger code-search benchmarks than older code embedders

Best use in cuecard:
- benchmark on PreToolUse hard subset
- especially for Bash/Edit/Write coding-security tasks

Pros:
- likely strongest drop-in code-oriented dense candidate
- Apache-2.0

Cons:
- integration path may be less turnkey than current fastembed setup
- workflow/process retrieval may still require a different model or route

#### Option 3: `Qwen3-Embedding-0.6B`

Why interesting:
- strong recent retrieval benchmarks for its size
- instruction-aware, which fits cuecard’s mixed task style

Best use in cuecard:
- benchmark on workflow/process and unified corpora
- use task-specific query prefixes

Pros:
- strong overall retrieval model
- may handle abstract workflow prompts better than a code-only embedder

Cons:
- heavier than current dense stage
- needs careful latency benchmarking

#### Option 4: `google/embeddinggemma-300m`

Why interesting:
- small enough to be plausible locally
- matryoshka-style dimensional truncation offers quality/speed tradeoffs
- task-prompted embedding setup is attractive for cuecard

Best use in cuecard:
- benchmark as a balanced general encoder
- especially if you want one model for both coding and workflow

Pros:
- strong balance candidate
- flexible vector size

Cons:
- license review needed
- may not beat code-specialized models on coding hard cases

#### Option 5: `BAAI/bge-m3`

Why interesting:
- dense + sparse + multi-vector in one family
- attractive if you want one model family to power hybrid retrieval and later interaction

Best use in cuecard:
- benchmark as a system simplifier, not necessarily as pure dense winner

Pros:
- can support dense/sparse/multivector expansion
- useful architecture family for research

Cons:
- not obviously the best pure code retriever
- probably more useful as a technique platform than as the one winner

### 6.2 Small rerankers before the 35B model

The goal here is not to replace the 35B model. It is to improve candidate precision so the 35B model sees a better list.

Candidates to test:
- `BAAI/bge-reranker-v2-m3`
- `mixedbread-ai/mxbai-rerank-xsmall-v1`
- `jinaai/jina-reranker-v1-turbo-en`

Pros:
- can re-score 20-80 candidate pairs far more cheaply than a 35B model
- good fit for tiny corpora

Cons:
- may still underperform on very indirect hard reasoning cases
- pairwise reranking needs strong candidate recall first

### 6.3 Late interaction

Candidates/stack ideas:
- ColBERTv2 / PLAID-style retrieval
- `jina-colbert` family
- PyLate for experimentation
- Qdrant multivector support if you want a practical backend

Pros:
- probably the best representation upgrade for hard retrieval short of full task-specific training
- strong fit for structured queries and localized evidence

Cons:
- more system change than dense swap or hybrid BM25

---

## 7. Open-Source Systems Worth Studying

### Qdrant

Current popularity: ~29.9k GitHub stars

Why study it:
- hybrid retrieval
- payload filtering
- multivector support
- operationally simple

Best cuecard use:
- dense + sparse + metadata payload filters
- possible future multivector support

### FlagEmbedding

Current popularity: ~11.5k GitHub stars

Why study it:
- active ecosystem for BGE models, rerankers, and hybrid retrieval
- useful for experimentation even if not all models become production choices

Best cuecard use:
- BGE-M3 experiments
- reranker experiments

### ColBERT / PyLate

Current popularity: ColBERT ~3.8k stars, PyLate ~776 stars

Why study them:
- late-interaction retrieval is one of the cleanest answers to “hard candidate recall”

Best cuecard use:
- hard-tier retrieval experiments
- token-level evidence matching without going full LLM

### Infinity

Current popularity: ~2.7k GitHub stars

Why study it:
- one practical serving layer for embeddings, reranking, and ColBERT-style models

Best cuecard use:
- operationally convenient benchmark environment for multiple model classes

### Vespa

Current popularity: ~6.9k GitHub stars

Why study it:
- multiphase retrieval/ranking design is aligned with the architecture cuecard is moving toward

Best cuecard use:
- as a conceptual reference for phased ranking and score fusion
- probably overkill for the current tiny corpus size

---

## 8. A 90%+ Plan by Difficulty

### Easy

Target path:
- deterministic tool parsing
- hybrid dense+sparse
- exact trigger matching
- rule metadata

Why:
- easy queries already have enough surface signal
- these should be the first bucket to push above 90 recall and below 10-15 noise

### Medium

Target path:
- better encoder
- query decomposition
- per-rule synthetic paraphrases
- small reranker before the 35B model

Why:
- medium failures are usually vocabulary gap and domain-gap failures, not deep reasoning failures

### Hard

Target path:
- route-specific retrieval
- document expansion
- hard-only query expansion
- late interaction or specialist small reranker
- 35B reranker reserved for final reasoning/selection

Why:
- hard failures are mostly candidate-discovery failures, not final ranking failures

Reality check:
- 90% hard recall + 90% precision is unlikely from zero-shot dense retrieval alone
- it becomes plausible only if you add richer candidate generation or task-specific supervision

### Negative

Target path:
- abstention gate
- route-aware thresholds
- negative-specialized classifier or calibration layer

Why:
- with 151 negatives in `combined.json`, silence is a major product-quality dimension

---

## 9. Recommended Experiment Order

This is the sequence I would actually run.

### Phase 1: Highest ROI, lowest risk

1. Keep separate coding/workflow indexes by default for quality mode.
2. Add rule metadata tags and use them as payload filters/score priors.
3. Add deterministic tool-aware query decomposition.
4. Add dense + sparse hybrid fusion.
5. Add offline per-rule paraphrase/trigger expansion.

Expected outcome:
- better easy/medium recall
- better negative silence
- lower unified/cross-domain noise

### Phase 2: Stronger retrieval stack

6. Benchmark `CodeRankEmbed`, `Qwen3-Embedding-0.6B`, `embeddinggemma-300m`, and `bge-m3`.
7. Add a small reranker over all rules or top 20-40 candidates.
8. Use 35B reranking only after the small reranker has narrowed the list.

Expected outcome:
- medium and hard candidate quality should improve materially
- latency becomes more controllable than sending broad noisy lists to 35B

### Phase 3: Hard-tier push

9. Add hard-only query expansion using a small local generator or teacher model.
10. Benchmark late interaction retrieval.
11. Distill 35B judgments into a small routing/ranking model.

Expected outcome:
- this is where hard-tier recall has the best chance of making a real jump

---

## 10. My Strongest Recommendations

If I had to choose only a few bets, they would be these:

### Bet 1: stop treating this like large-corpus ANN retrieval

Because the corpus is tiny, use richer exhaustive scoring and multiple views.

This is the most important architectural insight.

### Bet 2: invest heavily in rule expansion

The rule corpus is tiny and stable enough that you can enrich the document side aggressively.

That is cheaper and safer than asking one embedder to infer every latent relation from a single short rule sentence.

### Bet 3: add hybrid + metadata before changing the 35B reranker

This should improve both recall and silence faster than another reranker change.

### Bet 4: use separate or softly routed specialist indexes

Your own benchmarks already show the unified penalty.

### Bet 5: benchmark late interaction if hard-tier is the real goal

If you want a meaningful hard-tier jump without paying 35B cost on every candidate-discovery step, this is one of the most promising paths.

---

## 11. Concrete Metrics to Track Next

Add these metrics to every benchmark:

1. Candidate recall@5/@10/@20 before the 35B stage
2. Candidate recall split by route/domain/tool
3. Hard-only candidate recall for the gold rule in top-20
4. Negative false-positive rate before and after routing
5. Per-rule false negative rate
6. Cross-domain confusion rate
7. Latency split by:
   - routing
   - dense retrieval
   - sparse retrieval
   - small reranker
   - 35B reranker

Without candidate recall@K before the 35B stage, you will keep measuring the symptom instead of the real bottleneck.

---

## 12. Bottom Line

The right next move is not “swap the reranker.” The right next move is to widen and structure candidate generation.

The best near-term path is:
- routing
- hybrid retrieval
- query decomposition
- rule expansion
- small reranker before the 35B model

The best high-upside path is:
- late interaction retrieval
- hard-only expansion
- distillation from the current 35B reranker

If the target is truly 90%+ across easy, medium, hard, and negative, the likely winning architecture is not a better single dense embedder. It is a multi-view micro-corpus ranking system built for high abstention and high ambiguity.

---

## External Sources

Models and techniques:
- [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed)
- [EmbeddingGemma 300M](https://huggingface.co/google/embeddinggemma-300m)
- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- [BRIGHT benchmark](https://arxiv.org/pdf/2407.12883)

Systems and infrastructure:
- [Qdrant](https://github.com/qdrant/qdrant)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [ColBERT](https://github.com/stanford-futuredata/ColBERT)
- [PyLate](https://github.com/lightonai/pylate)
- [Infinity](https://github.com/michaelfeil/infinity)
- [Vespa phased ranking](https://docs.vespa.ai/en/ranking/phased-ranking.html)
