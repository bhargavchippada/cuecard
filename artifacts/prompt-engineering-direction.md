# Prompt Engineering Direction

> Key insight: Prompt size is NOT a latency concern. Prompt caching makes the system prompt free after first call. Optimize for quality, not brevity.

## Why Prompt Size Doesn't Matter

### Expansion Prompt (offline)
- Runs once per `cuecard rules expand`
- User waits ~1-2s per rule regardless of prompt size
- A 10K-token prompt with 20 golden examples costs the same as a 2K-token prompt
- **Maximize example diversity and instruction clarity**

### Reranker Prompt (online, cached)
- llama-server caches the system prompt after first call
- First call: system prompt (~5-15K tokens) + query+candidates (~200 tokens) = full processing
- Subsequent calls: only process the ~200 variable tokens (cache hit on system prompt)
- A 15K system prompt adds ~0ms after first call
- **Pack the system prompt with examples covering every failure pattern**

## Current Problems to Fix via Prompt Engineering

### 1. Easy tier over-filtering (94.2% → 85.2% with LLM)
The LLM rejects obvious matches. Fix: add examples showing high-confidence matches that should be KEPT:
- "Bash: git push --force" + "Never force-push to main" → RELEVANT (obvious)
- "Bash: pip install requests" + "Use uv for all Python package operations" → RELEVANT (direct violation)

### 2. Negative silence gap (91% vs 95% target)
~6 negative fixtures still get rules. Fix: add more negative examples:
- "Bash: nvidia-smi" → NO rules (system monitoring, not development)
- "Bash: pactl list sources" → NO rules (audio config, not code)
- "Bash: htop" → NO rules (process monitoring)

### 3. Cross-domain noise (workflow rules matching code queries)
- "Bash: git status" + "Update README when behavior changes" → NOT relevant (git status ≠ updating docs)
- "Read: package.json" + "Search GitHub for existing implementations" → NOT relevant (reading a file ≠ searching)

### 4. Hard tier recall (50.1% — barely meeting target)
- Add examples of indirect but valid matches
- "Bash: python -c 'import subprocess; subprocess.call(user_input)'" + "Always validate user input" → RELEVANT (injection risk)

## Prompt Engineering Principles

1. **Loss-pattern driven:** Every few-shot example should address a measured failure mode
2. **Exhaustive negatives:** More negative examples than positive (real queries are 54% negative)
3. **Calibration examples:** Show the model the BOUNDARY between relevant and irrelevant
4. **Generic and extensible:** Examples should generalize to any rule set, not hardcode specific rules
5. **Per-tier coverage:** Include easy (obvious), medium (requires reasoning), hard (indirect), and negative examples

## Future Iteration Process

1. Run benchmark → identify per-fixture failures
2. Categorize failures: over-filter (false negative) vs under-filter (false positive)
3. Find representative examples of each failure category
4. Add to the prompt as few-shot examples
5. Re-benchmark → verify improvement without regression
6. Repeat until convergence

This is a continuous process — each prompt version should be versioned and benchmarked.

## Applicability to Small Models

The same prompt engineering applies to small models (0.6B-3.8B):
- Larger prompts with more examples may help small models MORE (they need more guidance)
- Context window is the only constraint (Qwen3-0.6B: 40K context, plenty of room)
- Prompt caching works the same way
- The question is whether the model can follow complex multi-example instructions at all

## Key Constraint: Keep It Generic

The prompt must work for ANY rule set, not just our eval corpora:
- Don't reference specific rule texts in the prompt
- Use abstract categories: "coding rules about security", "workflow rules about planning"
- Show the pattern of reasoning, not memorized answers
- The few-shot examples teach the model HOW to think, not WHAT to answer
