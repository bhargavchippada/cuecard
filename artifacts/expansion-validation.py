"""Expansion validation experiment.

Tests whether LLM-generated expansions bridge the vocabulary gap
on hard fixtures. Compares cosine similarity of raw rule text vs
rule + expansions against hard queries.

Usage: uv run python artifacts/expansion-validation.py
"""

from __future__ import annotations

import json
import sys

import numpy as np

# --- Hand-crafted expansions for rules that hard fixtures should match ---
# These simulate what an LLM would generate via `cuecard rules expand`

RULE_EXPANSIONS: dict[str, list[str]] = {
    "Review all dependencies for known vulnerabilities before adding": [
        "docker build with untrusted base image",
        "pip install new package check CVEs",
        "npm audit before adding dependency",
        "cargo add verify crate security advisories",
        "check dependency license compatibility",
        "Dockerfile adding packages without vulnerability scan",
    ],
    "Never commit secrets (API keys, tokens, passwords) to git": [
        "AKIA AWS access key found in source code",
        "sk_live_ Stripe API key hardcoded",
        "grep for leaked credentials in repository",
        "curl command with embedded API key",
        "ssh private key exposed in codebase",
        "hardcoded password in configuration file",
        "Bearer token embedded in source",
    ],
    "Rotate any secrets that may have been exposed": [
        "searching for leaked AWS access keys AKIA",
        "grep for exposed credentials in codebase",
        "API key found in commit history",
        "rotate compromised tokens immediately",
        "secret scanning found exposed password",
    ],
    "Set file permissions to 0o600 for sensitive files": [
        "SSH private key file needs restrictive permissions",
        "cron job writing to /etc with wrong permissions",
        "Dockerfile should set file permissions on sensitive configs",
        "private key generated without chmod 600",
        "sensitive config file world-readable",
    ],
    "Use environment variables for configuration, not hardcoded values": [
        "docker run without passing environment variables",
        "hardcoded API key in curl command",
        "sk_live_ key embedded in source code",
        "config values hardcoded instead of env vars",
        "docker compose missing env_file directive",
    ],
    "Sanitize all HTML output to prevent XSS": [
        "React component rendering user input without escaping",
        "JSX template with unsanitized props",
        "f-string building HTML from user-provided data",
        "innerHTML with user content",
        "Button component rendering label without sanitization",
    ],
    "Use uv for all Python package operations, never pip": [
        "pip install in GitHub Actions CI workflow",
        "requirements.txt with pip instead of uv",
        "Dockerfile using pip install instead of uv",
        "CI/CD pipeline using pip for Python packages",
    ],
}

# --- Hard fixtures to validate against ---

HARD_FIXTURES = [
    {
        "id": "docker-build",
        "query": 'Bash: docker build -t myapp:latest .',
        "should_match": ["Review all dependencies for known vulnerabilities before adding"],
    },
    {
        "id": "secret-rotation",
        "query": "Bash: grep -r 'AKIA' . --include='*.py'",
        "should_match": [
            "Never commit secrets (API keys, tokens, passwords) to git",
            "Rotate any secrets that may have been exposed",
        ],
    },
    {
        "id": "write-dockerfile",
        "query": 'Write: {"file_path": "Dockerfile", "content": "FROM python:3.12\\nRUN pip install flask gunicorn\\nCOPY . /app\\nCMD [\\"gunicorn\\"]"}',
        "should_match": [
            "Review all dependencies for known vulnerabilities before adding",
            "Set file permissions to 0o600 for sensitive files",
        ],
    },
    {
        "id": "write-typescript-file",
        "query": 'Write: {"file_path": "src/components/Button.tsx", "content": "export function Button({ label }) { return <button>{label}</button> }"}',
        "should_match": ["Sanitize all HTML output to prevent XSS"],
    },
    {
        "id": "docker-run",
        "query": "Bash: docker run -p 8080:80 myapp:latest",
        "should_match": ["Use environment variables for configuration, not hardcoded values"],
    },
    {
        "id": "hard-curl-stripe-api",
        "query": "Bash: curl -X POST https://api.stripe.com/v1/charges -u sk_live_abc123: -d amount=2000 -d currency=usd",
        "should_match": [
            "Never commit secrets (API keys, tokens, passwords) to git",
            "Use environment variables for configuration, not hardcoded values",
        ],
    },
    {
        "id": "hard-ssh-key-generation",
        "query": "Bash: ssh-keygen -t ed25519 -C 'deploy@app.com' -f /app/deploy_key -N '' && cat /app/deploy_key",
        "should_match": [
            "Never commit secrets (API keys, tokens, passwords) to git",
            "Set file permissions to 0o600 for sensitive files",
        ],
    },
    {
        "id": "hard-github-actions-workflow",
        "query": 'Write: {"file_path": ".github/workflows/ci.yml", "content": "name: CI\\non: push\\njobs:\\n  test:\\n    runs-on: ubuntu-latest\\n    steps:\\n      - uses: actions/checkout@v4\\n      - run: pip install -r requirements.txt\\n      - run: pytest"}',
        "should_match": ["Use uv for all Python package operations, never pip"],
    },
]


def main() -> None:
    try:
        from fastembed import TextEmbedding
    except ImportError:
        print("ERROR: fastembed not installed. Run: uv pip install fastembed")
        sys.exit(1)

    model_name = "jinaai/jina-embeddings-v2-base-code"
    print(f"Loading {model_name}...")
    model = TextEmbedding(model_name)
    print("Model loaded.\n")

    # Load all rules from corpus
    with open("eval/corpora/rules_basic.txt") as f:
        rules = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
    print(f"Loaded {len(rules)} rules from corpus.\n")

    # Embed all rules (canonical only)
    print("Embedding canonical rules...")
    rule_embeddings = np.array(
        list(model.passage_embed(rules)), dtype=np.float32,
    )
    rule_norms = np.linalg.norm(rule_embeddings, axis=1, keepdims=True)
    rule_embeddings = rule_embeddings / rule_norms

    # Build expansion embeddings
    # For each rule, we embed canonical + expansions, take max score
    print("Embedding expansions...")
    expanded_texts: list[list[str]] = []  # per rule: [canonical, exp1, exp2, ...]
    for rule in rules:
        texts = [rule]
        if rule in RULE_EXPANSIONS:
            texts.extend(RULE_EXPANSIONS[rule])
        expanded_texts.append(texts)

    # Flatten for batch embedding
    all_texts: list[str] = []
    rule_map: list[int] = []  # maps each text to parent rule index
    for rule_idx, texts in enumerate(expanded_texts):
        for text in texts:
            all_texts.append(text)
            rule_map.append(rule_idx)

    all_embeddings = np.array(
        list(model.passage_embed(all_texts)), dtype=np.float32,
    )
    all_norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / all_norms
    rule_map_arr = np.array(rule_map, dtype=np.int32)

    print(f"Total embeddings: {len(all_texts)} ({len(rules)} canonical + {len(all_texts) - len(rules)} expansions)\n")

    # Now test each hard fixture
    threshold = 0.30
    print("=" * 90)
    print(f"{'FIXTURE':<30} {'EXPECTED RULE':<55} {'RAW':>5} {'EXP':>5} {'DELTA':>6}")
    print("=" * 90)

    improvements = []
    regressions = []

    for fixture in HARD_FIXTURES:
        # Embed query
        query_vec = np.array(
            list(model.query_embed([fixture["query"]])), dtype=np.float32,
        )
        query_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec / query_norm  # (1, dim)

        for expected_rule in fixture["should_match"]:
            rule_idx = rules.index(expected_rule)

            # Raw score: canonical only
            raw_score = float((query_vec @ rule_embeddings[rule_idx:rule_idx+1].T).item())

            # Expanded score: max across canonical + expansions
            mask = rule_map_arr == rule_idx
            exp_embs = all_embeddings[mask]  # (N, dim)
            exp_scores = (exp_embs @ query_vec.T).flatten()
            expanded_score = float(exp_scores.max())

            delta = expanded_score - raw_score
            marker = ""
            if delta > 0.02:
                marker = " ✓"
                improvements.append((fixture["id"], expected_rule, raw_score, expanded_score, delta))
            elif delta < -0.02:
                marker = " ✗"
                regressions.append((fixture["id"], expected_rule, raw_score, expanded_score, delta))

            above_raw = "Y" if raw_score >= threshold else "N"
            above_exp = "Y" if expanded_score >= threshold else "N"

            print(
                f"{fixture['id']:<30} "
                f"{expected_rule[:53]:<55} "
                f"{raw_score:>5.3f} "
                f"{expanded_score:>5.3f} "
                f"{delta:>+6.3f}"
                f"{marker}"
            )

    print("=" * 90)
    print(f"\nThreshold: {threshold}")
    print(f"Improvements (delta > +0.02): {len(improvements)}")
    print(f"Regressions (delta < -0.02): {len(regressions)}")
    print(f"Total query-rule pairs: {sum(len(f['should_match']) for f in HARD_FIXTURES)}")

    # Show cases where expansion crosses the threshold
    print("\n--- Cases where expansion crosses threshold (N → Y) ---")
    for fixture in HARD_FIXTURES:
        query_vec = np.array(
            list(model.query_embed([fixture["query"]])), dtype=np.float32,
        )
        query_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec / query_norm

        for expected_rule in fixture["should_match"]:
            rule_idx = rules.index(expected_rule)
            raw_score = float((query_vec @ rule_embeddings[rule_idx:rule_idx+1].T).item())

            mask = rule_map_arr == rule_idx
            exp_embs = all_embeddings[mask]
            exp_scores = (exp_embs @ query_vec.T).flatten()
            expanded_score = float(exp_scores.max())

            if raw_score < threshold <= expanded_score:
                # Find which expansion text gave the best score
                best_exp_idx = int(exp_scores.argmax())
                best_text = [t for i, t in enumerate(all_texts) if rule_map[i] == rule_idx][best_exp_idx]
                print(f"\n  {fixture['id']}: {raw_score:.3f} → {expanded_score:.3f}")
                print(f"    Rule: {expected_rule}")
                print(f"    Best expansion: \"{best_text}\"")

    # Show which expansion texts are most useful
    print("\n--- Per-expansion contribution analysis ---")
    for rule, expansions in RULE_EXPANSIONS.items():
        rule_idx = rules.index(rule)
        print(f"\nRule: \"{rule}\"")
        for exp_text in expansions:
            # Find this expansion's embedding
            exp_idx = all_texts.index(exp_text)
            exp_emb = all_embeddings[exp_idx:exp_idx+1]

            # Check which hard fixtures benefit
            for fixture in HARD_FIXTURES:
                if rule not in fixture["should_match"]:
                    continue
                query_vec = np.array(
                    list(model.query_embed([fixture["query"]])), dtype=np.float32,
                )
                query_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
                query_vec = query_vec / query_norm

                raw_score = float((query_vec @ rule_embeddings[rule_idx:rule_idx+1].T).item())
                exp_score = float((query_vec @ exp_emb.T).item())

                if exp_score > raw_score + 0.02:
                    print(f"  \"{exp_text[:60]}\"")
                    print(f"    +{exp_score - raw_score:.3f} on {fixture['id']} ({raw_score:.3f} → {exp_score:.3f})")


if __name__ == "__main__":
    main()
