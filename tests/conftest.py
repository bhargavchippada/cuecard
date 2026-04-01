"""Shared test fixtures for cuecard."""

from __future__ import annotations

import numpy as np
import pytest

from cuecard.models import Index, Provenance, Rule, SourceMeta


@pytest.fixture
def sample_provenance() -> Provenance:
    return Provenance(file="/tmp/rules.txt", line_start=1, line_end=1)


@pytest.fixture
def sample_rule(sample_provenance: Provenance) -> Rule:
    return Rule(text="Never commit secrets to git", provenance=sample_provenance)


@pytest.fixture
def sample_rules() -> tuple[Rule, ...]:
    rules = []
    texts = [
        "Never commit secrets to git",
        "Use uv for all Python package operations",
        "Send Enter after every tmux send-keys command",
        "Always validate user input at system boundaries",
        "Run quality checks before every commit",
    ]
    for i, text in enumerate(texts):
        prov = Provenance(file="/tmp/rules.txt", line_start=i + 1, line_end=i + 1)
        rules.append(Rule(text=text, provenance=prov))
    return tuple(rules)


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """5 rules x 384 dim, L2-normalized."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((5, 384)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


@pytest.fixture
def sample_index(
    sample_rules: tuple[Rule, ...],
    sample_embeddings: np.ndarray,
) -> Index:
    return Index(
        embeddings=sample_embeddings,
        rules=sample_rules,
        model_name="BAAI/bge-small-en-v1.5",
        dim=384,
        sources={
            "/tmp/rules.txt": SourceMeta(
                mtime=1711929600.0,
                content_hash="sha256:abc123",
                rule_count=5,
            ),
        },
    )
