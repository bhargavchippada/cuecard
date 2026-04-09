"""Shared test fixtures for cuecard."""

from __future__ import annotations

import socket
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from cuecard.models import Index, Provenance, Rule, SourceMeta

# ---------------------------------------------------------------------------
# Network guard — block all real outbound connections
# ---------------------------------------------------------------------------

_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_real_create_connection = socket.create_connection
_real_getaddrinfo = socket.getaddrinfo


def _guarded_create_connection(
    address: tuple[str, int], *args: Any, **kwargs: Any,
) -> socket.socket:
    host = str(address[0])
    if host not in _ALLOWED_HOSTS:
        msg = f"Test attempted real network connection to {host!r}"
        raise RuntimeError(msg)
    return _real_create_connection(address, *args, **kwargs)


def _guarded_getaddrinfo(
    host: str | None, *args: Any, **kwargs: Any,
) -> list[Any]:
    if host is not None and str(host) not in _ALLOWED_HOSTS:
        msg = f"Test attempted DNS lookup for {host!r}"
        raise RuntimeError(msg)
    return _real_getaddrinfo(host, *args, **kwargs)


@pytest.fixture(autouse=True)
def _block_network() -> Any:  # noqa: PT004
    """Prevent tests from making real outbound network connections."""
    with (
        patch("socket.create_connection", side_effect=_guarded_create_connection),
        patch("socket.getaddrinfo", side_effect=_guarded_getaddrinfo),
    ):
        yield


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
