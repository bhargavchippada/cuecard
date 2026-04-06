"""Tests for cuecard.indexer — build_index, checksum, and expansion-aware builds."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from cuecard.indexer import (
    _compute_checksum,
    build_index,
)
from cuecard.models import Provenance, Rule, SourceMeta


def _make_mock_model(dim: int = 384, n_rules: int = 5) -> MagicMock:
    """Create a mock embedding model that returns deterministic vectors."""
    model = MagicMock()
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n_rules, dim)).astype(np.float32)
    # Return a generator (like real fastembed)
    model.passage_embed.return_value = iter(embeddings)
    model.query_embed = MagicMock()
    return model


def _make_rules(n: int = 5) -> tuple[Rule, ...]:
    """Create n sample rules."""
    texts = [
        "Never commit secrets to git",
        "Use uv for all Python package operations",
        "Send Enter after every tmux send-keys command",
        "Always validate user input at system boundaries",
        "Run quality checks before every commit",
    ]
    return tuple(
        Rule(
            text=texts[i % len(texts)],
            provenance=Provenance(
                file="/tmp/rules.txt", line_start=i + 1, line_end=i + 1,
            ),
        )
        for i in range(n)
    )


class TestBuildIndex:
    def test_build_with_rules(self) -> None:
        rules = _make_rules(3)
        model = _make_mock_model(dim=384, n_rules=3)
        index = build_index(rules, {}, "test-model", model=model)

        assert index.size == 3
        assert index.model_name == "test-model"
        assert index.dim == 384
        assert index.embeddings.shape == (3, 384)
        model.passage_embed.assert_called_once()

    def test_build_empty_rules(self) -> None:
        index = build_index((), {}, "test-model", dim=128)

        assert index.size == 0
        assert index.dim == 128
        assert index.embeddings.shape == (0, 128)

    def test_build_empty_rules_default_dim(self) -> None:
        index = build_index((), {}, "test-model")

        assert index.dim == 384
        assert index.embeddings.shape == (0, 384)

    def test_l2_normalization(self) -> None:
        rules = _make_rules(3)
        model = _make_mock_model(dim=384, n_rules=3)
        index = build_index(rules, {}, "test-model", model=model)

        norms = np.linalg.norm(index.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_passage_embed_called_not_query_embed(self) -> None:
        rules = _make_rules(2)
        model = _make_mock_model(dim=384, n_rules=2)
        build_index(rules, {}, "test-model", model=model)

        model.passage_embed.assert_called_once()
        model.query_embed.assert_not_called()

    def test_build_preserves_sources(self) -> None:
        sources = {
            "/tmp/rules.txt": SourceMeta(
                mtime=1711929600.0,
                content_hash="sha256:abc123",
                rule_count=3,
            ),
        }
        rules = _make_rules(3)
        model = _make_mock_model(dim=384, n_rules=3)
        index = build_index(rules, sources, "test-model", model=model)

        assert "/tmp/rules.txt" in index.sources
        assert index.sources["/tmp/rules.txt"].rule_count == 3

    def test_build_no_model_raises(self) -> None:
        rules = _make_rules(1)
        with pytest.raises(ValueError, match="model is required"):
            build_index(rules, {}, "test-model", model=None)

    def test_build_single_rule_1d_embedding(self) -> None:
        """When model returns a flat 1D array (single rule), it gets reshaped to 2D."""
        rules = _make_rules(1)
        model = MagicMock()
        # Return a bare 1D array — list() iterates over elements, producing
        # np.array([scalar, scalar, ...]) which is 1D, triggering the reshape.
        rng = np.random.default_rng(42)
        flat_embedding = rng.standard_normal(384).astype(np.float32)
        model.passage_embed.return_value = flat_embedding
        index = build_index(rules, {}, "test-model", model=model)

        assert index.size == 1
        assert index.embeddings.shape == (1, 384)
        assert abs(np.linalg.norm(index.embeddings[0]) - 1.0) < 1e-6

    def test_build_detects_dim_from_model(self) -> None:
        rules = _make_rules(2)
        model = _make_mock_model(dim=128, n_rules=2)
        index = build_index(rules, {}, "test-model", model=model, dim=384)

        # dim should be detected from actual embeddings, not the default
        assert index.dim == 128


class TestComputeChecksum:
    def test_correct_sha256(self, tmp_path: object) -> None:
        import hashlib

        test_file = tmp_path / "test.bin"  # type: ignore[operator]
        content = b"hello world"
        test_file.write_bytes(content)

        result = _compute_checksum(str(test_file))

        expected_hash = hashlib.sha256(content).hexdigest()
        assert result == f"sha256:{expected_hash}"

    def test_empty_file(self, tmp_path: object) -> None:
        import hashlib

        test_file = tmp_path / "empty.bin"  # type: ignore[operator]
        test_file.write_bytes(b"")

        result = _compute_checksum(str(test_file))

        expected_hash = hashlib.sha256(b"").hexdigest()
        assert result == f"sha256:{expected_hash}"

    def test_deterministic(self, tmp_path: object) -> None:
        test_file = tmp_path / "det.bin"  # type: ignore[operator]
        test_file.write_bytes(b"deterministic content")

        first = _compute_checksum(str(test_file))
        second = _compute_checksum(str(test_file))
        assert first == second

    def test_different_content_different_checksum(self, tmp_path: object) -> None:
        file_a = tmp_path / "a.bin"  # type: ignore[operator]
        file_b = tmp_path / "b.bin"  # type: ignore[operator]
        file_a.write_bytes(b"content A")
        file_b.write_bytes(b"content B")

        assert _compute_checksum(str(file_a)) != _compute_checksum(str(file_b))


class TestBuildIndexExpansions:
    """Tests for expansion-aware build_index."""

    def test_build_with_expansions(self) -> None:
        """Expansions produce extra embedding rows + rule_map + bm25_corpus."""
        rules = (
            Rule(
                text="Never commit secrets",
                provenance=Provenance(
                    file="/tmp/r.txt", line_start=1, line_end=1,
                ),
                expansions=("hardcoded API key", "AKIA in source"),
            ),
        )
        model = _make_mock_model(dim=384, n_rules=3)
        index = build_index(rules, {}, "test-model", model=model)

        assert index.size == 1  # 1 rule
        assert index.embeddings.shape[0] == 3  # 1 text + 2 expansions
        assert index.rule_map == (0, 0, 0)
        assert index.bm25_corpus == (
            "Never commit secrets",
            "hardcoded API key",
            "AKIA in source",
        )

    def test_build_mixed_expansions(self) -> None:
        """Rules with varying expansion counts produce correct mapping."""
        rules = (
            Rule(
                text="Rule A",
                provenance=Provenance(
                    file="/tmp/r.txt", line_start=1, line_end=1,
                ),
                expansions=("exp A1",),
            ),
            Rule(
                text="Rule B",
                provenance=Provenance(
                    file="/tmp/r.txt", line_start=2, line_end=2,
                ),
            ),
        )
        # 3 embeddings: rule A (text + 1 expansion) + rule B (text only)
        model = _make_mock_model(dim=384, n_rules=3)
        index = build_index(rules, {}, "test-model", model=model)

        assert index.embeddings.shape[0] == 3
        assert index.rule_map == (0, 0, 1)
        assert index.bm25_corpus == ("Rule A", "exp A1", "Rule B")

    def test_build_empty_rules_has_empty_fields(self) -> None:
        """Empty rules produce empty rule_map and bm25_corpus."""
        index = build_index((), {}, "test-model")

        assert index.rule_map == ()
        assert index.bm25_corpus == ()
