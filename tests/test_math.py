"""Tests for cuecard._math — shared numeric utilities."""

from __future__ import annotations

import numpy as np

from cuecard._math import l2_normalize


class TestL2Normalize:
    """Tests for the unified l2_normalize function."""

    def test_2d_normal_vectors(self) -> None:
        """2D batch: each row is L2-normalized."""
        rng = np.random.default_rng(99)
        emb = rng.standard_normal((4, 16)).astype(np.float32)
        normed = l2_normalize(emb)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_2d_zero_vector_handled(self) -> None:
        """2D: zero-norm rows stay zero (no NaN)."""
        emb = np.zeros((2, 8), dtype=np.float32)
        emb[1] = np.ones(8, dtype=np.float32)
        normed = l2_normalize(emb)
        # Zero row stays near-zero (divided by epsilon)
        assert not np.any(np.isnan(normed))
        # Non-zero row gets normalized
        assert abs(np.linalg.norm(normed[1]) - 1.0) < 1e-6

    def test_1d_vector_normalized(self) -> None:
        """1D vector is L2-normalized correctly."""
        vec = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        result = l2_normalize(vec)
        assert result.ndim == 1
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0], atol=1e-6)

    def test_1d_zero_vector(self) -> None:
        """1D zero vector does not produce NaN."""
        vec = np.zeros(4, dtype=np.float32)
        result = l2_normalize(vec)
        assert result.ndim == 1
        assert not np.any(np.isnan(result))

    def test_preserves_dtype(self) -> None:
        """Output dtype matches input dtype."""
        vec_32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert l2_normalize(vec_32).dtype == np.float32
        vec_64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert l2_normalize(vec_64).dtype == np.float64
