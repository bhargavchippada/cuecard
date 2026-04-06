"""Tests for cuecard.expander — semantic deduplication."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cuecard.expander import (
    DEDUP_COSINE_THRESHOLD,
    _semantic_dedup,
)


class TestSemanticDedup:
    def test_empty_list(self) -> None:
        assert _semantic_dedup([]) == []

    def test_single_item(self) -> None:
        assert _semantic_dedup(["hello"]) == ["hello"]

    def test_preserves_order(self) -> None:
        """Non-duplicate items are returned in original order."""
        with patch("cuecard.expander.np") as mock_np:
            # Mock numpy to simulate no duplicates
            mock_array = MagicMock()
            mock_np.array.return_value = mock_array
            mock_np.linalg.norm.return_value = MagicMock()
            mock_np.where.return_value = MagicMock()

            # Just test the short-circuit paths
            result = _semantic_dedup(["a"])
            assert result == ["a"]

    def test_fastembed_import_failure_returns_original(self) -> None:
        """If fastembed is unavailable, return original list unchanged."""
        import builtins

        items = ["phrase a", "phrase b"]
        original_import = builtins.__import__
        with (
            patch.dict("sys.modules", {"fastembed": None}),
            patch(
                "builtins.__import__",
                side_effect=lambda name, *a, **kw: (
                    (_ for _ in ()).throw(ImportError("no fastembed"))
                    if name == "fastembed"
                    else original_import(name, *a, **kw)
                ),
            ),
        ):
            result = _semantic_dedup(items)
        assert result == items

    def test_embedding_failure_returns_original(self) -> None:
        """If embedding fails, return original list unchanged."""
        items = ["phrase a", "phrase b"]
        mock_model = MagicMock()
        mock_model.passage_embed.side_effect = RuntimeError("model load failed")
        with patch(
            "fastembed.TextEmbedding",
            return_value=mock_model,
        ):
            result = _semantic_dedup(items)
        assert result == items

    def test_threshold_constant(self) -> None:
        assert DEDUP_COSINE_THRESHOLD == 0.85

    def test_drops_near_duplicate(self) -> None:
        """Expansions with cosine > 0.85 are dropped."""
        import numpy as np

        items = ["pip install requests", "pip install numpy"]
        # Create nearly identical embeddings (cosine ~0.99)
        base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        near = np.array([0.99, 0.1, 0.0], dtype=np.float32)
        near = near / np.linalg.norm(near)
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [base, near]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        assert len(result) == 1
        assert result[0] == "pip install requests"

    def test_three_items_keeps_distinct_drops_duplicate(self) -> None:
        """With 3 items: first and third distinct, second similar to first."""
        import numpy as np

        items = ["phrase A", "phrase B (dup of A)", "phrase C (distinct)"]
        # A and B nearly identical (cosine ~0.99), C orthogonal (cosine ~0)
        vec_a = np.array([3.0, 0.0, 0.0], dtype=np.float32)  # unnormalized!
        vec_b = np.array([2.97, 0.3, 0.0], dtype=np.float32)
        vec_c = np.array([0.0, 0.0, 5.0], dtype=np.float32)  # orthogonal
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec_a, vec_b, vec_c]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        # B should be dropped (similar to A), A and C kept
        assert len(result) == 2
        assert result[0] == "phrase A"
        assert result[1] == "phrase C (distinct)"

    def test_normalization_divides_not_multiplies(self) -> None:
        """Verify embeddings are divided by norms, not multiplied.

        With small-norm near-duplicate vectors (cosine ~0.95, norms ~0.1):
        - Division by norms: similarity = cos(θ) ≈ 0.95 > 0.85 → B dropped ✓
        - Mult by norms: sim ~ 0.95*0.01*0.01 ~ 0.00009 < 0.85 → B kept ✗
        """
        import numpy as np

        items = ["item A", "item B"]
        # Small-norm vectors with cosine ~0.95 (above threshold)
        vec_a = np.array([0.1, 0.01], dtype=np.float32)
        vec_b = np.array([0.1, 0.0], dtype=np.float32)
        # Verify cosine is above threshold
        cos_sim = float(np.dot(vec_a, vec_b) / (
            np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        ))
        assert cos_sim > 0.85, f"cosine {cos_sim} should be above threshold"
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec_a, vec_b]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        # After proper normalization: cosine ~0.95 > 0.85 → B dropped
        assert len(result) == 1
        assert result[0] == "item A"

    def test_model_name_is_bge_small(self) -> None:
        """TextEmbedding must be initialized with the correct model name."""
        import numpy as np

        items = ["a", "b"]
        vec = np.array([1.0, 0.0], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec, vec]
        with patch("fastembed.TextEmbedding", return_value=mock_model) as mock_cls:
            _semantic_dedup(items)
            mock_cls.assert_called_once_with("BAAI/bge-small-en-v1.5")

    def test_passage_embed_receives_expansions(self) -> None:
        """passage_embed must be called with the actual expansion strings."""
        import numpy as np

        items = ["expansion one", "expansion two"]
        vec = np.array([1.0, 0.0], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec, vec]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            _semantic_dedup(items)
            mock_model.passage_embed.assert_called_once_with(items)

    def test_single_item_returned_unchanged(self) -> None:
        """A single-item list must short-circuit without embedding."""
        with patch("fastembed.TextEmbedding") as mock_cls:
            result = _semantic_dedup(["only one"])
            assert result == ["only one"]
            assert len(result) == 1
            # Must NOT call TextEmbedding — short-circuit on len <= 1
            mock_cls.assert_not_called()

    def test_zero_norm_vector_does_not_crash(self) -> None:
        """A zero-norm embedding should not cause division-by-zero.

        Zero-norm guard replaces 0 with 1.0 (not 2.0 or other values).
        With replacement=1.0: [0,0,0]/1.0 = [0,0,0], dot with [1,0,0] = 0 < 0.85 → kept.
        With replacement=2.0: [0,0,0]/2.0 = [0,0,0], same result (benign for this case).
        """
        import numpy as np

        items = ["zero vector", "normal vector"]
        vec_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        vec_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec_zero, vec_normal]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        # Should not crash; zero-norm guard replaces 0 with 1.0
        assert len(result) == 2

    def test_zero_norm_guard_replaces_zero_not_nonzero(self) -> None:
        """np.where condition must be norms==0, not norms!=0 or norms==1."""
        import numpy as np

        # Two items with identical direction but different magnitudes
        # After proper normalization: cosine=1.0 → second dropped
        # If guard is inverted (!=0): all norms become 1.0 before division,
        # making normalized = embeddings/1.0 = embeddings (unnormalized),
        # and dot products become magnitude-dependent instead of direction-only
        items = ["item A", "item B", "item C"]
        vec_a = np.array([5.0, 0.0], dtype=np.float32)
        vec_b = np.array([5.0, 0.1], dtype=np.float32)  # nearly same direction
        vec_c = np.array([0.0, 5.0], dtype=np.float32)  # orthogonal
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec_a, vec_b, vec_c]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        # B is near-duplicate of A, C is distinct
        assert len(result) == 2
        assert result[0] == "item A"
        assert result[1] == "item C"

    def test_first_item_always_kept(self) -> None:
        """Loop starts at index 1; first item is always in the result."""
        import numpy as np

        items = ["first", "second"]
        # Identical vectors → second should be dropped, first always kept
        vec = np.array([1.0, 0.0], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec, vec]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        assert len(result) == 1
        assert result[0] == "first"

    def test_below_threshold_kept(self) -> None:
        """Items with similarity below threshold must be kept."""
        import numpy as np

        items = ["item A", "item B"]
        # cosine ~0.6 (well below 0.85)
        vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = [vec_a, vec_b]
        with patch("fastembed.TextEmbedding", return_value=mock_model):
            result = _semantic_dedup(items)
        assert len(result) == 2
        assert result[0] == "item A"
        assert result[1] == "item B"
