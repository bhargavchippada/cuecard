"""Tests for cuecard.indexer."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from cuecard.indexer import (
    _compute_checksum,
    build_index,
    load_index,
    save_index,
)
from cuecard.models import Index, Provenance, Rule, SourceMeta


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


class TestSaveIndex:
    def test_creates_directory(self, tmp_path: Path, sample_index: Index) -> None:
        cache_dir = str(tmp_path / "new_cache")
        save_index(sample_index, cache_dir)

        assert (Path(cache_dir) / "embeddings.npz").exists()
        assert (Path(cache_dir) / "metadata.json").exists()

    def test_directory_permissions(self, tmp_path: Path, sample_index: Index) -> None:
        cache_dir = str(tmp_path / "perms_cache")
        save_index(sample_index, cache_dir)

        dir_stat = os.stat(cache_dir)
        assert stat.S_IMODE(dir_stat.st_mode) == 0o700

    def test_file_permissions(self, tmp_path: Path, sample_index: Index) -> None:
        cache_dir = str(tmp_path / "file_perms")
        save_index(sample_index, cache_dir)

        npz_stat = os.stat(Path(cache_dir) / "embeddings.npz")
        meta_stat = os.stat(Path(cache_dir) / "metadata.json")
        assert stat.S_IMODE(npz_stat.st_mode) == 0o600
        assert stat.S_IMODE(meta_stat.st_mode) == 0o600

    def test_lock_file_created(self, tmp_path: Path, sample_index: Index) -> None:
        cache_dir = str(tmp_path / "lock_test")
        save_index(sample_index, cache_dir)

        assert (Path(cache_dir) / ".lock").exists()

    def test_metadata_format(self, tmp_path: Path, sample_index: Index) -> None:
        cache_dir = str(tmp_path / "meta_test")
        save_index(sample_index, cache_dir)

        with open(Path(cache_dir) / "metadata.json") as f:
            meta = json.load(f)

        assert meta["version"] == 1
        assert meta["model"] == "BAAI/bge-small-en-v1.5"
        assert meta["dim"] == 384
        assert meta["checksum"].startswith("sha256:")
        assert "created" in meta
        assert len(meta["rules"]) == 5
        assert "/tmp/rules.txt" in meta["sources"]

    def test_metadata_rule_fields(self, tmp_path: Path, sample_index: Index) -> None:
        cache_dir = str(tmp_path / "rule_fields")
        save_index(sample_index, cache_dir)

        with open(Path(cache_dir) / "metadata.json") as f:
            meta = json.load(f)

        rule_entry = meta["rules"][0]
        assert "text" in rule_entry
        assert "file" in rule_entry
        assert "line_start" in rule_entry
        assert "line_end" in rule_entry
        assert "section_path" in rule_entry
        assert "chunk_type" in rule_entry

    def test_metadata_sources_format(
        self, tmp_path: Path, sample_index: Index,
    ) -> None:
        cache_dir = str(tmp_path / "sources_fmt")
        save_index(sample_index, cache_dir)

        with open(Path(cache_dir) / "metadata.json") as f:
            meta = json.load(f)

        src = meta["sources"]["/tmp/rules.txt"]
        assert src["mtime"] == 1711929600.0
        assert src["content_hash"] == "sha256:abc123"
        assert src["rule_count"] == 5

    def test_checksum_matches_npz(self, tmp_path: Path, sample_index: Index) -> None:
        cache_dir = str(tmp_path / "checksum_test")
        save_index(sample_index, cache_dir)

        with open(Path(cache_dir) / "metadata.json") as f:
            meta = json.load(f)

        actual = _compute_checksum(str(Path(cache_dir) / "embeddings.npz"))
        assert meta["checksum"] == actual

    def test_atomic_overwrite(self, tmp_path: Path, sample_index: Index) -> None:
        cache_dir = str(tmp_path / "overwrite_test")
        save_index(sample_index, cache_dir)

        # Save again — should overwrite atomically
        save_index(sample_index, cache_dir)

        with open(Path(cache_dir) / "metadata.json") as f:
            meta = json.load(f)
        assert len(meta["rules"]) == 5

    def test_temp_files_cleaned_on_failure(
        self, tmp_path: Path, sample_index: Index,
    ) -> None:
        """Temp files are removed when save_index fails partway through."""
        from unittest.mock import patch

        cache_dir = str(tmp_path / "cleanup_test")
        Path(cache_dir).mkdir(parents=True)
        os.chmod(cache_dir, 0o700)

        # Make json.dump fail after npz is written
        with (
            patch("json.dump", side_effect=RuntimeError("simulated")),
            pytest.raises(RuntimeError, match="simulated"),
        ):
            save_index(sample_index, cache_dir)

        # No temp files should remain
        remaining = [
            f for f in Path(cache_dir).iterdir()
            if f.suffix in (".npz", ".json") and f.name.startswith("tmp")
        ]
        assert remaining == []

    def test_temp_cleanup_tolerates_unlink_failure(
        self, tmp_path: Path, sample_index: Index,
    ) -> None:
        """If os.unlink fails during cleanup, the error is silently ignored."""
        from unittest.mock import patch

        cache_dir = str(tmp_path / "unlink_fail_test")
        Path(cache_dir).mkdir(parents=True)
        os.chmod(cache_dir, 0o700)

        original_unlink = os.unlink

        def failing_unlink(path: str) -> None:
            if str(path).endswith(".json"):
                raise OSError("simulated unlink failure")
            original_unlink(path)

        with (
            patch("json.dump", side_effect=RuntimeError("simulated")),
            patch("os.unlink", side_effect=failing_unlink),
            pytest.raises(RuntimeError, match="simulated"),
        ):
            save_index(sample_index, cache_dir)


class TestLoadIndex:
    def test_successful_roundtrip(
        self, tmp_path: Path, sample_index: Index,
    ) -> None:
        cache_dir = str(tmp_path / "roundtrip")
        save_index(sample_index, cache_dir)

        loaded = load_index(cache_dir)

        assert loaded is not None
        assert loaded.size == 5
        assert loaded.model_name == "BAAI/bge-small-en-v1.5"
        assert loaded.dim == 384
        np.testing.assert_allclose(
            loaded.embeddings, sample_index.embeddings, atol=1e-6,
        )

    def test_rules_preserved(self, tmp_path: Path, sample_index: Index) -> None:
        cache_dir = str(tmp_path / "rules_roundtrip")
        save_index(sample_index, cache_dir)

        loaded = load_index(cache_dir)
        assert loaded is not None
        assert len(loaded.rules) == 5
        assert loaded.rules[0].text == sample_index.rules[0].text
        assert loaded.rules[0].provenance.file == "/tmp/rules.txt"
        assert loaded.rules[0].provenance.section_path == ()
        assert loaded.rules[0].provenance.chunk_type == "rule"

    def test_provenance_roundtrip_with_section_path(self, tmp_path: Path) -> None:
        """section_path and chunk_type survive save/load."""
        rules = (
            Rule(
                text="test rule",
                provenance=Provenance(
                    file="/tmp/rules.txt",
                    line_start=1,
                    line_end=1,
                    section_path=("heading", "subheading"),
                    chunk_type="section",
                ),
            ),
        )
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((1, 384)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms
        idx = Index(
            embeddings=emb,
            rules=rules,
            model_name="test-model",
            dim=384,
            sources={},
        )
        cache_dir = str(tmp_path / "provenance_roundtrip")
        save_index(idx, cache_dir)
        loaded = load_index(cache_dir)
        assert loaded is not None
        assert loaded.rules[0].provenance.section_path == ("heading", "subheading")
        assert loaded.rules[0].provenance.chunk_type == "section"

    def test_sources_preserved(self, tmp_path: Path, sample_index: Index) -> None:
        cache_dir = str(tmp_path / "sources_roundtrip")
        save_index(sample_index, cache_dir)

        loaded = load_index(cache_dir)
        assert loaded is not None
        assert "/tmp/rules.txt" in loaded.sources
        assert loaded.sources["/tmp/rules.txt"].mtime == 1711929600.0

    def test_missing_metadata_returns_none(self, tmp_path: Path) -> None:
        cache_dir = str(tmp_path / "missing_meta")
        Path(cache_dir).mkdir()
        # No files — should return None
        result = load_index(cache_dir)
        assert result is None

    def test_missing_npz_returns_none(self, tmp_path: Path) -> None:
        cache_dir = str(tmp_path / "missing_npz")
        Path(cache_dir).mkdir()
        meta_path = Path(cache_dir) / "metadata.json"
        meta_path.write_text('{"version": 1, "model": "test", "dim": 384, '
                             '"checksum": "sha256:bad", "rules": [], "sources": {}}')

        result = load_index(cache_dir)
        assert result is None

    def test_corrupted_checksum_returns_none(
        self, tmp_path: Path, sample_index: Index,
    ) -> None:
        cache_dir = str(tmp_path / "bad_checksum")
        save_index(sample_index, cache_dir)

        # Tamper with checksum in metadata
        meta_path = Path(cache_dir) / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["checksum"] = "sha256:0000000000000000000000000000000000000000"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        result = load_index(cache_dir)
        assert result is None

    def test_mismatched_rule_count_returns_none(
        self, tmp_path: Path, sample_index: Index,
    ) -> None:
        cache_dir = str(tmp_path / "mismatch")
        save_index(sample_index, cache_dir)

        # Tamper with rules list — remove some rules but keep valid checksum
        meta_path = Path(cache_dir) / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        # Keep only 2 rules but embeddings still has 5 rows
        meta["rules"] = meta["rules"][:2]
        # Fix the checksum so checksum check passes
        meta["checksum"] = _compute_checksum(
            str(Path(cache_dir) / "embeddings.npz"),
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        result = load_index(cache_dir)
        assert result is None

    def test_corrupted_json_returns_none(self, tmp_path: Path) -> None:
        cache_dir = str(tmp_path / "bad_json")
        Path(cache_dir).mkdir()
        (Path(cache_dir) / "metadata.json").write_text("{invalid json")
        (Path(cache_dir) / "embeddings.npz").write_bytes(b"not a real npz")

        result = load_index(cache_dir)
        assert result is None

    def test_nonexistent_dir_returns_none(self, tmp_path: Path) -> None:
        result = load_index(str(tmp_path / "does_not_exist"))
        assert result is None

    def test_empty_index_roundtrip(self, tmp_path: Path) -> None:
        empty_index = Index(
            embeddings=np.zeros((0, 384), dtype=np.float32),
            rules=(),
            model_name="test-model",
            dim=384,
            sources={},
        )
        cache_dir = str(tmp_path / "empty_roundtrip")
        save_index(empty_index, cache_dir)

        loaded = load_index(cache_dir)
        assert loaded is not None
        assert loaded.size == 0
        assert loaded.dim == 384


class TestComputeChecksum:
    def test_correct_sha256(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.bin"
        content = b"hello world"
        test_file.write_bytes(content)

        result = _compute_checksum(str(test_file))

        expected_hash = hashlib.sha256(content).hexdigest()
        assert result == f"sha256:{expected_hash}"

    def test_empty_file(self, tmp_path: Path) -> None:
        test_file = tmp_path / "empty.bin"
        test_file.write_bytes(b"")

        result = _compute_checksum(str(test_file))

        expected_hash = hashlib.sha256(b"").hexdigest()
        assert result == f"sha256:{expected_hash}"

    def test_deterministic(self, tmp_path: Path) -> None:
        test_file = tmp_path / "det.bin"
        test_file.write_bytes(b"deterministic content")

        first = _compute_checksum(str(test_file))
        second = _compute_checksum(str(test_file))
        assert first == second

    def test_different_content_different_checksum(self, tmp_path: Path) -> None:
        file_a = tmp_path / "a.bin"
        file_b = tmp_path / "b.bin"
        file_a.write_bytes(b"content A")
        file_b.write_bytes(b"content B")

        assert _compute_checksum(str(file_a)) != _compute_checksum(str(file_b))
