"""Tests for cuecard.indexer — save_index, load_index, and v2 metadata."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cuecard.freshness import compute_file_hash
from cuecard.indexer import (
    load_index,
    save_index,
)
from cuecard.models import Index, Provenance, Rule


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

        assert meta["version"] == 2
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

        actual = compute_file_hash(str(Path(cache_dir) / "embeddings.npz"))
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
        meta["checksum"] = compute_file_hash(
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


class TestSaveLoadV2:
    """Tests for v2 metadata format with rule_map and bm25_corpus."""

    def test_v2_roundtrip(self, tmp_path: Path) -> None:
        """V2 index saves and loads rule_map, bm25_corpus, and expansions."""
        rules = (
            Rule(
                text="Rule A",
                provenance=Provenance(
                    file="/tmp/r.txt", line_start=1, line_end=1,
                ),
                expansions=("exp 1", "exp 2"),
            ),
        )
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((3, 384)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms

        idx = Index(
            embeddings=emb,
            rules=rules,
            model_name="test-model",
            dim=384,
            sources={},
            rule_map=(0, 0, 0),
            bm25_corpus=("Rule A", "exp 1", "exp 2"),
        )

        cache_dir = str(tmp_path / "v2_roundtrip")
        save_index(idx, cache_dir)

        loaded = load_index(cache_dir)
        assert loaded is not None
        assert loaded.rule_map == (0, 0, 0)
        assert loaded.bm25_corpus == ("Rule A", "exp 1", "exp 2")
        assert loaded.rules[0].expansions == ("exp 1", "exp 2")
        assert loaded.embeddings.shape[0] == 3

    def test_v1_metadata_loads_with_identity_rule_map(
        self, tmp_path: Path, sample_index: Index,
    ) -> None:
        """V1 metadata (no rule_map) loads with identity mapping."""
        cache_dir = str(tmp_path / "v1_compat")
        save_index(sample_index, cache_dir)

        # Downgrade metadata to v1
        meta_path = Path(cache_dir) / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["version"] = 1
        del meta["rule_map"]
        del meta["bm25_corpus"]
        # Fix checksum
        meta["checksum"] = compute_file_hash(
            str(Path(cache_dir) / "embeddings.npz"),
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        loaded = load_index(cache_dir)
        assert loaded is not None
        assert loaded.rule_map == tuple(range(5))
        assert loaded.bm25_corpus is None

    def test_v2_bad_rule_map_returns_none(self, tmp_path: Path) -> None:
        """V2 metadata with out-of-range rule_map returns None."""
        rules = (
            Rule(
                text="Rule A",
                provenance=Provenance(
                    file="/tmp/r.txt", line_start=1, line_end=1,
                ),
            ),
        )
        emb = np.zeros((1, 384), dtype=np.float32)
        idx = Index(
            embeddings=emb,
            rules=rules,
            model_name="test",
            dim=384,
            sources={},
        )
        cache_dir = str(tmp_path / "bad_rulemap")
        save_index(idx, cache_dir)

        # Tamper rule_map to have out-of-range index
        meta_path = Path(cache_dir) / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["rule_map"] = [5]  # out of range
        # Fix checksum
        meta["checksum"] = compute_file_hash(
            str(Path(cache_dir) / "embeddings.npz"),
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        loaded = load_index(cache_dir)
        assert loaded is None

    def test_v2_rule_map_length_mismatch_returns_none(
        self, tmp_path: Path,
    ) -> None:
        """V2 with rule_map length != embedding rows returns None."""
        rules = (
            Rule(
                text="Rule A",
                provenance=Provenance(
                    file="/tmp/r.txt", line_start=1, line_end=1,
                ),
            ),
        )
        emb = np.zeros((1, 384), dtype=np.float32)
        idx = Index(
            embeddings=emb,
            rules=rules,
            model_name="test",
            dim=384,
            sources={},
        )
        cache_dir = str(tmp_path / "mismatched_map")
        save_index(idx, cache_dir)

        # Tamper rule_map to have wrong length
        meta_path = Path(cache_dir) / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["rule_map"] = [0, 0, 0]  # length 3 but only 1 embedding
        meta["checksum"] = compute_file_hash(
            str(Path(cache_dir) / "embeddings.npz"),
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        loaded = load_index(cache_dir)
        assert loaded is None

    def test_v1_mismatched_rule_count_returns_none(
        self, tmp_path: Path, sample_index: Index,
    ) -> None:
        """V1 metadata with rule count != embedding rows returns None."""
        cache_dir = str(tmp_path / "v1_mismatch")
        save_index(sample_index, cache_dir)

        meta_path = Path(cache_dir) / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        # Downgrade to v1 and remove v2 fields
        meta["version"] = 1
        del meta["rule_map"]
        del meta["bm25_corpus"]
        # Keep only 2 rules but embeddings still has 5 rows
        meta["rules"] = meta["rules"][:2]
        meta["checksum"] = compute_file_hash(
            str(Path(cache_dir) / "embeddings.npz"),
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        loaded = load_index(cache_dir)
        assert loaded is None

    def test_v2_bm25_corpus_none(self, tmp_path: Path) -> None:
        """V2 with null bm25_corpus loads correctly."""
        rules = (
            Rule(
                text="Rule A",
                provenance=Provenance(
                    file="/tmp/r.txt", line_start=1, line_end=1,
                ),
            ),
        )
        emb = np.zeros((1, 384), dtype=np.float32)
        idx = Index(
            embeddings=emb,
            rules=rules,
            model_name="test",
            dim=384,
            sources={},
            bm25_corpus=None,
        )
        cache_dir = str(tmp_path / "no_bm25")
        save_index(idx, cache_dir)

        loaded = load_index(cache_dir)
        assert loaded is not None
        assert loaded.bm25_corpus is None
