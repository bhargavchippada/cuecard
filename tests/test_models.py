"""Tests for cuecard.models."""

from __future__ import annotations

import numpy as np
import pytest

from cuecard.models import (
    Index,
    Provenance,
    RankedResult,
    ResolvedConfig,
    Rule,
    SourceMeta,
)


class TestProvenance:
    def test_frozen(self, sample_provenance: Provenance) -> None:
        with pytest.raises(AttributeError):
            sample_provenance.file = "other.txt"  # type: ignore[misc]

    def test_defaults(self) -> None:
        p = Provenance(file="f.txt", line_start=1, line_end=1)
        assert p.section_path == ()
        assert p.chunk_type == "rule"

    def test_with_section_path(self) -> None:
        p = Provenance(
            file="security.md",
            line_start=10,
            line_end=15,
            section_path=("Security", "Secrets"),
            chunk_type="paragraph",
        )
        assert p.section_path == ("Security", "Secrets")
        assert p.chunk_type == "paragraph"


class TestRule:
    def test_frozen(self, sample_rule: Rule) -> None:
        with pytest.raises(AttributeError):
            sample_rule.text = "changed"  # type: ignore[misc]

    def test_max_length_constant(self) -> None:
        r = Rule(
            text="test",
            provenance=Provenance(file="f.txt", line_start=1, line_end=1),
        )
        assert r.MAX_LENGTH == 500

    def test_summary_default_none(self, sample_rule: Rule) -> None:
        assert sample_rule.summary is None

    def test_with_summary(self, sample_provenance: Provenance) -> None:
        r = Rule(text="long text", provenance=sample_provenance, summary="short")
        assert r.summary == "short"


class TestRankedResult:
    def test_frozen(self, sample_rule: Rule) -> None:
        rr = RankedResult(rule=sample_rule, score=0.85)
        with pytest.raises(AttributeError):
            rr.score = 0.5  # type: ignore[misc]

    def test_fields(self, sample_rule: Rule) -> None:
        rr = RankedResult(rule=sample_rule, score=0.85)
        assert rr.rule is sample_rule
        assert rr.score == 0.85


class TestSourceMeta:
    def test_frozen(self) -> None:
        sm = SourceMeta(mtime=1.0, content_hash="sha256:abc", rule_count=5)
        with pytest.raises(AttributeError):
            sm.mtime = 2.0  # type: ignore[misc]

    def test_fields(self) -> None:
        sm = SourceMeta(mtime=1.0, content_hash="sha256:abc", rule_count=5)
        assert sm.mtime == 1.0
        assert sm.content_hash == "sha256:abc"
        assert sm.rule_count == 5


class TestResolvedConfig:
    def test_frozen(self) -> None:
        cfg = ResolvedConfig(
            source_paths=("/tmp/rules.txt",),
            global_source_paths=("/tmp/rules.txt",),
            project_source_paths=(),
            model_name="BAAI/bge-small-en-v1.5",
            top_k=5,
            threshold=0.30,
            dedup_threshold=0.95,
            query_max_length=500,
            hook_events=("PreToolUse",),
            verbose=False,
            redact=True,
            max_log_size_mb=10,
            global_cache_dir="~/.cuecard/index/",
            project_cache_dir=".cuecard/index/",
            allowed_dirs=(),
        )
        with pytest.raises(AttributeError):
            cfg.top_k = 10  # type: ignore[misc]

    def test_all_fields(self) -> None:
        cfg = ResolvedConfig(
            source_paths=("/a.txt", "/b.txt"),
            global_source_paths=("/a.txt",),
            project_source_paths=("/b.txt",),
            model_name="test-model",
            top_k=10,
            threshold=0.5,
            dedup_threshold=0.9,
            query_max_length=200,
            hook_events=("PreToolUse", "UserPromptSubmit"),
            verbose=True,
            redact=False,
            max_log_size_mb=50,
            global_cache_dir="/cache/global",
            project_cache_dir="/cache/project",
            allowed_dirs=("/extra",),
        )
        assert cfg.source_paths == ("/a.txt", "/b.txt")
        assert cfg.model_name == "test-model"
        assert cfg.top_k == 10
        assert cfg.threshold == 0.5
        assert cfg.verbose is True
        assert cfg.allowed_dirs == ("/extra",)


class TestIndex:
    def test_construction(self, sample_index: Index) -> None:
        assert sample_index.size == 5
        assert sample_index.model_name == "BAAI/bge-small-en-v1.5"
        assert sample_index.dim == 384
        assert sample_index.embeddings.shape == (5, 384)

    def test_mismatched_shape_raises(self, sample_rules: tuple[Rule, ...]) -> None:
        bad_emb = np.zeros((3, 384), dtype=np.float32)
        with pytest.raises(ValueError, match="Embedding rows"):
            Index(
                embeddings=bad_emb,
                rules=sample_rules,
                model_name="test",
                dim=384,
                sources={},
            )

    def test_repr(self, sample_index: Index) -> None:
        r = repr(sample_index)
        assert "BAAI/bge-small-en-v1.5" in r
        assert "rules=5" in r
        assert "dim=384" in r

    def test_empty_index(self) -> None:
        idx = Index(
            embeddings=np.zeros((0, 384), dtype=np.float32),
            rules=(),
            model_name="test",
            dim=384,
            sources={},
        )
        assert idx.size == 0

    def test_sources(self, sample_index: Index) -> None:
        assert "/tmp/rules.txt" in sample_index.sources
        assert sample_index.sources["/tmp/rules.txt"].rule_count == 5
