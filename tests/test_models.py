"""Tests for cuecard.models."""

from __future__ import annotations

import numpy as np
import pytest

from cuecard.models import (
    AffinityIndex,
    Index,
    LoadedIndex,
    Provenance,
    RankedResult,
    ResolvedConfig,
    Rule,
    RuleAffinity,
    SourceMeta,
    _hash_rule_text,
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

    def test_expansions_default_empty(self, sample_rule: Rule) -> None:
        assert sample_rule.expansions == ()

    def test_with_expansions(self, sample_provenance: Provenance) -> None:
        r = Rule(
            text="Never commit secrets",
            provenance=sample_provenance,
            expansions=("hardcoded API key", "AKIA in source"),
        )
        assert r.expansions == ("hardcoded API key", "AKIA in source")

    def test_expansions_frozen(self, sample_provenance: Provenance) -> None:
        r = Rule(
            text="test",
            provenance=sample_provenance,
            expansions=("one",),
        )
        with pytest.raises(AttributeError):
            r.expansions = ("changed",)  # type: ignore[misc]

    def test_backwards_compatible_construction(
        self, sample_provenance: Provenance,
    ) -> None:
        """Old code constructing Rule(text=, provenance=) still works."""
        r = Rule(text="old style", provenance=sample_provenance)
        assert r.expansions == ()
        assert r.summary is None
        assert r.events == frozenset()
        assert r.tools == frozenset()

    def test_events_and_tools(self, sample_provenance: Provenance) -> None:
        r = Rule(
            text="Use uv",
            provenance=sample_provenance,
            events=frozenset({"PreToolUse", "Stop"}),
            tools=frozenset({"Bash"}),
        )
        assert r.events == frozenset({"PreToolUse", "Stop"})
        assert r.tools == frozenset({"Bash"})

    def test_events_tools_frozen(
        self, sample_provenance: Provenance,
    ) -> None:
        r = Rule(
            text="test",
            provenance=sample_provenance,
            events=frozenset({"PreToolUse"}),
        )
        with pytest.raises(AttributeError):
            r.events = frozenset()  # type: ignore[misc]
        with pytest.raises(AttributeError):
            r.tools = frozenset()  # type: ignore[misc]


class TestKnownHookEvents:
    def test_contains_all_events(self) -> None:
        from cuecard.models import KNOWN_HOOK_EVENTS

        expected = {
            "PreToolUse", "UserPromptSubmit",
            "SubagentStart", "Stop",
        }
        assert expected == KNOWN_HOOK_EVENTS

    def test_is_frozenset(self) -> None:
        from cuecard.models import KNOWN_HOOK_EVENTS

        assert isinstance(KNOWN_HOOK_EVENTS, frozenset)


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

    def test_default_rule_map(self, sample_index: Index) -> None:
        """Default rule_map is identity mapping."""
        assert sample_index.rule_map == tuple(range(5))
        assert sample_index.bm25_corpus is None

    def test_explicit_rule_map(self) -> None:
        """Explicit rule_map and bm25_corpus are preserved."""
        rules = (
            Rule(
                text="rule A",
                provenance=Provenance(file="f.txt", line_start=1, line_end=1),
            ),
        )
        emb = np.zeros((3, 4), dtype=np.float32)
        idx = Index(
            embeddings=emb,
            rules=rules,
            model_name="test",
            dim=4,
            sources={},
            rule_map=(0, 0, 0),
            bm25_corpus=("rule A", "expansion 1", "expansion 2"),
        )
        assert idx.rule_map == (0, 0, 0)
        assert idx.bm25_corpus == ("rule A", "expansion 1", "expansion 2")

    def test_rule_map_out_of_range_raises(self) -> None:
        """rule_map index beyond rules count raises ValueError."""
        rules = (
            Rule(
                text="only rule",
                provenance=Provenance(file="f.txt", line_start=1, line_end=1),
            ),
        )
        emb = np.zeros((2, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="rule_map indices"):
            Index(
                embeddings=emb,
                rules=rules,
                model_name="test",
                dim=4,
                sources={},
                rule_map=(0, 1),  # index 1 is out of range for 1 rule
            )

    def test_rule_map_length_mismatch_raises(self) -> None:
        """rule_map length not matching embeddings raises ValueError."""
        rules = (
            Rule(
                text="rule",
                provenance=Provenance(file="f.txt", line_start=1, line_end=1),
            ),
        )
        emb = np.zeros((2, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="rule_map length"):
            Index(
                embeddings=emb,
                rules=rules,
                model_name="test",
                dim=4,
                sources={},
                rule_map=(0,),  # length 1 but 2 embeddings
            )

    def test_bm25_corpus_length_mismatch_raises(self) -> None:
        """bm25_corpus length not matching rule_map raises ValueError."""
        rules = (
            Rule(
                text="rule",
                provenance=Provenance(file="f.txt", line_start=1, line_end=1),
            ),
        )
        emb = np.zeros((2, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="bm25_corpus length"):
            Index(
                embeddings=emb,
                rules=rules,
                model_name="test",
                dim=4,
                sources={},
                rule_map=(0, 0),
                bm25_corpus=("only one",),  # length 1 but rule_map is 2
            )


class TestRuleAffinity:
    def test_frozen(self) -> None:
        ra = RuleAffinity(
            events=frozenset({"PreToolUse"}),
            tools=frozenset({"Bash"}),
            source="explicit",
        )
        with pytest.raises(AttributeError):
            ra.events = frozenset()  # type: ignore[misc]

    def test_fields(self) -> None:
        ra = RuleAffinity(
            events=frozenset({"PreToolUse", "Stop"}),
            tools=frozenset({"Bash", "Edit"}),
            source="explicit+inferred",
            explicit_events=frozenset({"PreToolUse"}),
            explicit_tools=frozenset({"Bash"}),
            reasoning="Git operations",
        )
        assert ra.events == frozenset({"PreToolUse", "Stop"})
        assert ra.tools == frozenset({"Bash", "Edit"})
        assert ra.source == "explicit+inferred"
        assert ra.explicit_events == frozenset({"PreToolUse"})
        assert ra.explicit_tools == frozenset({"Bash"})
        assert ra.reasoning == "Git operations"

    def test_defaults(self) -> None:
        ra = RuleAffinity(
            events=frozenset(),
            tools=frozenset(),
            source="default",
        )
        assert ra.explicit_events == frozenset()
        assert ra.explicit_tools == frozenset()
        assert ra.reasoning == ""

    def test_events_tools_are_frozenset(self) -> None:
        ra = RuleAffinity(
            events=frozenset({"PreToolUse"}),
            tools=frozenset({"Bash"}),
            source="explicit",
        )
        assert isinstance(ra.events, frozenset)
        assert isinstance(ra.tools, frozenset)


class TestAffinityIndex:
    def _make_index(self) -> AffinityIndex:
        ra = RuleAffinity(
            events=frozenset({"PreToolUse"}),
            tools=frozenset({"Bash"}),
            source="explicit",
        )
        text_hash = _hash_rule_text("Never commit secrets")
        return AffinityIndex(
            version=1,
            mode="infer",
            model="local",
            affinities=((text_hash, ra),),
        )

    def test_get_returns_correct_affinity(
        self, sample_provenance: Provenance,
    ) -> None:
        idx = self._make_index()
        rule = Rule(
            text="Never commit secrets",
            provenance=sample_provenance,
        )
        result = idx.get(rule)
        assert result is not None
        assert result.events == frozenset({"PreToolUse"})

    def test_get_returns_none_for_unknown(
        self, sample_provenance: Provenance,
    ) -> None:
        idx = self._make_index()
        rule = Rule(
            text="Unknown rule text",
            provenance=sample_provenance,
        )
        assert idx.get(rule) is None

    def test_get_by_hash(self) -> None:
        idx = self._make_index()
        text_hash = _hash_rule_text("Never commit secrets")
        result = idx.get_by_hash(text_hash)
        assert result is not None
        assert result.source == "explicit"

    def test_get_by_hash_unknown(self) -> None:
        idx = self._make_index()
        assert idx.get_by_hash("nonexistent") is None

    def test_not_hashable(self) -> None:
        idx = self._make_index()
        with pytest.raises(TypeError):
            hash(idx)

    def test_items_property(self) -> None:
        idx = self._make_index()
        items = idx.items
        assert isinstance(items, tuple)
        assert len(items) == 1

    def test_repr(self) -> None:
        idx = self._make_index()
        r = repr(idx)
        assert "infer" in r
        assert "rules=1" in r

    def test_fields(self) -> None:
        idx = self._make_index()
        assert idx.version == 1
        assert idx.mode == "infer"
        assert idx.model == "local"


class TestLoadedIndex:
    def test_frozen(self, sample_index: Index) -> None:
        li = LoadedIndex(index=sample_index)
        with pytest.raises(AttributeError):
            li.index = sample_index  # type: ignore[misc]

    def test_default_affinity_none(self, sample_index: Index) -> None:
        li = LoadedIndex(index=sample_index)
        assert li.affinity is None

    def test_with_affinity(self, sample_index: Index) -> None:
        aff_idx = AffinityIndex(
            version=1, mode="strict", model="",
            affinities=(),
        )
        li = LoadedIndex(index=sample_index, affinity=aff_idx)
        assert li.index is sample_index
        assert li.affinity is aff_idx
