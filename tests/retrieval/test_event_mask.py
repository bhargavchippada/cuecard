"""Tests for event mask building and integration with retrievers and pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np

from cuecard.models import (
    AffinityIndex,
    Index,
    LoadedIndex,
    PipelineResult,
    Provenance,
    Rule,
    RuleAffinity,
    StageTrace,
    _hash_rule_text,
)
from cuecard.retrieval.affinity import build_event_mask, build_strict_affinity
from cuecard.retrieval.dense import DenseRetriever
from cuecard.retrieval.fusion import ScoredCandidate
from cuecard.retrieval.sparse import SparseRetriever

if TYPE_CHECKING:
    from pathlib import Path


# --- Helpers ---


def _make_rule(
    text: str,
    events: frozenset[str] | None = None,
    tools: frozenset[str] | None = None,
) -> Rule:
    return Rule(
        text=text,
        provenance=Provenance(file="test.txt", line_start=1, line_end=1),
        events=events or frozenset(),
        tools=tools or frozenset(),
    )


def _make_affinity(
    rules: list[Rule],
    affinities: list[RuleAffinity],
) -> AffinityIndex:
    """Build an AffinityIndex from parallel lists of rules and affinities."""
    items = tuple(
        (_hash_rule_text(rule.text), aff)
        for rule, aff in zip(rules, affinities, strict=True)
    )
    return AffinityIndex(
        version=1, mode="strict", model="", affinities=items,
    )


def _make_index(
    rules: tuple[Rule, ...],
    rule_map: tuple[int, ...] | None = None,
    bm25_corpus: tuple[str, ...] | None = None,
) -> Index:
    """Build a minimal Index for testing."""
    n = len(rule_map) if rule_map else len(rules)
    emb = np.zeros((n, 8), dtype=np.float32)
    # Make each embedding slightly different so scores vary
    for i in range(n):
        emb[i, i % 8] = 1.0
    return Index(
        embeddings=emb,
        rules=rules,
        model_name="test",
        dim=8,
        sources={},
        rule_map=rule_map,
        bm25_corpus=bm25_corpus,
    )


# --- build_event_mask tests ---


class TestBuildEventMask:
    """Tests for build_event_mask() in affinity.py."""

    def test_basic_mask_filters_by_event(self) -> None:
        """Rules with matching event are True; others False."""
        rules = [
            _make_rule("Rule A"),
            _make_rule("Rule B"),
        ]
        affinities = [
            RuleAffinity(
                events=frozenset({"PreToolUse"}),
                tools=frozenset(),
                source="explicit",
            ),
            RuleAffinity(
                events=frozenset({"PostToolUse"}),
                tools=frozenset(),
                source="explicit",
            ),
        ]
        index = _make_index(tuple(rules))
        aff_idx = _make_affinity(rules, affinities)

        mask = build_event_mask(index, aff_idx, "PreToolUse")

        assert mask[0] is np.True_
        assert mask[1] is np.False_

    def test_unclassified_rules_pass_through(self) -> None:
        """Rules without affinity entry get True (pass through)."""
        rules = [_make_rule("Rule A"), _make_rule("Rule B")]
        # Only classify Rule A
        affinities = [
            RuleAffinity(
                events=frozenset({"Stop"}),
                tools=frozenset(),
                source="explicit",
            ),
        ]
        items = ((_hash_rule_text(rules[0].text), affinities[0]),)
        aff_idx = AffinityIndex(
            version=1, mode="strict", model="", affinities=items,
        )
        index = _make_index(tuple(rules))

        mask = build_event_mask(index, aff_idx, "PreToolUse")

        # Rule A has Stop only -> False for PreToolUse
        assert mask[0] is np.False_
        # Rule B has no affinity -> pass through (True)
        assert mask[1] is np.True_

    def test_all_true_when_all_match(self) -> None:
        """All rules match the event -> all True."""
        rules = [_make_rule("Rule A"), _make_rule("Rule B")]
        affinities = [
            RuleAffinity(
                events=frozenset({"PreToolUse"}),
                tools=frozenset(),
                source="explicit",
            ),
            RuleAffinity(
                events=frozenset({"PreToolUse", "PostToolUse"}),
                tools=frozenset(),
                source="explicit",
            ),
        ]
        index = _make_index(tuple(rules))
        aff_idx = _make_affinity(rules, affinities)

        mask = build_event_mask(index, aff_idx, "PreToolUse")
        assert mask.all()

    def test_all_false_when_none_match(self) -> None:
        """No rules match the event -> all False."""
        rules = [_make_rule("Rule A"), _make_rule("Rule B")]
        affinities = [
            RuleAffinity(
                events=frozenset({"Stop"}),
                tools=frozenset(),
                source="explicit",
            ),
            RuleAffinity(
                events=frozenset({"Stop"}),
                tools=frozenset(),
                source="explicit",
            ),
        ]
        index = _make_index(tuple(rules))
        aff_idx = _make_affinity(rules, affinities)

        mask = build_event_mask(index, aff_idx, "PreToolUse")
        assert not mask.any()

    def test_tool_name_filtering(self) -> None:
        """When tool_name specified, only matching tools pass."""
        rules = [
            _make_rule("Rule A"),
            _make_rule("Rule B"),
            _make_rule("Rule C"),
        ]
        affinities = [
            RuleAffinity(
                events=frozenset({"PreToolUse"}),
                tools=frozenset({"Bash"}),
                source="explicit",
            ),
            RuleAffinity(
                events=frozenset({"PreToolUse"}),
                tools=frozenset({"Edit"}),
                source="explicit",
            ),
            RuleAffinity(
                events=frozenset({"PreToolUse"}),
                tools=frozenset(),  # No tool filter -> matches all tools
                source="explicit",
            ),
        ]
        index = _make_index(tuple(rules))
        aff_idx = _make_affinity(rules, affinities)

        mask = build_event_mask(index, aff_idx, "PreToolUse", tool_name="Bash")

        assert mask[0] is np.True_   # Bash matches
        assert mask[1] is np.False_  # Edit doesn't match Bash
        assert mask[2] is np.True_   # No tool filter -> matches

    def test_tool_name_none_passes_all_tools(self) -> None:
        """When tool_name is None, all rules with matching event pass."""
        rules = [_make_rule("Rule A")]
        affinities = [
            RuleAffinity(
                events=frozenset({"PreToolUse"}),
                tools=frozenset({"Bash"}),
                source="explicit",
            ),
        ]
        index = _make_index(tuple(rules))
        aff_idx = _make_affinity(rules, affinities)

        mask = build_event_mask(index, aff_idx, "PreToolUse", tool_name=None)
        assert mask[0] is np.True_

    def test_expansion_rule_map(self) -> None:
        """Mask expands from rule-level to embedding-level via rule_map."""
        rules = [_make_rule("Rule A"), _make_rule("Rule B")]
        affinities = [
            RuleAffinity(
                events=frozenset({"PreToolUse"}),
                tools=frozenset(),
                source="explicit",
            ),
            RuleAffinity(
                events=frozenset({"Stop"}),
                tools=frozenset(),
                source="explicit",
            ),
        ]
        # Rule A has 2 expansions, Rule B has 1: rule_map = [0,0,1]
        rule_map = (0, 0, 1)
        index = _make_index(tuple(rules), rule_map=rule_map)
        aff_idx = _make_affinity(rules, affinities)

        mask = build_event_mask(index, aff_idx, "PreToolUse")

        assert len(mask) == 3  # embedding-level
        assert mask[0] is np.True_   # Rule A expansion 1
        assert mask[1] is np.True_   # Rule A expansion 2
        assert mask[2] is np.False_  # Rule B


# --- Dense retriever with mask ---


class TestDenseRetrieverMask:
    """Dense retriever respects the event mask post-scoring."""

    def test_mask_filters_results(self) -> None:
        """Masked embeddings are excluded from results."""
        rules = (_make_rule("Rule 0"), _make_rule("Rule 1"))
        emb = np.eye(2, 8, dtype=np.float32)
        index = Index(
            embeddings=emb,
            rules=rules,
            model_name="test",
            dim=8,
            sources={},
        )

        # Query matches Rule 0 perfectly
        query_vec = emb[0:1].copy()
        model = MagicMock()
        model.query_embed.return_value = iter([query_vec.flatten()])

        # Mask out Rule 0
        mask = np.array([False, True], dtype=bool)

        retriever = DenseRetriever(model=model, dedup_threshold=0.95)
        results = retriever.retrieve(
            "test", index, top_k=5, threshold=0.0, mask=mask,
        )

        # Rule 0 is masked out despite perfect match
        rule_texts = [r.rule.text for r in results]
        assert "Rule 0" not in rule_texts

    def test_no_mask_returns_all(self) -> None:
        """Without mask, all rules returned as normal."""
        rules = (_make_rule("Rule 0"), _make_rule("Rule 1"))
        emb = np.eye(2, 8, dtype=np.float32)
        index = Index(
            embeddings=emb,
            rules=rules,
            model_name="test",
            dim=8,
            sources={},
        )

        query_vec = emb[0:1].copy()
        model = MagicMock()
        model.query_embed.return_value = iter([query_vec.flatten()])

        retriever = DenseRetriever(model=model, dedup_threshold=0.95)
        results = retriever.retrieve(
            "test", index, top_k=5, threshold=0.0, mask=None,
        )

        assert len(results) >= 1

    def test_mask_does_not_mutate_embeddings(self) -> None:
        """Applying mask must not modify the index embeddings."""
        rules = (_make_rule("Rule 0"),)
        emb = np.ones((1, 8), dtype=np.float32)
        emb_copy = emb.copy()
        index = Index(
            embeddings=emb,
            rules=rules,
            model_name="test",
            dim=8,
            sources={},
        )

        query_vec = np.ones((1, 8), dtype=np.float32)
        model = MagicMock()
        model.query_embed.return_value = iter([query_vec.flatten()])

        mask = np.array([False], dtype=bool)
        retriever = DenseRetriever(model=model, dedup_threshold=0.95)
        retriever.retrieve("test", index, top_k=5, threshold=0.0, mask=mask)

        # Original embeddings unchanged
        np.testing.assert_array_equal(index.embeddings, emb_copy)


# --- Sparse retriever with mask ---


class TestSparseRetrieverMask:
    """Sparse retriever respects the event mask post-scoring."""

    def test_mask_filters_results(self) -> None:
        """Masked documents are excluded from sparse results."""
        rules = (_make_rule("cat sat mat"), _make_rule("dog played yard"))
        corpus = ("the cat sat on the mat", "the dog played in the yard")
        index = _make_index(rules, bm25_corpus=corpus)

        # Mask out doc 0 (cat)
        mask = np.array([False, True], dtype=bool)

        retriever = SparseRetriever()
        results = retriever.retrieve(
            "cat sat", index, top_k=5, threshold=0.0, mask=mask,
        )

        rule_texts = [r.rule.text for r in results]
        assert "cat sat mat" not in rule_texts

    def test_no_mask_returns_all(self) -> None:
        """Without mask, BM25 returns matching results normally."""
        rules = (_make_rule("cat sat mat"), _make_rule("dog played yard"))
        corpus = ("the cat sat on the mat", "the dog played in the yard")
        index = _make_index(rules, bm25_corpus=corpus)

        retriever = SparseRetriever()
        results = retriever.retrieve(
            "cat sat", index, top_k=5, threshold=0.0, mask=None,
        )

        assert len(results) >= 1

    def test_idf_unchanged_by_mask(self) -> None:
        """BM25 IDF statistics must not change when mask is applied.

        Post-scoring means the BM25 model sees the full corpus for IDF
        computation, then the mask zeroes scores after.
        """
        rules = (_make_rule("Rule A"), _make_rule("Rule B"))
        corpus = ("alpha beta gamma", "alpha delta epsilon")
        index = _make_index(rules, bm25_corpus=corpus)

        retriever = SparseRetriever()

        # Get unmasked scores
        unmasked = retriever.retrieve(
            "alpha", index, top_k=5, threshold=0.0, mask=None,
        )

        # Get masked scores (mask out doc 1)
        mask = np.array([True, False], dtype=bool)
        masked = retriever.retrieve(
            "alpha", index, top_k=5, threshold=0.0, mask=mask,
        )

        # Rule A should have same score in both (IDF unchanged)
        unmasked_a = [r for r in unmasked if r.rule.text == "Rule A"]
        masked_a = [r for r in masked if r.rule.text == "Rule A"]
        assert len(unmasked_a) == 1
        assert len(masked_a) == 1
        assert abs(unmasked_a[0].score - masked_a[0].score) < 1e-10

    def test_mask_with_expansion_rule_map(self) -> None:
        """Mask at embedding level collapses correctly via rule_map."""
        rules = (_make_rule("Rule A"), _make_rule("Rule B"))
        # Rule A has 2 expansion rows, Rule B has 1
        rule_map = (0, 0, 1)
        corpus = ("rule a text", "rule a expansion", "rule b text")
        index = _make_index(rules, rule_map=rule_map, bm25_corpus=corpus)

        # Mask: True for Rule A rows, False for Rule B
        mask = np.array([True, True, False], dtype=bool)

        retriever = SparseRetriever()
        results = retriever.retrieve(
            "rule text", index, top_k=5, threshold=0.0, mask=mask,
        )

        rule_texts = [r.rule.text for r in results]
        assert "Rule B" not in rule_texts


# --- Pipeline with affinity ---


class TestPipelineEventMask:
    """Pipeline builds event mask and passes to retrievers."""

    def _make_config(self) -> object:
        from cuecard.models import PipelineConfig, ResolvedConfig

        return ResolvedConfig(
            source_paths=(),
            global_source_paths=(),
            project_source_paths=(),
            model_name="test",
            top_k=5,
            threshold=0.30,
            dedup_threshold=0.95,
            query_max_length=500,
            hook_events=(),
            verbose=False,
            redact=False,
            max_log_size_mb=10,
            global_cache_dir="/tmp",
            project_cache_dir=None,
            allowed_dirs=(),
            pipeline=PipelineConfig(),
        )

    def test_pipeline_with_affinity(self) -> None:
        """Pipeline applies event mask when affinity + event provided."""
        rules = [
            _make_rule("Rule A", events=frozenset({"PreToolUse"})),
            _make_rule("Rule B", events=frozenset({"Stop"})),
        ]
        aff_idx = build_strict_affinity(rules)
        index = _make_index(tuple(rules))
        config = self._make_config()

        candidates = [
            ScoredCandidate(rule=rules[0], score=0.9, retriever="dense"),
        ]

        from cuecard.retrieval.pipeline import run_pipeline

        with patch(
            "cuecard.retrieval.dense.DenseRetriever.retrieve",
            return_value=candidates,
        ):
            result = run_pipeline(
                "test", index, config,  # type: ignore[arg-type]
                mode="embedding",
                event="PreToolUse",
                affinity=aff_idx,
            )

        assert isinstance(result, PipelineResult)
        assert result.event == "PreToolUse"
        assert result.event_mask_applied is True
        assert result.rules_masked >= 0

    def test_pipeline_without_affinity_backwards_compat(self) -> None:
        """Pipeline works without affinity (backwards compat)."""
        rules = [_make_rule("Rule A")]
        index = _make_index(tuple(rules))
        config = self._make_config()

        candidates = [
            ScoredCandidate(rule=rules[0], score=0.9, retriever="dense"),
        ]

        from cuecard.retrieval.pipeline import run_pipeline

        with patch(
            "cuecard.retrieval.dense.DenseRetriever.retrieve",
            return_value=candidates,
        ):
            result = run_pipeline(
                "test", index, config,  # type: ignore[arg-type]
                mode="embedding",
            )

        assert isinstance(result, PipelineResult)
        assert result.event == ""
        assert result.event_mask_applied is False
        assert result.rules_masked == 0

    def test_pipeline_affinity_without_event_no_mask(self) -> None:
        """Affinity provided but empty event -> no mask applied."""
        rules = [_make_rule("Rule A")]
        aff_idx = build_strict_affinity(rules)
        index = _make_index(tuple(rules))
        config = self._make_config()

        candidates = [
            ScoredCandidate(rule=rules[0], score=0.9, retriever="dense"),
        ]

        from cuecard.retrieval.pipeline import run_pipeline

        with patch(
            "cuecard.retrieval.dense.DenseRetriever.retrieve",
            return_value=candidates,
        ):
            result = run_pipeline(
                "test", index, config,  # type: ignore[arg-type]
                mode="embedding",
                event="",
                affinity=aff_idx,
            )

        assert result.event_mask_applied is False

    def test_pipeline_mask_passed_to_retrievers(self) -> None:
        """Verify mask kwarg is actually passed to retrievers."""
        rules = [_make_rule("Rule A")]
        aff_idx = build_strict_affinity(rules)
        index = _make_index(tuple(rules))
        config = self._make_config()

        from cuecard.retrieval.pipeline import run_pipeline

        with patch(
            "cuecard.retrieval.dense.DenseRetriever.retrieve",
            return_value=[],
        ) as mock_dense:
            run_pipeline(
                "test", index, config,  # type: ignore[arg-type]
                mode="embedding",
                event="PreToolUse",
                affinity=aff_idx,
            )

        # Check that mask kwarg was passed
        call_kwargs = mock_dense.call_args
        assert "mask" in call_kwargs.kwargs
        assert call_kwargs.kwargs["mask"] is not None

    def test_pipeline_tool_name_passed(self) -> None:
        """tool_name parameter flows to event mask builder."""
        rules = [
            _make_rule("Rule A", events=frozenset({"PreToolUse"})),
        ]
        affinities = [
            RuleAffinity(
                events=frozenset({"PreToolUse"}),
                tools=frozenset({"Edit"}),
                source="explicit",
            ),
        ]
        aff_idx = _make_affinity(rules, affinities)
        index = _make_index(tuple(rules))
        config = self._make_config()

        from cuecard.retrieval.pipeline import run_pipeline

        with patch(
            "cuecard.retrieval.dense.DenseRetriever.retrieve",
            return_value=[],
        ) as mock_dense:
            result = run_pipeline(
                "test", index, config,  # type: ignore[arg-type]
                mode="embedding",
                event="PreToolUse",
                tool_name="Bash",
                affinity=aff_idx,
            )

        # Rule A only applies to Edit, not Bash -> should be masked
        call_kwargs = mock_dense.call_args
        mask = call_kwargs.kwargs["mask"]
        assert mask is not None
        # The single rule should be masked (Edit != Bash)
        assert not mask[0]
        assert result.rules_masked == 1


# --- Loader affinity composition ---


class TestLoaderAffinityMerge:
    """Tests for _load_and_merge_affinity in loader.py."""

    def test_no_affinity_returns_none(self, tmp_path: Path) -> None:
        from cuecard.indexing.loader import _load_and_merge_affinity

        result = _load_and_merge_affinity(
            str(tmp_path / "global"), None,
        )
        assert result is None

    def test_global_only(self, tmp_path: Path) -> None:
        from cuecard.retrieval.affinity import save_affinity

        rule = _make_rule("test rule")
        aff = build_strict_affinity([rule])
        global_dir = str(tmp_path / "global")
        save_affinity(aff, global_dir)

        from cuecard.indexing.loader import _load_and_merge_affinity

        result = _load_and_merge_affinity(global_dir, None)
        assert result is not None
        assert result.get(rule) is not None

    def test_project_only(self, tmp_path: Path) -> None:
        from cuecard.retrieval.affinity import save_affinity

        rule = _make_rule("project rule")
        aff = build_strict_affinity([rule])
        proj_dir = str(tmp_path / "project")
        save_affinity(aff, proj_dir)

        from cuecard.indexing.loader import _load_and_merge_affinity

        result = _load_and_merge_affinity(
            str(tmp_path / "nonexistent"), proj_dir,
        )
        assert result is not None
        assert result.get(rule) is not None

    def test_project_wins_on_duplicate(self, tmp_path: Path) -> None:
        from cuecard.retrieval.affinity import save_affinity

        rule = _make_rule("shared rule")
        text_hash = _hash_rule_text(rule.text)

        global_aff = AffinityIndex(
            version=1, mode="strict", model="",
            affinities=(
                (text_hash, RuleAffinity(
                    events=frozenset({"PreToolUse"}),
                    tools=frozenset(),
                    source="explicit",
                )),
            ),
        )
        project_aff = AffinityIndex(
            version=1, mode="strict", model="",
            affinities=(
                (text_hash, RuleAffinity(
                    events=frozenset({"Stop"}),
                    tools=frozenset(),
                    source="explicit",
                )),
            ),
        )

        global_dir = str(tmp_path / "global")
        proj_dir = str(tmp_path / "project")
        save_affinity(global_aff, global_dir)
        save_affinity(project_aff, proj_dir)

        from cuecard.indexing.loader import _load_and_merge_affinity

        result = _load_and_merge_affinity(global_dir, proj_dir)
        assert result is not None
        ra = result.get(rule)
        assert ra is not None
        # Project wins
        assert ra.events == frozenset({"Stop"})

    def test_load_or_build_returns_loaded_index(
        self, tmp_path: Path,
    ) -> None:
        """load_or_build returns LoadedIndex with affinity field."""
        index = _make_index((_make_rule("Rule 1"),))

        from cuecard.models import PipelineConfig, ResolvedConfig

        cfg = ResolvedConfig(
            source_paths=(),
            global_source_paths=("/tmp/rules.txt",),
            project_source_paths=(),
            model_name="test",
            top_k=5,
            threshold=0.30,
            dedup_threshold=0.95,
            query_max_length=500,
            hook_events=("PreToolUse",),
            verbose=False,
            redact=True,
            max_log_size_mb=10,
            global_cache_dir=str(tmp_path / "global_index"),
            project_cache_dir=None,
            allowed_dirs=(),
            pipeline=PipelineConfig(),
        )

        from cuecard.indexing.loader import load_or_build

        with patch(
            "cuecard.indexing.loader._load_or_rebuild_scope",
            return_value=index,
        ):
            result = load_or_build(cfg)

        assert isinstance(result, LoadedIndex)
        assert result.index is index
        # Affinity is None when no affinity.json exists
        assert result.affinity is None


# --- PipelineResult new fields ---


class TestPipelineResultFields:
    """PipelineResult has event mask fields."""

    def test_default_values(self) -> None:
        result = PipelineResult(
            results=(), stages=(), mode="embedding",
        )
        assert result.event == ""
        assert result.event_mask_applied is False
        assert result.rules_masked == 0

    def test_custom_values(self) -> None:
        result = PipelineResult(
            results=(),
            stages=(
                StageTrace(
                    stage="retrieval",
                    input_count=10,
                    output_count=5,
                    latency_ms=1.0,
                ),
            ),
            mode="embedding",
            event="PreToolUse",
            event_mask_applied=True,
            rules_masked=3,
        )
        assert result.event == "PreToolUse"
        assert result.event_mask_applied is True
        assert result.rules_masked == 3
