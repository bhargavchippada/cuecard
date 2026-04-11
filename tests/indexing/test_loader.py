"""Tests for cuecard.loader — load_or_build with scoped freshness."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np

from cuecard.indexing.freshness import FreshnessResult
from cuecard.indexing.loader import _load_or_rebuild_scope, load_or_build
from cuecard.models import (
    Index,
    LoadedIndex,
    PipelineConfig,
    Provenance,
    ResolvedConfig,
    Rule,
    SourceMeta,
)

if TYPE_CHECKING:
    from pathlib import Path


# --- helpers ---


def _make_config(
    tmp_path: Path,
    *,
    global_sources: tuple[str, ...] = (),
    project_sources: tuple[str, ...] = (),
    project_cache: str | None = None,
) -> ResolvedConfig:
    return ResolvedConfig(
        source_paths=(*global_sources, *project_sources),
        global_source_paths=global_sources,
        project_source_paths=project_sources,
        model_name="BAAI/bge-small-en-v1.5",
        top_k=5,
        threshold=0.30,
        dedup_threshold=0.95,
        query_max_length=500,
        hook_events=("PreToolUse",),
        verbose=False,
        redact=True,
        max_log_size_mb=10,
        global_cache_dir=str(tmp_path / "global_index"),
        project_cache_dir=project_cache,
        allowed_dirs=(),
        pipeline=PipelineConfig(),
    )


def _make_index(n: int = 2, dim: int = 384) -> Index:
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    return Index(
        embeddings=emb,
        rules=tuple(
            Rule(
                text=f"Rule {i}",
                provenance=Provenance(
                    file="/tmp/rules.txt", line_start=i, line_end=i,
                ),
            )
            for i in range(1, n + 1)
        ),
        model_name="BAAI/bge-small-en-v1.5",
        dim=dim,
        sources={
            "/tmp/rules.txt": SourceMeta(
                mtime=1.0, content_hash="sha256:abc", rule_count=n,
            ),
        },
    )


def _fresh_result() -> FreshnessResult:
    return FreshnessResult(
        is_stale=False,
        updated_sources=MappingProxyType({}),
        changed_files=(),
        removed_files=(),
        new_files=(),
    )


def _stale_result(
    sources: dict[str, SourceMeta] | None = None,
) -> FreshnessResult:
    return FreshnessResult(
        is_stale=True,
        updated_sources=MappingProxyType(sources or {}),
        changed_files=("/tmp/rules.txt",),
        removed_files=(),
        new_files=(),
    )


# --- _load_or_rebuild_scope tests ---


class TestLoadOrRebuildScope:
    def test_no_sources_returns_none(self, tmp_path: Path) -> None:
        result = _load_or_rebuild_scope(
            str(tmp_path / "cache"), (), "test", None, reindex=True,
        )
        assert result is None

    def test_cached_fresh_returns_cached(self, tmp_path: Path) -> None:
        index = _make_index()
        with (
            patch("cuecard.indexing.loader.load_index", return_value=index),
            patch(
                "cuecard.indexing.loader.check_freshness",
                return_value=_fresh_result(),
            ),
        ):
            result = _load_or_rebuild_scope(
                str(tmp_path / "cache"),
                ("/tmp/rules.txt",),
                "test",
                None,
                reindex=True,
            )
        assert result is index

    def test_cached_no_reindex_returns_cached(self, tmp_path: Path) -> None:
        index = _make_index()
        with patch("cuecard.indexing.loader.load_index", return_value=index):
            result = _load_or_rebuild_scope(
                str(tmp_path / "cache"),
                ("/tmp/rules.txt",),
                "test",
                None,
                reindex=False,
            )
        assert result is index

    def test_stale_rebuilds(self, tmp_path: Path) -> None:
        old_index = _make_index()
        new_index = _make_index(n=3)
        mock_model = MagicMock()
        sources = {
            "/tmp/rules.txt": SourceMeta(
                mtime=2.0, content_hash="sha256:def", rule_count=3,
            ),
        }

        with (
            patch("cuecard.indexing.loader.load_index", return_value=old_index),
            patch(
                "cuecard.indexing.loader.check_freshness",
                side_effect=[
                    _stale_result(),
                    FreshnessResult(
                        is_stale=False,
                        updated_sources=MappingProxyType(sources),
                        changed_files=(),
                        removed_files=(),
                        new_files=(),
                    ),
                ],
            ),
            patch(
                "cuecard.indexing.loader.parse_rules",
                return_value=[
                    Rule(
                        text="New rule",
                        provenance=Provenance(
                            file="/tmp/rules.txt",
                            line_start=1, line_end=1,
                        ),
                    ),
                ],
            ),
            patch(
                "cuecard.indexing.loader.build_index",
                return_value=new_index,
            ) as mock_build,
            patch("cuecard.indexing.loader.save_index") as mock_save,
        ):
            result = _load_or_rebuild_scope(
                str(tmp_path / "cache"),
                ("/tmp/rules.txt",),
                "test",
                mock_model,
                reindex=True,
            )

        assert result is new_index
        mock_build.assert_called_once()
        mock_save.assert_called_once()

    def test_no_cached_builds_fresh(self, tmp_path: Path) -> None:
        new_index = _make_index()
        mock_model = MagicMock()
        sources = {
            "/tmp/rules.txt": SourceMeta(
                mtime=1.0, content_hash="sha256:abc", rule_count=2,
            ),
        }

        with (
            patch("cuecard.indexing.loader.load_index", return_value=None),
            patch(
                "cuecard.indexing.loader.check_freshness",
                return_value=FreshnessResult(
                    is_stale=False,
                    updated_sources=MappingProxyType(sources),
                    changed_files=(),
                    removed_files=(),
                    new_files=(),
                ),
            ),
            patch(
                "cuecard.indexing.loader.parse_rules",
                return_value=[
                    Rule(
                        text="rule",
                        provenance=Provenance(
                            file="/tmp/rules.txt",
                            line_start=1, line_end=1,
                        ),
                    ),
                ],
            ),
            patch(
                "cuecard.indexing.loader.build_index",
                return_value=new_index,
            ),
            patch("cuecard.indexing.loader.save_index"),
        ):
            result = _load_or_rebuild_scope(
                str(tmp_path / "cache"),
                ("/tmp/rules.txt",),
                "test",
                mock_model,
                reindex=True,
            )

        assert result is new_index

    def test_no_rules_parsed_returns_none(self, tmp_path: Path) -> None:
        with (
            patch("cuecard.indexing.loader.load_index", return_value=None),
            patch("cuecard.indexing.loader.parse_rules", return_value=[]),
        ):
            result = _load_or_rebuild_scope(
                str(tmp_path / "cache"),
                ("/tmp/rules.txt",),
                "test",
                MagicMock(),
                reindex=True,
            )
        assert result is None

    def test_no_model_returns_none(self, tmp_path: Path) -> None:
        with (
            patch("cuecard.indexing.loader.load_index", return_value=None),
            patch(
                "cuecard.indexing.loader.parse_rules",
                return_value=[
                    Rule(
                        text="rule",
                        provenance=Provenance(
                            file="/tmp/r.txt", line_start=1, line_end=1,
                        ),
                    ),
                ],
            ),
        ):
            result = _load_or_rebuild_scope(
                str(tmp_path / "cache"),
                ("/tmp/r.txt",),
                "test",
                None,
                reindex=True,
            )
        assert result is None


# --- load_or_build tests ---


class TestLoadOrBuild:
    def test_no_sources_returns_none(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        result = load_or_build(cfg)
        assert result is None

    def test_global_only(self, tmp_path: Path) -> None:
        index = _make_index()
        cfg = _make_config(
            tmp_path, global_sources=("/tmp/rules.txt",),
        )
        with patch(
            "cuecard.indexing.loader._load_or_rebuild_scope",
            return_value=index,
        ):
            result = load_or_build(cfg)
        assert isinstance(result, LoadedIndex)
        assert result.index is index

    def test_global_plus_project_composes(self, tmp_path: Path) -> None:
        global_idx = _make_index(n=2)
        project_idx = _make_index(n=1)
        # Give project_idx different rule text so merge doesn't dedup
        project_idx = Index(
            embeddings=project_idx.embeddings,
            rules=(
                Rule(
                    text="Project rule",
                    provenance=Provenance(
                        file="/proj/rules.txt", line_start=1, line_end=1,
                    ),
                ),
            ),
            model_name=project_idx.model_name,
            dim=project_idx.dim,
            sources={
                "/proj/rules.txt": SourceMeta(
                    mtime=1.0, content_hash="sha256:xyz", rule_count=1,
                ),
            },
        )

        cfg = _make_config(
            tmp_path,
            global_sources=("/tmp/rules.txt",),
            project_sources=("/proj/rules.txt",),
            project_cache=str(tmp_path / "project_index"),
        )

        with patch(
            "cuecard.indexing.loader._load_or_rebuild_scope",
            side_effect=[global_idx, project_idx],
        ):
            result = load_or_build(cfg)

        assert result is not None
        idx = result.index
        # Should have 3 rules (2 global + 1 project)
        assert idx.size == 3
        rule_texts = {r.text for r in idx.rules}
        assert "Rule 1" in rule_texts
        assert "Rule 2" in rule_texts
        assert "Project rule" in rule_texts
        # Sources should be merged
        assert "/tmp/rules.txt" in idx.sources
        assert "/proj/rules.txt" in idx.sources

    def test_project_scope_skipped_when_no_project_cache(
        self, tmp_path: Path,
    ) -> None:
        index = _make_index()
        cfg = _make_config(
            tmp_path,
            global_sources=("/tmp/rules.txt",),
            project_sources=("/proj/rules.txt",),
            project_cache=None,  # No project cache dir
        )
        with patch(
            "cuecard.indexing.loader._load_or_rebuild_scope",
            return_value=index,
        ) as mock_scope:
            result = load_or_build(cfg)

        # Only called once (for global)
        assert mock_scope.call_count == 1
        assert isinstance(result, LoadedIndex)
        assert result.index is index

    def test_project_scope_skipped_when_no_project_sources(
        self, tmp_path: Path,
    ) -> None:
        index = _make_index()
        cfg = _make_config(
            tmp_path,
            global_sources=("/tmp/rules.txt",),
            project_sources=(),
            project_cache=str(tmp_path / "proj"),
        )
        with patch(
            "cuecard.indexing.loader._load_or_rebuild_scope",
            return_value=index,
        ) as mock_scope:
            result = load_or_build(cfg)

        assert mock_scope.call_count == 1
        assert isinstance(result, LoadedIndex)
        assert result.index is index

    def test_both_scopes_empty_returns_none(
        self, tmp_path: Path,
    ) -> None:
        cfg = _make_config(
            tmp_path,
            global_sources=("/tmp/rules.txt",),
            project_sources=("/proj/rules.txt",),
            project_cache=str(tmp_path / "proj"),
        )
        with patch(
            "cuecard.indexing.loader._load_or_rebuild_scope",
            return_value=None,
        ):
            result = load_or_build(cfg)
        assert result is None

    def test_reindex_false_passed_through(self, tmp_path: Path) -> None:
        index = _make_index()
        cfg = _make_config(
            tmp_path, global_sources=("/tmp/rules.txt",),
        )
        with patch(
            "cuecard.indexing.loader._load_or_rebuild_scope",
            return_value=index,
        ) as mock_scope:
            load_or_build(cfg, reindex=False)

        call_kwargs = mock_scope.call_args
        assert call_kwargs[1]["reindex"] is False

    def test_merge_dedup_identical_rules(self, tmp_path: Path) -> None:
        """When global and project have same rule text, dedup keeps one."""
        global_idx = _make_index(n=1)
        # Project has same rule text as global
        project_idx = Index(
            embeddings=global_idx.embeddings,
            rules=global_idx.rules,
            model_name=global_idx.model_name,
            dim=global_idx.dim,
            sources={
                "/proj/rules.txt": SourceMeta(
                    mtime=1.0, content_hash="sha256:xyz", rule_count=1,
                ),
            },
        )

        cfg = _make_config(
            tmp_path,
            global_sources=("/tmp/rules.txt",),
            project_sources=("/proj/rules.txt",),
            project_cache=str(tmp_path / "proj"),
        )

        with patch(
            "cuecard.indexing.loader._load_or_rebuild_scope",
            side_effect=[global_idx, project_idx],
        ):
            result = load_or_build(cfg)

        assert result is not None
        assert result.index.size == 1  # Deduped

    def test_merge_empty_rules_returns_none(self, tmp_path: Path) -> None:
        """If merge produces zero rules, return None."""
        empty_idx = Index(
            embeddings=np.zeros((0, 384), dtype=np.float32),
            rules=(),
            model_name="test",
            dim=384,
            sources={},
        )

        cfg = _make_config(
            tmp_path,
            global_sources=("/tmp/rules.txt",),
            project_sources=("/proj/rules.txt",),
            project_cache=str(tmp_path / "proj"),
        )

        with patch(
            "cuecard.indexing.loader._load_or_rebuild_scope",
            side_effect=[empty_idx, empty_idx],
        ):
            result = load_or_build(cfg)

        assert result is None


class TestCrossProjectIsolation:
    """End-to-end test proving no rule leakage between projects."""

    def test_no_cross_project_leakage(self, tmp_path: Path) -> None:
        """Rules from project A must not appear when querying project B."""
        project_b_idx = Index(
            embeddings=np.random.default_rng(2).standard_normal(
                (1, 384),
            ).astype(np.float32),
            rules=(
                Rule(
                    text="Project B rule",
                    provenance=Provenance(
                        file="/proj_b/rules.txt",
                        line_start=1, line_end=1,
                    ),
                ),
            ),
            model_name="BAAI/bge-small-en-v1.5",
            dim=384,
            sources={
                "/proj_b/rules.txt": SourceMeta(
                    mtime=1.0, content_hash="sha256:b", rule_count=1,
                ),
            },
        )

        global_idx = _make_index(n=1)

        # Config for project B — should NOT see project A rules
        cfg_b = _make_config(
            tmp_path,
            global_sources=("/tmp/rules.txt",),
            project_sources=("/proj_b/rules.txt",),
            project_cache=str(tmp_path / "proj_b_index"),
        )

        with patch(
            "cuecard.indexing.loader._load_or_rebuild_scope",
            side_effect=[global_idx, project_b_idx],
        ):
            result = load_or_build(cfg_b)

        assert result is not None
        rule_texts = {r.text for r in result.index.rules}
        assert "Project A secret rule" not in rule_texts
        assert "Project B rule" in rule_texts


class TestRulesJsonIntegration:
    """Test that _load_or_rebuild_scope preserves expansions via rules.json."""

    def test_rebuild_preserves_expansions(self, tmp_path: Path) -> None:
        """When rebuilding, expansions from cached rules.json are preserved."""
        cache_dir = str(tmp_path / "cache")
        rules_txt = tmp_path / "rules.txt"
        rules_txt.write_text("Never commit secrets\n")

        # Pre-populate rules.json with expansions
        import json
        import os

        cache_path = tmp_path / "cache"
        cache_path.mkdir()
        rules_json = {
            "version": 1,
            "rules": [
                {
                    "text": "Never commit secrets",
                    "expansions": ["hardcoded API key", "AKIA in source"],
                    "source": {
                        "file": str(rules_txt.resolve()),
                        "line_start": 1,
                        "line_end": 1,
                        "chunk_type": "rule",
                    },
                },
            ],
        }
        json_path = cache_path / "rules.json"
        json_path.write_text(json.dumps(rules_json))
        os.chmod(str(json_path), 0o600)

        # Mock the model + freshness to force rebuild
        # 3 embeddings: 1 rule text + 2 expansions
        mock_model = MagicMock()
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((3, 384)).astype(np.float32)
        mock_model.passage_embed.return_value = iter(emb)

        with patch("cuecard.indexing.loader.check_freshness") as mock_fresh:
            mock_fresh.return_value = FreshnessResult(
                is_stale=False,
                changed_files=(),
                new_files=(),
                removed_files=(),
                updated_sources=MappingProxyType({
                    str(rules_txt.resolve()): SourceMeta(
                        mtime=1.0, content_hash="sha256:abc", rule_count=1,
                    ),
                }),
            )

            result = _load_or_rebuild_scope(
                cache_dir=cache_dir,
                source_paths=(str(rules_txt),),
                model_name="test-model",
                model=mock_model,
                reindex=True,
            )

        assert result is not None
        # Verify the rules.json was updated with preserved expansions
        from cuecard.indexing.indexer import load_rules_json

        loaded_result = load_rules_json(cache_dir)
        assert loaded_result is not None
        loaded_rules, _aff = loaded_result
        assert loaded_rules[0].expansions == ("hardcoded API key", "AKIA in source")



class TestAffinityLoading:
    def test_load_affinity_for_scope_prefers_inline_rules_json(
        self, tmp_path: Path,
    ) -> None:
        from cuecard.indexing.loader import _load_affinity_for_scope
        from cuecard.models import AffinityIndex, RuleAffinity

        inline_aff = AffinityIndex(
            version=1,
            mode="strict",
            model="",
            affinities=((
                "hash",
                RuleAffinity(
                    events=frozenset({"PreToolUse"}),
                    tools=frozenset(),
                    source="explicit",
                ),
            ),),
        )

        with patch(
            "cuecard.indexing.loader.load_rules_json",
            return_value=([], inline_aff),
        ):
            result = _load_affinity_for_scope(str(tmp_path))

        assert result is inline_aff
