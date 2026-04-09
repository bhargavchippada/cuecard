"""Tests for cuecard.freshness."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from cuecard.indexing.freshness import (
    FreshnessResult,
    check_freshness,
    compute_file_hash,
)
from cuecard.models import SourceMeta

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _meta_for(path: Path, rule_count: int = 1) -> SourceMeta:
    """Build a SourceMeta matching the current on-disk state of *path*."""
    return SourceMeta(
        mtime=os.stat(str(path)).st_mtime,
        content_hash=compute_file_hash(str(path)),
        rule_count=rule_count,
    )


# ---------------------------------------------------------------------------
# TestComputeFileHash
# ---------------------------------------------------------------------------

class TestComputeFileHash:
    def test_correct_sha256(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.md"
        _write(f, "hello world")
        result = compute_file_hash(str(f))
        assert result.startswith("sha256:")
        # Verify deterministic
        assert result == compute_file_hash(str(f))

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        _write(a, "alpha")
        _write(b, "beta")
        assert compute_file_hash(str(a)) != compute_file_hash(str(b))

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.md"
        _write(f, "")
        result = compute_file_hash(str(f))
        assert result.startswith("sha256:")

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_file_hash(str(tmp_path / "nonexistent.md"))


# ---------------------------------------------------------------------------
# TestCheckFreshness
# ---------------------------------------------------------------------------

class TestCheckFreshness:
    def test_all_fresh_mtime_matches(self, tmp_path: Path) -> None:
        """mtime matches stored -> is_stale=False, no changes."""
        f = tmp_path / "rules.md"
        _write(f, "rule one")
        stored = {str(f): _meta_for(f)}

        result = check_freshness((str(f),), stored)

        assert result.is_stale is False
        assert result.changed_files == ()
        assert result.removed_files == ()
        assert result.new_files == ()

    def test_mtime_differs_hash_matches(self, tmp_path: Path) -> None:
        """mtime changed but content identical -> fresh, mtime updated."""
        f = tmp_path / "rules.md"
        _write(f, "rule one")
        meta = _meta_for(f)
        # Simulate stored mtime being stale (different from current).
        old_meta = SourceMeta(
            mtime=meta.mtime - 100.0,
            content_hash=meta.content_hash,
            rule_count=meta.rule_count,
        )
        stored = {str(f): old_meta}

        result = check_freshness((str(f),), stored)

        assert result.is_stale is False
        assert result.changed_files == ()
        # mtime should be updated to current value.
        updated = result.updated_sources[str(f)]
        assert updated.mtime == meta.mtime
        assert updated.content_hash == meta.content_hash

    def test_mtime_and_hash_differ(self, tmp_path: Path) -> None:
        """Both mtime and hash changed -> stale."""
        f = tmp_path / "rules.md"
        _write(f, "original content")
        meta = _meta_for(f)

        # Change the file content.
        _write(f, "modified content")

        stored = {str(f): meta}

        result = check_freshness((str(f),), stored)

        assert result.is_stale is True
        assert str(f) in result.changed_files

    def test_new_file_not_in_stored(self, tmp_path: Path) -> None:
        """File in source_paths but not in stored -> stale (new file)."""
        f = tmp_path / "new_rules.md"
        _write(f, "brand new")

        result = check_freshness((str(f),), {})

        assert result.is_stale is True
        assert str(f) in result.new_files
        assert result.changed_files == ()
        assert result.removed_files == ()

    def test_removed_file(self, tmp_path: Path) -> None:
        """File in stored but not in source_paths -> stale (removed)."""
        f = tmp_path / "old_rules.md"
        _write(f, "old")
        stored = {str(f): _meta_for(f)}

        result = check_freshness((), stored)

        assert result.is_stale is True
        assert str(f) in result.removed_files
        assert result.changed_files == ()
        assert result.new_files == ()

    def test_missing_file_on_disk_skipped(self, tmp_path: Path) -> None:
        """File in source_paths but doesn't exist on disk -> skip, not stale."""
        missing = str(tmp_path / "ghost.md")

        result = check_freshness((missing,), {})

        # Not stale because the missing file is also absent from stored.
        assert result.is_stale is False
        assert result.changed_files == ()
        assert result.new_files == ()

    def test_missing_file_on_disk_was_stored(self, tmp_path: Path) -> None:
        """File was in stored AND in source_paths but no longer on disk.

        The file is skipped (not stale from the disk-check perspective),
        BUT it is still in stored_sources and still in source_paths,
        so it is NOT in removed_files either. Net effect: not stale.
        """
        missing = str(tmp_path / "ghost.md")
        stored = {
            missing: SourceMeta(
                mtime=1000.0,
                content_hash="sha256:aaa",
                rule_count=1,
            ),
        }

        result = check_freshness((missing,), stored)

        # It is in both source_paths and stored — not removed.
        # It can't be stat'd — skipped, not flagged stale.
        assert result.is_stale is False

    def test_empty_source_paths_empty_stored(self) -> None:
        """No sources at all -> fresh (nothing to check)."""
        result = check_freshness((), {})

        assert result.is_stale is False
        assert result.changed_files == ()
        assert result.removed_files == ()
        assert result.new_files == ()

    def test_mixed_some_fresh_some_stale(self, tmp_path: Path) -> None:
        """Multiple files: one fresh, one changed, one new, one removed."""
        fresh_f = tmp_path / "fresh.md"
        changed_f = tmp_path / "changed.md"
        new_f = tmp_path / "new.md"
        removed_f = tmp_path / "removed.md"

        _write(fresh_f, "unchanged")
        _write(changed_f, "before")
        _write(new_f, "brand new")
        _write(removed_f, "will be removed")

        stored = {
            str(fresh_f): _meta_for(fresh_f),
            str(changed_f): _meta_for(changed_f),
            str(removed_f): _meta_for(removed_f),
        }

        # Modify the changed file.
        _write(changed_f, "after")

        source_paths = (str(fresh_f), str(changed_f), str(new_f))

        result = check_freshness(source_paths, stored)

        assert result.is_stale is True
        assert str(changed_f) in result.changed_files
        assert str(new_f) in result.new_files
        assert str(removed_f) in result.removed_files
        # fresh_f should not appear in any stale list.
        assert str(fresh_f) not in result.changed_files
        assert str(fresh_f) not in result.new_files
        assert str(fresh_f) not in result.removed_files


class TestFreshnessResultFrozen:
    def test_immutable(self) -> None:
        r = FreshnessResult(
            is_stale=False,
            updated_sources={},
            changed_files=(),
            removed_files=(),
            new_files=(),
        )
        with pytest.raises(AttributeError):
            r.is_stale = True  # type: ignore[misc]
