"""Index freshness checking — determines if on-disk index is up-to-date."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from types import MappingProxyType

from cuecard.models import SourceMeta

logger = logging.getLogger(__name__)

_HASH_PREFIX = "sha256:"
_READ_CHUNK_SIZE = 65536


@dataclass(frozen=True)
class FreshnessResult:
    """Outcome of a freshness check against source files."""

    is_stale: bool
    updated_sources: MappingProxyType[str, SourceMeta]
    changed_files: tuple[str, ...]
    removed_files: tuple[str, ...]
    new_files: tuple[str, ...]


def compute_file_hash(path: str) -> str:
    """Read *path* and return ``'sha256:<hex>'``.

    Raises ``FileNotFoundError`` if *path* does not exist.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_READ_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return f"{_HASH_PREFIX}{h.hexdigest()}"


def check_freshness(
    source_paths: tuple[str, ...],
    stored_sources: dict[str, SourceMeta],
) -> FreshnessResult:
    """Compare *source_paths* against *stored_sources* and return staleness info.

    Rules
    -----
    * New file (in *source_paths* but not in *stored_sources*) -> stale.
    * Removed file (in *stored_sources* but not in *source_paths*) -> stale.
    * mtime matches stored -> fresh (fast path).
    * mtime differs, hash matches -> fresh (update mtime in *updated_sources*).
    * mtime differs, hash differs -> stale.
    * File missing on disk -> logged warning, skipped (not stale by itself).
    """
    changed: list[str] = []
    new: list[str] = []
    updated = dict(stored_sources)

    source_set = set(source_paths)

    for path in source_paths:
        # --- file missing on disk ---
        try:
            st = os.stat(path)
        except OSError:
            logger.warning("Source file not found, skipping: %s", path)
            continue

        current_mtime = st.st_mtime

        # --- new file ---
        if path not in stored_sources:
            new.append(path)
            continue

        stored = stored_sources[path]

        # --- fast path: mtime unchanged ---
        if current_mtime == stored.mtime:
            continue

        # --- mtime changed, check hash ---
        current_hash = compute_file_hash(path)

        if current_hash == stored.content_hash:
            # Content identical — just update the cached mtime.
            updated[path] = SourceMeta(
                mtime=current_mtime,
                content_hash=stored.content_hash,
                rule_count=stored.rule_count,
            )
        else:
            changed.append(path)

    # --- removed files ---
    removed = [p for p in stored_sources if p not in source_set]

    is_stale = bool(changed or removed or new)

    return FreshnessResult(
        is_stale=is_stale,
        updated_sources=MappingProxyType(updated),
        changed_files=tuple(changed),
        removed_files=tuple(removed),
        new_files=tuple(new),
    )
