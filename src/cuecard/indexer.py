"""Embed rules via fastembed, build index, persist with atomic writes."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from cuecard._math import l2_normalize
from cuecard.models import Index, Provenance, Rule, SourceMeta

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)

_DEFAULT_DIM = 384
_METADATA_VERSION = 1
_FILE_PERMS = 0o600
_DIR_PERMS = 0o700


class EmbeddingModel(Protocol):
    """Protocol for fastembed-compatible embedding models."""

    def passage_embed(
        self, texts: Iterable[str], **kwargs: Any,
    ) -> Iterable[npt.NDArray[np.floating]]: ...


def _compute_checksum(path: str) -> str:
    """Compute sha256 hex digest of a file, prefixed with 'sha256:'."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"



def build_index(
    rules: tuple[Rule, ...],
    sources: Mapping[str, SourceMeta],
    model_name: str,
    model: EmbeddingModel | None = None,
    dim: int = _DEFAULT_DIM,
) -> Index:
    """Embed all rule texts and return an Index.

    Uses asymmetric passage_embed for indexing (never query_embed).
    If rules is empty, returns an Index with a (0, dim) embedding array.
    """
    if not rules:
        empty_emb = np.zeros((0, dim), dtype=np.float32)
        return Index(
            embeddings=empty_emb,
            rules=rules,
            model_name=model_name,
            dim=dim,
            sources=sources,
        )

    if model is None:
        msg = "model is required when rules is non-empty"
        raise ValueError(msg)

    texts = [r.text for r in rules]
    raw = np.array(list(model.passage_embed(texts)), dtype=np.float32)

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    detected_dim = raw.shape[1]
    normalized = l2_normalize(raw)

    return Index(
        embeddings=normalized,
        rules=rules,
        model_name=model_name,
        dim=detected_dim,
        sources=sources,
    )


def save_index(index: Index, cache_dir: str) -> None:
    """Persist index to disk with atomic writes and file locking.

    Creates two files in cache_dir:
      - embeddings.npz: numpy compressed array (N, dim)
      - metadata.json: model info, checksum, rules, sources
    """
    dir_path = Path(cache_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    os.chmod(dir_path, _DIR_PERMS)

    lock_path = dir_path / ".lock"
    npz_path = dir_path / "embeddings.npz"
    meta_path = dir_path / "metadata.json"

    tmp_npz_path: str | None = None
    tmp_meta_path: str | None = None

    lock_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    lock_fd = os.open(str(lock_path), lock_flags, _FILE_PERMS)
    with os.fdopen(lock_fd, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Atomic write: embeddings.npz
            with tempfile.NamedTemporaryFile(
                dir=cache_dir, suffix=".npz", delete=False,
            ) as tmp_npz:
                tmp_npz_path = tmp_npz.name
                np.savez_compressed(tmp_npz, embeddings=index.embeddings)
            os.chmod(tmp_npz_path, _FILE_PERMS)
            os.replace(tmp_npz_path, npz_path)
            tmp_npz_path = None  # successfully replaced

            # Compute checksum of the written npz
            checksum = _compute_checksum(str(npz_path))

            # Build metadata
            rules_list = [
                {
                    "text": r.text,
                    "file": r.provenance.file,
                    "line_start": r.provenance.line_start,
                    "line_end": r.provenance.line_end,
                    "section_path": list(r.provenance.section_path),
                    "chunk_type": r.provenance.chunk_type,
                }
                for r in index.rules
            ]

            sources_dict = {
                path: {
                    "mtime": sm.mtime,
                    "content_hash": sm.content_hash,
                    "rule_count": sm.rule_count,
                }
                for path, sm in index.sources.items()
            }

            metadata = {
                "version": _METADATA_VERSION,
                "model": index.model_name,
                "dim": index.dim,
                "checksum": checksum,
                "created": datetime.now(tz=UTC).isoformat(),
                "sources": sources_dict,
                "rules": rules_list,
            }

            # Atomic write: metadata.json
            with tempfile.NamedTemporaryFile(
                dir=cache_dir, suffix=".json", mode="w",
                delete=False,
            ) as tmp_meta:
                tmp_meta_path = tmp_meta.name
                json.dump(metadata, tmp_meta, indent=2)
            os.chmod(tmp_meta_path, _FILE_PERMS)
            os.replace(tmp_meta_path, meta_path)
            tmp_meta_path = None  # successfully replaced

        finally:
            # Clean up leaked temp files on partial failure
            for tmp in (tmp_npz_path, tmp_meta_path):
                if tmp is not None:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    # lock_fd was already created with _FILE_PERMS — no chmod needed


def load_index(cache_dir: str) -> Index | None:
    """Load a persisted index from disk, verifying integrity.

    Returns None (with a logged warning) on:
      - Missing files
      - Checksum mismatch
      - Rule count / embedding row mismatch
      - Any corruption or parse error
    """
    dir_path = Path(cache_dir)
    npz_path = dir_path / "embeddings.npz"
    meta_path = dir_path / "metadata.json"

    try:
        with open(meta_path) as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load index metadata from %s: %s", cache_dir, exc)
        return None

    try:
        with np.load(str(npz_path)) as data:
            embeddings = data["embeddings"]
    except (FileNotFoundError, OSError, KeyError) as exc:
        logger.warning("Failed to load embeddings from %s: %s", cache_dir, exc)
        return None

    # Integrity: checksum
    expected_checksum = metadata.get("checksum", "")
    actual_checksum = _compute_checksum(str(npz_path))
    if actual_checksum != expected_checksum:
        logger.warning(
            "Checksum mismatch in %s: expected %s, got %s",
            cache_dir, expected_checksum, actual_checksum,
        )
        return None

    # Integrity: rule count matches embedding rows
    rules_data = metadata.get("rules", [])
    if embeddings.shape[0] != len(rules_data):
        logger.warning(
            "Rule count mismatch in %s: %d embeddings vs %d rules",
            cache_dir, embeddings.shape[0], len(rules_data),
        )
        return None

    # Reconstruct Rule objects
    rules = tuple(
        Rule(
            text=rd["text"],
            provenance=Provenance(
                file=rd["file"],
                line_start=rd["line_start"],
                line_end=rd["line_end"],
                section_path=tuple(rd.get("section_path", ())),
                chunk_type=rd.get("chunk_type", "rule"),
            ),
        )
        for rd in rules_data
    )

    # Reconstruct SourceMeta
    sources = {
        path: SourceMeta(
            mtime=sd["mtime"],
            content_hash=sd["content_hash"],
            rule_count=sd["rule_count"],
        )
        for path, sd in metadata.get("sources", {}).items()
    }

    return Index(
        embeddings=embeddings,
        rules=rules,
        model_name=metadata["model"],
        dim=metadata["dim"],
        sources=sources,
    )
