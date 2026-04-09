"""Embed rules via fastembed, build index, persist with atomic writes."""

from __future__ import annotations

import contextlib
import fcntl
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
from cuecard.freshness import compute_file_hash
from cuecard.models import (
    KNOWN_HOOK_EVENTS,
    MAX_EXPANSION_LENGTH,
    MAX_EXPANSIONS_PER_RULE,
    VALID_AFFINITY_SOURCES,
    AffinityIndex,
    AffinitySource,
    Index,
    Provenance,
    Rule,
    RuleAffinity,
    SourceMeta,
    _hash_rule_text,
)
from cuecard.security import scrub_secrets

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)

_DEFAULT_DIM = 384
_METADATA_VERSION = 1
_METADATA_VERSION_V2 = 2
_FILE_PERMS = 0o600
_DIR_PERMS = 0o700


class EmbeddingModel(Protocol):
    """Protocol for fastembed-compatible embedding models."""

    def passage_embed(
        self, texts: Iterable[str], **kwargs: Any,
    ) -> Iterable[npt.NDArray[np.floating]]: ...


_RULES_JSON_VERSION = 2


def save_rules_json(
    rules: list[Rule],
    cache_dir: str,
    affinity: AffinityIndex | None = None,
) -> None:
    """Persist rules to a canonical JSON intermediate format.

    Writes ``rules.json`` with atomic write + 0o600 permissions.
    If ``affinity`` is provided, each rule entry includes an inline
    ``affinity`` dict — no separate sidecar file needed.
    """
    dir_path = Path(cache_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    os.chmod(dir_path, _DIR_PERMS)

    # Build affinity lookup if provided
    aff_lookup: dict[str, RuleAffinity] = {}
    if affinity is not None:
        aff_lookup = dict(affinity.items)

    rules_list: list[dict[str, object]] = []
    for r in rules:
        entry: dict[str, object] = {
            "text": r.text,
            "expansions": list(r.expansions),
            "events": sorted(r.events),
            "tools": sorted(r.tools),
            "source": {
                "file": r.provenance.file,
                "line_start": r.provenance.line_start,
                "line_end": r.provenance.line_end,
                "chunk_type": r.provenance.chunk_type,
            },
        }
        # Inline affinity if available for this rule
        text_hash = _hash_rule_text(r.text)
        ra = aff_lookup.get(text_hash)
        if ra is not None:
            entry["affinity"] = {
                "events": sorted(ra.events),
                "tools": sorted(ra.tools),
                "source": ra.source,
                "reasoning": ra.reasoning,
            }
        rules_list.append(entry)

    data: dict[str, object] = {"version": _RULES_JSON_VERSION, "rules": rules_list}
    if affinity is not None:
        data["affinity_mode"] = affinity.mode
        data["affinity_model"] = affinity.model
    json_path = dir_path / "rules.json"

    with tempfile.NamedTemporaryFile(
        dir=cache_dir, suffix=".json", mode="w", delete=False,
    ) as tmp:
        tmp_path = tmp.name
        json.dump(data, tmp, indent=2)

    os.chmod(tmp_path, _FILE_PERMS)
    os.replace(tmp_path, json_path)


def load_rules_json(
    cache_dir: str,
) -> tuple[list[Rule], AffinityIndex | None] | None:
    """Load rules from the canonical JSON intermediate format.

    Returns ``(rules, affinity)`` where affinity is extracted from
    inline ``affinity`` dicts if present, or None if absent.
    Returns None if rules.json does not exist or is corrupt.
    """
    json_path = Path(cache_dir) / "rules.json"

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.debug("No rules.json in %s: %s", cache_dir, exc)
        return None

    version = data.get("version")
    if version not in (1, _RULES_JSON_VERSION):
        logger.warning(
            "Unsupported rules.json version %s in %s",
            version, cache_dir,
        )
        return None

    rules: list[Rule] = []
    affinities: list[tuple[str, RuleAffinity]] = []
    has_affinity = False

    for entry in data.get("rules", []):
        text = entry.get("text", "").strip()
        if not text:
            continue

        raw_expansions = entry.get("expansions", [])
        expansions: list[str] = []
        for exp in raw_expansions:
            if not isinstance(exp, str):
                continue
            exp = exp.strip()
            if not exp:
                continue
            if len(exp) > MAX_EXPANSION_LENGTH:
                exp = exp[:MAX_EXPANSION_LENGTH]
            expansions.append(exp)
            if len(expansions) >= MAX_EXPANSIONS_PER_RULE:
                break

        source = entry.get("source", {})
        provenance = Provenance(
            file=source.get("file", ""),
            line_start=source.get("line_start", 0),
            line_end=source.get("line_end", 0),
            chunk_type=source.get("chunk_type", "rule"),
        )

        # V2: read events/tools (default to empty for v1)
        events = frozenset(entry.get("events", ()))
        tools = frozenset(entry.get("tools", ()))

        rules.append(Rule(
            text=text,
            provenance=provenance,
            expansions=tuple(expansions),
            events=events,
            tools=tools,
        ))

        # Extract inline affinity if present
        raw_aff = entry.get("affinity")
        if isinstance(raw_aff, dict):
            has_affinity = True
            text_hash = _hash_rule_text(text)
            aff_events = frozenset(
                e for e in raw_aff.get("events", [])
                if isinstance(e, str) and e in KNOWN_HOOK_EVENTS
            )
            aff_tools = frozenset(
                t[:50] for t in raw_aff.get("tools", [])
                if isinstance(t, str) and t.strip() and len(t) <= 50
            )
            raw_source = raw_aff.get("source", "inferred")
            aff_source: AffinitySource = (
                raw_source if raw_source in VALID_AFFINITY_SOURCES
                else "inferred"
            )
            raw_reasoning = raw_aff.get("reasoning", "")
            aff_reasoning = (
                scrub_secrets(raw_reasoning.strip()[:500])
                if isinstance(raw_reasoning, str) else ""
            )
            affinities.append((text_hash, RuleAffinity(
                events=aff_events,
                tools=aff_tools,
                source=aff_source,
                reasoning=aff_reasoning,
            )))

    affinity_index: AffinityIndex | None = None
    if has_affinity and affinities:
        raw_mode = data.get("affinity_mode", "inferred")
        raw_model = data.get("affinity_model", "")
        aff_mode = (
            raw_mode if isinstance(raw_mode, str) and len(raw_mode) <= 50
            else "inferred"
        )
        aff_model = raw_model[:100] if isinstance(raw_model, str) else ""
        affinity_index = AffinityIndex(
            version=1,
            mode=aff_mode,
            model=aff_model,
            affinities=tuple(affinities),
        )

    return rules, affinity_index


def merge_rules_json(
    fresh_rules: list[Rule],
    cached_rules: list[Rule],
) -> list[Rule]:
    """Merge fresh-parsed rules with cached rules, preserving expansions.

    For rules whose canonical text hasn't changed (exact string equality),
    expansions from the cached version are preserved. Rules with changed
    text get empty expansions (with a warning).
    """
    cached_by_text = {r.text: r for r in cached_rules}
    merged: list[Rule] = []

    for rule in fresh_rules:
        cached = cached_by_text.get(rule.text)
        if cached is not None and cached.expansions:
            # Preserve cached expansions, use fresh events/tools
            merged.append(Rule(
                text=rule.text,
                provenance=rule.provenance,
                summary=rule.summary,
                expansions=cached.expansions,
                events=rule.events,
                tools=rule.tools,
            ))
        else:
            merged.append(rule)

    # Warn about rules that lost expansions
    fresh_texts = {r.text for r in fresh_rules}
    for cached_rule in cached_rules:
        if cached_rule.expansions and cached_rule.text not in fresh_texts:
            logger.warning(
                "Rule text changed or removed — expansions lost: %s",
                scrub_secrets(cached_rule.text[:80]),
            )

    return merged


def build_index(
    rules: tuple[Rule, ...],
    sources: Mapping[str, SourceMeta],
    model_name: str,
    model: EmbeddingModel | None = None,
    dim: int = _DEFAULT_DIM,
) -> Index:
    """Embed all rule texts (plus expansions) and return an Index.

    Uses asymmetric passage_embed for indexing (never query_embed).
    For each rule, embeds ``[rule.text] + list(rule.expansions)``,
    building a ``rule_map`` that maps each embedding row to its parent
    rule index and a ``bm25_corpus`` with all texts in embedding order.

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
            rule_map=(),
            bm25_corpus=(),
        )

    if model is None:
        msg = "model is required when rules is non-empty"
        raise ValueError(msg)

    # Collect all texts and build rule_map
    all_texts: list[str] = []
    rule_map_list: list[int] = []
    for i, rule in enumerate(rules):
        texts = [rule.text, *rule.expansions]
        for text in texts:
            all_texts.append(text)
            rule_map_list.append(i)

    raw = np.array(list(model.passage_embed(all_texts)), dtype=np.float32)

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
        rule_map=tuple(rule_map_list),
        bm25_corpus=tuple(all_texts),
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
            checksum = compute_file_hash(str(npz_path))

            # Build metadata
            rules_list = [
                {
                    "text": r.text,
                    "file": r.provenance.file,
                    "line_start": r.provenance.line_start,
                    "line_end": r.provenance.line_end,
                    "section_path": list(r.provenance.section_path),
                    "chunk_type": r.provenance.chunk_type,
                    "expansions": list(r.expansions),
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

            metadata: dict[str, Any] = {
                "version": _METADATA_VERSION_V2,
                "model": index.model_name,
                "dim": index.dim,
                "checksum": checksum,
                "created": datetime.now(tz=UTC).isoformat(),
                "sources": sources_dict,
                "rules": rules_list,
                "rule_map": list(index.rule_map),
                "bm25_corpus": list(index.bm25_corpus)
                if index.bm25_corpus is not None
                else None,
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
    actual_checksum = compute_file_hash(str(npz_path))
    if actual_checksum != expected_checksum:
        logger.warning(
            "Checksum mismatch in %s: expected %s, got %s",
            cache_dir, expected_checksum, actual_checksum,
        )
        return None

    version = metadata.get("version", 1)
    rules_data = metadata.get("rules", [])

    if version >= _METADATA_VERSION_V2:
        # V2: rule_map present, embeddings match rule_map length
        rule_map_data = metadata.get("rule_map", [])
        if embeddings.shape[0] != len(rule_map_data):
            logger.warning(
                "rule_map length mismatch in %s: %d embeddings vs %d rule_map",
                cache_dir, embeddings.shape[0], len(rule_map_data),
            )
            return None
        rule_map: tuple[int, ...] = tuple(rule_map_data)
        # Validate rule_map indices against rules count
        n_rules = len(rules_data)
        if not all(0 <= i < n_rules for i in rule_map):
            logger.warning(
                "rule_map contains out-of-range indices in %s", cache_dir,
            )
            return None
        raw_corpus = metadata.get("bm25_corpus")
        bm25_corpus: tuple[str, ...] | None = (
            tuple(raw_corpus) if raw_corpus is not None else None
        )
    else:
        # V1: identity rule_map, no bm25_corpus
        if embeddings.shape[0] != len(rules_data):
            logger.warning(
                "Rule count mismatch in %s: %d embeddings vs %d rules",
                cache_dir, embeddings.shape[0], len(rules_data),
            )
            return None
        rule_map = tuple(range(len(rules_data)))
        bm25_corpus = None

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
            expansions=tuple(rd.get("expansions", ())),
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
        rule_map=rule_map,
        bm25_corpus=bm25_corpus,
    )
