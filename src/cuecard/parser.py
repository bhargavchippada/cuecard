"""Parse rule files into Rule objects."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cuecard.models import (
    MAX_EXPANSION_LENGTH,
    MAX_EXPANSIONS_PER_RULE,
    MAX_RULE_LENGTH,
    Provenance,
    Rule,
)

logger = logging.getLogger(__name__)


def parse_rules(paths: tuple[str, ...]) -> list[Rule]:
    """Parse multiple rule files into a flat list of Rules.

    Dispatches by file suffix:
      - .txt  -> _parse_txt
      - .json -> _parse_json
      - .md   -> NotImplementedError (v0.2)
      - other -> ValueError

    Args:
        paths: Tuple of resolved absolute file paths.

    Returns:
        List of Rule objects with provenance.
    """
    rules: list[Rule] = []
    for path_str in paths:
        suffix = Path(path_str).suffix.lower()
        if suffix == ".txt":
            rules.extend(_parse_txt(path_str))
        elif suffix == ".json":
            rules.extend(_parse_json(path_str))
        elif suffix == ".md":
            msg = "Markdown parsing not yet implemented (v0.2)"
            raise NotImplementedError(msg)
        else:
            msg = f"Unsupported file type: {suffix!r} for {path_str}"
            raise ValueError(msg)
    return rules


def _parse_json(path: str) -> list[Rule]:
    """Parse a JSON rule file in the canonical intermediate format.

    Expected schema (version 1):
      {"version": 1, "rules": [{"text": "...", "expansions": [...], "source": {...}}]}

    Validates expansion length and count. Skips rules with empty text.
    """
    resolved = str(Path(path).resolve())

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    version = data.get("version")
    if version != 1:
        msg = f"Unsupported rules.json version: {version} in {path}"
        raise ValueError(msg)

    rules_data = data.get("rules", [])
    rules: list[Rule] = []

    for entry in rules_data:
        text = entry.get("text", "").strip()
        if not text:
            continue

        if len(text) > MAX_RULE_LENGTH:
            logger.warning(
                "Rule in %s exceeds %d chars (%d), truncating",
                path, MAX_RULE_LENGTH, len(text),
            )
            text = text[:MAX_RULE_LENGTH]

        # Parse expansions with validation
        raw_expansions = entry.get("expansions", [])
        expansions: list[str] = []
        for exp in raw_expansions:
            if not isinstance(exp, str):
                continue
            exp = exp.strip()
            if not exp:
                continue
            if len(exp) > MAX_EXPANSION_LENGTH:
                logger.warning(
                    "Expansion in %s exceeds %d chars (%d), truncating",
                    path, MAX_EXPANSION_LENGTH, len(exp),
                )
                exp = exp[:MAX_EXPANSION_LENGTH]
            expansions.append(exp)

        if len(expansions) > MAX_EXPANSIONS_PER_RULE:
            logger.warning(
                "Rule in %s has %d expansions (max %d), dropping extras",
                path, len(expansions), MAX_EXPANSIONS_PER_RULE,
            )
            expansions = expansions[:MAX_EXPANSIONS_PER_RULE]

        # Parse source provenance — always use the JSON file's own
        # resolved path, never trust embedded source.file from untrusted
        # JSON (prevents arbitrary file overwrite via cli remove)
        source = entry.get("source", {})
        line_start = source.get("line_start", 0)
        line_end = source.get("line_end", 0)
        chunk_type = source.get("chunk_type", "rule")

        provenance = Provenance(
            file=resolved,
            line_start=line_start,
            line_end=line_end,
            chunk_type=chunk_type,
        )
        rules.append(Rule(
            text=text,
            provenance=provenance,
            expansions=tuple(expansions),
        ))

    return rules


def _parse_txt(path: str) -> list[Rule]:
    """Parse a plain text rule file.

    One rule per line. Empty lines and '#' comments are skipped.
    Lines exceeding 500 characters are truncated with a warning.

    Args:
        path: Resolved absolute path to the .txt file.

    Returns:
        List of Rule objects.
    """
    resolved = str(Path(path).resolve())
    rules: list[Rule] = []

    with open(path, encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Enforce length limit
            if len(line) > MAX_RULE_LENGTH:
                logger.warning(
                    "Rule at %s:%d exceeds %d chars (%d), truncating",
                    path,
                    line_num,
                    MAX_RULE_LENGTH,
                    len(line),
                )
                line = line[:MAX_RULE_LENGTH]

            provenance = Provenance(
                file=resolved,
                line_start=line_num,
                line_end=line_num,
                chunk_type="rule",
            )
            rules.append(Rule(text=line, provenance=provenance))

    return rules
