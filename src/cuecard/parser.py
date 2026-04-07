"""Parse rule files into Rule objects."""

from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path

from cuecard.models import (
    KNOWN_HOOK_EVENTS,
    MAX_EXPANSION_LENGTH,
    MAX_EXPANSIONS_PER_RULE,
    MAX_RULE_LENGTH,
    Provenance,
    Rule,
)

logger = logging.getLogger(__name__)

_MAX_RULES_PER_FILE = 500


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
        elif suffix == ".toml":
            rules.extend(_parse_toml(path_str))
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
    if version not in (1, 2):
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

        # V2: read events/tools if present
        events = frozenset(entry.get("events", ()))
        tools = frozenset(entry.get("tools", ()))

        rules.append(Rule(
            text=text,
            provenance=provenance,
            expansions=tuple(expansions),
            events=events,
            tools=tools,
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


def _parse_toml(path: str) -> list[Rule]:
    """Parse TOML rule file with event/tool annotations.

    Expected schema::

        [[rules]]
        text = "Rule text"
        events = ["PreToolUse", "PostToolUse"]
        tools = ["Bash"]

    Uses ``chunk_type="toml_rule"`` — line_start is the rule ordinal
    (1-based), not the actual file line number (tomllib doesn't expose lines).
    """
    resolved = str(Path(path).resolve())

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        msg = f"Malformed TOML rule file {path!r}: {exc}"
        raise ValueError(msg) from exc

    rules_data = data.get("rules", [])
    if len(rules_data) > _MAX_RULES_PER_FILE:
        logger.warning(
            "Rule file %s has %d rules, capped at %d",
            path, len(rules_data), _MAX_RULES_PER_FILE,
        )
        rules_data = rules_data[:_MAX_RULES_PER_FILE]

    rules: list[Rule] = []
    for i, entry in enumerate(rules_data):
        text = entry.get("text", "").strip()
        if not text:
            continue

        if len(text) > MAX_RULE_LENGTH:
            logger.warning(
                "Rule %d in %s exceeds %d chars (%d), truncating",
                i + 1, path, MAX_RULE_LENGTH, len(text),
            )
            text = text[:MAX_RULE_LENGTH]

        events = frozenset(entry.get("events", ()))
        tools = frozenset(entry.get("tools", ()))

        # Validate event names (warn, don't error — forward compat)
        for ev in events:
            if ev not in KNOWN_HOOK_EVENTS:
                logger.warning(
                    "Unknown event %r in rule %d of %s", ev, i + 1, path,
                )

        provenance = Provenance(
            file=resolved,
            line_start=i + 1,
            line_end=i + 1,
            chunk_type="toml_rule",
        )
        rules.append(Rule(
            text=text,
            provenance=provenance,
            events=events,
            tools=tools,
        ))

    return rules
