"""Parse rule files into Rule objects."""

from __future__ import annotations

import logging
from pathlib import Path

from cuecard.models import MAX_RULE_LENGTH, Provenance, Rule

logger = logging.getLogger(__name__)


def parse_rules(paths: tuple[str, ...]) -> list[Rule]:
    """Parse multiple rule files into a flat list of Rules.

    Dispatches by file suffix:
      - .txt -> _parse_txt
      - .md  -> NotImplementedError (v0.2)
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
        elif suffix == ".md":
            msg = "Markdown parsing not yet implemented (v0.2)"
            raise NotImplementedError(msg)
        else:
            msg = f"Unsupported file type: {suffix!r} for {path_str}"
            raise ValueError(msg)
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

    with open(path) as f:
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
