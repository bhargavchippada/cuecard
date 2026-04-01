"""Format ranked retrieval results for context injection and CLI display."""

from __future__ import annotations

from os.path import basename
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuecard.models import RankedResult

_BOUNDARY_LABEL = "[cuecard \u2014 user-defined guidelines relevant to this action]"


def format_rules(results: list[RankedResult], *, scrub: bool = True) -> str:
    """Format results for injection into agent context.

    Returns the labeled boundary prefix followed by one rule per line.
    Uses rule.summary when available, otherwise rule.text.
    Scrubs secrets from rule text before injection (defense-in-depth).
    No scores, no provenance — clean and directive.

    Returns empty string when *results* is empty.
    """
    if not results:
        return ""

    from cuecard.security import scrub_secrets

    lines: list[str] = [_BOUNDARY_LABEL]
    for r in results:
        text = r.rule.summary if r.rule.summary is not None else r.rule.text
        if scrub:
            text = scrub_secrets(text)
        lines.append(f"- {text}")
    return "\n".join(lines)


def format_rules_verbose(results: list[RankedResult]) -> str:
    """Format results for CLI output with scores and provenance.

    Each result is numbered and shows its retrieval score plus the
    source file basename and line range.

    Returns ``"No matching rules found."`` when *results* is empty.
    """
    if not results:
        return "No matching rules found."

    lines: list[str] = []
    for idx, r in enumerate(results, start=1):
        text = r.rule.summary if r.rule.summary is not None else r.rule.text
        prov = r.rule.provenance
        filename = basename(prov.file)
        if prov.line_start == prov.line_end:
            location = f"{filename}:{prov.line_start}"
        else:
            location = f"{filename}:{prov.line_start}-{prov.line_end}"
        lines.append(f"[{idx}] ({r.score:.2f}) {text}")
        lines.append(f"    \u2514\u2500 {location}")
    return "\n".join(lines)
