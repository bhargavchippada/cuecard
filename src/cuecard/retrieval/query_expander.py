"""Query-side expansion for tag-to-tag matching.

Uses the same unified expansion prompt as rule expansion so that rules and
queries produce consistent abstract tag vocabulary. When both sides generate
the same tag phrasing, cosine similarity approaches 1.0 — the tag-to-tag
bridge that enables cross-language retrieval.
"""

from __future__ import annotations

import logging
import secrets

from cuecard.indexing.expander import (
    _build_expansion_prompt,
    _parse_expansion_response,
)
from cuecard.retrieval.llm_utils import call_local

logger = logging.getLogger(__name__)


def expand_query(
    query: str,
    *,
    endpoint: str,
    event: str = "PreToolUse",
    max_per_query: int = 5,
    timeout: float = 2.0,
) -> tuple[str, ...]:
    """Generate expansion phrases for a query via LLM.

    Uses the same unified expansion prompt as rule expansion so both sides
    produce consistent tag vocabulary.

    Args:
        query: Raw query string (e.g., "Bash: cargo add serde").
        endpoint: Local LLM endpoint URL (llama-server).
        event: Hook event for prompt targeting (tool-style vs workflow-style).
        max_per_query: Max expansions to return after balanced selection.
        timeout: Max seconds to wait for LLM response.

    Returns:
        Tuple of expansion strings. Empty on failure, timeout, or parse error.
    """
    if not query.strip():
        return ()

    nonce = secrets.token_hex(6)
    try:
        system, user = _build_expansion_prompt(
            query, nonce, event_type=event,
        )
        raw = call_local(
            system, user, endpoint, False,
            timeout=timeout,
            temperature=0.7,
        )
        expansions = _parse_expansion_response(
            raw, max_per_rule=max_per_query,
        )
        return tuple(expansions)
    except Exception as exc:
        logger.warning(
            "Query expansion failed: %s — falling back to raw query",
            type(exc).__name__,
        )
        return ()
