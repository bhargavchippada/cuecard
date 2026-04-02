"""Multi-retriever framework: protocols, data types, and RRF fusion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cuecard.models import Index, Rule


@dataclass(frozen=True)
class ScoredCandidate:
    """A rule with a retriever-specific score."""

    rule: Rule
    score: float
    retriever: str  # "dense", "sparse", etc.


@runtime_checkable
class Retriever(Protocol):
    """A retrieval strategy that scores rules against a query."""

    def retrieve(
        self,
        query: str,
        index: Index,
        *,
        top_k: int,
        threshold: float,
    ) -> list[ScoredCandidate]: ...


def fuse(
    results: list[list[ScoredCandidate]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[ScoredCandidate]:
    """Reciprocal Rank Fusion across multiple retriever result lists.

    Score = sum(1 / (k + rank_i)) across retrievers, where rank_i
    is the 1-based position in retriever i's list.

    Args:
        results: One list of ScoredCandidates per retriever.
        k: RRF smoothing parameter (default 60).
        top_k: If set, truncate output to this many results.

    Returns:
        Merged list sorted by RRF score descending.  The
        ``retriever`` field on each output candidate is set to
        ``"fused"``.
    """
    if not results:
        return []

    # Accumulate RRF scores per rule (keyed by rule text for dedup)
    rrf_scores: dict[str, float] = defaultdict(float)
    rule_lookup: dict[str, Rule] = {}

    for ranked_list in results:
        for rank_zero, candidate in enumerate(ranked_list):
            key = candidate.rule.text
            rrf_scores[key] += 1.0 / (k + rank_zero + 1)
            # First occurrence wins for the Rule reference
            if key not in rule_lookup:
                rule_lookup[key] = candidate.rule

    # Sort descending by RRF score
    sorted_keys = sorted(rrf_scores, key=lambda t: rrf_scores[t], reverse=True)

    if top_k is not None:
        sorted_keys = sorted_keys[:top_k]

    return [
        ScoredCandidate(
            rule=rule_lookup[key],
            score=rrf_scores[key],
            retriever="fused",
        )
        for key in sorted_keys
    ]


__all__ = [
    "Retriever",
    "ScoredCandidate",
    "fuse",
]
