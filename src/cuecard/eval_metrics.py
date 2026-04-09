"""Evaluation metric functions: precision, recall, MRR, nDCG, noise, quality."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

import numpy as np


def precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Fraction of retrieved items that are relevant.

    Returns 0.0 if retrieved is empty.
    """
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(retrieved)


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Fraction of relevant items that appear in retrieved.

    Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant result.

    Returns 0.0 if no relevant result is found.
    """
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Normalized Discounted Cumulative Gain with binary relevance.

    Uses DCG formula: sum(rel_i / log2(i+1)) for i=1..k.
    Returns 0.0 if no relevant items exist.
    """
    if not relevant or not retrieved:
        return 0.0

    # DCG of the actual ranking
    dcg = 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: all relevant items ranked first
    n_relevant_in_k = min(len(relevant), len(retrieved))
    idcg = 0.0
    for i in range(1, n_relevant_in_k + 1):
        idcg += 1.0 / math.log2(i + 1)

    # idcg > 0 guaranteed: we early-return when relevant or retrieved is empty
    return dcg / idcg


def anti_precision(retrieved: list[str], anti_relevant: set[str]) -> float:
    """Fraction of retrieved items that are in the anti-relevant set.

    Desired value is always 0.0. Returns 0.0 if retrieved is empty.
    """
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in anti_relevant)
    return hits / len(retrieved)


def noise_ratio(retrieved: list[str], relevant: set[str]) -> float:
    """Fraction of retrieved items that are NOT relevant.

    Returns 0.0 if retrieved is empty.
    """
    if not retrieved:
        return 0.0
    irrelevant = sum(1 for r in retrieved if r not in relevant)
    return irrelevant / len(retrieved)


def context_waste_ratio(retrieved: list[str], relevant: set[str]) -> float:
    """Char-weighted waste: fraction of injected chars that are irrelevant.

    Returns 0.0 if retrieved is empty or total chars is 0.
    """
    if not retrieved:
        return 0.0
    total_chars = sum(len(r) for r in retrieved)
    if total_chars == 0:
        return 0.0
    waste_chars = sum(len(r) for r in retrieved if r not in relevant)
    return waste_chars / total_chars


def quality_score(
    retrieved: list[str],
    relevant: set[str],
    is_negative: bool,
) -> float:
    """Single quality score per fixture: F2 with correct-abstention convention.

    Uses F-beta with beta=2 (recall-weighted), extended with the
    empty-set convention for retrieval systems with negative queries.

    For positive fixtures (has should_match):
        F2 = 5 * P * R / (4P + R), where:
          P = precision (hits / retrieved)
          R = recall (hits / relevant)
        Returns 0.0 if both P and R are 0.

    For negative fixtures (no should_match):
        1.0 if silent (correct abstention), 0.0 otherwise.
        This is the standard empty-set convention — correct silence
        is a perfect retrieval outcome.

    The F2 formulation naturally penalizes noise (low precision) while
    weighting recall 4x higher than precision. It handles partial
    matches and noisy retrieval in a single principled number.
    """
    if is_negative:
        return 1.0 if not retrieved else 0.0

    if not relevant:
        return 1.0 if not retrieved else 0.0

    p = precision_at_k(retrieved, relevant)
    r = recall_at_k(retrieved, relevant)

    if p == 0.0 and r == 0.0:
        return 0.0

    # F2: beta=2 -> (1 + 4) * P * R / (4 * P + R)
    return 5.0 * p * r / (4.0 * p + r)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean(values: Iterable[float]) -> float:
    """Compute mean from an iterable. Returns 0.0 for empty input."""
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile from a non-empty list of values."""
    arr = np.array(sorted(values), dtype=np.float64)
    return float(np.percentile(arr, pct))
