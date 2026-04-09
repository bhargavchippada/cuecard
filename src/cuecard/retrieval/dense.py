"""Dense retriever: cosine similarity via dot product with parent collapse."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from cuecard._math import l2_normalize
from cuecard.retrieval.fusion import ScoredCandidate
from cuecard.retrieval.retriever import normalize_query

if TYPE_CHECKING:
    import numpy.typing as npt
    from fastembed import TextEmbedding

    from cuecard.models import Index

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Embedding-based retrieval with parent-child collapse via rule_map.

    Wraps the core cosine-similarity logic from ``retriever.py``,
    adding parent-child collapse so expansion embeddings are folded
    back to their parent rule (max score wins).
    """

    def __init__(
        self,
        model: TextEmbedding | None = None,
        *,
        dedup_threshold: float = 0.95,
        max_query_length: int = 500,
    ) -> None:
        self._model = model
        self._dedup_threshold = dedup_threshold
        self._max_query_length = max_query_length

    def retrieve(
        self,
        query: str,
        index: Index,
        *,
        top_k: int = 5,
        threshold: float = 0.30,
        mask: npt.NDArray[np.bool_] | None = None,
    ) -> list[ScoredCandidate]:
        """Retrieve top-k rules via dense cosine similarity.

        Uses ``index.rule_map`` to collapse expansion embeddings
        back to parent rules (max score per parent).
        """
        if index.size == 0:
            return []

        query = normalize_query(query)

        if len(query) > self._max_query_length:
            logger.warning(
                "Query truncated from %d to %d characters",
                len(query),
                self._max_query_length,
            )
            query = query[: self._max_query_length]

        if self._model is None:
            msg = "model is required for non-empty index"
            raise ValueError(msg)

        # Asymmetric encoding: query_embed for queries
        query_vec = np.array(
            list(self._model.query_embed([query])), dtype=np.float32,
        )
        query_vec = l2_normalize(query_vec)  # (1, dim)

        # Dot product == cosine (embeddings are L2-normalized)
        scores = (index.embeddings @ query_vec.T).flatten()

        # Event mask: zero out non-matching embeddings (post-scoring)
        if mask is not None:
            scores = scores.copy()
            scores[~mask] = -np.inf

        # Parent collapse: max score per parent rule via rule_map
        num_rules = len(index.rules)
        parent_scores = np.full(num_rules, -np.inf, dtype=np.float64)
        np.maximum.at(parent_scores, list(index.rule_map), scores)

        # Threshold filter on parent scores
        above_threshold = parent_scores >= threshold
        candidate_parent_idxs = np.where(above_threshold)[0]

        if candidate_parent_idxs.size == 0:
            return []

        # Sort descending by parent score
        order = np.argsort(-parent_scores[candidate_parent_idxs])
        sorted_parent_idxs = candidate_parent_idxs[order]

        # Precompute canonical row per parent (first embedding row)
        canonical_row: dict[int, int] = {}
        for emb_idx, parent_idx_val in enumerate(index.rule_map):
            if parent_idx_val not in canonical_row:
                canonical_row[parent_idx_val] = emb_idx

        # Semantic dedup on canonical embeddings
        accepted: list[int] = []
        for parent_idx in sorted_parent_idxs:
            pidx = int(parent_idx)
            if self._is_near_duplicate(
                pidx, accepted, index, canonical_row,
            ):
                continue
            accepted.append(pidx)
            if len(accepted) == top_k:
                break

        return [
            ScoredCandidate(
                rule=index.rules[i],
                score=float(parent_scores[i]),
                retriever="dense",
            )
            for i in accepted
        ]

    def _is_near_duplicate(
        self,
        candidate_idx: int,
        accepted_idxs: list[int],
        index: Index,
        canonical_row: dict[int, int],
    ) -> bool:
        """Check near-duplicate against accepted parents using canonical embeddings."""
        if not accepted_idxs:
            return False

        cand_row = canonical_row[candidate_idx]
        cand_emb = index.embeddings[cand_row]

        for acc_idx in accepted_idxs:
            acc_row = canonical_row[acc_idx]
            sim = float(np.dot(cand_emb, index.embeddings[acc_row]))
            if sim > self._dedup_threshold:
                return True
        return False
