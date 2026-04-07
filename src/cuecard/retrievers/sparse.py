"""Sparse retriever: inline BM25Okapi with parent collapse."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import numpy as np

from cuecard.retriever import normalize_query
from cuecard.retrievers import ScoredCandidate

if TYPE_CHECKING:
    import numpy.typing as npt

    from cuecard.models import Index

# Tokenizer: whitespace split + lowercasing + dot/paren/bracket splitting
_SPLIT_RE = re.compile(r"[\s.()\[\]{}]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, split on whitespace and code punctuation."""
    return [tok for tok in _SPLIT_RE.split(text.lower()) if tok]


class BM25Okapi:
    """Minimal inline BM25Okapi implementation (~30 lines of core logic).

    Avoids external dependency on rank_bm25. Operates on a pre-tokenized
    corpus built at index time.
    """

    def __init__(
        self,
        corpus: tuple[str, ...],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = k1
        self._b = b
        tokenized = [_tokenize(doc) for doc in corpus]
        self._doc_lens = [len(doc) for doc in tokenized]
        self._avgdl = sum(self._doc_lens) / max(len(self._doc_lens), 1)
        self._n_docs = len(tokenized)

        # Build inverted index: term -> list of (doc_idx, term_freq)
        self._idf: dict[str, float] = {}
        self._tf: dict[str, list[tuple[int, int]]] = {}

        df: dict[str, int] = {}
        for doc_idx, tokens in enumerate(tokenized):
            term_counts: dict[str, int] = {}
            for tok in tokens:
                term_counts[tok] = term_counts.get(tok, 0) + 1
            for term, count in term_counts.items():
                df[term] = df.get(term, 0) + 1
                if term not in self._tf:
                    self._tf[term] = []
                self._tf[term].append((doc_idx, count))

        # IDF with smoothing: log((N - df + 0.5) / (df + 0.5) + 1)
        for term, doc_freq in df.items():
            self._idf[term] = math.log(
                (self._n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0,
            )

    def score(self, query: str) -> list[float]:
        """Score all documents against the query. Returns list of floats."""
        query_tokens = _tokenize(query)
        scores = [0.0] * self._n_docs

        for token in query_tokens:
            idf = self._idf.get(token, 0.0)
            if idf == 0.0:
                continue
            for doc_idx, tf in self._tf.get(token, []):
                dl = self._doc_lens[doc_idx]
                numerator = tf * (self._k1 + 1.0)
                denominator = tf + self._k1 * (
                    1.0 - self._b + self._b * dl / self._avgdl
                )
                scores[doc_idx] += idf * numerator / denominator

        return scores


class SparseRetriever:
    """BM25-based sparse retrieval with parent collapse via rule_map.

    Uses ``index.bm25_corpus`` for the document texts. If the corpus
    is None (old indexes), retrieval returns an empty list.

    Caches the BM25 index for the lifetime of this retriever instance.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._corpus_id: int = -1

    def retrieve(
        self,
        query: str,
        index: Index,
        *,
        top_k: int = 5,
        threshold: float = 0.0,
        mask: npt.NDArray[np.bool_] | None = None,
    ) -> list[ScoredCandidate]:
        """Retrieve top-k rules via BM25 scoring with parent collapse."""
        if index.size == 0 or index.bm25_corpus is None:
            return []

        query = normalize_query(query)

        # Cache BM25 index — rebuild only if corpus identity changes
        corpus_id = id(index.bm25_corpus)
        if self._bm25 is None or self._corpus_id != corpus_id:
            self._bm25 = BM25Okapi(index.bm25_corpus)
            self._corpus_id = corpus_id
        bm25 = self._bm25
        doc_scores = bm25.score(query)

        # Event mask: zero out non-matching entries (post-scoring)
        if mask is not None:
            doc_scores_arr = np.array(doc_scores, dtype=np.float64)
            doc_scores_arr[~mask] = -np.inf
            doc_scores = doc_scores_arr.tolist()

        # Parent collapse: max score per parent rule via rule_map
        num_rules = len(index.rules)
        parent_scores = np.full(num_rules, -np.inf, dtype=np.float64)
        np.maximum.at(parent_scores, list(index.rule_map), doc_scores)

        # Threshold filter
        above_threshold = parent_scores > threshold
        candidate_parent_idxs = np.where(above_threshold)[0]

        if candidate_parent_idxs.size == 0:
            return []

        # Sort descending
        order = np.argsort(-parent_scores[candidate_parent_idxs])
        sorted_parent_idxs = candidate_parent_idxs[order]

        # Top-k
        top_parent_idxs = sorted_parent_idxs[:top_k]

        return [
            ScoredCandidate(
                rule=index.rules[int(i)],
                score=float(parent_scores[int(i)]),
                retriever="sparse",
            )
            for i in top_parent_idxs
        ]
