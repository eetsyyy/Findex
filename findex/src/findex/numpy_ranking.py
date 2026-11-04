"""NumPy vectorized BM25 and TF-IDF; preserves Findex Lab 03 ranking semantics.

The index's serialized posting representation stays unchanged for backwards
compatibility. Arrays are materialized once per NumPyRankedSearcher instance.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np

from findex.index import posting_pairs
from findex.models import Index
from findex.query import parse
from findex.ranking import SearchResult, make_snippet

ScoreKind = Literal["bm25", "tfidf"]


class NumPyRankedSearcher:
    def __init__(self, index: Index) -> None:
        self.index = index
        self.doc_ids = np.array(sorted(index.doc_lengths), dtype=np.int32)
        self.doc_lengths = np.zeros(int(self.doc_ids.max()) + 1 if self.doc_ids.size else 0, dtype=np.int32)
        for doc_id, length in index.doc_lengths.items():
            self.doc_lengths[doc_id] = length
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term in index.postings:
            pairs = posting_pairs(index, term)
            self.postings[term] = (
                np.fromiter((doc for doc, _ in pairs), dtype=np.int32, count=len(pairs)),
                np.fromiter((tf for _, tf in pairs), dtype=np.int32, count=len(pairs)),
            )

    @classmethod
    def from_memmap(cls, index: Index, directory: Path) -> NumPyRankedSearcher:
        """Load memory-mapped int32 postings instead of copying numeric data."""
        from findex.memmap_index import load_mmap

        ranker = cls.__new__(cls)
        ranker.index = index
        ranker.doc_ids = np.array(sorted(index.doc_lengths), dtype=np.int32)
        ranker.doc_lengths, ranker.postings = load_mmap(directory)
        return ranker

    def search(
        self, query: str, *, scorer: ScoreKind = "bm25", k: int = 5,
        corpus: Path | None = None,
    ) -> list[SearchResult]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if scorer not in ("bm25", "tfidf"):
            raise ValueError("scorer must be bm25 or tfidf")
        if k == 0 or self.doc_ids.size == 0:
            return []
        parsed = parse(query)
        matched = parsed.evaluate(self.index)
        if not matched:
            return []
        terms = tuple(dict.fromkeys(parsed.terms()))
        # Keep only query-matching documents, including matches with score zero.
        ids = np.array(sorted(matched), dtype=np.int32)
        total_scores = np.zeros(len(self.doc_lengths), dtype=np.float64)
        n_docs = self.index.num_docs
        avg_len = self.index.avg_doc_length or 1.0
        for term in terms:
            postings = self.postings.get(term)
            if postings is None:
                continue
            docs, tfs = postings
            df = len(docs)
            include = np.isin(docs, ids, assume_unique=True)
            docs, tfs = docs[include], tfs[include]
            if not len(docs):
                continue
            tf = tfs.astype(np.float64)
            if scorer == "bm25":
                k1, b = 1.5, 0.75
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * self.doc_lengths[docs] / avg_len)
                scores = idf * tf * (k1 + 1) / denom
            else:
                idf = math.log(1 + n_docs / (1 + df))
                scores = (1 + np.log(tf)) * idf
            np.add.at(total_scores, docs, scores)
        scores = total_scores[ids]
        # np.argpartition is O(n) average for top-k. Include all ties at the
        # cutoff before the final stable sort by (-score, doc_id).
        count = min(k, len(ids))
        if count < len(ids):
            top = np.argpartition(scores, -count)[-count:]
            cutoff = float(np.min(scores[top]))
            eligible = np.flatnonzero(scores >= cutoff)
        else:
            eligible = np.arange(len(ids))
        order = np.lexsort((ids[eligible], -scores[eligible]))[:count]
        winners = eligible[order]
        results: list[SearchResult] = []
        for position in winners:
            doc_id = int(ids[position])
            meta = self.index.doc_meta[doc_id]
            snippet = make_snippet(meta.path, terms, corpus) if corpus is not None else ""
            results.append(SearchResult(float(scores[position]), doc_id, meta.title, meta.path, snippet))
        return results


def cached_numpy_searcher(index: Index) -> NumPyRankedSearcher:
    """Reuse native arrays across requests to the same in-memory index."""
    ranker: NumPyRankedSearcher | None = getattr(index, "_numpy_ranker", None)
    if ranker is None:
        ranker = NumPyRankedSearcher(index)
        index.__dict__["_numpy_ranker"] = ranker
    return ranker
