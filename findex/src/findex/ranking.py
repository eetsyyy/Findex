"""TF-IDF / BM25 rankers, snippets, timing and query caching."""
from __future__ import annotations

import heapq
import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from findex.index import posting_pairs
from findex.models import Index
from findex.query import parse
from findex.timing import timed

log = logging.getLogger("findex.timing")


class Scorer(Protocol):
    def __call__(
        self, tf: int, df: int, num_docs: int, doc_len: int, avg_len: float
    ) -> float: ...


@dataclass(frozen=True)
class TfIdf:
    def __call__(
        self, tf: int, df: int, num_docs: int, doc_len: int, avg_len: float
    ) -> float:
        return (1 + math.log(tf)) * math.log(1 + num_docs / (1 + df)) if tf else 0.0


@dataclass(frozen=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def __call__(
        self, tf: int, df: int, num_docs: int, doc_len: int, avg_len: float
    ) -> float:
        if not tf:
            return 0.0
        idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
        denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (avg_len or 1))
        return idf * tf * (self.k1 + 1) / denom


@dataclass(order=True, frozen=True)
class SearchResult:
    score: float
    doc_id: int
    title: str = field(compare=False)
    path: str = field(compare=False)
    snippet: str = field(compare=False, default="")


def make_snippet(path: str, words: tuple[str, ...], corpus: Path, radius: int = 80) -> str:
    source = corpus / path
    if not source.is_file():
        return "(Текст документа недоступний: вкажіть --corpus)"
    text = unicodedata.normalize("NFC", source.read_text(encoding="utf-8", errors="replace"))
    hits: list[tuple[int, int]] = []
    for word in words:
        if word:
            match = re.search(re.escape(unicodedata.normalize("NFC", word)), text, re.IGNORECASE)
            if match:
                hits.append(match.span())
    begin, end = min(hits) if hits else (0, min(30, len(text)))
    left, right = max(0, begin - radius), min(len(text), end + radius)
    excerpt = text[left:right].replace("\n", " ")
    for word in sorted(set(words), key=len, reverse=True):
        excerpt = re.sub(re.escape(word), lambda m: f"**{m.group()}**", excerpt, flags=re.IGNORECASE)
    return ("…" if left else "") + excerpt + ("…" if right < len(text) else "")


class RankedSearcher:
    def __init__(self, index: Index) -> None:
        self.index: Index = index

    @lru_cache(maxsize=256)  # noqa: B019 - educational per-instance searcher, cache cleared on disposal
    def matching_ids(self, query: str) -> frozenset[int]:
        return frozenset(parse(query).evaluate(self.index))

    @timed
    def search(
        self, query: str, scorer: Scorer | None = None, k: int = 5,
        corpus: Path | None = None,
    ) -> list[SearchResult]:
        if k < 0:
            raise ValueError("k must be non-negative")
        scorer = scorer if scorer is not None else BM25()
        before = self.matching_ids.cache_info()
        ids = self.matching_ids(query)
        after = self.matching_ids.cache_info()
        log.info("cache %s: %s", "HIT" if after.hits > before.hits else "MISS", query)
        terms = tuple(dict.fromkeys(parse(query).terms()))
        totals: dict[int, float] = {doc_id: 0.0 for doc_id in ids}
        for term in terms:
            pairs = posting_pairs(self.index, term)
            df = len(pairs)
            for doc_id, tf in pairs:
                if doc_id in totals:
                    totals[doc_id] += scorer(
                        tf, df, self.index.num_docs,
                        self.index.doc_lengths[doc_id], self.index.avg_doc_length,
                    )
        ranked = heapq.nlargest(
            k,
            (
                SearchResult(score, doc_id, self.index.doc_meta[doc_id].title,
                             self.index.doc_meta[doc_id].path)
                for doc_id, score in totals.items()
            ),
            key=lambda result: (result.score, -result.doc_id),
        )
        if corpus is not None:
            ranked = [
                SearchResult(r.score, r.doc_id, r.title, r.path,
                             make_snippet(r.path, terms, corpus))
                for r in ranked
            ]
        return ranked
