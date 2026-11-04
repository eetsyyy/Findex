"""Lab 08: NumPy scorer must preserve the legacy ranking and scores."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findex.index import build_index
from findex.numpy_ranking import NumPyRankedSearcher
from findex.ranking import BM25, RankedSearcher, TfIdf


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    for name, content in {
        "a.txt": "alpha beta beta common",
        "b.txt": "alpha beta common common",
        "c.txt": "gamma rare common",
        "d.txt": "alpha gamma common",
        "e.txt": "another independent text",
    }.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("scorer,original", [("bm25", BM25()), ("tfidf", TfIdf())])
@pytest.mark.parametrize("query", ["alpha", "common", "rare", "alpha OR rare", "alpha AND beta", "gamma NOT alpha", "alpha beta gamma"])
def test_ranking_matches_legacy(corpus: Path, scorer: str, original: BM25 | TfIdf, query: str) -> None:
    index = build_index(corpus)
    before = RankedSearcher(index).search(query, scorer=original, k=10)
    after = NumPyRankedSearcher(index).search(query, scorer=scorer, k=10)  # type: ignore[arg-type]
    assert [item.doc_id for item in after] == [item.doc_id for item in before]
    assert np.allclose([item.score for item in after], [item.score for item in before], atol=1e-12)


def test_topk_ties_stable(corpus: Path) -> None:
    searcher = NumPyRankedSearcher(build_index(corpus))
    all_hits = searcher.search("common", k=10)
    assert [hit.doc_id for hit in searcher.search("common", k=2)] == [hit.doc_id for hit in all_hits[:2]]


def test_invalid_k(corpus: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NumPyRankedSearcher(build_index(corpus)).search("alpha", k=-1)
