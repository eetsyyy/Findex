from pathlib import Path

import pytest

from findex.index import DocMeta, Posting, build_index, posting_pairs
from findex.search import evaluate, merge_and, merge_not, merge_or
from findex.store import load, save


def test_hashable():
    assert Posting(1, 2) in {Posting(1, 2)}
    assert DocMeta("a", "b") in {DocMeta("a", "b")}


def test_merges():
    assert merge_and([1, 3, 5], [2, 3, 5]) == [3, 5]
    assert merge_or([1, 3], [2, 3]) == [1, 2, 3]
    assert merge_not([1, 2, 3], [2]) == [1, 3]


@pytest.fixture
def corpus(tmp_path: Path):
    (tmp_path / "a.txt").write_text("Київ Python Python", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Київ море", encoding="utf-8")
    (tmp_path / "c.txt").write_text("море", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("rep", ["plain", "slots", "array"])
def test_build_and_search(corpus, rep):
    ix = build_index(corpus, rep)
    assert len(ix.doc_meta) == 3
    assert ix.doc_lengths == {0: 3, 1: 2, 2: 1}
    assert posting_pairs(ix, "python") == [(0, 2)]
    assert posting_pairs(ix, "київ") == [(0, 1), (1, 1)]
    for engine in ("merge", "set"):
        assert evaluate(ix, "київ python", engine) == [0]
        assert evaluate(ix, "київ OR море", engine) == [0, 1, 2]
        assert evaluate(ix, "київ NOT python", engine) == [1]
        assert evaluate(ix, "NOT море", engine) == [0]


@pytest.mark.parametrize("rep", ["plain", "slots", "array"])
@pytest.mark.parametrize("fmt,ext", [("pickle", ".bin"), ("json", ".json")])
def test_roundtrip(corpus, tmp_path, rep, fmt, ext):
    ix = build_index(corpus, rep)
    path = tmp_path / ("index" + ext)
    saved, size = save(ix, path, fmt)
    assert saved >= 0 and size > 0
    result, elapsed = load(path, fmt)
    assert elapsed >= 0
    assert result.doc_meta == ix.doc_meta
    assert result.doc_lengths == ix.doc_lengths
    assert posting_pairs(result, "київ") == posting_pairs(ix, "київ")
