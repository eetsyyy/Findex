"""Property-based tests: index round-trip, order of postings, Boolean merge."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

from findex.index import build_index, posting_pairs
from findex.search import merge_and, merge_not, merge_or
from findex.store import load, save


@settings(
    max_examples=25,
    deadline=None,
)
@given(st.lists(st.text(alphabet="abc абв", min_size=1, max_size=15), min_size=1, max_size=6))
def test_roundtrip(texts: list[str]) -> None:
    with TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        corpus = tmp_path / "texts"
        corpus.mkdir()
        for i, content in enumerate(texts):
            (corpus / f"{i:03}.txt").write_text(content, encoding="utf-8")
        idx = build_index(corpus)
        file = tmp_path / "index.json"
        save(idx, file, "json")
        restored, _ = load(file)
        assert restored.doc_lengths == idx.doc_lengths
        assert restored.doc_meta == idx.doc_meta
        assert {t: posting_pairs(restored, t) for t in restored} == {t: posting_pairs(idx, t) for t in idx}

@settings(
    max_examples=25,
    deadline=None,
)
@given(st.lists(st.text(alphabet="abc def", min_size=1, max_size=15), min_size=1, max_size=6))
def test_postings_sorted(texts: list[str]) -> None:
    with TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        for i, content in enumerate(texts):
            (tmp_path / f"{i:03}.txt").write_text(content, encoding="utf-8")
        idx = build_index(tmp_path)
        for term in idx:
            ids = [doc_id for doc_id, _ in posting_pairs(idx, term)]
            assert ids == sorted(set(ids))


@given(st.sets(st.integers(0, 100)), st.sets(st.integers(0, 100)))
def test_merge_matches_set_ops(left: set[int], right: set[int]) -> None:
    a, b = sorted(left), sorted(right)
    assert merge_and(a, b) == sorted(left & right)
    assert merge_or(a, b) == sorted(left | right)
    assert merge_not(a, b) == sorted(left - right)
