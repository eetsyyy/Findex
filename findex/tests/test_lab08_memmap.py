"""Memory-mapped array storage returns the same search rankings."""
from __future__ import annotations

from pathlib import Path

from findex.index import build_index
from findex.memmap_index import load_mmap, save_mmap
from findex.numpy_ranking import NumPyRankedSearcher


def test_mmap_rankings(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(10):
        (docs / f"{i}.txt").write_text(f"alpha beta unique{i}", encoding="utf-8")
    index = build_index(docs)
    destination = tmp_path / "arrays"
    save_mmap(index, destination)
    lengths, postings = load_mmap(destination)
    assert lengths.dtype.name == "int32"
    assert all(ids.dtype.name == "int32" and tfs.dtype.name == "int32" for ids, tfs in postings.values())
    baseline = NumPyRankedSearcher(index).search("alpha OR beta", k=5)
    mapped = NumPyRankedSearcher.from_memmap(index, destination).search("alpha OR beta", k=5)
    assert [r.doc_id for r in baseline] == [r.doc_id for r in mapped]
