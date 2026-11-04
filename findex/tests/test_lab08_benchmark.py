"""Pytest-benchmark smoke cases: run locally and in CI with --benchmark-only."""
from __future__ import annotations

from pathlib import Path

from findex.index import build_index
from findex.numpy_ranking import NumPyRankedSearcher


def test_benchmark_build_index(benchmark, tmp_path: Path) -> None:
    for n in range(25):
        (tmp_path / f"{n:03d}.txt").write_text(f"alpha beta gamma document {n}")
    index = benchmark(build_index, tmp_path)
    assert index.num_docs == 25


def test_benchmark_numpy_search(benchmark, tmp_path: Path) -> None:
    for n in range(25):
        (tmp_path / f"{n:03d}.txt").write_text(f"alpha beta gamma document {n}")
    ranker = NumPyRankedSearcher(build_index(tmp_path))
    hits = benchmark(ranker.search, "alpha OR beta OR gamma", scorer="bm25", k=5)
    assert len(hits) == 5
