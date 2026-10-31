"""Core Lab 05 invariants: equivalence, global IDs, sorted lists and worker errors."""
from pathlib import Path

import pytest

from findex.index import build_index, posting_pairs
from findex.parallel import build_parallel, build_partial, merge, partition_documents


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("python тест python", encoding="utf-8")
    (tmp_path / "b.txt").write_text("тест дані", encoding="utf-8")
    (tmp_path / "c.txt").write_text("python дані", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("representation", ["plain", "slots", "array"])
@pytest.mark.parametrize("positions", [False, True])
def test_serial_matches_lab04(corpus: Path, representation: str, positions: bool) -> None:
    reference = build_index(corpus, representation=representation, with_positions=positions)  # type: ignore[arg-type]
    chunks = partition_documents(corpus, 3)
    result = merge([build_partial(chunk, positions) for chunk in chunks], representation)  # type: ignore[arg-type]
    assert result.doc_meta == reference.doc_meta
    assert result.doc_lengths == reference.doc_lengths
    assert result.positions == reference.positions
    assert set(result.postings) == set(reference.postings)
    for term in result.postings:
        assert posting_pairs(result, term) == posting_pairs(reference, term)


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_all_executors(corpus: Path, executor: str) -> None:
    idx, metrics = build_parallel(corpus, workers=2, executor=executor, with_positions=True)  # type: ignore[arg-type]
    assert idx.num_docs == 3
    assert metrics.wall_seconds >= metrics.merge_seconds >= 0
    assert posting_pairs(idx, "python") == [(0, 2), (2, 1)]


def test_worker_failure_propagates(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        build_partial([(0, missing, "missing.txt")])


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("one", encoding="utf-8")
    partial = build_partial([(0, f, "x.txt")])
    with pytest.raises(ValueError, match="Duplicate"):
        merge([partial, partial])
