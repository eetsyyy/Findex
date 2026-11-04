"""Lab 08 benchmark: legacy scorer versus NumPy on corpus and 10x corpus.

Run in project folder: uv run python scripts/benchmark_lab08.py web_corpus
"""
from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import tempfile
import time
from functools import partial
from pathlib import Path

from findex.index import build_index
from findex.memmap_index import load_mmap, save_mmap
from findex.numpy_ranking import NumPyRankedSearcher
from findex.ranking import BM25, RankedSearcher


def median_seconds(action, repeats: int = 15) -> float:
    return statistics.median(_time_one(action) for _ in range(repeats))


def _time_one(action) -> float:
    start = time.perf_counter()
    action()
    return time.perf_counter() - start


def build_expanded_corpus(source: Path, target: Path, factor: int) -> None:
    files = sorted(source.rglob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in {source}")
    for repeat in range(factor):
        part = target / f"batch_{repeat:02d}"
        part.mkdir(parents=True, exist_ok=True)
        for file in files:
            # Do not hardlink, copy from the source: one source path = one document.
            shutil.copy2(file, part / f"{repeat:02d}_{file.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/lab08"))
    parser.add_argument("--factor", type=int, default=10)
    parser.add_argument("--runs", type=int, default=15)
    args = parser.parse_args()
    if args.factor < 2 or args.runs < 3:
        parser.error("factor must be >= 2 and runs must be >= 3")
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="findex_lab08_") as temp:
        large = Path(temp) / "large"
        build_expanded_corpus(args.corpus, large, args.factor)
        for label, root in (("original", args.corpus), (f"x{args.factor}", large)):
            build_s = median_seconds(partial(build_index, root), repeats=3)
            index = build_index(root)
            old = RankedSearcher(index)
            before = time.perf_counter()
            new = NumPyRankedSearcher(index)
            array_build_s = time.perf_counter() - before
            terms = sorted(index.postings, key=lambda t: index.df(t))
            if len(terms) < 3:
                raise ValueError("Need at least three unique terms")
            common, rare = terms[-1], terms[0]
            three = " OR ".join(terms[-3:])
            for kind, query in (("common", common), ("rare", rare), ("three_terms", three)):
                # Warm both paths before measuring.
                old.search(query, BM25(), k=10)
                new.search(query, scorer="bm25", k=10)
                a = median_seconds(partial(old.search, query, BM25(), k=10), args.runs)
                b = median_seconds(partial(new.search, query, scorer="bm25", k=10), args.runs)
                old_ids = [r.doc_id for r in old.search(query, BM25(), k=10)]
                numpy_ids = [r.doc_id for r in new.search(query, scorer="bm25", k=10)]
                if old_ids != numpy_ids:
                    raise AssertionError(f"Rankings differ: {query!r}")
                row = {
                    "corpus": label, "docs": index.num_docs, "query_type": kind,
                    "query": query, "matches": len(old.matching_ids(query)),
                    "legacy_ms": round(a * 1000, 5), "numpy_ms": round(b * 1000, 5),
                    "speedup": round(a / b, 4) if b else 0,
                    "build_index_s": round(build_s, 5),
                    "numpy_conversion_s": round(array_build_s, 5),
                }
                rows.append(row)
                print(row, flush=True)
            mmap_dir = args.out / f"mmap_{label}"
            writing = _time_one(partial(save_mmap, index, mmap_dir))
            loading = _time_one(partial(load_mmap, mmap_dir))
            lengths, postings = load_mmap(mmap_dir)
            assert len(postings) == len(index.postings)
            assert len(lengths) == max(index.doc_lengths) + 1
            mapped = NumPyRankedSearcher.from_memmap(index, mmap_dir)
            assert [r.doc_id for r in mapped.search(common, k=10)] == [r.doc_id for r in new.search(common, k=10)]
            mapped_search = median_seconds(partial(mapped.search, common, k=10), args.runs)
            print(f"memmap {label}: write {writing:.4f}s, reopen {loading:.4f}s, search {mapped_search*1000:.4f}ms")
    output = args.out / "benchmark.csv"
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
