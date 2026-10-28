"""Lab 03 CLI: ranked queries against a saved positional index."""
import argparse
import logging
from contextlib import contextmanager
from pathlib import Path

from findex.ranking import BM25, RankedSearcher, TfIdf
from findex.store import load


@contextmanager
def open_index(path: Path):
    """Release file-backed resources even when the body raises an exception."""
    index, _ = load(path)
    try:
        yield index
    finally:
        # Files are opened/closed inside load(), so no descriptor stays open.
        pass


def main():
    p = argparse.ArgumentParser(description="Lab 03: ranked full-text search")
    p.add_argument("index", type=Path)
    p.add_argument("query")
    p.add_argument("--corpus", type=Path, default=Path("data"))
    p.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25")
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with open_index(args.index) as idx:
        searcher = RankedSearcher(idx)
        for run in range(2):
            print(f"ПОШУК #{run+1} — {args.scorer} — {args.query}")
            results = searcher.search(args.query, BM25() if args.scorer == "bm25"
                                      else TfIdf(), args.top, args.corpus)
            for r in results:
                print(f"[{r.doc_id}] {r.title} | score={r.score:.5f}\n  {r.snippet}")
            print(f"Знайдено: {len(searcher.matching_ids(args.query))} документів")


if __name__ == "__main__":
    main()
