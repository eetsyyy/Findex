"""Long-running workload for py-spy / Scalene to capture meaningful samples."""
from __future__ import annotations

import argparse
from pathlib import Path

from findex.ranking import BM25, RankedSearcher
from findex.store import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("--repeat", type=int, default=5000)
    args = parser.parse_args()
    index, _ = load(args.index)
    terms = sorted(index.postings, key=lambda term: index.df(term))[-3:]
    query = " OR ".join(terms)
    scorer = RankedSearcher(index)
    for _ in range(args.repeat):
        scorer.search(query, BM25(), k=10)
    print(f"Profiled {args.repeat} searches for: {query}")


if __name__ == "__main__":
    main()
