"""Profile indexing and three-term search; save cProfile artifacts and top functions."""
from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

from findex.index import build_index
from findex.ranking import BM25, RankedSearcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    def index_work():
        return build_index(args.corpus)
    profile = cProfile.Profile()
    index = profile.runcall(index_work)
    profile.dump_stats(str(args.out / "index.prof"))
    with (args.out / "index_profile.txt").open("w", encoding="utf-8") as output:
        pstats.Stats(profile, stream=output).sort_stats("cumtime").print_stats(30)
    terms = sorted(index.postings, key=lambda t: index.df(t))[-3:]
    query = " OR ".join(terms)
    searcher = RankedSearcher(index)
    profile = cProfile.Profile()
    profile.runcall(searcher.search, query, BM25(), 10)
    profile.dump_stats(str(args.out / "search.prof"))
    with (args.out / "search_profile.txt").open("w", encoding="utf-8") as output:
        pstats.Stats(profile, stream=output).sort_stats("cumtime").print_stats(30)
    print("Profiled index and search query:", query)
    print("Saved docs/index_profile.txt and docs/search_profile.txt")


if __name__ == "__main__":
    main()
