"""Corpus statistics and reproducible lazy-vs-eager benchmark CLI."""

import argparse
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass
class Result:
    documents: int
    tokens: int
    frequencies: Counter[str]

    @property
    def vocabulary(self) -> int:
        return len(self.frequencies)


def lazy_stats(root: Path, limit: int | None = None) -> Result:
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)
    counts: Counter[str] = Counter()
    document_count = 0
    token_count = 0
    for document in docs:
        document_count += 1
        for token in tokenize(document.text):
            counts[token] += 1
            token_count += 1
    return Result(document_count, token_count, counts)


def eager_stats(root: Path, limit: int | None = None) -> Result:
    """Deliberately materialize documents AND per-document token lists."""
    source = iter_documents(root)
    if limit is not None:
        source = islice(source, limit)
    documents = list(source)
    token_lists = [list(tokenize(doc.text)) for doc in documents]
    counts = Counter(token for tokens in token_lists for token in tokens)
    token_count = sum(map(len, token_lists))
    return Result(len(documents), token_count, counts)


def measure(function, root: Path, limit: int | None) -> tuple[Result, float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = function(root, limit)
        elapsed = time.perf_counter() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, elapsed, peak_bytes / 1024**2


def print_result(name: str, result: Result, elapsed: float, peak_mb: float) -> None:
    print(f"\n{name.upper()}")
    print(f"Documents: {result.documents}")
    print(f"Tokens: {result.tokens}")
    print(f"Vocabulary: {result.vocabulary}")
    print(f"Elapsed: {elapsed:.4f} s")
    print(f"Peak traced memory: {peak_mb:.3f} MiB")
    print("Top 50 terms:")
    for word, count in result.frequencies.most_common(50):
        print(f"  {word:<24} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory with .txt documents")
    parser.add_argument("--limit", type=int, default=None, help="Process first N documents")
    parser.add_argument("--mode", choices=("lazy", "eager", "compare"), default="lazy")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    modes = {"lazy": lazy_stats, "eager": eager_stats}
    selected = modes.items() if args.mode == "compare" else [(args.mode, modes[args.mode])]
    results = {}
    for name, func in selected:
        result, elapsed, peak_mb = measure(func, args.root, args.limit)
        results[name] = result
        print_result(name, result, elapsed, peak_mb)
    if len(results) == 2 and results["lazy"] != results["eager"]:
        raise RuntimeError("Eager and lazy implementations produced different statistics")


if __name__ == "__main__":
    main()
