"""Boolean search using sorted two-pointer merges or Python sets."""

import argparse
import time
import tracemalloc
from pathlib import Path
from typing import Literal

from findex.index import Index, posting_pairs
from findex.store import load
from findex.tokenize import tokenize


def merge_and(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def merge_or(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and a[i] < b[j]):
            out.append(a[i]); i += 1
        elif i >= len(a) or b[j] < a[i]:
            out.append(b[j]); j += 1
        else:
            out.append(a[i]); i += 1; j += 1
    return out


def merge_not(a: list[int], b: list[int]) -> list[int]:
    """Sorted difference: a minus b."""
    i = j = 0
    out: list[int] = []
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i]); i += 1
        elif a[i] == b[j]:
            i += 1; j += 1
        else:
            j += 1
    return out


def evaluate(index: Index, query: str, engine: Literal["merge", "set"] = "merge") -> list[int]:
    """Left-to-right Boolean evaluator; adjacent terms imply AND; NOT is unary."""
    if engine not in {"merge", "set"}:
        raise ValueError(engine)
    raw = query.split()
    parts: list[str] = []
    for word in raw:
        if word.upper() in {"AND", "OR", "NOT"}:
            parts.append(word.upper())
        else:
            parts.extend(tokenize(word))
    if not parts:
        return []
    universe = sorted(index.doc_meta)

    def combine(a: list[int], b: list[int], op: Literal["AND", "OR"]) -> list[int]:
        if engine == "set":
            sa, sb = set(a), set(b)
            return sorted((sa | sb) if op == "OR" else (sa & sb))
        return merge_or(a, b) if op == "OR" else merge_and(a, b)

    def operand(at: int) -> tuple[list[int], int]:
        negate = False
        while at < len(parts) and parts[at] == "NOT":
            negate = not negate
            at += 1
        if at >= len(parts) or parts[at] in {"AND", "OR"}:
            raise ValueError("Expected search term")
        found = [doc_id for doc_id, _ in posting_pairs(index, parts[at])]
        if negate:
            found = (sorted(set(universe) - set(found)) if engine == "set"
                     else merge_not(universe, found))
        return found, at + 1

    result, pos = operand(0)
    while pos < len(parts):
        op: Literal["AND", "OR"] = "AND"
        if parts[pos] in {"AND", "OR"}:
            op = "OR" if parts[pos] == "OR" else "AND"
            pos += 1
        rhs, pos = operand(pos)
        result = combine(result, rhs, op)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Search a saved index")
    p.add_argument("index", type=Path)
    p.add_argument("query")
    p.add_argument("--engine", choices=["merge", "set"], default="merge")
    p.add_argument("--format", choices=["pickle", "json"], default=None)
    args = p.parse_args()
    tracemalloc.start()
    t0 = time.perf_counter()
    index, load_time = load(args.index, args.format)
    matches = evaluate(index, args.query, args.engine)
    total = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Пошук: {args.query!r}; алгоритм: {args.engine}; знайдено: {len(matches)}")
    for doc_id in matches[:20]:
        meta = index.doc_meta[doc_id]
        print(f"  [{doc_id}] {meta.title} — {meta.path}")
    if len(matches) > 20:
        print(f"  ...ще {len(matches) - 20}")
    print(f"Завантаження: {load_time:.4f} с; загалом: {total:.4f} с; "
          f"пік tracemalloc: {peak / 1048576:.3f} MiB")


if __name__ == "__main__":
    main()
