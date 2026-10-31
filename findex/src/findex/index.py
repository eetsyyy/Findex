"""One-pass inverted-index builder and CLI."""

import argparse
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from findex.corpus import iter_documents
from findex.models import DocMeta, Index, PlainPosting, Posting
from findex.timing import timed
from findex.tokenize import tokenize

Representation = Literal["plain", "slots", "array"]


def posting_pairs(index: Index, term: str) -> list[tuple[int, int]]:
    """Return pairs independently of representation, ordered by document id."""
    postings = index.postings.get(term)
    if postings is None:
        return []
    if isinstance(postings, tuple):
        return list(zip(postings[0], postings[1]))
    return [(p.doc_id, p.tf) for p in postings]


@timed
def build_index(root: Path, representation: Representation = "slots", with_positions: bool = False) -> Index:
    """Read each document once, only retaining its Counter and final postings."""
    if representation not in {"plain", "slots", "array"}:
        raise ValueError(f"Unknown representation: {representation}")
    plain: defaultdict[str, list[PlainPosting]] = defaultdict(list)
    slots: defaultdict[str, list[Posting]] = defaultdict(list)
    compact: defaultdict[str, tuple[array[int], array[int]]] = defaultdict(
        lambda: (array("I"), array("I"))
    )
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    positions: defaultdict[str, dict[int, tuple[int, ...]]] | None = (
        defaultdict(dict) if with_positions else None
    )
    for doc_id, doc in enumerate(iter_documents(Path(root))):
        if with_positions:
            tokens = list(tokenize(doc.text))
            counts: Counter[str] = Counter(tokens)
            per_term: defaultdict[str, list[int]] = defaultdict(list)
            for offset, term in enumerate(tokens):
                per_term[term].append(offset)
            if positions is not None:
                for term, offsets in per_term.items():
                    positions[term][doc_id] = tuple(offsets)
        else:
            counts = Counter(tokenize(doc.text))
        doc_lengths[doc_id] = counts.total()
        doc_meta[doc_id] = DocMeta(path=doc.doc_id, title=doc.path.stem)
        for term, tf in counts.items():
            if representation == "plain":
                plain[term].append(PlainPosting(doc_id, tf))
            elif representation == "slots":
                slots[term].append(Posting(doc_id, tf))
            else:
                ids, frequencies = compact[term]
                ids.append(doc_id)
                frequencies.append(tf)
    if representation == "plain":
        final: dict[str, list[Posting] | list[PlainPosting] | tuple[array[int], array[int]]] = dict(plain)
    elif representation == "slots":
        final = dict(slots)
    else:
        final = dict(compact)
    return Index(final, doc_lengths, doc_meta, representation,
                 dict(positions) if positions is not None else None)


def measured_build(root: Path, representation: Representation) -> tuple[Index, float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    index = build_index(root, representation)
    duration = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return index, duration, peak / 1048576


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save an inverted index")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, default=Path("index.bin"))
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle")
    parser.add_argument("--positions", action="store_true", help="Store token positions for phrases")
    parser.add_argument("--representation", choices=["slots", "plain", "array"], default="slots")
    args = parser.parse_args()
    from findex.store import save

    if args.positions:
        tracemalloc.start()
        tick = time.perf_counter()
        index = build_index(args.corpus, args.representation, with_positions=True)
        duration = time.perf_counter() - tick
        _, memory = tracemalloc.get_traced_memory()
        peak = memory / 1048576
        tracemalloc.stop()
    else:
        index, duration, peak = measured_build(args.corpus, args.representation)
    store_time, size = save(index, args.out, args.format)
    print(f"Документів: {len(index.doc_meta)}; термінів: {len(index.postings)}")
    print(f"Побудова: {duration:.4f} с; пік tracemalloc: {peak:.3f} MiB")
    print(f"Збережено: {args.out} ({args.format}), {size} байт за {store_time:.4f} с")


if __name__ == "__main__":
    main()
