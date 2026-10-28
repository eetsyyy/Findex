"""One-pass inverted-index builder and CLI."""

import argparse
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from pathlib import Path

from findex.corpus import iter_documents
from findex.models import DocMeta, Index, PlainPosting, Posting
from findex.timing import timed
from findex.tokenize import tokenize


def posting_pairs(index: Index, term: str) -> list[tuple[int, int]]:
    """Return pairs independently of representation, ordered by document id."""
    postings = index.postings.get(term, ())
    if index.representation == "array":
        return list(zip(postings[0], postings[1])) if postings else []
    return [(p.doc_id, p.tf) for p in postings]


@timed
def build_index(root: Path, representation: str = "slots", with_positions: bool = False) -> Index:
    """Read each document once, only retaining its Counter and final postings."""
    if representation not in {"plain", "slots", "array"}:
        raise ValueError(f"Unknown representation: {representation}")
    postings = defaultdict(list) if representation != "array" else defaultdict(
        lambda: (array("I"), array("I"))
    )
    doc_lengths = {}
    doc_meta = {}
    positions = defaultdict(dict) if with_positions else None
    for doc_id, doc in enumerate(iter_documents(Path(root))):
        if with_positions:
            tokens = list(tokenize(doc.text))
            counts = Counter(tokens)
            per_term = defaultdict(list)
            for offset, term in enumerate(tokens):
                per_term[term].append(offset)
            for term, offsets in per_term.items():
                positions[term][doc_id] = tuple(offsets)
        else:
            counts = Counter(tokenize(doc.text))
        doc_lengths[doc_id] = counts.total()
        doc_meta[doc_id] = DocMeta(path=doc.doc_id, title=doc.path.stem)
        for term, tf in counts.items():
            if representation == "plain":
                postings[term].append(PlainPosting(doc_id, tf))
            elif representation == "slots":
                postings[term].append(Posting(doc_id, tf))
            else:
                ids, frequencies = postings[term]
                ids.append(doc_id)
                frequencies.append(tf)
    final = dict(postings)
    return Index(final, doc_lengths, doc_meta, representation, dict(positions) if positions is not None else None)


def measured_build(root: Path, representation: str):
    tracemalloc.start()
    start = time.perf_counter()
    index = build_index(root, representation)
    duration = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return index, duration, peak / 1048576


def main():
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
