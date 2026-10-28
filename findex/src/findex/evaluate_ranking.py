"""Ten manually labeled topic queries and three ranking sanity checks."""
import argparse
from pathlib import Path

from findex.index import build_index
from findex.ranking import BM25, RankedSearcher, TfIdf

# Labels describe the generated 8-topic demonstration corpus.
# Each query must retrieve documents belonging to the listed topical group.
CASES = [
    ("python", 0), ("генератори", 0),
    ("індекс", 1), ("київ", 1),
    ("unicode", 2), ("токенізація", 2),
    ("продуктивність", 3), ("пам'яті", 3),
    ("ліниве", 4), ("жадібне", 4),
]


def evaluate(corpus: Path):
    index = build_index(corpus, with_positions=True)
    searcher = RankedSearcher(index)
    print("ЗАПИТ | TF-IDF P@5 | BM25 P@5 | позначка релевантності")
    total = [0.0, 0.0]
    for query, group in CASES:
        row = []
        for scorer in (TfIdf(), BM25()):
            results = searcher.search(query, scorer, k=5)
            # Article numbers were generated in 8-topic cycles; this is the
            # explicit relevance rule for the synthetic demonstration corpus.
            hits = sum(int(r.doc_id % 8 == group) for r in results)
            row.append(hits / 5)
        total[0] += row[0]
        total[1] += row[1]
        print(f"{query} | {row[0]:.2f} | {row[1]:.2f} | doc_id % 8 = {group}")
    print(f"СЕРЕДНЄ | {total[0]/len(CASES):.2f} | {total[1]/len(CASES):.2f}")

    print("\nSANITY CHECK BM25 (синтетичні контрольовані умови)")
    bm = BM25()
    rare = bm(1, 1, 100, 100, 100)
    common = bm(1, 70, 100, 100, 100)
    first = bm(1, 10, 100, 100, 100)
    twentieth = bm(20, 10, 100, 100, 100)
    nineteenth = bm(19, 10, 100, 100, 100)
    short = bm(1, 10, 100, 50, 100)
    long = bm(1, 10, 100, 1000, 100)
    print(f"Рідкісний {rare:.6f} > частий {common:.6f}: {rare > common}")
    print(f"Приріст від 19 до 20: {twentieth - nineteenth:.6f}; від 0 до 1: {first:.6f}")
    print(f"Короткий {short:.6f} > довгий {long:.6f}: {short > long}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    args = ap.parse_args()
    evaluate(args.corpus)


if __name__ == "__main__":
    main()
