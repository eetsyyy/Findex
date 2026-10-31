"""Reproducible memory, persistence and Boolean-query benchmarks."""

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from findex.index import measured_build
from findex.models import Index
from findex.search import evaluate
from findex.store import load, save


def benchmark(root: Path, runs: int = 1000) -> None:
    print("ПРЕДСТАВЛЕННЯ | ДОКУМЕНТІВ | ПІК MiB | PICKLE БАЙТ | LOAD c | JSON БАЙТ | LOAD c")
    indexes: dict[str, Index] = {}
    for kind in ("plain", "slots", "array"):
        index, sec, mib = measured_build(root, kind)
        indexes[kind] = index
        with tempfile.TemporaryDirectory() as tmp:
            row: list[tuple[int, float, float]] = []
            for fmt, ext in (("pickle", ".bin"), ("json", ".json")):
                name = Path(tmp) / ("index" + ext)
                saved, size = save(index, name, fmt)
                reloaded, loaded = load(name, fmt)
                assert len(reloaded.doc_meta) == len(index.doc_meta)
                row.append((size, saved, loaded))
        print(f"{kind} | {len(index.doc_meta)} | {mib:.4f} | "
              f"{row[0][0]} | {row[0][2]:.5f} | {row[1][0]} | {row[1][2]:.5f} "
              f"(build {sec:.4f} c; save pickle {row[0][1]:.5f} c; "
              f"save JSON {row[1][1]:.5f} c)")
    index = indexes["slots"]
    common = sorted(index.postings, key=lambda t: (-len(index.postings[t]), t))[:2]
    rare = sorted(index.postings, key=lambda t: (len(index.postings[t]), t))[:2]
    print("\nЗАПИТ | ENGINE | середній час пошуку, мс | результати")
    for label, terms in (("часті", common), ("рідкі", rare)):
        query = " ".join(terms)
        for engine in ("merge", "set"):
            times: list[float] = []
            result: list[int] = []
            for _ in range(runs):
                start = time.perf_counter()
                result = evaluate(index, query, engine)
                times.append((time.perf_counter() - start) * 1000)
            print(f"{label}: {query} | {engine} | {statistics.mean(times):.6f} | {len(result)}")
    print("\nПРИМІТКА: два найчастіші/найрідкісніші терміни визначені за document frequency.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()
    benchmark(args.corpus, args.runs)


if __name__ == "__main__":
    main()
