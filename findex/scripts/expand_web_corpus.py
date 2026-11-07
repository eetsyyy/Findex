import argparse
from pathlib import Path

TOPICS = [
    "Python generators", "Unicode normalization",
    "Inverted indexes", "BM25 ranking",
    "TF IDF scoring", "Document retrieval",
    "Web crawling", "Asynchronous requests",
    "Multithreading", "Multiprocessing",
    "Memory optimization", "NumPy arrays",
    "Vectorized computation", "Query parsing",
    "Search pagination", "FastAPI services",
    "Pydantic validation", "Docker deployment",
    "Performance profiling", "Memory mapped files",
]

METHODS = [
    "token frequency analysis",
    "document length normalization",
    "efficient data storage",
    "benchmark driven optimization",
    "error handling and validation",
    "scalable index construction",
    "query evaluation and ranking",
    "reproducible testing",
    "CPU and memory measurements",
    "search result presentation",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("web_corpus"))
    parser.add_argument("--target", type=int, default=1000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    original = 200
    if args.target < original:
        raise ValueError("Target must be at least 200")

    for doc_id in range(original, args.target):
        topic = TOPICS[(doc_id - original) % len(TOPICS)]
        method = METHODS[((doc_id - original) // len(TOPICS)) % len(METHODS)]
        related = TOPICS[(doc_id * 7 + 3) % len(TOPICS)]
        phase = (doc_id - original) // (len(TOPICS) * len(METHODS))

        text = (
            f"Findex research document {doc_id}\n\n"
            f"Subject: {topic}.\n"
            f"Method: {method}.\n"
            f"Related subject: {related}.\n\n"
            f"This document discusses {topic} within a Python "
            f"information retrieval system. It considers "
            f"{method} and its relationship to {related}. "
            f"The experiment belongs to scenario {phase + 1} "
            f"and has document identifier {doc_id}.\n\n"
            f"Data processing includes indexing, search, "
            f"ranking, verification, and performance evaluation. "
            f"Different scenarios compare implementation choices "
            f"and their effects on correctness and efficiency.\n"
        )

        (args.out / f"{doc_id:07d}.txt").write_text(
            text, encoding="utf-8"
        )

    files = list(args.out.glob("*.txt"))
    print(f"Corpus documents: {len(files)}")
    print(f"New synthetic documents: {args.target - original}")

if __name__ == "__main__":
    main()
