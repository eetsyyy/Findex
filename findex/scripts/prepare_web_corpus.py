"""Reconstruct persistent text files from trusted crawl JSONL for the web demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out", type=Path, default=Path("web_corpus"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.jsonl.open(encoding="utf-8") as source:
        for count, line in enumerate(source, start=1):
            row = json.loads(line)
            (args.out / f"{count - 1:07d}.txt").write_text(
                str(row.get("title", "")) + "\n" + str(row.get("text", "")),
                encoding="utf-8",
            )
    print(f"Created {count} documents in {args.out}")


if __name__ == "__main__":
    main()
