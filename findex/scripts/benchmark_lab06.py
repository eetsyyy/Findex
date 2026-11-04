"""Repeatable local 200-page crawl benchmark: no third-party sites."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

from findex.crawler.crawl import CrawlStats, crawl


async def measure(seed: str, concurrency: int, output: Path, max_pages: int) -> dict[str, int | float]:
    stats = CrawlStats()
    parent = psutil.Process()
    max_rss = parent.memory_info().rss
    done = False

    async def sampler() -> None:
        nonlocal max_rss
        while not done:
            max_rss = max(max_rss, parent.memory_info().rss)
            await asyncio.sleep(0.01)

    count = 0
    started = time.perf_counter()
    task = asyncio.create_task(sampler())
    try:
        with output.open("w", encoding="utf-8") as file:
            async for page in crawl(
                [seed], max_pages=max_pages, concurrency=concurrency, per_host=20,
                min_delay=0.0, stats=stats,
                log_path=str(output.with_suffix(".log")),
            ):
                file.write(json.dumps(page.__dict__, ensure_ascii=False) + "\n")
                count += 1
    finally:
        done = True
        await task
    wall = time.perf_counter() - started
    return {"concurrency": concurrency, "pages": count, "wall_seconds": round(wall, 4),
            "pages_per_second": round(count / wall, 4), "errors": stats.errors,
            "peak_parent_rss_mib": round(max_rss / (1024 * 1024), 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("results/lab06"))
    args = parser.parse_args()
    if args.max_pages < 200:
        parser.error("The assignment requires at least 200 pages")
    args.out.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([sys.executable, "scripts/local_crawl_site.py", "--port", "8765"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    seed = "http://127.0.0.1:8765/page/0"
    try:
        for _ in range(50):
            try:
                with httpx.Client(timeout=0.3) as client:
                    if client.get("http://127.0.0.1:8765/robots.txt").status_code == 200:
                        break
            except httpx.RequestError:
                time.sleep(0.1)
        else:
            raise RuntimeError("Local test site did not start (port 8765 occupied?)")
        rows = []
        for level in (1, 5, 20):
            row = asyncio.run(measure(seed, level, args.out / f"crawl_{level}.jsonl", args.max_pages))
            if row["pages"] != args.max_pages:
                raise RuntimeError(f"Incomplete crawl: {row}")
            rows.append(row)
            print(row, flush=True)
        with (args.out / "benchmark.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {args.out / 'benchmark.csv'}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
