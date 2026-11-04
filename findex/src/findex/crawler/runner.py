"""Streaming JSONL output and Rich live counters."""
from __future__ import annotations

import json
import time
from pathlib import Path

from rich.live import Live
from rich.table import Table

from findex.crawler.crawl import CrawlStats, crawl


async def run_crawl(
    seed: str, out: Path, *, max_pages: int, concurrency: int,
    per_host: int, min_delay: float, log_path: Path,
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stats = CrawlStats()
    start = time.perf_counter()

    def display() -> Table:
        elapsed = max(time.perf_counter() - start, 0.001)
        table = Table(title="Findex: асинхронний краулер")
        table.add_column("Зібрано")
        table.add_column("Сторінок/с")
        table.add_column("Запитів в очікуванні")
        table.add_column("Помилок")
        table.add_column("Черга")
        table.add_row(str(stats.pages), f"{stats.pages / elapsed:.2f}", str(stats.in_flight),
                      str(stats.errors), str(stats.queue_depth))
        return table

    with out.open("w", encoding="utf-8") as target, Live(display(), refresh_per_second=4) as live:
        async for page in crawl(
            [seed], max_pages=max_pages, concurrency=concurrency, per_host=per_host,
            min_delay=min_delay, stats=stats, log_path=str(log_path),
        ):
            target.write(json.dumps({"url": page.url, "title": page.title,
                                     "text": page.text, "fetched_at": page.fetched_at}, ensure_ascii=False) + "\n")
            target.flush()
            live.update(display())
        live.update(display())
    return stats.pages
