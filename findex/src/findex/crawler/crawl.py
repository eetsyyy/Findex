"""Bounded async crawl generator using TaskGroup, queues and semaphores."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from findex.crawler.fetch import fetch
from findex.crawler.parse import canonicalize, parse_page
from findex.crawler.robots import RobotsCache

logger = logging.getLogger("findex.crawl")
DEFAULT_UA = "FindexCourseCrawler/0.6 (+https://github.com/eetsyyy/Findex/issues)"


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


@dataclass
class CrawlStats:
    pages: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    requests: int = 0


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    allowed_domains: set[str] | None = None,
    min_delay: float = 0.1,
    user_agent: str = DEFAULT_UA,
    client: httpx.AsyncClient | None = None,
    stats: CrawlStats | None = None,
    log_path: str | None = None,
) -> AsyncIterator[Page]:
    """Stream pages; enforce a finite URL budget and site-specific politeness."""
    if max_pages < 1 or concurrency < 1 or per_host < 1 or min_delay < 0:
        raise ValueError("max_pages/concurrency/per_host must be >=1 and delay >=0")
    seed_urls = [canonicalize(seed) for seed in seeds]
    if not seed_urls:
        return
    domains = {d.lower() for d in allowed_domains} if allowed_domains is not None else {
        urlsplit(seed).hostname or "" for seed in seed_urls
    }
    state = stats if stats is not None else CrawlStats()
    frontier: asyncio.Queue[str] = asyncio.Queue()
    outputs: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()
    for seed in seed_urls:
        if urlsplit(seed).hostname in domains and seed not in seen and len(seen) < max_pages:
            seen.add(seed)
            frontier.put_nowait(seed)
    state.queue_depth = frontier.qsize()
    global_sem = asyncio.Semaphore(concurrency)
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_locks: dict[str, asyncio.Lock] = {}
    host_next: dict[str, float] = {}
    # A single AsyncClient and a log file are owned by the generator.
    owned_client = client is None
    active_client = client or httpx.AsyncClient(headers={"User-Agent": user_agent}, follow_redirects=False)
    log_file = open(log_path, "w", encoding="utf-8") if log_path else None  # noqa: ASYNC230, SIM115
    robots = RobotsCache(active_client, user_agent)

    async def worker() -> None:
        while True:
            url = await frontier.get()
            try:
                host = urlsplit(url).netloc
                if not await robots.allowed(url):
                    state.errors += 1
                    if log_file:
                        log_file.write(f"{url}\tROBOTS_DENIED\t0\t0.000\n")
                        log_file.flush()
                    continue
                host_sem = host_sems.setdefault(host, asyncio.Semaphore(per_host))
                host_lock = host_locks.setdefault(host, asyncio.Lock())
                async with global_sem, host_sem:
                    async with host_lock:
                        now = time.monotonic()
                        pause = max(0.0, host_next.get(host, 0.0) - now)
                        if pause:
                            await asyncio.sleep(pause)
                        host_next[host] = time.monotonic() + min_delay
                    state.in_flight += 1
                    state.requests += 1
                    try:
                        result = await fetch(active_client, url)
                    finally:
                        state.in_flight -= 1
                if log_file:
                    log_file.write(
                        f"{url}\t{result.status}\t{result.size}\t{result.elapsed:.3f}"
                        f"\tattempts={result.attempts}\t{result.error or ''}\n"
                    )
                    log_file.flush()
                if result.status != 200:
                    state.errors += 1
                    continue
                if "html" not in result.content[:200].decode("utf-8", errors="ignore").lower() and not result.content.lstrip().startswith(b"<"):
                    state.errors += 1
                    continue
                title, text, links = parse_page(result.content.decode("utf-8", errors="replace"), url)
                state.pages += 1
                await outputs.put(Page(url, title, text, datetime.now(UTC).isoformat()))
                for link in links:
                    if len(seen) >= max_pages:
                        break
                    if urlsplit(link).hostname in domains and link not in seen:
                        seen.add(link)
                        frontier.put_nowait(link)
                state.queue_depth = frontier.qsize()
            except (httpx.HTTPError, UnicodeError, ValueError) as exc:
                state.errors += 1
                logger.warning("Unable to crawl %s: %s", url, exc)
                if log_file:
                    log_file.write(f"{url}\tERROR\t0\t0\t{exc}\n")
                    log_file.flush()
            finally:
                frontier.task_done()
                state.queue_depth = frontier.qsize()

    async def finalize() -> None:
        await frontier.join()
        await outputs.put(None)

    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(worker()) for _ in range(concurrency)]
            tg.create_task(finalize())
            while True:
                item = await outputs.get()
                if item is None:
                    break
                yield item
            for task in tasks:
                task.cancel()
    finally:
        if log_file:
            log_file.close()
        if owned_client:
            await active_client.aclose()

