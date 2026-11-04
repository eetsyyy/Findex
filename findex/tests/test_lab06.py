"""All crawler tests are local: never touch the live internet."""
from __future__ import annotations

import asyncio

import httpx

from findex.crawler.crawl import crawl
from findex.crawler.fetch import fetch
from findex.crawler.parse import canonicalize
from findex.crawler.robots import RobotsCache


def run(coro):
    return asyncio.run(coro)


def test_canonicalize():
    assert canonicalize("HTTPS://EXAMPLE.COM/a/?z=3&a=1#frag") == "https://example.com/a?a=1&z=3"


def test_fetch_404_no_retry():
    calls = []
    async def handler(request):
        calls.append(request)
        return httpx.Response(404)
    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            r = await fetch(c, "https://example.com/missing", backoff=0)
            assert r.status == 404 and r.attempts == 1 and len(calls) == 1
    run(check())


def test_fetch_429_retries_and_retry_after():
    calls = 0
    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, text="<html>ok</html>")
    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            r = await fetch(c, "https://example.com", backoff=0)
            assert r.status == 200 and r.attempts == 2 and calls == 2
    run(check())


def test_robots_caches_denial():
    calls = 0
    async def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            robots = RobotsCache(c, "FindexCourseCrawler")
            assert not await robots.allowed("https://example.com/private")
            assert await robots.allowed("https://example.com/public")
            assert calls == 1
    run(check())


def test_crawl_limit_dedup_and_allowed_host():
    async def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        num = int(request.url.path.split("/")[-1])
        links = "".join(f'<a href="/page/{n}">next</a>' for n in range(8))
        links += '<a href="https://foreign.test/page/9">outside</a>'
        return httpx.Response(200, text=f"<html><title>{num}</title>{links}</html>")
    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            pages = [page async for page in crawl(["https://example.com/page/0"], max_pages=5, concurrency=3, per_host=2, min_delay=0, client=client)]
            assert len(pages) == 5
            assert len({page.url for page in pages}) == 5
            assert all("example.com" in page.url for page in pages)
    run(check())
