"""Per-host robots.txt cache; conservative fail-closed policy on fetch errors."""
from __future__ import annotations

import asyncio
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx


class RobotsCache:
    def __init__(self, client: httpx.AsyncClient, user_agent: str) -> None:
        self._client = client
        self._user_agent = user_agent
        self._cache: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return False
        origin = f"{parts.scheme}://{parts.netloc.lower()}"
        if origin not in self._cache:
            lock = self._locks.setdefault(origin, asyncio.Lock())
            async with lock:
                if origin not in self._cache:
                    try:
                        async with asyncio.timeout(10):
                            response = await self._client.get(origin + "/robots.txt", follow_redirects=False)
                        if response.status_code == 404:
                            parser = RobotFileParser()
                            parser.parse([])
                            self._cache[origin] = parser
                        elif response.status_code == 200:
                            parser = RobotFileParser()
                            parser.parse(response.text.splitlines())
                            self._cache[origin] = parser
                        else:
                            self._cache[origin] = None
                    except (TimeoutError, httpx.RequestError):
                        self._cache[origin] = None
        rule = self._cache[origin]
        return rule is not None and rule.can_fetch(self._user_agent, url)
