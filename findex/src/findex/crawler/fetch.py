"""Nonblocking HTTP fetches with bounded transient-failure retries."""
from __future__ import annotations

import asyncio
import email.utils
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx


@dataclass(frozen=True)
class FetchResult:
    url: str
    status: int
    content: bytes
    elapsed: float
    attempts: int
    error: str | None = None

    @property
    def size(self) -> int:
        return len(self.content)


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            return max(0.0, (when - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_attempts: int = 3,
    backoff: float = 0.2,
    backoff_cap: float = 10.0,
) -> FetchResult:
    """Try at most three times; retry only 429, 5xx and timeouts."""
    if max_attempts < 1 or max_attempts > 3:
        raise ValueError("max_attempts must be 1..3")
    started = time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        try:
            async with asyncio.timeout(timeout):
                response = await client.get(url, follow_redirects=False)
            status = response.status_code
            retry = status == 429 or 500 <= status <= 599
            if retry and attempt < max_attempts:
                retry_after = retry_after_seconds(response.headers.get("Retry-After"))
                delay = min(backoff_cap, backoff * (2 ** (attempt - 1)) + random.uniform(0, backoff))
                await asyncio.sleep(max(delay, retry_after or 0))
                continue
            return FetchResult(url, status, response.content, time.perf_counter() - started, attempt)
        except (TimeoutError, httpx.TimeoutException) as exc:
            if attempt >= max_attempts:
                return FetchResult(url, 0, b"", time.perf_counter() - started, attempt, str(exc) or "timeout")
            await asyncio.sleep(min(backoff_cap, backoff * (2 ** (attempt - 1)) + random.uniform(0, backoff)))
        except httpx.RequestError as exc:
            # Connection errors are logged and not retried (required policy: only timeout, 429, 5xx).
            return FetchResult(url, 0, b"", time.perf_counter() - started, attempt, str(exc))
    raise AssertionError("unreachable")
