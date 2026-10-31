"""Reusable decorator for measuring execution time."""
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

log = logging.getLogger("findex.timing")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            log.info("@timed %s %.3f ms", func.__name__, (time.perf_counter() - start) * 1000)
    return wrapper
