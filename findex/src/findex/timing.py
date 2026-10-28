"""Reusable decorator for measuring execution time."""
import logging
import time
from functools import wraps

log = logging.getLogger("findex.timing")


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            log.info("@timed %s %.3f ms", func.__name__, (time.perf_counter() - start) * 1000)
    return wrapper
