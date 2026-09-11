# cache.py
"""
In-memory cache keyed by normalized query text, so a repeated
identical query skips a redundant crew invocation.
"""
import re

_CACHE: dict[str, dict] = {}
_STATS = {"calls_to_crew": 0, "cache_hits": 0}


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def get_or_compute(query: str, compute_fn):
    """
    compute_fn: zero-arg callable that actually invokes the crew.
    Returns (result, was_cache_hit: bool).
    """
    key = _normalize(query)
    if key in _CACHE:
        _STATS["cache_hits"] += 1
        return _CACHE[key], True

    result = compute_fn()
    _STATS["calls_to_crew"] += 1
    _CACHE[key] = result
    return result, False


def stats() -> dict:
    return dict(_STATS)


def clear() -> None:
    _CACHE.clear()
    _STATS["calls_to_crew"] = 0
    _STATS["cache_hits"] = 0
