import asyncio
import time
from collections.abc import Callable
from functools import wraps
from typing import Any


def async_cache(ttl_seconds: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for caching the result of an async function for `ttl_seconds`.

    Each unique combination of positional and keyword arguments gets its own
    entry.  After `ttl_seconds` have elapsed the entry is discarded and the
    wrapped coroutine is executed again on the next call.

    The wrapper gains a `.clear_cache()` method that wipes everything.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache: dict[tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]], tuple[Any, float]] = {}
        lock = asyncio.Lock()

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build an argument key that distinguishes calls.
            # `id(self)` ensures methods on different instances
            # don't collide while still sharing the same decorator.
            if args and hasattr(args[0], "__dict__"):
                key_args = (id(args[0]),) + args[1:]
            else:
                key_args = args
            key = (key_args, tuple(sorted(kwargs.items())))

            now = time.monotonic()

            # Fast path: cache hit.
            async with lock:
                cached = cache.get(key)
                if cached:
                    value, expires_at = cached
                    if now < expires_at:
                        return value
                    else:
                        del cache[key]

            # Cache miss / expired.
            value = await func(*args, **kwargs)

            async with lock:
                cache[key] = (value, now + ttl_seconds)
            return value

        # Optional helper so you can evict everything manually.
        def clear_cache():
            cache.clear()

        wrapper.clear_cache = clear_cache  # type: ignore[attr-defined]
        return wrapper

    return decorator
