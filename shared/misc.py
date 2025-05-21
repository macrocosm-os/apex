import asyncio
import functools
import time
import traceback
from functools import lru_cache, update_wrapper
from math import floor
from typing import Any, Callable

import bittensor as bt

from shared.exceptions import BittensorError


# decorator with options
def async_lru_cache(*lru_cache_args, **lru_cache_kwargs):
    def async_lru_cache_decorator(async_function):
        @functools.lru_cache(*lru_cache_args, **lru_cache_kwargs)
        def cached_async_function(*args, **kwargs):
            coroutine = async_function(*args, **kwargs)
            return asyncio.ensure_future(coroutine)

        return cached_async_function

    return async_lru_cache_decorator


class classproperty:
    def __init__(self, func: Callable):
        self.fget = func

    def __get__(self, instance, owner: Any):
        return self.fget(owner)


# LRU Cache with TTL
def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1):
    """
    Decorator that creates a cache of the most recently used function calls with a time-to-live (TTL) feature.
    The cache evicts the least recently used entries if the cache exceeds the `maxsize` or if an entry has
    been in the cache longer than the `ttl` period.

    Args:
        maxsize (int): Maximum size of the cache. Once the cache grows to this size, subsequent entries
                       replace the least recently used ones. Defaults to 128.
        typed (bool): If set to True, arguments of different types will be cached separately. For example,
                      f(3) and f(3.0) will be treated as distinct calls with distinct results. Defaults to False.
        ttl (int): The time-to-live for each cache entry, measured in seconds. If set to a non-positive value,
                   the TTL is set to a very large number, effectively making the cache entries permanent. Defaults to -1.

    Returns:
        Callable: A decorator that can be applied to functions to cache their return values.

    The decorator is useful for caching results of functions that are expensive to compute and are called
    with the same arguments frequently within short periods of time. The TTL feature helps in ensuring
    that the cached values are not stale.

    Example:
        @ttl_cache(ttl=10)
        def get_data(param):
            # Expensive data retrieval operation
            return data
    """
    if ttl <= 0:
        ttl = 65536
    hash_gen = _ttl_hash_gen(ttl)

    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def ttl_func(ttl_hash, *args, **kwargs):
            return func(*args, **kwargs)

        def wrapped(*args, **kwargs) -> Any:
            th = next(hash_gen)
            return ttl_func(th, *args, **kwargs)

        return update_wrapper(wrapped, func)

    return wrapper


def _ttl_hash_gen(seconds: int):
    """
    Internal generator function used by the `ttl_cache` decorator to generate a new hash value at regular
    time intervals specified by `seconds`.

    Args:
        seconds (int): The number of seconds after which a new hash value will be generated.

    Yields:
        int: A hash value that represents the current time interval.

    This generator is used to create time-based hash values that enable the `ttl_cache` to determine
    whether cached entries are still valid or if they have expired and should be recalculated.
    """
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)


# 12 seconds updating block.
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(subtensor: bt.Subtensor | None = None) -> int:
    """
    Retrieves the current block number from the blockchain. This method is cached with a time-to-live (TTL)
    of 12 seconds, meaning that it will only refresh the block number from the blockchain at most every 12 seconds,
    reducing the number of calls to the underlying blockchain interface.

    Returns:
        int: The current block number on the blockchain.

    This method is useful for applications that need to access the current block number frequently and can
    tolerate a delay of up to 12 seconds for the latest information. By using a cache with TTL, the method
    efficiently reduces the workload on the blockchain interface.

    Example:
        current_block = ttl_get_block(subtensor=subtensor)

    Note: self here is the miner or validator instance
    """
    try:
        return subtensor.get_current_block()
    except Exception as e:
        raise BittensorError(f"Bittensor error: {str(e)}") from e


def async_log(func):
    async def wrapper(*args, **kwargs):
        # Execute the wrapped function
        result = await func(*args, **kwargs)
        return result

    return wrapper


def serialize_exception_to_string(e):
    if isinstance(e, BaseException):
        # Format the traceback
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        # Combine type, message, and traceback into one string
        serialized_str = f"Exception Type: {type(e).__name__}, Message: {str(e)}, Traceback: {tb_str}"
        return serialized_str
    else:
        return e


def cached_property_with_expiration(expiration_seconds=1200):
    """
    Decorator that caches the property's value for `expiration_seconds` seconds.
    After this duration, the cached value is refreshed.
    """

    def decorator(func):
        attr_name = f"_cached_{func.__name__}"

        @property
        def wrapper(self):
            now = time.time()

            # Check if we have a cached value and if it's still valid
            if hasattr(self, attr_name):
                cached_value, timestamp = getattr(self, attr_name)

                # If valid, return cached value
                if now - timestamp < expiration_seconds:
                    return cached_value

            # Otherwise, compute the new value and cache it
            value = func(self)
            setattr(self, attr_name, (value, now))
            return value

        return wrapper

    return decorator


def is_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
