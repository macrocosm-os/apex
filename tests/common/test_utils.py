import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from apex.common.utils import async_cache


@pytest.mark.asyncio
async def test_async_cache_caches_result():
    mock_async_func = AsyncMock(return_value="test_result")

    @async_cache(ttl_seconds=10)
    async def cached_func() -> str:
        return str(await mock_async_func())

    # First call, should call the original function.
    result1 = await cached_func()
    assert result1 == "test_result"
    mock_async_func.assert_called_once()

    # Second call, should return cached result.
    result2 = await cached_func()
    assert result2 == "test_result"
    # Should not be called again.
    mock_async_func.assert_called_once()


@pytest.mark.asyncio
async def test_async_cache_ttl_expiration():
    mock_async_func = AsyncMock(return_value="test_result")

    @async_cache(ttl_seconds=1)
    async def cached_func() -> Any:
        return await mock_async_func()

    # First call.
    await cached_func()
    mock_async_func.assert_called_once()

    # Wait for TTL to expire.
    await asyncio.sleep(2)

    # Second call after expiration.
    await cached_func()
    assert mock_async_func.call_count == 2


@pytest.mark.asyncio
async def test_async_cache_clear_cache():
    mock_async_func = AsyncMock(return_value="test_result")

    @async_cache(ttl_seconds=10)
    async def cached_func() -> str:
        return str(await mock_async_func())

    # First call.
    await cached_func()
    mock_async_func.assert_called_once()

    # Clear the cache.
    cached_func.clear_cache()  # type: ignore

    # Second call after clearing cache.
    await cached_func()
    assert mock_async_func.call_count == 2


@pytest.mark.asyncio
async def test_async_cache_different_args():
    mock_async_func = AsyncMock(side_effect=lambda x, y: f"{x}-{y}")

    @async_cache(ttl_seconds=10)
    async def cached_func(arg1: Any, kwarg1: str = "default") -> Any:
        return await mock_async_func(arg1, kwarg1)

    # Call with different arguments.
    result1 = await cached_func("a", kwarg1="b")
    result2 = await cached_func("a", kwarg1="c")
    result3 = await cached_func("b", kwarg1="b")
    result4 = await cached_func("a", kwarg1="b")

    assert result1 == "a-b"
    assert result2 == "a-c"
    assert result3 == "b-b"
    assert result4 == "a-b"

    assert mock_async_func.call_count == 3


@pytest.mark.asyncio
async def test_async_cache_on_class_method():
    mock_method_1 = AsyncMock(return_value="instance1")
    mock_method_2 = AsyncMock(return_value="instance2")

    class MyClass:
        def __init__(self, mock_method: Any) -> None:
            self.mock_method = mock_method

        @async_cache(ttl_seconds=10)
        async def my_method(self, arg: Any) -> Any:
            return await self.mock_method(arg)

    instance1 = MyClass(mock_method_1)
    instance2 = MyClass(mock_method_2)

    # Call method on instance 1.
    res1_call1 = await instance1.my_method("a")
    res1_call2 = await instance1.my_method("a")

    # Call method on instance 2.
    res2_call1 = await instance2.my_method("a")
    res2_call2 = await instance2.my_method("a")

    assert res1_call1 == "instance1"
    assert res1_call2 == "instance1"
    assert res2_call1 == "instance2"
    assert res2_call2 == "instance2"

    mock_method_1.assert_called_once_with("a")
    mock_method_2.assert_called_once_with("a")
