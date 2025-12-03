import asyncio
from contextvars import copy_context
from typing import Callable, TypeVar


T = TypeVar("T")


async def to_thread_with_context(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a function in a thread while preserving the current loguru context.
    This is a drop-in replacement for asyncio.to_thread that preserves logging context.
    """
    ctx = copy_context()
    return await asyncio.to_thread(ctx.run, func, *args, **kwargs)
