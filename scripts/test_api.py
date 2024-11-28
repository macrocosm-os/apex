import openai
from httpx import Timeout
from typing import Optional
from prompting.base.epistula import create_header_hook
from prompting import settings

settings.settings = settings.Settings.load(mode="validator")
settings = settings.settings


def setup_miner_client(
    port: int = 8004,
    api_key: str = "123456",  # Default key from your api_keys.json
    hotkey: Optional[str] = None
) -> openai.AsyncOpenAI:
    """
    Setup an authenticated OpenAI client for the miner.
    
    Args:
        port: Port number for the local server
        api_key: API key for authentication
        hotkey: Optional wallet hotkey
    
    Returns:
        Configured AsyncOpenAI client
    """

    # Create headers with both API key and hotkey
    async def combined_header_hook(request):
        # Add API key header
        request.headers["api-key"] = api_key
        # Add any additional headers from the original header hook
        if hotkey:
            original_hook = create_header_hook(hotkey, None)
            await original_hook(request)
        return request

    return openai.AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        max_retries=0,
        timeout=Timeout(15, connect=5, read=10),
        http_client=openai.DefaultAsyncHttpxClient(
            event_hooks={"request": [combined_header_hook]}
        ),
    )


async def make_completion(
    miner: openai.AsyncOpenAI,
    prompt: str,
    stream: bool = False,
    seed: str = "1759348"
) -> str:
    """
    Make a completion request to the API.
    
    Args:
        miner: Configured AsyncOpenAI client
        prompt: Input prompt
        stream: Whether to stream the response
        seed: Random seed for reproducibility

    Returns:
        Generated completion text
    """
    result = await miner.chat.completions.create(
        model="Test-Model",
        messages=[{"role": "user", "content": prompt}],
        stream=stream,
        extra_body={"seed": seed, "sampling_parameters": settings.SAMPLING_PARAMS, "task": "QuestionAnsweringTask"}
    )
    
    if not stream:
        return result
    else:
        chunks = []
        async for chunk in result:
            print(chunk)
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        return "".join(chunks)


async def main():
    PORT = 8004
    API_KEY = "YOUR_API_KEY_HERE"
    miner = setup_miner_client(
        port=PORT,
        api_key=API_KEY,
        hotkey=settings.WALLET.hotkey if hasattr(settings, 'WALLET') else None
    )
    response = await make_completion(
        miner=miner,
        prompt="Say 10 random numbers between 1 and 100",
        stream=False
    )
    print(response)


# Run the async main function
import asyncio
asyncio.run(main())