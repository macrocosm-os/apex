from shared import settings

settings.shared_settings = settings.SharedSettings.load(mode="validator")
shared_settings = settings.shared_settings

import asyncio
import json

from loguru import logger

from shared.epistula import query_miners
from shared.settings import shared_settings

"""
This has assumed you have:
1. Registered your miner on the chain (finney/test)
2. Are serving your miner on an open port (e.g. 12345)

Steps:
- Instantiate your synapse subclass with the relevant information. E.g. messages, roles, etc.
- Instantiate your wallet and a dendrite client
- Query the dendrite client with your synapse object
- Iterate over the async generator to extract the yielded tokens on the server side
"""

TEST_MINER_IDS = [203]


async def query_and_print():
    body = {
        "seed": 0,
        "sampling_parameters": shared_settings.SAMPLING_PARAMS,
        "task": "inference",
        "model": "casperhansen/llama-3-8b-instruct-awq",
        "messages": [
            {"role": "user", "content": "what is the meaning of life?"},
        ],
    }
    res = await query_miners(TEST_MINER_IDS, json.dumps(body).encode("utf-8"))
    for token in res:
        logger.info(token)


if __name__ == "__main__":
    asyncio.run(query_and_print())
