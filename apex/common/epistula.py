import json
from typing import Any

import bittensor as bt


def generate_header(
    hotkey: bt.wallet.hotkey, body: dict[Any, Any] | list[Any]
) -> dict[str, Any]:
    return {"Body-Signature": "0x" + hotkey.sign(json.dumps(body)).hex()}
