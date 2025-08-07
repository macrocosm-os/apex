import json
from typing import Any, Dict, List, Union

import bittensor as bt


def generate_header(
    hotkey: bt.wallet.hotkey, body: Union[Dict[Any, Any], List[Any]]
) -> Dict[str, Any]:
    return {"Body-Signature": "0x" + hotkey.sign(json.dumps(body)).hex()}
