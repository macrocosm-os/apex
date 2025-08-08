import time
from hashlib import sha256
from math import ceil
from typing import Any
from uuid import uuid4

from substrateinterface import Keypair


async def generate_header(
    hotkey: Keypair,
    body: bytes,
    signed_for: str | None = None,
) -> dict[str, Any]:
    timestamp = round(time.time() * 1000)
    timestamp_interval = ceil(timestamp / 1e4) * 1e4
    uuid = str(uuid4())
    headers = {
        "Epistula-Version": "2",
        "Epistula-Timestamp": str(timestamp),
        "Epistula-Uuid": uuid,
        "Epistula-Signed-By": hotkey.ss58_address,
        "Epistula-Request-Signature": "0x"
        + hotkey.sign(f"{sha256(body).hexdigest()}.{uuid}.{timestamp}.{signed_for or ''}").hex(),
    }
    if signed_for:
        headers["Epistula-Signed-For"] = signed_for
        headers["Epistula-Secret-Signature-0"] = (
            "0x" + hotkey.sign(str(timestamp_interval - 1) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-1"] = "0x" + hotkey.sign(str(timestamp_interval) + "." + signed_for).hex()
        headers["Epistula-Secret-Signature-2"] = (
            "0x" + hotkey.sign(str(timestamp_interval + 1) + "." + signed_for).hex()
        )
    return headers
