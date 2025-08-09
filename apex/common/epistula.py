# import json
import time
from hashlib import sha256
from math import ceil
from typing import Any
from uuid import uuid4

# from fastapi import HTTPException, Request
# from loguru import logger
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


# def verify_signature(
#     signature: str, body: bytes, timestamp: int, uuid: str, signed_for: str, signed_by: str, now: float
# ) -> Annotated[str, "Error Message"] | None:
#     if not isinstance(signature, str):
#         return "Invalid Signature"
#     timestamp = int(timestamp)
#     if not isinstance(timestamp, int):
#         return "Invalid Timestamp"
#     if not isinstance(signed_by, str):
#         return "Invalid Sender key"
#     if not isinstance(signed_for, str):
#         return "Invalid receiver key"
#     if not isinstance(uuid, str):
#         return "Invalid uuid"
#     if not isinstance(body, bytes):
#         return "Body is not of type bytes"
#     allowed_delta_ms = 8000
#     keypair = Keypair(ss58_address=signed_by)
#     if timestamp + allowed_delta_ms < now:
#         return "Request is too stale"
#     message = f"{sha256(body).hexdigest()}.{uuid}.{timestamp}.{signed_for}"
#     verified = keypair.verify(message, signature)
#     if not verified:
#         return "Signature Mismatch"
#     return None
#
#
# async def verify_weight_signature(request: Request):
#     signed_by = request.headers.get("Epistula-Signed-By")
#     signed_for = request.headers.get("Epistula-Signed-For")
#     if not signed_by or not signed_for:
#         raise HTTPException(400, "Missing Epistula-Signed-* headers")
#
#     if signed_for != shared_settings.WALLET.hotkey.ss58_address:
#         logger.error("Bad Request, message is not intended for self")
#         raise HTTPException(status_code=400, detail="Bad Request, message is not intended for self")
#     validator_hotkeys = [shared_settings.METAGRAPH.hotkeys[uid] for uid in WHITELISTED_VALIDATORS_UIDS]
#     if signed_by not in validator_hotkeys:
#         logger.error(f"Signer not the expected ss58 address: {signed_by}")
#         raise HTTPException(status_code=401, detail="Signer not the expected ss58 address")
#
#     now = time.time()
#     body: bytes = await request.body()
#     try:
#         payload = json.loads(body)
#     except json.JSONDecodeError:
#         raise HTTPException(400, "Invalid JSON body")
#
#     if payload.get("uid") != get_uid_from_hotkey(signed_by):
#         raise HTTPException(400, "Invalid uid in body")
#
#     err = verify_signature(
#         request.headers.get("Epistula-Request-Signature"),
#         body,
#         request.headers.get("Epistula-Timestamp"),
#         request.headers.get("Epistula-Uuid"),
#         signed_for,
#         signed_by,
#         now,
#     )
#     if err:
#         logger.error(err)
#         raise HTTPException(status_code=400, detail=err)
