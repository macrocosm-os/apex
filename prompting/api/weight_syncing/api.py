import time

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from shared.constants import WHITELISTED_VALIDATORS_UIDS
from shared.epistula import verify_signature
from shared.settings import shared_settings

router = APIRouter()


def get_weight_dict(request: Request):
    return request.app.state.weight_dict


def get_uid_from_hotkey(hotkey: str):
    return shared_settings.METAGRAPH.hotkeys.index(hotkey)


async def verify_weight_signature(request: Request):
    signed_by = request.headers.get("Epistula-Signed-By")
    signed_for = request.headers.get("Epistula-Signed-For")
    if signed_for != shared_settings.WALLET.hotkey.ss58_address:
        logger.error("Bad Request, message is not intended for self")
        raise HTTPException(status_code=400, detail="Bad Request, message is not intended for self")
    validator_hotkeys = [shared_settings.METAGRAPH.hotkeys[uid] for uid in WHITELISTED_VALIDATORS_UIDS]
    if signed_by not in validator_hotkeys:
        logger.error("Signer not the expected ss58 address")
        raise HTTPException(status_code=401, detail="Signer not the expected ss58 address")
    now = time.time()
    body = await request.body()
    if body["uid"] != get_uid_from_hotkey(signed_by):
        logger.error("Invalid uid")
        raise HTTPException(status_code=400, detail="Invalid uid in body")
    err = verify_signature(
        request.headers.get("Epistula-Request-Signature"),
        body,
        request.headers.get("Epistula-Timestamp"),
        request.headers.get("Epistula-Uuid"),
        signed_for,
        signed_by,
        now,
    )
    if err:
        logger.error(err)
        raise HTTPException(status_code=400, detail=err)


@router.post("/receive_weight_matrix")
async def receive_weight_matrix(
    request: Request, verification_data: dict = Depends(verify_weight_signature), weight_dict=Depends(get_weight_dict)
):
    """Endpoint to receive weight matrix updates from validators."""
    await verify_weight_signature(request)

    body = await request.json()
    if not isinstance(body, dict) or "weights" not in body:
        raise HTTPException(status_code=400, detail="Invalid request body format")

    try:
        uid = body["uid"]
        weights = list(body["weights"])
        weight_dict[uid] = weights
        return {"status": "success", "message": "Weight matrix updated successfully"}
    except Exception as e:
        logger.error(f"Error processing weight matrix: {e}")
        raise HTTPException(status_code=500, detail="Error processing weight matrix")
