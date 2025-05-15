import asyncio
import time

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from shared.constants import WHITELISTED_VALIDATORS_UIDS
from shared.epistula import create_header_hook, verify_signature


class WeightSynchronizer:
    """The weight syncronizer is responsible for syncing the weights of the miners with the weight setter."""

    def __init__(self, metagraph: "bt.metagraph.Metagraph", wallet: "bt.wallet.Wallet"):
        self.wallet = wallet
        self.uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

        self.weight_matrix = np.zeros((len(WHITELISTED_VALIDATORS_UIDS), 1))
        self.stake_matrix = np.zeros((len(WHITELISTED_VALIDATORS_UIDS), 1))

        self.validator_uids = np.array(WHITELISTED_VALIDATORS_UIDS)
        self.validator_hotkeys = np.array([metagraph.hotkeys[uid] for uid in WHITELISTED_VALIDATORS_UIDS])
        self.validator_addresses = np.array(
            [f"{metagraph.axons[uid].ip}:{metagraph.axons[uid].port}" for uid in WHITELISTED_VALIDATORS_UIDS]
        )

        self.router = APIRouter()
        self.router.post("/receive_weights")(self.receive_weight_matrix)

    async def receive_weight_matrix(self, request: Request):
        """Endpoint to receive weight matrix updates from validators."""
        await self.verify_weight_signature(request)

        body = await request.json()

        if not isinstance(body, dict) or "weights" not in body:
            raise HTTPException(status_code=400, detail="Invalid request body format")

        try:
            uid = body["uid"]
            weights = np.array(body["weights"])
            if weights.shape != self.weight_matrix.shape:
                raise HTTPException(status_code=400, detail="Invalid weight matrix shape")

            # Update the weight matrix
            self.weight_matrix[uid] = weights
            return {"status": "success", "message": "Weight matrix updated successfully"}

        except Exception as e:
            logger.error(f"Error processing weight matrix: {e}")
            raise HTTPException(status_code=500, detail="Error processing weight matrix")

    async def verify_weight_signature(self, request: Request):
        signed_by = request.headers.get("Epistula-Signed-By")
        signed_for = request.headers.get("Epistula-Signed-For")
        if signed_for != self.wallet.hotkey.ss58_address:
            logger.error("Bad Request, message is not intended for self")
            raise HTTPException(status_code=400, detail="Bad Request, message is not intended for self")
        if signed_by not in self.validator_hotkeys:
            logger.error("Signer not the expected ss58 address")
            raise HTTPException(status_code=401, detail="Signer not the expected ss58 address")
        now = time.time()
        body = await request.body()
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

    async def make_epistula_request(self, weight_matrix: np.ndarray, validator_address: str, validator_hotkey: str):
        """Make an epistula request to the validator at the given address."""
        try:
            vali_url = f"http://{validator_address}/receive_weights"
            timeout = httpx.Timeout(timeout=120.0)
            async with httpx.AsyncClient(
                timeout=timeout,
                event_hooks={"request": [create_header_hook(self.wallet.hotkey, validator_hotkey)]},
            ) as client:
                response = await client.post(
                    url=vali_url,
                    content={"weights": weight_matrix, "uid": self.uid},
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code != 200:
                    raise Exception(
                        f"Status code {response.status_code} response for validator {validator_hotkey} - {vali_url}: "
                        f"{response.status_code} for uids {len(weight_matrix)}"
                    )
                logger.debug(f"Successfully forwarded response to uid {validator_hotkey} - {vali_url}")
        except httpx.ConnectError as e:
            logger.warning(
                f"Couldn't connect to validator {validator_hotkey} {vali_url} for weight setting. Exception: {e}"
            )
        except Exception as e:
            logger.warning(
                f"Error while forwarding weight matrix to validator {validator_hotkey} {vali_url}. Exception: {e}"
            )

    async def get_augmented_weights(self, weights: np.ndarray, uid: int) -> np.ndarray:
        """Get the augmented weights for the given uid, sends the weights to the validators."""
        await self.send_weight_matrixes(weights)

        self.weight_matrix[uid] = weights

        return np.average(self.weight_matrix, axis=0, weights=self.stake_matrix)

    async def send_weight_matrixes(self, weight_matrix: np.ndarray):
        tasks = [
            self.make_epistula_request(weight_matrix, validator_address, validator_hotkey)
            for validator_address, validator_hotkey in zip(self.validator_addresses, self.validator_hotkeys)
        ]
        await asyncio.gather(*tasks)
