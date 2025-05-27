import asyncio

import bittensor as bt
import httpx
import numpy as np
from loguru import logger

from shared.constants import WHITELISTED_VALIDATORS_UIDS
from shared.epistula import create_header_hook


class WeightSynchronizer:
    """The weight syncronizer is responsible for syncing the weights of the miners with the weight setter."""

    def __init__(self, metagraph: bt.Metagraph, wallet: bt.Wallet, weight_dict: dict[int, list[float]]):
        self.wallet = wallet
        self.uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

        self.weight_matrix = np.zeros((len(WHITELISTED_VALIDATORS_UIDS), 1024))
        self.stake_matrix = np.array([metagraph.S[uid] for uid in WHITELISTED_VALIDATORS_UIDS])

        self.validator_uids = np.array(WHITELISTED_VALIDATORS_UIDS)
        self.validator_hotkeys = np.array([metagraph.hotkeys[uid] for uid in WHITELISTED_VALIDATORS_UIDS])
        self.validator_addresses = np.array(
            [f"{metagraph.axons[uid].ip}:{metagraph.axons[uid].port}" for uid in WHITELISTED_VALIDATORS_UIDS]
        )

        self.weight_dict = weight_dict

        self.request_tracker = np.zeros(len(WHITELISTED_VALIDATORS_UIDS))

    async def make_epistula_request(self, weight_matrix: np.ndarray, validator_address: str, validator_hotkey: str):
        """Make an epistula request to the validator at the given address."""
        try:
            vali_url = f"http://{validator_address}/receive_weight_matrix"
            timeout = httpx.Timeout(timeout=40.0)
            async with httpx.AsyncClient(
                timeout=timeout,
                event_hooks={"request": [create_header_hook(self.wallet.hotkey, validator_hotkey)]},
            ) as client:
                response = await client.post(
                    url=vali_url,
                    json={"weights": weight_matrix.tolist(), "uid": self.uid},
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

        await self.process_weight_dict()

        return np.average(self.weight_matrix, axis=0, weights=self.stake_matrix * self.request_tracker)

    async def send_weight_matrixes(self, weight_matrix: np.ndarray):
        tasks = [
            self.make_epistula_request(weight_matrix, validator_address, validator_hotkey)
            for validator_address, validator_hotkey in zip(self.validator_addresses, self.validator_hotkeys)
        ]

        await asyncio.gather(*tasks)

    async def process_weight_dict(self):
        for uid, weights in self.weight_dict.items():
            # Verify uid is in whitelist
            validator_indices = np.where(self.validator_uids == uid)[0]
            if len(validator_indices) == 0:
                logger.error(f"Invalid validator UID {uid}, not in whitelist")
                continue

            # Update the weight matrix for this validator
            validator_idx = validator_indices[0]
            self.weight_matrix[validator_idx] = weights
            self.request_tracker[validator_idx] = 1
