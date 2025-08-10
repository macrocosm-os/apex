import asyncio
import time
from typing import cast

import httpx
import netaddr
import requests
import uvicorn
from bittensor.core.async_subtensor import AsyncMetagraph
from bittensor.core.extrinsics.asyncex.serving import serve_extrinsic
from fastapi import APIRouter, FastAPI, HTTPException, Request
from loguru import logger
from pydantic import BaseModel

from apex.common.async_chain import AsyncChain
from apex.common.constants import VALIDATOR_VERIFIED_HOTKEYS
from apex.common.epistula import generate_header, verify_weight_signature


class ValidatorInfo(BaseModel):
    uid: int
    hotkey: str
    address: str
    stake: float


class WeightSyncer:
    REWARD_EXPIRATION_SEC: float = 60 * 60

    def __init__(
        self,
        chain: AsyncChain,
        min_alpha_stake: float = 100_000,
        verified_hotkeys: dict[str, str | None] | None = None,
        enable_receive: bool = True,
        enable_send: bool = True,
        port: int = 8001,
    ) -> None:
        """Validator weight synchronizer."""
        self.chain = chain
        self._min_alpha_stake = min_alpha_stake
        self.verified_hotkeys = verified_hotkeys or VALIDATOR_VERIFIED_HOTKEYS
        self.wallet = self.chain.wallet
        self.current_hotkey = self.wallet.hotkey.ss58_address
        self.receive_enabled = enable_receive
        self.send_enabled = enable_send
        self.port = port
        self.server: uvicorn.Server | None = None
        self.server_task: asyncio.Task[None] | None = None
        self.hotkey_rewards: dict[str, float] | None = None
        self._last_update_time: float = 0

    async def start(self) -> None:
        if not self.send_enabled:
            logger.warning("Weight synchronization API is disabled for incoming reward requests")
            return

        try:
            # Setup and run the API server in the background.
            app = FastAPI()
            app.include_router(self.get_router())

            # Running uvicorn in a background task.
            config = uvicorn.Config(app=app, host="0.0.0.0", port=self.port, log_level="info")
            self.server = uvicorn.Server(config)
            self.server_task = asyncio.create_task(self.server.serve())

            logger.info(f"Started weight synchronization API on port {self.port}.")

            # Announce the axon on the network.
            external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
            netaddr.IPAddress(external_ip)
            sub = await self.chain.subtensor()
            serve_success = await serve_extrinsic(
                subtensor=sub,
                wallet=self.chain.wallet,
                ip=external_ip,
                port=self.port,
                protocol=4,
                netuid=self.chain.netuid,
            )
            if serve_success:
                logger.success(f"Serving weight syncer axon on subtensor at {external_ip}:{self.port}")
            else:
                logger.error("Failed to serve weight syncer axon on subtensor")
        except BaseException as e:
            logger.warning(f"Failed to announce weight syncer axon on subtensor: {e}")

    async def shutdown(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.server_task is not None:
            await self.server_task

    def get_router(self) -> APIRouter:
        """Creates and returns a FastAPI router with the endpoints for this class."""
        router = APIRouter()

        @router.post("/v1/get_rewards")
        async def get_rewards_endpoint(request: Request) -> dict[str, float]:
            """FastAPI endpoint to get the rewards of this validator."""
            await verify_weight_signature(request=request, chain=self.chain)

            if time.time() - self._last_update_time > self.REWARD_EXPIRATION_SEC or self.hotkey_rewards is None:
                raise HTTPException(status_code=404, detail="Rewards not available or expired")
            if not self.send_enabled:
                raise HTTPException(status_code=404, detail="API is disabled")
            return self.hotkey_rewards

        return router

    async def compute_weighted_rewards(self, hotkey_rewards: dict[str, float]) -> dict[str, float]:
        """Computes weighted rewards by fetching rewards from other validators and averaging them by stake."""
        self.hotkey_rewards = hotkey_rewards
        self._last_update_time = time.time()
        if not self.receive_enabled:
            logger.warning("Rewards weight averaging is disable, using raw rewards")
            return hotkey_rewards

        metagraph = await self.chain.metagraph()

        try:
            own_uid = metagraph.hotkeys.index(self.current_hotkey)
        except ValueError:
            logger.error(f"Could not find own hotkey {self.current_hotkey} in metagraph, returning raw rewards")
            return hotkey_rewards

        validator_uids: list[int] = []
        for uid in metagraph.uids:
            if uid == own_uid:
                continue

            stake = metagraph.stake[uid]
            hotkey = metagraph.hotkeys[uid]
            is_verified = hotkey in self.verified_hotkeys
            is_validator = metagraph.validator_permit[uid]

            if stake >= self._min_alpha_stake or is_verified or is_validator:
                validator_uids.append(uid)

        validator_rewards_tasks: dict[int, asyncio.Task[dict[str, float]]] = {
            uid: asyncio.create_task(self.receive_rewards(metagraph, uid)) for uid in validator_uids
        }

        results = await asyncio.gather(*validator_rewards_tasks.values(), return_exceptions=True)

        validator_rewards: dict[int, dict[str, float]] = {}
        for uid, result in zip(validator_uids, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(f"Cannot receive rewards from uid {uid}: {result}")
                continue
            validator_rewards[uid] = result

        all_validator_uids = [own_uid] + list(validator_rewards.keys())
        total_stake = sum(metagraph.stake[uid] for uid in all_validator_uids)

        if total_stake == 0:
            logger.warning("Total stake of responding validators is zero, returning original rewards")
            return hotkey_rewards

        own_stake = metagraph.stake[own_uid]

        weighted_rewards: dict[str, float] = {}
        for miner_hkey in hotkey_rewards:
            own_reward = hotkey_rewards.get(miner_hkey, 0.0)
            total_weighted_reward = own_reward * own_stake

            for uid, rewards in validator_rewards.items():
                validator_reward = rewards.get(miner_hkey, 0.0)
                validator_stake = metagraph.stake[uid].item()
                total_weighted_reward += validator_reward * validator_stake

            weighted_rewards[miner_hkey] = total_weighted_reward / total_stake

        logger.debug(
            f"Averaged rewards over {len(all_validator_uids)} validators. "
            f"Self stake percentage: {100 * own_stake / total_stake:.2f}"
        )
        return weighted_rewards

    async def receive_rewards(self, metagraph: AsyncMetagraph, uid: int) -> dict[str, float]:
        """Receive rewards from the given validator uid."""
        try:
            target_hotkey = metagraph.hotkeys[uid]
            if (address := VALIDATOR_VERIFIED_HOTKEYS.get(target_hotkey)) is None:
                axon = metagraph.axons[uid]
                address = f"{axon.ip}:{axon.port}"
            async with httpx.AsyncClient() as client:
                headers = await generate_header(
                    self.chain.wallet.hotkey,
                    body=b"",
                    signed_for=target_hotkey,
                )
                resp = await client.post(
                    f"http://{address}/v1/get_rewards",
                    headers=headers,
                    content=b"",
                )
                resp.raise_for_status()
                return cast(dict[str, float], resp.json())

        except BaseException as exc:
            logger.warning(f"Cannot receive rewards from uid {uid}: {exc}")
        return {}
