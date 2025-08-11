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
from apex.common.epistula import generate_header, verify_validator_signature


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
        self.min_alpha_stake = min_alpha_stake
        self.verified_hotkeys = verified_hotkeys or VALIDATOR_VERIFIED_HOTKEYS
        self.wallet = self.chain.wallet
        self.current_hotkey = self.wallet.hotkey.ss58_address
        self.receive_enabled = enable_receive
        self.send_enabled = enable_send
        self.port = int(port)
        self.server: uvicorn.Server | None = None
        self.server_task: asyncio.Task[None] | None = None
        self.hotkey_rewards: dict[str, float] | None = None
        self.last_update_time: float = 0

    async def start(self) -> None:
        if not self.send_enabled:
            logger.warning("Weight synchronization API is disabled for incoming reward requests")
            return

        try:
            app = FastAPI()
            app.include_router(self.get_router())

            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=self.port,
                log_level="info",
                workers=1,
                reload=False,
                loop="asyncio",
            )
            self.server = uvicorn.Server(config)
            self.server_task = asyncio.create_task(self.server.serve())
            logger.info(f"Started weight synchronization API on port {self.port}")

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
            await verify_validator_signature(request=request, chain=self.chain, min_stake=self.min_alpha_stake)

            outdated = time.time() - self.last_update_time
            if (outdated := time.time() - self.last_update_time) > self.REWARD_EXPIRATION_SEC:
                logger.warning(f"Rewards expired: {outdated:.2f}s - {self.last_update_time}")
                raise HTTPException(status_code=503, detail="Rewards expired")
            if self.hotkey_rewards is None:
                logger.warning("Rewards not available")
                raise HTTPException(status_code=503, detail="Rewards not available")
            if not self.send_enabled:
                logger.warning("API is disabled")
                raise HTTPException(status_code=405, detail="API is disabled")
            return self.hotkey_rewards

        return router

    async def compute_weighted_rewards(self, hotkey_rewards: dict[str, float]) -> dict[str, float]:
        """Computes weighted rewards by fetching rewards from other validators and averaging them by stake."""
        self.hotkey_rewards = hotkey_rewards
        self.last_update_time = time.time()
        if not self.receive_enabled:
            logger.warning("Rewards weight averaging is disable, using raw rewards")
            return hotkey_rewards

        metagraph = await self.chain.metagraph()

        try:
            own_uid = metagraph.hotkeys.index(self.current_hotkey)
        except ValueError:
            logger.error(f"Could not find own hotkey {self.current_hotkey} in metagraph, returning raw rewards")
            return hotkey_rewards

        validator_rewards_tasks: dict[int, asyncio.Task[dict[str, float]]] = {}
        for uid in metagraph.uids:
            if uid == own_uid:
                continue

            stake = metagraph.stake[uid]
            hotkey = metagraph.hotkeys[uid]
            is_verified = hotkey in self.verified_hotkeys
            is_validator = metagraph.validator_permit[uid]

            if (stake >= self.min_alpha_stake and is_validator) or is_verified:
                validator_rewards_tasks[uid] = asyncio.create_task(self.receive_rewards(metagraph, uid))

        results = await asyncio.gather(*validator_rewards_tasks.values(), return_exceptions=True)

        all_miner_hotkeys: set[str] = set()
        validator_rewards: dict[int, dict[str, float]] = {}
        for uid, result in zip(validator_rewards_tasks, results, strict=True):
            if isinstance(result, BaseException) or not result:
                logger.warning(f"Cannot receive rewards from uid {uid}: {result}")
                continue
            validator_rewards[uid] = result
            all_miner_hotkeys.update(result)
            logger.debug(f"Received rewards from validator {uid} with stake {metagraph.stake[uid]}")
        logger.debug(f"Total amount of unique hotkeys from all validators: {len(all_miner_hotkeys)}")

        all_validator_uids = [own_uid] + list(validator_rewards.keys())
        total_stake = sum(metagraph.stake[uid] for uid in all_validator_uids)

        if total_stake == 0:
            logger.warning("Total stake of responding validators is zero, returning original rewards")
            return hotkey_rewards

        own_stake = metagraph.stake[own_uid]

        weighted_rewards: dict[str, float] = {}
        for miner_hkey in all_miner_hotkeys:
            own_reward = hotkey_rewards.get(miner_hkey, 0.0)
            total_weighted_reward = own_reward * own_stake

            for uid, rewards in validator_rewards.items():
                validator_reward = rewards.get(miner_hkey, 0.0)
                total_weighted_reward += validator_reward * metagraph.stake[uid]

            weighted_rewards[miner_hkey] = total_weighted_reward / total_stake

        logger.debug(
            f"Averaged rewards over {len(all_validator_uids)} validators. "
            f"Self stake: {100 * own_stake / total_stake:.2f}%"
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
                body = b"{}"
                headers = await generate_header(
                    hotkey=self.chain.wallet.hotkey,
                    body=body,
                    signed_for=target_hotkey,
                )
                resp = await client.post(
                    f"http://{address}/v1/get_rewards",
                    headers=headers,
                    content=body,
                )
                resp.raise_for_status()
                return cast(dict[str, float], resp.json())

        except BaseException as exc:
            logger.warning(f"Cannot receive rewards from uid {uid}: {exc}")
        return {}
