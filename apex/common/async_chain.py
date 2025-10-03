from typing import Self

import bittensor as bt
from bittensor.core.async_subtensor import AsyncSubtensor
from bittensor.core.metagraph import AsyncMetagraph
from loguru import logger

from apex import __spec_version__
from apex.common.utils import async_cache

_METAGRAPH_TTL: int = 10 * 60


class AsyncChain:
    def __init__(self, coldkey: str, hotkey: str, netuid: int, network: list[str] | str = "finney"):
        if isinstance(network, str):
            network = [network]
        self._network: list[str] = network
        self._coldkey = coldkey
        self._hotkey = hotkey
        self._netuid = netuid
        self._wallet = bt.Wallet(hotkey=self._hotkey, name=self._coldkey)

        self._subtensor_cm: list[AsyncSubtensor] = []
        self._subtensors: list[AsyncSubtensor] = []
        self._subtensor_alive: list[bool] = []
        self._metagraph: AsyncMetagraph | None = None

    async def start(self) -> Self:
        """Open the AsyncSubtensor connection(s).

        Raises:
            ValueError: Failed to initialize any subtensor.
        """
        if self._subtensors:
            return self

        for url in self._network:
            try:
                subtensor_cm = AsyncSubtensor(network=url)
                self._subtensor_cm.append(subtensor_cm)
                subtensor = await subtensor_cm.__aenter__()
                self._subtensors.append(subtensor)
                self._subtensor_alive.append(True)
            except BaseException as exc:
                logger.error(f"Failed to initialize subtensor: {exc}")

        if not self._subtensors:
            network_masked: list[str] = await self.mask_network()
            raise ValueError(f"Failed to initialize any subtensor using networks: {network_masked}")
        return self

    async def shutdown(self) -> None:
        """Close the AsyncSubtensor(s) if it's open."""
        if not self._subtensors:
            return

        for subtensor_cm in self._subtensor_cm:
            await subtensor_cm.__aexit__(None, None, None)

        del self._subtensor_cm[:]
        del self._subtensors[:]
        del self._subtensor_alive[:]

    @async_cache(_METAGRAPH_TTL)
    async def metagraph(self) -> AsyncMetagraph:
        """Retrieve metagraph.

        Raises:
            ValueError: Failed to retrieve any metagraph from all subtensor endpoints.
        """
        for idx, subtensor in enumerate(self._subtensors):
            try:
                meta = await subtensor.metagraph(self._netuid)
                self._metagraph = meta
                self._subtensor_alive[idx] = True
                return self._metagraph
            except BaseException as exc:
                self._subtensor_alive[idx] = False
                logger.error(f"Error during metagraph instantiation: {exc}")

        network_masked: list[str] = await self.mask_network()
        raise ValueError(f"Failed to retrieve any metagraph from all subtensor endpoints: {network_masked}")

    async def subtensor(self) -> AsyncSubtensor:
        for idx, subtensor in enumerate(self._subtensors):
            if self._subtensor_alive[idx]:
                return subtensor

        network_masked: list[str] = await self.mask_network()
        raise ValueError(f"Failed to get any subtensor using networks: {network_masked}")

    @property
    def hotkey(self) -> str:
        return self._hotkey

    @property
    def coldkey(self) -> str:
        return self._coldkey

    @property
    def wallet(self) -> bt.Wallet:
        return self._wallet

    @property
    def netuid(self) -> int:
        return self._netuid

    @property
    def network(self) -> list[str]:
        return self._network

    async def full_burn(self) -> bool:
        try:
            metagraph = await self.metagraph()
            subtensor = await self.subtensor()
            netuid = 1
            owner_hotkey = await subtensor.query_subtensor("SubnetOwnerHotkey", params=[netuid])
            owner_uid = await metagraph.hotkeys.index(owner_hotkey)
            logger.info(f"Burning to {owner_hotkey} hotkey, {owner_uid} UID")
            uids: list[int] = [owner_uid]
            weights: list[float] = [1.0]
            success, message = await subtensor.set_weights(
                self.wallet,
                netuid,
                uids,
                weights,
                version_key=__spec_version__,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            if not success:
                print(f"Failed to apply full burn: {message}")
            return bool(success)
        except Exception as exc:
            logger.exception(f"Error during full burn: {exc}")
            return False

    async def set_weights(self, rewards: dict[str, float]) -> bool:
        try:
            metagraph = await self.metagraph()
            subtensor = await self.subtensor()
            weights: dict[int, float] = dict.fromkeys(range(int(metagraph.n)), 0.0)

            for hotkey, reward in rewards.items():
                try:
                    idx = metagraph.hotkeys.index(hotkey)
                except ValueError:
                    # Hotkey not found in the metagraph (e.g., deregistered). Skip it.
                    continue

                uid = metagraph.uids[idx]
                weights[uid] = reward

            success, err = await subtensor.set_weights(
                wallet=self._wallet,
                netuid=self._netuid,
                uids=list(weights.keys()),
                weights=list(weights.values()),
                version_key=__spec_version__,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            if not success:
                logger.error(f"Error during weight set: {err}")
            return bool(success)
        except Exception as exc:
            logger.exception(f"Error during weight set: {exc}")
            return False

    async def mask_network(self) -> list[str]:
        """Mask local subtensors urls, used to reduct sensetive information in public logs."""
        network_masked: list[str] = []
        for network in self._network:
            if "ws" in network:
                network = f"{network[:6]}***{network[-3:]}"
            network_masked.append(network)
        return network_masked
