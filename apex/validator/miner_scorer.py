import asyncio
import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import aiosqlite
from loguru import logger

from apex.common.async_chain import AsyncChain
from apex.common.constants import VALIDATOR_REFERENCE_LABEL

# Scoring moving average in hours. Set to be: immunity_period - post_reg_threshold.
SCORE_MA_WINDOW_HOURS = 23.75
SCORE_INTERVAL_DEFAULT = 22 * 60


class MinerScorer:
    def __init__(self, chain: AsyncChain, interval: float = SCORE_INTERVAL_DEFAULT, debug: bool = False):
        self.chain = chain
        self.interval = interval
        self._running = True
        self._debug = debug
        self._debug_rewards_path = Path("debug_rewards.jsonl")

    async def start_loop(self) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(self.interval)
            logger.debug("Attempting to set weights")
            success = await self.set_scores()
            logger.log("INFO" if success else "ERROR", f"Set weights: {'success' if success else 'fail'}")

    async def shutdown(self) -> None:
        self._running = False

    @staticmethod
    @asynccontextmanager
    async def _db() -> AsyncGenerator[aiosqlite.Connection, None]:
        async with aiosqlite.connect("results.db") as conn:
            await conn.execute("PRAGMA foreign_keys = ON")
            yield conn

    async def set_scores(self) -> bool:
        """Set weights based on the current miner scores.

        Iterate over all rows in the discriminator_results table from the last SCORE_WINDOW_HOURS,
        expose each one as plain python objects so that downstream code can work with them,
        and remove rows that are older than the time window.
        """
        logger.debug("Retrieving miner's performance history")
        async with self._db() as conn:  # type: aiosqlite.Connection
            # Calculate the cutoff timestamp (current time - window hours).
            cutoff_timestamp = int(time.time() - SCORE_MA_WINDOW_HOURS * 3600)

            # 1. Fetch every row from the last SCORE_MA_WINDOW_HOURS.
            try:
                async with conn.execute(
                    """
                    SELECT generator_hotkey, generator_score, discriminator_hotkeys, discriminator_scores
                    FROM discriminator_results
                    WHERE timestamp >= ?
                    """,
                    (cutoff_timestamp,),
                ) as cursor:
                    rows = await cursor.fetchall()
            except BaseException as exc:
                logger.exception(f"Exception during DB fetch: {exc}")
                return False

            # 2. Iterate over the in-memory list so that the caller can process freely.
            logger.debug("Pre-processing miner's rewards")
            hkey_agg_rewards: dict[str, float] = {}
            for generator_hotkey, generator_score, disc_hotkeys_json, disc_scores_json in rows:
                # Deserialize JSON columns.
                disc_hotkeys = json.loads(disc_hotkeys_json)
                disc_scores = json.loads(disc_scores_json)

                # Create reward dictionary with generator and discriminator scores.
                reward_dict = dict(zip(disc_hotkeys, disc_scores, strict=False))

                if generator_hotkey != VALIDATOR_REFERENCE_LABEL:
                    # Skip validator generated references in score calculation.
                    reward_dict[generator_hotkey] = generator_score

                # Update the aggregate rewards.
                for hotkey, reward in reward_dict.items():
                    hkey_agg_rewards[hotkey] = float(hkey_agg_rewards.get(hotkey, 0.0)) + float(reward)

            # 3. Delete rows that are older than the time window.
            logger.debug("Cleaning up miner's outdated history")
            await conn.execute(
                "DELETE FROM discriminator_results WHERE timestamp < ?",
                (cutoff_timestamp,),
            )

            if self._debug:
                record: dict[str, str | dict[str, float]] = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "rewards": hkey_agg_rewards,
                }
                with self._debug_rewards_path.open("a+") as fh:
                    record_str: str = json.dumps(record)
                    fh.write(f"{record_str}\n")
            # TODO: Flush the db only on set_weights_result is True.
            logger.debug("Setting weights")
            set_weights_result = await self.chain.set_weights(hkey_agg_rewards)

            # 4. Flush all deletions in a single commit.
            logger.debug("Updating rewards DB")
            await conn.commit()
            return set_weights_result
