import asyncio
import time

import bittensor as bt
import numpy as np
from loguru import logger
from bittensor_wallet import Wallet

from common import settings as common_settings
from common.models.api_models import SubnetScores
from validator.bt_utils import get_subtensor, get_wallet
from validator.validator_api_client import ValidatorAPIClient
from validator.validator_health import HealthServerMixin
from validator import settings as validator_settings


class Validator(HealthServerMixin):
    def __init__(self, wallet_name: str | None = None, wallet_hotkey: str | None = None, wallet: Wallet | None = None):
        super().__init__()
        self.wallet = wallet or get_wallet(
            wallet_name=wallet_name,
            wallet_hotkey=wallet_hotkey,
        )
        self.hotkey = self.wallet.hotkey.ss58_address
        self.available: bool = True

        # Circuit breaker state
        self._orchestrator_failure_count: int = 0
        self._last_orchestrator_failure_time: float = 0

        # Metrics
        self._tasks_failed: int = 0
        self._last_heartbeat: float = time.time()

        logger.info(
            f"Running Validator. Mock: {common_settings.MOCK}. Bittensor: {common_settings.BITTENSOR}. "
            f"Network: {common_settings.NETWORK}. Netuid: {common_settings.NETUID}"
        )
        self.subtensor = get_subtensor()
        self.metagraph = bt.metagraph(netuid=common_settings.NETUID, lite=False, network=common_settings.NETWORK)
        self.burn_factor: float = 0.0

        if common_settings.BITTENSOR:
            try:
                uid = (
                    self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address) if not common_settings.MOCK else None
                )
            except ValueError:
                logger.warning(
                    f"Hotkey {self.wallet.hotkey.ss58_address} not registered on Subnet {common_settings.NETUID} // network: {common_settings.NETWORK} // mock: {common_settings.MOCK}"
                )
        else:
            logger.info(f"Validator {self.hotkey[:8]} registered on metagraph")

    async def _validator_loop(self):
        """Main validator loop that handles registration and health checks."""
        logger.info(f"🔄 Starting validator loop for {self.hotkey[:8]}")

        while True:
            try:
                # Check orchestrator health before proceeding
                logger.debug("Checking orchestrator health")
                if not await self._check_orchestrator_health():
                    logger.warning(
                        f"⏳ Orchestrator health check failed for validator {self.hotkey[:8]}, "
                        f"sleeping for {validator_settings.ORCHESTRATOR_HEALTH_CHECK_INTERVAL} seconds"
                    )
                    await asyncio.sleep(validator_settings.ORCHESTRATOR_HEALTH_CHECK_INTERVAL)
                    continue

            except Exception as e:
                logger.exception(f"Error in validator main loop: {e}")

            finally:
                logger.info(
                    f"🔄 Validator loop sleeping for {validator_settings.ORCHESTRATOR_HEALTH_CHECK_INTERVAL} seconds"
                )
                await asyncio.sleep(validator_settings.ORCHESTRATOR_HEALTH_CHECK_INTERVAL)

    async def weight_loop(self):
        """Enhanced weight loop with better error handling and logging."""
        loop_count = 0
        logger.info(f"🔄 Starting weight loop for validator {self.hotkey[:8]}")

        while True:
            loop_count += 1
            try:
                logger.debug(f"Weight loop iteration {loop_count} starting")

                # Sync the metagraph to get the latest weights, must use lite=False to get the latest weights
                self.metagraph.sync(subtensor=self.subtensor, lite=False)

                logger.debug("VALIDATOR: WEIGHT LOOP RUNNING")
                if await ValidatorAPIClient.check_orchestrator_health(hotkey=self.wallet.hotkey):
                    logger.debug("VALIDATOR: GETTING GLOBAL MINER SCORES")
                    scores: dict | None = await ValidatorAPIClient.get_global_miner_scores(hotkey=self.wallet.hotkey)

                    if not scores or not isinstance(scores, dict):
                        raise Exception("No global weights received from orchestrator")

                    if "error_name" in scores:
                        logger.error(f"Error getting global weights: {scores['error_name']}")
                        scores = {}
                    else:
                        subnet_scores = SubnetScores.model_validate(scores)

                        logger.debug(f"VALIDATOR: GLOBAL MINER SCORES: {subnet_scores}")

                        # Safer type conversion
                        try:
                            weights = {int(m.uid): m.weight for m in subnet_scores.miner_scores}
                        except (ValueError, TypeError) as e:
                            logger.error(f"Invalid UID in global_weights: {e}")
                            weights = {}

                else:
                    logger.warning("Orchestrator is not healthy, skipping weight submission")
                    weights = {}

                # Submit global weights to Bittensor
                if len(weights) > 0:
                    logger.debug(f"Received global weights: {weights}")
                    await self.set_weights(weights=weights)
                else:
                    logger.warning("No global weights received, temporarily copying weights from the chain")
                    await self.set_weights(weights=self.copy_weights_from_chain())

                logger.debug(f"Weight loop iteration {loop_count} completed successfully")

            except TimeoutError as e:
                logger.error(f"TimeoutError in weight loop iteration {loop_count}: {e}")
                await self.set_weights(weights=self.copy_weights_from_chain())

            except Exception as e:
                logger.exception(f"Error in weight loop iteration {loop_count}: {e}")
                await self.set_weights(weights=self.copy_weights_from_chain())

            finally:
                logger.info(
                    f"💤 Weight submission loop sleeping for {validator_settings.WEIGHT_SUBMIT_INTERVAL} seconds 💤"
                )
                await asyncio.sleep(validator_settings.WEIGHT_SUBMIT_INTERVAL)

    async def run_validator(self):
        logger.info("🚀 Starting validator")
        # Start the healthcheck server
        if validator_settings.LAUNCH_HEALTH:
            await self._start_health_server()
            logger.info(f"🏥 Health server started for validator {self.hotkey[:8]}")
        else:
            logger.warning(
                "⚠️ Validator healthcheck API not configured in settings (VALIDATOR_HEALTH_PORT missing). Skipping."
            )

        # Task management state
        self._weight_task = None
        self._validator_task = None
        task_restart_count = {"weight_loop": 0, "validator_loop": 0}
        max_restarts = 10  # Prevent infinite restart loops
        restart_delay = 5  # Seconds to wait before restarting a failed task
        status_log_interval = 300  # Log status every 5 minutes
        last_status_log = 0

        # Main task monitoring loop
        while True:
            try:
                current_time = time.time()

                # Log task status periodically
                if current_time - last_status_log > status_log_interval:
                    self._log_task_status(
                        weight_task=self._weight_task,
                        validator_task=self._validator_task,
                        task_restart_count=task_restart_count,
                    )
                    last_status_log = current_time

                # Create tasks if they don't exist or have completed/failed
                if self._weight_task is None or self._weight_task.done():
                    if self._weight_task is not None and self._weight_task.done():
                        try:
                            # Check if the task completed with an exception
                            self._weight_task.result()
                            logger.info("Weight loop task completed normally")
                        except Exception as e:
                            logger.exception(f"❌ Weight loop task failed: {e}")
                            task_restart_count["weight_loop"] += 1

                            if task_restart_count["weight_loop"] >= max_restarts:
                                logger.critical(f"Weight loop has failed {max_restarts} times, giving up")
                                raise Exception(f"Weight loop exceeded maximum restart attempts ({max_restarts})")

                    logger.info(
                        f"🔄 Starting/restarting weight loop task (attempt {task_restart_count['weight_loop'] + 1})"
                    )
                    self._weight_task = asyncio.create_task(self.weight_loop())

                if self._validator_task is None or self._validator_task.done():
                    if self._validator_task is not None and self._validator_task.done():
                        try:
                            # Check if the task completed with an exception
                            self._validator_task.result()
                            logger.info("Validator loop task completed normally")
                        except Exception as e:
                            logger.exception(f"❌ Validator loop task failed: {e}")
                            task_restart_count["validator_loop"] += 1

                            if task_restart_count["validator_loop"] >= max_restarts:
                                logger.critical(f"Validator loop has failed {max_restarts} times, giving up")
                                raise Exception(f"Validator loop exceeded maximum restart attempts ({max_restarts})")

                    logger.info(
                        f"🔄 Starting/restarting validator loop task (attempt {task_restart_count['validator_loop'] + 1})"
                    )
                    self._validator_task = asyncio.create_task(self._validator_loop())

                # Wait for either task to complete (indicating failure since they run forever)
                logger.debug("🔍 Monitoring tasks for failures...")
                done, pending = await asyncio.wait(
                    [self._weight_task, self._validator_task], return_when=asyncio.FIRST_COMPLETED
                )

                # Log which task(s) completed
                for task in done:
                    if task == self._weight_task:
                        logger.warning("⚠️ Weight loop task completed unexpectedly")
                    elif task == self._validator_task:
                        logger.warning("⚠️ Validator loop task completed unexpectedly")

                # Wait a bit before restarting to prevent rapid restart loops
                if restart_delay > 0:
                    logger.info(f"⏳ Waiting {restart_delay} seconds before restarting failed tasks...")
                    await asyncio.sleep(restart_delay)

            except Exception as e:
                logger.exception(f"Critical error in validator task manager: {e}")

                # Cancel any running tasks before retrying
                if self._weight_task and not self._weight_task.done():
                    self._weight_task.cancel()
                    try:
                        await self._weight_task
                    except asyncio.CancelledError:
                        pass

                if self._validator_task and not self._validator_task.done():
                    self._validator_task.cancel()
                    try:
                        await self._validator_task
                    except asyncio.CancelledError:
                        pass

                # Reset tasks to None so they get recreated
                self._weight_task = None
                self._validator_task = None

                # Wait before retrying
                await asyncio.sleep(10)

    async def set_weights(self, weights: dict[int, float]):
        """
        Sets the validator weights to the metagraph hotkeys based on the global weights.
        """
        logger.info("Attempting to set weights to Bittensor.")
        if not common_settings.BITTENSOR:
            logger.warning("Bittensor is not enabled, skipping weight submission")
            return

        if not hasattr(self, "wallet") or not self.wallet:
            logger.warning("Wallet not initialized, skipping weight submission")
            return

        if not hasattr(self, "subtensor") or not self.subtensor:
            logger.warning("Subtensor not initialized, skipping weight submission")
            return

        if not hasattr(self, "metagraph") or not self.metagraph:
            logger.warning("Metagraph not initialized, skipping weight submission")
            return

        try:
            uids, scores = zip(*weights.items())
            uids = np.array(uids)
            scores = np.array(scores)

            # Check if scores contains any NaN values
            if np.isnan(scores).any():
                logger.warning("Scores contain NaN values. Replacing with 0.")
                scores = np.nan_to_num(scores, 0)

            # Check if we have any non-zero scores
            if np.sum(scores) == 0:
                logger.warning("All scores are zero, skipping weight submission")
                return

            (
                processed_weight_uids,
                processed_weights,
            ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids=uids,
                weights=scores,
                netuid=int(common_settings.NETUID),
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )

            # Log the weights being set
            weight_dict = dict(zip(processed_weight_uids.tolist(), processed_weights.tolist()))
            logger.info(f"Setting weights for {len(weight_dict)} miners")
            logger.debug(f"Weight details: {weight_dict}")

            # Submit weights to Bittensor chain
            success, response = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=int(common_settings.NETUID),
                uids=processed_weight_uids,
                weights=processed_weights,
                wait_for_finalization=False,
                version_key=common_settings.__SPEC_VERSION__,
            )

            if success:
                logger.success("Successfully submitted weights to Bittensor.")
                logger.debug(f"Response: {response}")
            else:
                logger.error("Failed to submit weights to Bittensor")
                logger.error(f"Response: {response}")

        except Exception as e:
            logger.exception(f"Error submitting weights to Bittensor: {e}")

    def copy_weights_from_chain(self) -> dict[int, float]:
        """Copy weights from the chain to the validator.

        Returns:
            dict[int, float]: A dictionary of weights for each miner.
        """
        # Sync the existing metagraph instead of creating a new one
        self.metagraph.sync(subtensor=self.subtensor, lite=False)

        valid_indices = np.where(self.metagraph.validator_permit)[0]
        valid_weights = self.metagraph.weights[valid_indices]
        valid_stakes = self.metagraph.stake[valid_indices]
        normalized_stakes = valid_stakes / np.sum(valid_stakes)
        stake_weighted_average = np.dot(normalized_stakes, valid_weights).astype(float).tolist()

        # This is for the special case of testnet.
        if len(self.metagraph.uids) == 0:
            logger.warning("No valid indices found in metagraph, returning empty weights")
            return {}

        return dict(zip(self.metagraph.uids, list(stake_weighted_average)))

    async def _check_orchestrator_health(self) -> bool:
        """
        Check if the orchestrator is healthy.
        """
        logger.info(f"🔄 Checking orchestrator health for validator {self.hotkey[:8]}")
        current_time = time.time()

        try:
            is_healthy = await ValidatorAPIClient.check_orchestrator_health(hotkey=self.wallet.hotkey)

            if is_healthy:
                logger.success(f"✅ Orchestrator health check passed for validator {self.hotkey[:8]}")
                return True
            else:
                self._orchestrator_failure_count += 1
                self._last_orchestrator_failure_time = current_time
                return False

        except Exception as e:
            logger.warning(f"Orchestrator health check failed: {e}")
            self._orchestrator_failure_count += 1
            self._last_orchestrator_failure_time = current_time
            return False

    async def get_validator_status(self) -> dict:
        """Get current validator status for monitoring, including task states."""
        status = {
            "hotkey": self.hotkey[:8] if hasattr(self, "hotkey") else "N/A",
            "available": self.available,
            "orchestrator_failure_count": self._orchestrator_failure_count,
            "last_heartbeat": self._last_heartbeat,
            "uptime": time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 0,
        }

        # Add task status if available
        if hasattr(self, "_weight_task") and self._weight_task:
            status["weight_task_running"] = not self._weight_task.done()
            status["weight_task_cancelled"] = self._weight_task.cancelled()
        else:
            status["weight_task_running"] = False

        if hasattr(self, "_validator_task") and self._validator_task:
            status["validator_task_running"] = not self._validator_task.done()
            status["validator_task_cancelled"] = self._validator_task.cancelled()
        else:
            status["validator_task_running"] = False

        return status

    def _log_task_status(self, weight_task: asyncio.Task, validator_task: asyncio.Task, task_restart_count: dict):
        """
        Log the current status of both tasks for debugging.
        """
        weight_status = "None"
        if weight_task:
            if weight_task.done():
                weight_status = "Done/Failed"
            elif weight_task.cancelled():
                weight_status = "Cancelled"
            else:
                weight_status = "Running"

        validator_status = "None"
        if validator_task:
            if validator_task.done():
                validator_status = "Done/Failed"
            elif validator_task.cancelled():
                validator_status = "Cancelled"
            else:
                validator_status = "Running"

        logger.info(
            f"📊 Task Status - Weight: {weight_status} (restarts: {task_restart_count['weight_loop']}), "
            f"Validator: {validator_status} (restarts: {task_restart_count['validator_loop']})"
        )


if __name__ == "__main__":
    gradient_validator = Validator(
        wallet_name=validator_settings.WALLET_NAME, wallet_hotkey=validator_settings.WALLET_HOTKEY
    )
    asyncio.run(gradient_validator.run_validator())
