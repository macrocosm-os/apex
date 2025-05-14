import random
from collections import defaultdict

import requests
from loguru import logger

from shared import settings
from shared.loop_runner import AsyncLoopRunner
from shared.uids import get_uids


class UpdateMinerAvailabilitiesForAPI(AsyncLoopRunner):
    interval: int = 120
    miner_availabilities: dict[int, dict] = {}
    _previous_availabilities: dict[str, dict[str, bool]] | None = None
    _previous_uids: list[int] | None = None

    def _fallback_availabilities(self, uids: list[int]) -> dict[str, dict[str, bool]]:
        return {
            str(uid): {
                "task_availabilities": defaultdict(lambda: True),
                "llm_model_availabilities": defaultdict(lambda: True),
            }
            for uid in uids
        }

    def _try_get_uids(self) -> list[int]:
        try:
            uids = get_uids(sampling_mode="all")
            self._previous_uids = uids
        except BaseException as e:
            logger.error(f"Error while getting miner UIDs from subtensor, using all UIDs: {e}")
            uids = self._previous_uids or settings.shared_settings.TEST_MINER_IDS or list(range(1024))
        return list(map(int, uids))

    async def run_step(self):
        logger.debug("Running update miner availabilities step")
        if settings.shared_settings.API_TEST_MODE:
            return
        uids = self._try_get_uids()
        try:
            response = requests.post(
                f"http://{settings.shared_settings.VALIDATOR_API}/miner_availabilities/miner_availabilities",
                headers={"accept": "application/json", "Content-Type": "application/json"},
                json=uids,
                timeout=15,
            )
            self.miner_availabilities = response.json()
        except Exception:
            # logger.error(f"Error while getting miner availabilities from validator API, fallback to all uids: {e}")
            self.miner_availabilities = self._fallback_availabilities(uids=uids)
        tracked_availabilities = [m for m in self.miner_availabilities.values() if m is not None]
        logger.info(f"Availabilities updated, tracked: {len(tracked_availabilities)}")


update_miner_availabilities_for_api = UpdateMinerAvailabilitiesForAPI()


def filter_available_uids(
    task: str | None = None,
    model: str | None = None,
    test: bool = False,
    n_miners: int = 10,
    n_top_incentive: int = 400,
    explore: float = 0.0,
) -> list[int]:
    """Filter UIDs based on task and model availability.

    Args:
        task (str | None, optional): The task to filter miners by. Defaults to None.
        model (str | None, optional): The LLM model to filter miners by. Defaults to None.
        test (bool, optional): Whether to run in test mode. Defaults to False.
        n_miners (int, optional): Number of miners to return. Defaults to 10.
        n_top_incentive (int, optional): Number of top incentivized miners to consider. Defaults to 10.

    Returns:
        list[int]: List of filtered UIDs that match the criteria.
    """
    if test:
        return get_uids(sampling_mode="top_incentive", k=n_miners)

    filtered_uids = []

    for uid in get_uids(sampling_mode="top_incentive", k=max(n_top_incentive, n_miners), explore=explore):
        # Skip if miner data is None/unavailable
        if update_miner_availabilities_for_api.miner_availabilities.get(str(uid)) is None:
            continue
        miner_data = update_miner_availabilities_for_api.miner_availabilities[str(uid)]

        # Check task availability if specified
        if task is not None:
            if not miner_data["task_availabilities"].get(task, False):
                continue

        # Check model availability if specified
        if model is not None:
            if not miner_data["llm_model_availabilities"].get(model, False):
                continue

        filtered_uids.append(uid)

    if len(filtered_uids) == 0:
        # logger.error(
        #     "Got an empty list of available UIDs, falling back to all uids. "
        #     "Check VALIDATOR_API and SCORING_KEY in .env.api"
        # )
        filtered_uids = get_uids(sampling_mode="top_incentive", k=n_top_incentive, explore=explore)

    # logger.info(f"Filtered UIDs: {filtered_uids}")
    filtered_uids = random.sample(filtered_uids, min(len(filtered_uids), n_miners))

    # logger.info(f"Filtered UIDs after sampling: {filtered_uids}")
    return filtered_uids
