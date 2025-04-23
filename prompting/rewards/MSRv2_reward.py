from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from pydantic import ConfigDict

from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from shared import settings
from shared.dendrite import DendriteResponseEvent
from shared.uids import get_uids

if TYPE_CHECKING:
    from prompting.tasks.MSRv2_task import MSRv2Task

shared_settings = settings.shared_settings


uids_to_sample = get_uids(sampling_mode="all")


class MSRv2RewardModel(BaseRewardModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def reward(
        self,
        reference: str,
        response_event: DendriteResponseEvent,
        task: "MSRv2Task",
        task_queue: list,
        **kwargs,
    ) -> BatchRewardOutput:
        completions: list[str] = response_event.completions

        if task.stage == "generative":
            if len(completions) > 1:
                logger.warning(f"Received {len(completions)} completions in generative stage, only using the first one")

            if completions:
                task.generative_miner_answer = completions[0] if completions[0] else "Miner did not return a response"
                task.generator_uid = response_event.uids[0]

            # Add task back to the task queue but now in the discriminative stage
            task_queue.append(task)

            logger.debug(f"Generate stage with answer: {task.generative_miner_answer} scored and re-appended")
            output = BatchRewardOutput(rewards=np.array([]), timings=np.array([]), threshold=None, uids=np.array([]))

            return output

        elif task.stage == "discriminative":
            discriminator_rewards = []
            for comp in completions:
                try:
                    # discriminator reward is (1-Squared Error)/N_Discriminators
                    comp_value = float(comp)
                    discriminator_rewards.append((1 - (task.ground_truth - comp_value) ** 2) / len(completions))
                except (ValueError, TypeError):
                    #logger.error(f"Error converting completion to float: {e}")
                    discriminator_rewards.append(0.0)  # Assign zero reward for invalid responses
            generator_reward = 1 - sum(discriminator_rewards)

            # If the answer was 'real' (hence no generator uid), we need to average the reward over all miners
            if task.generator_uid is None:
                assert task.ground_truth == 1, "If the answer was 'real', there should NOT be a generator uid"
                generator_uids = get_uids(sampling_mode="all", exclude=response_event.uids)
                generator_reward /= len(generator_uids)
            else:
                generator_uids = [task.generator_uid]

            logger.debug(
                f"Discriminative stage for task: {task.task_id} Generator rewards: {generator_reward} Discriminator rewards: {discriminator_rewards}, Ground truth: {task.ground_truth}"
            )
            return BatchRewardOutput(
                rewards=np.array([generator_reward] * len(generator_uids) + discriminator_rewards),
                timings=np.array([0] * (len(generator_uids) + len(discriminator_rewards))),
                threshold=None,
                uids=np.array(generator_uids + response_event.uids),
            )
        else:
            raise ValueError(f"Invalid task stage: {task.stage}")
