from loguru import logger

from prompting.rewards.exact_match import LogitsRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from prompting.tasks.base_task import BaseTextTask
from shared.dendrite import DendriteResponseEvent


class InferenceRewardModel(BaseRewardModel):
    async def reward(
        self,
        reference: str,
        response_event: DendriteResponseEvent,
        model_id: str | None = None,
        task: BaseTextTask | None = None,
        **kwargs,
    ) -> BatchRewardOutput:
        """Gives an exact reward of 1 if the response matches the reference, 0 otherwise"""
        logger.info(f"model_id: {model_id}")

        if model_id or task.organic:
            logger.info("Using logits reward model")
            logits_reward_model = LogitsRewardModel()
            return await logits_reward_model.reward(reference, response_event, task)

        relevance_reward_model = RelevanceRewardModel()
        return await relevance_reward_model.reward(reference, response_event)
