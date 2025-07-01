import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import macrocosmos as mc
from loguru import logger
from pydantic import BaseModel, ConfigDict

import prompting
from prompting.rewards.reward import WeightedRewardEvent
from shared import settings
from shared.dendrite import DendriteResponseEvent
from shared.logging.serializer_registry import recursive_model_dump

MC_LOGGER = None


@dataclass
class Log:
    validator_model_id: str
    challenge: str
    challenge_prompt: str
    reference: str
    miners_ids: list[str]
    responses: list[str]
    miners_time: list[float]
    challenge_time: float
    reference_time: float
    rewards: list[float]
    task: dict


def export_logs(logs: list[Log]):
    logger.info("📝 Exporting logs...")

    # Create logs folder if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Get the current date and time for logging purposes
    date_string = datetime.now().strftime("%Y-%m-%d_%H:%M")

    all_logs_dict = [asdict(log) for log in logs]

    for logs in all_logs_dict:
        task_dict = logs.pop("task")
        prefixed_task_dict = {f"task_{k}": v for k, v in task_dict.items()}
        logs.update(prefixed_task_dict)

    log_file = f"./logs/{date_string}_output.json"
    with open(log_file, "w") as file:
        json.dump(all_logs_dict, file)

    return log_file


async def init_run():
    global MC_LOGGER
    mcl_client = mc.AsyncLoggerClient(app_name="my_app")
    MC_LOGGER = mcl_client.logger
    # Start a new run
    tags = [
        f"Version: {prompting.__version__}",
        f"Netuid: {settings.shared_settings.NETUID}",
    ]
    if settings.shared_settings.NEURON_DISABLE_SET_WEIGHTS:
        tags.append("Disable weights set")
    if settings.shared_settings.MOCK:
        tags.append("Mock")

    run = await MC_LOGGER.init(
        project=settings.shared_settings.LOGGING_PROJECT,
        tags=tags,
    )


class BaseEvent(BaseModel):
    forward_time: float | None = None


class WeightSetEvent(BaseEvent):
    weight_set_event: list[float]


class ErrorLoggingEvent(BaseEvent):
    error: str
    forward_time: float | None = None


class ValidatorLoggingEvent(BaseEvent):
    block: int
    step: int
    step_time: float
    response_event: DendriteResponseEvent
    task_id: str
    forward_time: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self):
        sample_completions = [completion for completion in self.response_event.completions if len(completion) > 0]
        forward_time = round(self.forward_time, 4) if self.forward_time else self.forward_time
        return f"""ValidatorLoggingEvent:
            Block: {self.block}
            Step: {self.step}
            Step time: {self.step_time:.4f}
            Forward time: {forward_time}
            Task id: {self.task_id}
            Number of total completions: {len(self.response_event.completions)}
            Number of non-empty completions: {len(sample_completions)}
            Sample 1 completion: {sample_completions[:1]}
        """


class RewardLoggingEvent(BaseEvent):
    block: int | None
    step: int
    response_event: DendriteResponseEvent
    reward_events: list[WeightedRewardEvent]
    task_id: str
    reference: str | None
    challenge: str | list[dict]
    task: str
    task_dict: dict
    source: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self):
        # Return everthing
        return f"""RewardLoggingEvent:
            block: {self.block}
            step: {self.step}
            response_event: {self.response_event}
            reward_events: {self.reward_events}
            task_id: {self.task_id}
            task: {self.task}
            task_dict: {self.task_dict}
            source: {self.source}
            reference: {self.reference}
            challenge: {self.challenge}
        """

    # Override the model_dump method to return a dictionary like the __str__ method
    def model_dump(self) -> dict:
        return {
            "block": self.block,
            "step": self.step,
            "response_event": self.response_event,
            "reward_events": self.reward_events,
            "task_id": self.task_id,
            "task": self.task,
            "task_dict": self.task_dict,
            "source": self.source,
            "reference": self.reference,
            "challenge": self.challenge,
        }


class MinerLoggingEvent(BaseEvent):
    epoch_time: float
    messages: int
    accumulated_chunks: int
    accumulated_chunks_timings: float
    validator_uid: int
    validator_ip: str
    validator_coldkey: str
    validator_hotkey: str
    validator_stake: float
    validator_trust: float
    validator_incentive: float
    validator_consensus: float
    validator_dividends: float
    model_config = ConfigDict(arbitrary_types_allowed=True)


async def log_event(event: BaseEvent):
    global MC_LOGGER

    if not settings.shared_settings.LOGGING_DONT_SAVE_EVENTS:
        logger.info(f"{event}")

    if settings.shared_settings.LOGGING_PROJECT == "":
        return

    if MC_LOGGER is None:
        await init_run()

    slim_event = {
        "task": event.task.__class__.__name__,
        "block": event.block,
        "reward_events": recursive_model_dump(event.reward_events[0]),
    }

    await MC_LOGGER.log(slim_event)
