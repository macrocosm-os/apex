import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Literal

import wandb
from loguru import logger
from pydantic import BaseModel, ConfigDict
from wandb.wandb_run import Run

import prompting
from prompting.rewards.reward import WeightedRewardEvent
from prompting.tasks.task_registry import TaskRegistry
from shared import settings
from shared.dendrite import DendriteResponseEvent
from shared.logging.serializer_registry import recursive_model_dump

# TODO: Get rid of global variables.
WANDB: Run | None = None


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


def should_reinit_wandb():
    """Checks if 24 hours have passed since the last wandb initialization."""
    # Get the start time from the wandb config
    if wandb.run is None:
        return False
    wandb_start_time = wandb.run.config.get("wandb_start_time", None)

    if wandb_start_time:
        # Convert the stored time (string) back to a datetime object
        wandb_start_time = datetime.strptime(wandb_start_time, "%Y-%m-%d %H:%M:%S")
        current_time = datetime.now()
        elapsed_time = current_time - wandb_start_time
        # Check if more than 24 hours have passed
        if elapsed_time > timedelta(hours=settings.shared_settings.MAX_WANDB_DURATION):
            return True
    return False


def init_wandb(reinit=False, neuron: Literal["validator", "miner", "api"] = "validator", custom_tags: list = []):
    """Starts a new wandb run."""
    # global WANDB
    tags = [
        f"Wallet: {settings.shared_settings.WALLET.hotkey.ss58_address}",
        f"Version: {prompting.__version__}",
        f"Netuid: {settings.shared_settings.NETUID}",
    ]

    if settings.shared_settings.MOCK:
        tags.append("Mock")
    if settings.shared_settings.NEURON_DISABLE_SET_WEIGHTS:
        tags.append("Disable weights set")

    tags += custom_tags

    task_list = []
    for task_config in TaskRegistry.task_configs:
        task_list.append(task_config.task.__name__)

    wandb_config = {
        "HOTKEY_SS58": settings.shared_settings.WALLET.hotkey.ss58_address,
        "NETUID": settings.shared_settings.NETUID,
        "wandb_start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "TASKS": task_list,
    }
    wandb.login(anonymous="allow", key=settings.shared_settings.WANDB_API_KEY, verify=True)
    logger.info(
        f"Logging in to wandb on entity: {settings.shared_settings.WANDB_ENTITY} and project: "
        f"{settings.shared_settings.WANDB_PROJECT_NAME}"
    )
    wandb_run_name = f"{neuron}{settings.shared_settings.UID}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize the wandb run with the custom name.
    wandb_obj = wandb.init(
        reinit=reinit,
        name=wandb_run_name,
        project=settings.shared_settings.WANDB_PROJECT_NAME,
        entity=settings.shared_settings.WANDB_ENTITY,
        mode="offline" if settings.shared_settings.WANDB_OFFLINE else "online",
        tags=tags,
        notes=settings.shared_settings.WANDB_NOTES,
        config=wandb_config,
    )
    signature = settings.shared_settings.WALLET.hotkey.sign(wandb_obj.id.encode()).hex()
    wandb_config["SIGNATURE"] = signature
    wandb_obj.config.update(wandb_config)
    logger.success(f"Started a new wandb run <blue> {wandb_obj.name} </blue>")


def reinit_wandb():
    """Reinitializes wandb, rolling over the run."""
    wandb.finish()
    init_wandb(reinit=True)


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

    model_config = ConfigDict(arbitrary_types_allowed=True, copy_on_model_validation=False)

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
    reference: str
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


def log_event(event: BaseEvent):
    if not settings.shared_settings.LOGGING_DONT_SAVE_EVENTS:
        logger.info(f"{event}")

    if settings.shared_settings.WANDB_ON:
        if should_reinit_wandb():
            reinit_wandb()
        unpacked_event = recursive_model_dump(event)
        try:
            wandb.log(unpacked_event)
        except BaseException as e:
            logger.error(f"Error during wandb log {e}: {unpacked_event}")
