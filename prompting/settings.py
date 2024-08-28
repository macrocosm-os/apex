import os
import torch
import dotenv
from loguru import logger
import bittensor as bt
from pydantic import BaseModel, model_validator, ConfigDict
from typing import Literal, Optional
from prompting.utils.config import config

# TODO: Remove in future as we deprecate config
bt_config = config()
logger.info(f"Config: {bt_config}")


class Settings(BaseModel):
    mode: Literal["miner", "validator"]
    MOCK: bool = False
    NO_BACKGROUND_THREAD: bool = True

    # WANDB
    WANDB_ON: bool = True
    WANDB_ENTITY: Optional[str] = None
    WANDB_PROJECT_NAME: Optional[str] = None
    WANDB_RUN_STEP_LENGTH: int = 100
    WANDB_API_KEY: Optional[str] = None
    WANDB_OFFLINE: bool = False
    WANDB_NOTES: str = ""
    SAVE_PATH: str | None = None

    # NEURON
    NEURON_EPOCH_LENGTH: int = 1
    NEURON_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NEURON_GPUS: int = 1

    # LOGGING
    LOGGING_DONT_SAVE_EVENTS: bool = False
    LOG_WEIGHTS: bool = False

    # NEURON PARAMETERS
    NEURON_TIMEOUT: int = 15
    NEURON_DISABLE_SET_WEIGHTS: bool = False
    NEURON_MOVING_AVERAGE_ALPHA: float = 0.1
    NEURON_DECAY_ALPHA: float = 0.001
    NEURON_AXON_OFF: bool = False
    NEURON_VPERMIT_TAO_LIMIT: int = 4096
    NEURON_QUERY_UNIQUE_COLDKEYS: bool = False
    NEURON_QUERY_UNIQUE_IPS: bool = False
    NEURON_FORWARD_MAX_TIME: int = 120
    NEURON_MAX_TOKENS: int = 256

    # ORGANIC
    ORGANIC_TIMEOUT: int = 15
    ORGANIC_SAMPLE_SIZE: int = 5
    ORGANIC_REUSE_RESPONSE_DISABLED: bool = False
    ORGANIC_REFERENCE_MAX_TOKENS: int = 1024
    ORGANIC_SYNTH_REWARD_SCALE: float = 1.0
    ORGANIC_SET_WEIGHTS_ENABLED: bool = True
    ORGANIC_DISABLED: bool = False
    ORGANIC_TRIGGER_FREQUENCY: int = 120
    ORGANIC_TRIGGER_FREQUENCY_MIN: int = 5
    ORGANIC_TRIGGER: str = "seconds"
    ORGANIC_SCALING_FACTOR: int = 1
    HF_TOKEN: Optional[str] = None

    # ADDITIONAL FIELDS FROM model_validator
    NETUID: int
    TEST: bool
    OPENAI_API_KEY: Optional[str] = None
    WALLET_NAME: Optional[str] = None
    HOTKEY: Optional[str] = None
    AXON_PORT: int
    ORGANIC_WHITELIST_HOTKEY: Optional[str] = None
    TEST_MINER_IDS: Optional[list[int]] = None
    SUBTENSOR_NETWORK: Optional[str] = None
    WALLET: bt.wallet
    SUBTENSOR: bt.subtensor
    METAGRAPH: bt.metagraph
    NEURON_LLM_MAX_ALLOWED_MEMORY_IN_GB: int
    LLM_MAX_MODEL_LEN: int
    NEURON_MODEL_ID_VALIDATOR: str

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)  # freeze all variables

    @model_validator(mode="before")
    def load_env(values):
        mode = values.get("mode")
        if mode == "miner":
            values["WANDB_ON"] = False
            if not dotenv.load_dotenv(dotenv.find_dotenv(filename=".env.miner")):
                logger.warning(
                    "No .env.miner file found. The use of args when running a miner will be deprecated in the near future."
                )
        else:
            if not dotenv.load_dotenv(dotenv.find_dotenv(filename=".env.validator")):
                logger.warning(
                    "No .env.validator file found. The use of args when running a validator will be deprecated in the near future."
                )

        bt_config = config()  # Re-fetch config as it might depend on .env values

        values["WANDB_ENTITY"] = os.environ.get("WANDB_ENTITY", "macrocosmos")
        values["WANDB_PROJECT_NAME"] = os.environ.get("WANDB_PROJECT_NAME", "prompting-validators")
        values["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY")

        values["NETUID"] = bt_config.netuid or int(os.environ.get("NETUID"))
        values["TEST"] = values["NETUID"] != 1
        values["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
        values["WALLET_NAME"] = bt_config.wallet.name or os.environ.get("WALLET_NAME")
        values["HOTKEY"] = bt_config.wallet.hotkey or os.environ.get("HOTKEY")
        values["NEURON_DISABLE_SET_WEIGHTS"] = os.environ.get("NEURON_DISABLE_SET_WEIGHTS", False)

        values["AXON_PORT"] = bt_config.axon.port or int(os.environ.get("AXON_PORT"))
        values["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        values["ORGANIC_WHITELIST_HOTKEY"] = os.environ.get(
            "ORGANIC_WHITELIST_HOTKEY",
            # OTF hotkey.
            "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3",
        )
        values["ORGANIC_TIMEOUT"] = os.environ.get("ORGANIC_TIMEOUT", 15)
        values["ORGANIC_SAMPLE_SIZE"] = os.environ.get("ORGANIC_SAMPLE_SIZE", 5)
        values["ORGANIC_REUSE_RESPONSE_DISABLED"] = os.environ.get("ORGANIC_REUSE_RESPONSE_DISABLED", False)
        values["ORGANIC_REFERENCE_MAX_TOKENS"] = os.environ.get("ORGANIC_REFERENCE_MAX_TOKENS", 1024)
        # TODO: Set to 0.1 when task-isolated rewards are implemented.
        values["ORGANIC_SYNTH_REWARD_SCALE"] = os.environ.get("ORGANIC_SYNTH_REWARD_SCALE", 1.0)
        values["ORGANIC_SET_WEIGHTS_ENABLED"] = os.environ.get("ORGANIC_SET_WEIGHTS_ENABLED", True)
        values["ORGANIC_DISABLED"] = os.environ.get("ORGANIC_DISABLED", False)
        values["ORGANIC_TRIGGER_FREQUENCY"] = os.environ.get("ORGANIC_TRIGGER_FREQUENCY", 120)
        values["ORGANIC_TRIGGER_FREQUENCY_MIN"] = os.environ.get("ORGANIC_TRIGGER_FREQUENCY_MIN", 5)
        values["ORGANIC_TRIGGER"] = os.environ.get("ORGANIC_TRIGGER", "seconds")
        values["ORGANIC_SCALING_FACTOR"] = os.environ.get("ORGANIC_SCALING_FACTOR", 1)

        values["LOG_WEIGHTS"] = os.environ.get("LOG_WEIGHTS", False)
        if values["TEST"] and os.environ.get("TEST_MINER_IDS"):
            values["TEST_MINER_IDS"] = [int(miner_id) for miner_id in os.environ.get("TEST_MINER_IDS").split(",")]
        values["NEURON_MODEL_ID_VALIDATOR"] = os.environ.get("LLM_MODEL", "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
        values["LLM_MAX_MODEL_LEN"] = int(os.environ.get("LLM_MAX_MODEL_LEN", 4096))
        values["NEURON_LLM_MAX_ALLOWED_MEMORY_IN_GB"] = os.environ.get("MAX_ALLOWED_VRAM_GB", 62)
        values["NEURON_GPUS"] = os.environ.get("NEURON_GPUS", 1)
        if int(values["NEURON_GPUS"]) > 1:
            # Set the start method to spawn for vllm
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        if os.environ.get("SUBTENSOR_NETWORK") == "local":
            values["SUBTENSOR_NETWORK"] = bt_config.subtensor.chain_endpoint or os.environ.get(
                "SUBTENSOR_CHAIN_ENDPOINT"
            )
        else:
            values["SUBTENSOR_NETWORK"] = bt_config.subtensor.network or os.environ.get("SUBTENSOR_NETWORK")

        logger.info(
            f"Instantiating bittensor objects with NETUID: {values['NETUID']}, WALLET_NAME: {values['WALLET_NAME']}, HOTKEY: {values['HOTKEY']}"
        )
        values["WALLET"] = bt.wallet(name=values["WALLET_NAME"], hotkey=values["HOTKEY"])
        values["SUBTENSOR"] = bt.subtensor(network=values["SUBTENSOR_NETWORK"])
        values["METAGRAPH"] = bt.metagraph(
            netuid=values["NETUID"], network=values["SUBTENSOR_NETWORK"], sync=True, lite=True
        )

        logger.info(
            f"Bittensor objects instantiated... WALLET: {values['WALLET']}, SUBTENSOR: {values['SUBTENSOR']}, METAGRAPH: {values['METAGRAPH']}"
        )
        values["SAVE_PATH"] = os.environ.get("SAVE_PATH") or "./storage"
        if not os.path.exists(values["SAVE_PATH"]):
            os.makedirs(values["SAVE_PATH"])

        return values


settings: Settings
