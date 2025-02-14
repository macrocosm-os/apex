import sys

# Need to delete logging from modules and load in standard python logging
if "logging" in sys.modules:
    del sys.modules["logging"]
current_directory = sys.path.pop(0)
import logging

sys.path.insert(0, current_directory)

import os
from functools import cached_property
from typing import Any, Literal, Optional

import bittensor as bt
import dotenv
from loguru import logger
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from shared.misc import cached_property_with_expiration

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class SharedSettings(BaseSettings):
    _instance: Optional["SharedSettings"] = None
    _instance_mode: Optional[str] = None

    mode: Literal["api", "validator", "miner", "mock"] = Field("validator", env="MODE")
    MOCK: bool = False
    NO_BACKGROUND_THREAD: bool = True
    SAVE_PATH: Optional[str] = Field("./storage", env="SAVE_PATH")

    # W&B.
    WANDB_ON: bool = Field(True, env="WANDB_ON")
    WANDB_ENTITY: Optional[str] = Field("macrocosmos", env="WANDB_ENTITY")
    WANDB_PROJECT_NAME: Optional[str] = Field("prompting-validators", env="WANDB_PROJECT_NAME")
    WANDB_RUN_STEP_LENGTH: int = Field(100, env="WANDB_RUN_STEP_LENGTH")
    WANDB_API_KEY: Optional[str] = Field(None, env="WANDB_API_KEY")
    WANDB_OFFLINE: bool = Field(False, env="WANDB_OFFLINE")
    WANDB_NOTES: str = Field("", env="WANDB_NOTES")
    MAX_WANDB_DURATION: int = 24

    # Neuron.
    NEURON_EPOCH_LENGTH: int = Field(100, env="NEURON_EPOCH_LENGTH")
    NEURON_DEVICE: str | None = Field("cuda", env="NEURON_DEVICE")
    NEURON_GPUS: int = Field(1, env="NEURON_GPUS")

    # Logging.
    LOGGING_DONT_SAVE_EVENTS: bool = Field(True, env="LOGGING_DONT_SAVE_EVENTS")
    LOG_WEIGHTS: bool = Field(False, env="LOG_WEIGHTS")

    # Neuron parameters.
    NEURON_TIMEOUT: int = Field(20, env="NEURON_TIMEOUT")
    INFERENCE_TIMEOUT: int = Field(60, env="INFERENCE_TIMEOUT")
    NEURON_DISABLE_SET_WEIGHTS: bool = Field(False, env="NEURON_DISABLE_SET_WEIGHTS")
    NEURON_MOVING_AVERAGE_ALPHA: float = Field(0.1, env="NEURON_MOVING_AVERAGE_ALPHA")
    NEURON_DECAY_ALPHA: float = Field(0.001, env="NEURON_DECAY_ALPHA")
    NEURON_AXON_OFF: bool = Field(False, env="NEURON_AXON_OFF")
    NEURON_VPERMIT_TAO_LIMIT: int = Field(4096, env="NEURON_VPERMIT_TAO_LIMIT")
    NEURON_QUERY_UNIQUE_COLDKEYS: bool = Field(False, env="NEURON_QUERY_UNIQUE_COLDKEYS")
    NEURON_QUERY_UNIQUE_IPS: bool = Field(False, env="NEURON_QUERY_UNIQUE_IPS")
    NEURON_FORWARD_MAX_TIME: int = Field(240, env="NEURON_FORWARD_MAX_TIME")
    NEURON_MAX_TOKENS: int = Field(512, env="NEURON_MAX_TOKENS")
    REWARD_STEEPNESS: float = Field(0.7, env="STEEPNESS")

    # Organic.
    ORGANIC_TIMEOUT: int = Field(30, env="ORGANIC_TIMEOUT")
    ORGANIC_SAMPLE_SIZE: int = Field(5, env="ORGANIC_SAMPLE_SIZE")
    ORGANIC_REUSE_RESPONSE_DISABLED: bool = Field(False, env="ORGANIC_REUSE_RESPONSE_DISABLED")
    ORGANIC_REFERENCE_MAX_TOKENS: int = Field(1024, env="ORGANIC_REFERENCE_MAX_TOKENS")
    ORGANIC_SYNTH_REWARD_SCALE: float = Field(1.0, env="ORGANIC_SYNTH_REWARD_SCALE")
    ORGANIC_SET_WEIGHTS_ENABLED: bool = Field(True, env="ORGANIC_SET_WEIGHTS_ENABLED")
    ORGANIC_DISABLED: bool = Field(False, env="ORGANIC_DISABLED")
    ORGANIC_TRIGGER_FREQUENCY: int = Field(120, env="ORGANIC_TRIGGER_FREQUENCY")
    ORGANIC_TRIGGER_FREQUENCY_MIN: int = Field(5, env="ORGANIC_TRIGGER_FREQUENCY_MIN")
    ORGANIC_TRIGGER: str = Field("seconds", env="ORGANIC_TRIGGER")
    ORGANIC_SCALING_FACTOR: int = Field(1, env="ORGANIC_SCALING_FACTOR")
    TASK_QUEUE_LENGTH_THRESHOLD: int = Field(10, env="TASK_QUEUE_LENGTH_THRESHOLD")
    SCORING_QUEUE_LENGTH_THRESHOLD: int = Field(10, env="SCORING_QUEUE_LENGTH_THRESHOLD")
    HF_TOKEN: Optional[str] = Field(None, env="HF_TOKEN")
    DEPLOY_VALIDATOR: bool = Field(True, env="DEPLOY_VALDITAOR")

    # ==== API =====
    # API key used to access validator organic scoring mechanism (both .env.validator and .env.api).
    SCORING_KEY: str | None = Field(None, env="SCORING_KEY")
    # Scoring request rate limit in seconds.
    SCORING_RATE_LIMIT_SEC: float = Field(0.5, env="SCORING_RATE_LIMIT_SEC")
    # Scoring queue threshold when rate-limit start to kick in, used to query validator API with scoring requests.
    SCORING_QUEUE_API_THRESHOLD: int = Field(5, env="SCORING_QUEUE_API_THRESHOLD")
    API_TEST_MODE: bool = Field(False, env="API_TEST_MODE")

    # Validator scoring API (.env.validator).
    DEPLOY_SCORING_API: bool = Field(False, env="DEPLOY_SCORING_API")
    SCORING_API_PORT: int = Field(8094, env="SCORING_API_PORT")
    SCORING_ADMIN_KEY: str | None = Field(None, env="SCORING_ADMIN_KEY")
    SCORE_ORGANICS: bool = Field(False, env="SCORE_ORGANICS")
    WORKERS: int = Field(2, env="WORKERS")

    # API Management (.env.api).
    API_PORT: int = Field(8005, env="API_PORT")
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    # Validator scoring API address.
    VALIDATOR_API: str = Field("0.0.0.0:8094", env="VALIDATOR_API")
    # Default SN1 API address
    DEFAULT_SN1_API: str = Field("http://sn1.api.macrocosmos.ai:11198/v1", env="DEFAULT_SN1_API")
    # File with keys used to access API.
    API_KEYS_FILE: str = Field("api_keys.json", env="API_KEYS_FILE")
    # Admin key used to generate API keys.
    ADMIN_KEY: str | None = Field(None, env="ADMIN_KEY")
    # ==============

    # Additional Validator Fields.
    NETUID: Optional[int] = Field(61, env="NETUID")
    TEST: bool = False
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    WALLET_NAME: Optional[str] = Field(None, env="WALLET_NAME")
    HOTKEY: Optional[str] = Field(None, env="HOTKEY")
    AXON_PORT: Optional[int] = Field(None, env="AXON_PORT")
    ORGANIC_WHITELIST_HOTKEY: Optional[str] = Field(
        "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3", env="ORGANIC_WHITELIST_HOTKEY"
    )
    TEST_MINER_IDS: list[int] = Field([], env="TEST_MINER_IDS")
    SUBTENSOR_NETWORK: Optional[str] = Field(None, env="SUBTENSOR_NETWORK")
    MAX_ALLOWED_VRAM_GB: int = Field(62, env="MAX_ALLOWED_VRAM_GB")
    LLM_MAX_MODEL_LEN: int = Field(4096, env="LLM_MAX_MODEL_LEN")
    PROXY_URL: Optional[str] = Field(None, env="PROXY_URL")
    LLM_MODEL: str = Field("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", env="LLM_MODEL")
    SAMPLING_PARAMS: dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": 512,
        "do_sample": True,
    }
    MINER_LLM_MODEL: Optional[str] = Field(None, env="MINER_LLM_MODEL")
    LLM_MODEL_RAM: float = Field(70, env="LLM_MODEL_RAM")
    OPENAI_API_KEY: str | None = Field(None, env="OPENAI_API_KEY")
    SN19_API_KEY: str | None = Field(None, env="SN19_API_KEY")
    SN19_API_URL: str | None = Field(None, env="SN19_API_URL")
    GPT_MODEL_CONFIG: dict[str, dict[str, Any]] = {
        "gpt-3.5-turbo": {
            "context_window": 16_385,
            "max_tokens": 4096,
            "vision": False,
            "score": 100,
            "upgrade": "gpt-4-turbo",
            "input_token_cost": 0.0005,
            "output_token_cost": 0.0015,
        },
        "gpt-4-turbo": {
            "context_window": 128_000,
            "max_tokens": 4096,
            "vision": True,
            "score": 200,
            "upgrade": "gpt-4o",
            "input_token_cost": 0.01,
            "output_token_cost": 0.03,
        },
        "gpt-4o": {
            "context_window": 128_000,
            "max_tokens": 4096,
            "vision": True,
            "score": 300,
            "input_token_cost": 0.005,
            "output_token_cost": 0.015,
        },
    }
    model_config = {"frozen": True, "arbitrary_types_allowed": False}

    @model_validator(mode="before")
    def validate_mode(cls, v):
        if v["mode"] == "api":
            if not dotenv.load_dotenv(".env.api"):
                logger.warning("No .env.api file found. Please create one.")
            if not os.getenv("SCORING_KEY"):
                logger.warning(
                    "No SCORING_KEY found in .env.api file. You must add a scoring key that will allow us to forward miner responses to the validator for scoring."
                )
            if not os.getenv("SCORE_ORGANICS"):
                logger.warning("Not scoring organics. This means that miners may not respond as consistently.")
        elif v["mode"] == "miner":
            if not dotenv.load_dotenv(".env.miner"):
                logger.warning("No .env.miner file found. Please create one.")
        elif v["mode"] == "validator":
            if not dotenv.load_dotenv(".env.validator"):
                logger.warning("No .env.validator file found. Please create one.")

        return v

    @classmethod
    def load_env_file(cls, mode: Literal["miner", "validator", "mock", "api"]):
        """Load the appropriate .env file based on the mode."""
        if mode == "miner":
            dotenv_file = ".env.miner"
        elif mode == "validator":
            dotenv_file = ".env.validator"
        # For mock testing, still make validator env vars available where possible.
        elif mode == "mock":
            dotenv_file = ".env.validator"
        elif mode == "api":
            dotenv_file = ".env.api"
            if os.getenv("API_TEST_MODE"):
                logger.warning("API_TEST_MODE is set to true - THE API IS RUNNING IN TEST MODE.")
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if dotenv_file:
            if not dotenv.load_dotenv(dotenv.find_dotenv(filename=dotenv_file)):
                logger.warning(
                    f"No {dotenv_file} file found. The use of args when running a {mode} will be deprecated "
                    "in the near future."
                )

    @classmethod
    def load(cls, mode: Literal["miner", "validator", "mock", "api"]) -> "SharedSettings":
        """Load or retrieve the Settings instance based on the mode."""
        if cls._instance is not None and cls._instance_mode == mode:
            return cls._instance
        else:
            cls.load_env_file(mode)
            cls._instance = cls(mode=mode)
            cls._instance_mode = mode
            return cls._instance

    @model_validator(mode="before")
    def complete_settings(cls, values: dict[str, Any]) -> dict[str, Any]:
        mode = values["mode"]
        netuid = values.get("NETUID", 61)
        if netuid is None:
            raise ValueError("NETUID must be specified")
        values["TEST"] = netuid != 1
        if values.get("TEST_MINER_IDS"):
            values["TEST_MINER_IDS"] = str(values["TEST_MINER_IDS"]).split(",")
        if mode == "mock":
            values["MOCK"] = True
            values["NEURON_DEVICE"] = "cpu"
            logger.info("Running in mock mode. Bittensor objects will not be initialized.")
            return values

        return values

    @cached_property
    def WALLET(self):
        wallet_name = self.WALLET_NAME  # or config().wallet.name
        hotkey = self.HOTKEY  # or config().wallet.hotkey
        logger.info(f"Instantiating wallet with name: {wallet_name}, hotkey: {hotkey}")
        return bt.wallet(name=wallet_name, hotkey=hotkey)

    @cached_property
    def SUBTENSOR(self) -> bt.subtensor:
        subtensor_network = self.SUBTENSOR_NETWORK or os.environ.get("SUBTENSOR_NETWORK", "local")
        # bt_config = config()
        if subtensor_network.lower() == "local":
            subtensor_network = os.environ.get("SUBTENSOR_CHAIN_ENDPOINT")  # bt_config.subtensor.chain_endpoint or
        else:
            subtensor_network = subtensor_network.lower()  # bt_config.subtensor.network or
        logger.info(f"Instantiating subtensor with network: {subtensor_network}")
        return bt.subtensor(network=subtensor_network)

    @cached_property_with_expiration(expiration_seconds=1200)
    def METAGRAPH(self) -> bt.metagraph:
        logger.info(f"Instantiating metagraph with NETUID: {self.NETUID}")
        return self.SUBTENSOR.metagraph(netuid=self.NETUID)

    @cached_property
    def UID(self) -> int:
        return self.METAGRAPH.hotkeys.index(self.WALLET.hotkey.ss58_address)

    @cached_property
    def DENDRITE(self) -> bt.dendrite:
        logger.info(f"Instantiating dendrite with wallet: {self.WALLET}")
        return bt.dendrite(wallet=self.WALLET)


shared_settings: Optional[SharedSettings] = None
try:
    shared_settings = SharedSettings.load(mode="mock")
    pass
except Exception as e:
    logger.exception(f"Error loading settings: {e}")
    shared_settings = None
logger.info("Shared settings loaded.")
