import os

from common.settings import BITTENSOR, MOCK

WEIGHT_SUBMIT_INTERVAL: int = 10 if (MOCK or not BITTENSOR) else 60 * 21  # submit weight every 21 minutes
ORCHESTRATOR_HEALTH_CHECK_INTERVAL: int = 60  # check orchestrator health every 1 minute

# Health settings
LAUNCH_HEALTH = os.getenv("LAUNCH_HEALTH") == "True"
VALIDATOR_HEALTH_HOST = os.getenv("VALIDATOR_HEALTH_HOST", "0.0.0.0")
VALIDATOR_HEALTH_PORT = int(os.getenv("VALIDATOR_HEALTH_PORT", 9100))
VALIDATOR_HEALTH_ENDPOINT = os.getenv("VALIDATOR_HEALTH_ENDPOINT", "/health")

WALLET_NAME = os.getenv("WALLET_NAME", "test")
WALLET_HOTKEY = os.getenv("WALLET_HOTKEY", "m1")

REQUEST_RETRY_COUNT = int(os.getenv("REQUEST_RETRY_COUNT", 3))
CLIENT_REQUEST_TIMEOUT = int(os.getenv("CLIENT_REQUEST_TIMEOUT", 30))
