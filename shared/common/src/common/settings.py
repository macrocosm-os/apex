import os
from dotenv import load_dotenv
from loguru import logger

__SPEC_VERSION__ = 100000

COMMON_DOTENV_PATH = os.getenv("COMMON_DOTENV_PATH", ".env")
if not load_dotenv(dotenv_path=COMMON_DOTENV_PATH):
    logger.warning("No .env file found for common settings")

# System
MOCK = os.getenv("MOCK") == "True"  # initialize with mock competition
ENV = os.getenv("ENV", "local")
LOG_FILE_ENABLED = os.getenv("LOG_FILE_ENABLED") == "True"
TEST_MODE = os.getenv("TEST_MODE") == "True"

# Bittensor settings
BITTENSOR = os.getenv("BITTENSOR") == "True"
NETUID = int(os.getenv("NETUID", 1))
NETWORK = os.getenv("NETWORK", "finney")
OWNER_UID = 395

# Orchestrator
if ENV == "local":
    # Local testing
    ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", 8000))
    ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "localhost")
    ORCHESTRATOR_SCHEMA = os.getenv("ORCHESTRATOR_SCHEMA", "http")
elif NETWORK == "test":
    # Testnet
    ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", 443))
    ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "apex-stage.api.macrocosmos.ai")
    ORCHESTRATOR_SCHEMA = os.getenv("ORCHESTRATOR_SCHEMA", "https")
else:
    # Mainnet
    ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", 443))
    ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "apex.api.macrocosmos.ai")
    ORCHESTRATOR_SCHEMA = os.getenv("ORCHESTRATOR_SCHEMA", "https")

# Scoring
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.01"))
