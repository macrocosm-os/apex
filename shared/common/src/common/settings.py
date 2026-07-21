import os
from dotenv import load_dotenv
from loguru import logger

__SPEC_VERSION__ = 100000
COMMIT_HASH = os.getenv("COMMIT_HASH", "DUMMY")
COMMON_DOTENV_PATH = os.getenv("COMMON_DOTENV_PATH", ".env")
if not load_dotenv(dotenv_path=COMMON_DOTENV_PATH):
    logger.warning("No .env file found for common settings")

# System
MOCK = os.getenv("MOCK") == "True"  # initialize with mock competition
ENV = os.getenv("ENV", "prod")
SB_PREFIX = os.getenv("SB_PREFIX", "main")
LOG_FILE_ENABLED = os.getenv("LOG_FILE_ENABLED") == "True"
TEST_MODE = os.getenv("TEST_MODE") == "True"


def sandbox_image_tag(competition_pkg: str, variant: str | None = None) -> str:
    """Canonical sandbox image tag: sb-{SB_PREFIX}-{pkg}[-{variant}]-{COMMIT_HASH}.

    Single source of truth for the tag CI builds and the worker pulls.
    variant="groundtruth" selects the round-gen bake image; None = miner image.
    """
    infix = f"-{variant}" if variant else ""
    return f"sb-{SB_PREFIX}-{competition_pkg}{infix}-{COMMIT_HASH}"


# Bittensor settings
BITTENSOR = os.getenv("BITTENSOR") == "True"
NETUID = int(os.getenv("NETUID", 1))
NETWORK = os.getenv("NETWORK", "finney")
OWNER_UID = 248

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

# Competition settings
DEFAULT_BASE_BURN_RATE = 0.9

# Submission fee settings
SUBMISSION_FEE_ADDRESS = os.getenv(
    "SUBMISSION_FEE_ADDRESS", "5EtauUg5ZyHYuRN8MP1hBSejvFjXsKoCKcDr3FJrdy8dZepK"
)  # prod ss58 coldkey that receives submission fees

# Primary network for payment verification — defaults to finney lite for low-latency
# verification of recent blocks. State-pruned blocks (older than ~256 blocks on lite)
# transparently fail over to SUBMISSION_FEE_VERIFICATION_ARCHIVE_NETWORK if set.
SUBMISSION_FEE_VERIFICATION_NETWORK = os.getenv(
    "SUBMISSION_FEE_VERIFICATION_NETWORK", "wss://entrypoint-finney.opentensor.ai:443"
)

SUBMISSION_FEE_VERIFICATION_ARCHIVE_NETWORK = os.getenv("SUBMISSION_FEE_VERIFICATION_ARCHIVE_NETWORK", "")
