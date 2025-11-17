from hashlib import sha256
import bittensor as bt
from bittensor_wallet import Keypair
from bittensor_wallet.mock import get_mock_wallet
from loguru import logger
import tenacity

from common import settings as common_settings


def _log_retry_attempt(retry_state):
    """Log when a retry attempt is made."""
    attempt_number = retry_state.attempt_number
    logger.warning(f"🔄 Retry attempt {attempt_number} for getting subtensor on network {common_settings.NETWORK}")


# retry but if it fails, it will raise an error
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=_log_retry_attempt,
)
def get_subtensor() -> bt.subtensor:
    logger.info(f"🔄 Getting subtensor for network: {common_settings.NETWORK}")
    if common_settings.MOCK:
        logger.info("🔄 Using mock subtensor")
        from bittensor.utils.mock.subtensor_mock import Subtensor

        try:
            subtensor = Subtensor("test")
            logger.info("Using Mock subtensor with network test")
            return subtensor
        except Exception as e:
            logger.error(f"Error loading subtensor(test) while in Mock mode: {e}")
            subtensor = Subtensor()
            logger.info("Using Mock subtensor with network Finney")
            return subtensor

    elif common_settings.BITTENSOR:
        logger.info("🔄 Using subtensor")
        return bt.subtensor(network=common_settings.NETWORK)
    else:
        raise Exception("No subtensor found")


def get_wallet(wallet_name: str, wallet_hotkey: str) -> bt.wallet:
    """Get a Bittensor wallet.

    Args:
        wallet_name: The name of the wallet
        wallet_hotkey: The hotkey of the wallet
    """
    logger.info(
        f"Initializing Bittensor wallet: {wallet_name} and hotkey: {wallet_hotkey}. Bittensor is set to {common_settings.BITTENSOR}"
    )
    if common_settings.BITTENSOR:
        wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        return wallet
    else:
        return get_mock_wallet(
            hotkey=Keypair.create_from_seed(seed=sha256(wallet_name.encode()).hexdigest()),
            coldkey=Keypair.create_from_seed(seed=sha256(wallet_hotkey.encode()).hexdigest()),
        )
