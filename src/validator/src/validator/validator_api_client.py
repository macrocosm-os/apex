import asyncio
from loguru import logger
from bittensor_wallet import Keypair
from aiohttp import ClientSession, ClientTimeout

from common.models.api_models import SubnetScores
from common.settings import ORCHESTRATOR_HOST, ORCHESTRATOR_PORT, ORCHESTRATOR_SCHEMA
from validator.settings import REQUEST_RETRY_COUNT, CLIENT_REQUEST_TIMEOUT

from common.utils.epistula import create_message_body, generate_header
from common.utils.exceptions import APIException, RateLimitException

HEADER_REQUEST_ID = "X-Request-Id"


class ValidatorAPIClient:
    """Common API client for all entities."""

    @classmethod
    async def orchestrator_request(
        cls, method: str, path: str, body: dict | None = None, hotkey: Keypair | None = None
    ) -> dict:
        logger.opt(colors=True).debug(
            f"\n<magenta>Making orchestrator request | method: {method} | path: {path}</magenta>"
        )

        headers = None
        response = None
        request_id = None  # Will be extracted from response
        body_bytes = create_message_body(data={} if not body else body)
        response_text = None

        if hotkey:
            headers = generate_header(hotkey, body_bytes)
            # Don't add request ID to headers - let orchestrator generate it

        for i in range(REQUEST_RETRY_COUNT):
            try:
                if i:
                    logger.warning(f"Retrying request to endpoint {path} (attempt {i + 1})")

                timeout = ClientTimeout(total=CLIENT_REQUEST_TIMEOUT)
                async with ClientSession(timeout=timeout) as session:
                    async with session.request(
                        method,
                        f"{ORCHESTRATOR_SCHEMA}://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}{path}",
                        json=body,
                        headers=headers,
                    ) as response:
                        # Extract request ID from response headers
                        request_id = response.headers.get(HEADER_REQUEST_ID, "unknown")
                        response_text = None

                        # Add request ID to logger context for all subsequent logs
                        with logger.contextualize(request_id=request_id):
                            if response.status == 429:
                                logger.warning(f"Rate limited on request to endpoint {path}")
                                await asyncio.sleep(2**i)
                                continue

                            if response.status != 200:
                                # Handle non-JSON error responses first
                                response_text = await response.text() if not response_text else response_text
                                msg = f"{response.status} - {response_text}"
                                raise APIException(f"Error making orchestrator request to endpoint {path}: {msg}")

                            # Success.
                            response_json = await response.json()
                            logger.debug(
                                f"Successfully completed request to {path}; response: {str(response_json)[:100]}"
                            )
                            return response_json
            except RateLimitException:
                logger.error("Rate limit exception, applying exponential backoff")
                await asyncio.sleep(2**i)
            except Exception as e:
                logger.error(e)
                await asyncio.sleep(1)

        # The only time you get here is because you've exhausted all retries.
        error_msg = (
            f"Failed request after {REQUEST_RETRY_COUNT} attempts: {response.status if response else 'No response'}, "
            f"{response_text if response_text else 'No response text'}"
        )
        if request_id:
            with logger.contextualize(request_id=request_id):
                logger.error(error_msg)
        raise RateLimitException(error_msg)

    @classmethod
    async def check_orchestrator_health(cls, hotkey: Keypair) -> bool | dict:
        try:
            response = await cls.orchestrator_request(method="GET", path="/healthcheck", hotkey=hotkey)
            if "error_name" in response:
                return response
            return response
        except Exception as e:
            logger.exception(f"Error checking orchestrator health: {e}")
            raise e

    @classmethod
    async def get_global_miner_scores(cls, hotkey: Keypair) -> SubnetScores | None:
        """Get the global scores for all miners from the orchestrator."""
        try:
            response: SubnetScores = await cls.orchestrator_request(
                method="GET", path="/validator/global_miner_scores", hotkey=hotkey
            )
            if hasattr(response, "error_name"):
                logger.error(f"Error getting global miner scores: {response}")
                return
            return response

        except Exception as e:
            logger.error(f"Error getting global miner scores: {e}")
            raise e
