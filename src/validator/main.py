import asyncio
from validator.validator import Validator
from validator import settings as validator_settings

if __name__ == "__main__":
    gradient_validator = Validator(
        wallet_name=validator_settings.WALLET_NAME, wallet_hotkey=validator_settings.WALLET_HOTKEY
    )
    asyncio.run(gradient_validator.run_validator())
