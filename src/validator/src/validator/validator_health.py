import time

from aiohttp import web
from loguru import logger

from validator import settings as validator_settings


class HealthServerMixin:
    health_app_runner: web.AppRunner | None = None
    health_site: web.TCPSite | None = None

    async def _start_health_server(self):
        """Starts the aiohttp web server for healthchecks."""
        app = web.Application()

        async def health_handler(request):
            return web.json_response(
                {
                    "status": "healthy",
                    "hotkey": getattr(self, "hotkey", "N/A"),
                    "layer": getattr(self, "layer", "N/A"),
                    "uid": getattr(self, "uid", "N/A"),
                    "registered": getattr(self, "reregister_needed", True) is False,
                    "timestamp": time.time(),
                }
            )

        app.router.add_get(validator_settings.VALIDATOR_HEALTH_ENDPOINT, health_handler)

        self.health_app_runner = web.AppRunner(app)
        await self.health_app_runner.setup()

        self.health_site = web.TCPSite(
            self.health_app_runner, validator_settings.VALIDATOR_HEALTH_HOST, validator_settings.VALIDATOR_HEALTH_PORT
        )
        if validator_settings.LAUNCH_HEALTH:
            await self.health_site.start()
            logger.info(
                f"Miner {getattr(self, 'hotkey', 'N/A')} healthcheck API started on "
                f"http://{validator_settings.VALIDATOR_HEALTH_HOST}:{validator_settings.VALIDATOR_HEALTH_PORT}{validator_settings.VALIDATOR_HEALTH_ENDPOINT}"
            )
