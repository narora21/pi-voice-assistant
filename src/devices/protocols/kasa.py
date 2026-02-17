import logging

logger = logging.getLogger(__name__)


class KasaDevice:
    """Kasa smart plug backend using python-kasa."""

    def __init__(self, username: str = "", password: str = "") -> None:
        self._username = username
        self._password = password

    async def _connect(self, ip: str):
        from kasa import Discover

        credentials = None
        if self._username and self._password:
            from kasa import Credentials
            credentials = Credentials(self._username, self._password)

        return await Discover.discover_single(ip, credentials=credentials)

    async def turn_on(self, ip: str) -> None:
        dev = await self._connect(ip)
        try:
            await dev.turn_on()
            await dev.update()
            logger.info(f"Kasa device at {ip} turned on")
        finally:
            await dev.disconnect()

    async def turn_off(self, ip: str) -> None:
        dev = await self._connect(ip)
        try:
            await dev.turn_off()
            await dev.update()
            logger.info(f"Kasa device at {ip} turned off")
        finally:
            await dev.disconnect()
