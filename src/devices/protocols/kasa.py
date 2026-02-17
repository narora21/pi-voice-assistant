import logging

logger = logging.getLogger(__name__)


class KasaDevice:
    """Kasa smart plug backend using python-kasa."""

    async def turn_on(self, ip: str) -> None:
        from kasa import Discover

        dev = await Discover.discover_single(ip)
        await dev.turn_on()
        await dev.update()
        logger.info(f"Kasa device at {ip} turned on")

    async def turn_off(self, ip: str) -> None:
        from kasa import Discover

        dev = await Discover.discover_single(ip)
        await dev.turn_off()
        await dev.update()
        logger.info(f"Kasa device at {ip} turned off")
