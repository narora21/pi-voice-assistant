from typing import Protocol


class DeviceProtocol(Protocol):
    """Protocol for smart home device backends."""

    async def turn_on(self, ip: str) -> None: ...

    async def turn_off(self, ip: str) -> None: ...
