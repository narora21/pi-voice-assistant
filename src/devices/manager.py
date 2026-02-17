import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.devices.protocols.base import DeviceProtocol
from src.devices.protocols.kasa import KasaDevice

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    KASA = "kasa"


PROTOCOL_MAP: dict[DeviceType, type] = {
    DeviceType.KASA: KasaDevice,
}


@dataclass(frozen=True)
class DeviceEntry:
    name: str
    type: DeviceType
    ip: str
    aliases: list[str] = field(default_factory=list)


class DeviceManager:
    """Loads device config and routes commands to the correct protocol backend."""

    def __init__(
        self,
        config_path: str = "devices.json",
        kasa_username: str = "",
        kasa_password: str = "",
    ) -> None:
        self._devices: dict[str, DeviceEntry] = {}
        self._protocols: dict[str, DeviceProtocol] = {}
        self._kasa_username = kasa_username
        self._kasa_password = kasa_password
        self._load(config_path)

    def _load(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Device config not found: {config_path}")
            return

        with open(path) as f:
            data = json.load(f)

        for raw in data.get("devices", []):
            try:
                device_type = DeviceType(raw["type"])
            except ValueError:
                logger.warning(f"Unknown device type '{raw['type']}', skipping {raw['name']}")
                continue

            entry = DeviceEntry(
                name=raw["name"],
                type=device_type,
                ip=raw["ip"],
                aliases=raw.get("aliases", []),
            )

            # Register under primary name and all aliases
            self._devices[entry.name] = entry
            for alias in entry.aliases:
                self._devices[alias] = entry

            # Lazily create one protocol instance per type
            if device_type not in self._protocols:
                if device_type == DeviceType.KASA:
                    self._protocols[device_type] = KasaDevice(
                        self._kasa_username, self._kasa_password,
                    )
                else:
                    self._protocols[device_type] = PROTOCOL_MAP[device_type]()

            logger.info(f"Loaded device: {entry.name} ({device_type.value} @ {entry.ip})")

    @property
    def device_names(self) -> list[str]:
        """All valid device names (primary + aliases)."""
        return list(self._devices.keys())

    def _resolve(self, name: str) -> DeviceEntry | None:
        return self._devices.get(name)

    async def turn_on(self, device_name: str) -> str:
        entry = self._resolve(device_name)
        if entry is None:
            return f"Error: Unknown device '{device_name}'. Available: {self.device_names}"
        protocol = self._protocols[entry.type]
        await protocol.turn_on(entry.ip)
        return f"OK: '{entry.name}' turned on."

    async def turn_off(self, device_name: str) -> str:
        entry = self._resolve(device_name)
        if entry is None:
            return f"Error: Unknown device '{device_name}'. Available: {self.device_names}"
        protocol = self._protocols[entry.type]
        await protocol.turn_off(entry.ip)
        return f"OK: '{entry.name}' turned off."
