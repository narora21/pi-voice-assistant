import logging
from typing import Any

from src.devices.manager import DeviceManager
from src.tools.base import ToolDefinition, ToolParameter

logger = logging.getLogger(__name__)


class DeviceControlTool:
    """Control smart home devices via the device manager."""

    def __init__(self, manager: DeviceManager) -> None:
        self._manager = manager

    @property
    def definition(self) -> ToolDefinition:
        device_names = self._manager.device_names
        return ToolDefinition(
            name="device_control",
            description=(
                "Control a smart home device. Can turn devices on or off. "
                f"Available devices: {', '.join(device_names)}."
            ),
            parameters=[
                ToolParameter(
                    name="device_name",
                    type="string",
                    description="Name of the device to control",
                    enum=device_names if device_names else None,
                ),
                ToolParameter(
                    name="action",
                    type="string",
                    description="Action to perform on the device",
                    enum=["on", "off"],
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> str:
        device_name: str = kwargs.get("device_name", "")
        action: str = kwargs.get("action", "")

        if not device_name:
            return "Error: 'device_name' is required."
        if action not in ("on", "off"):
            return f"Error: Invalid action '{action}'. Must be 'on' or 'off'."

        if action == "on":
            return await self._manager.turn_on(device_name)
        else:
            return await self._manager.turn_off(device_name)
