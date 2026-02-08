from typing import Any

from src.core.signal_bus import SignalBus
from src.tools.base import ToolDefinition

END_CONVERSATION = "end_conversation"


class EndConversationTool:
    """Tool that lets the agent signal the conversation is over."""

    def __init__(self, signal_bus: SignalBus) -> None:
        self._bus = signal_bus

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="end_conversation",
            description="End the conversation when the exchange is complete.",
        )

    async def execute(self, **kwargs: Any) -> str:
        self._bus.emit(END_CONVERSATION)
        return "Conversation ended."
