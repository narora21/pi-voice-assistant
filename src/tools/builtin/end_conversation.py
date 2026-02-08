from typing import Any

from src.tools.base import ToolDefinition


class EndConversationTool:
    """Tool that lets the agent signal the conversation is over."""

    def __init__(self) -> None:
        self.triggered = False

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="end_conversation",
            description="End the conversation when the exchange is complete.",
        )

    async def execute(self, **kwargs: Any) -> str:
        self.triggered = True
        return "Conversation ended."

    def reset(self) -> None:
        self.triggered = False
