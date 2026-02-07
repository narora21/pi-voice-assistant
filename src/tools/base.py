from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)


class Tool(Protocol):
    @property
    def definition(self) -> ToolDefinition: ...

    async def execute(self, **kwargs: Any) -> str: ...
