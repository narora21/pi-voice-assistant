import logging
from typing import Any

from src.tools.base import Tool, ToolDefinition, ToolParameter

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for agent tools with Ollama schema conversion."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError if name already taken."""
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool
        logger.info(f"Registered tool: {name}")

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    def to_ollama_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to Ollama's tool calling format."""
        result: list[dict[str, Any]] = []
        for tool in self._tools.values():
            defn = tool.definition
            properties: dict[str, Any] = {}
            required: list[str] = []

            for param in defn.parameters:
                prop: dict[str, Any] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.enum:
                    prop["enum"] = param.enum
                properties[param.name] = prop
                if param.required:
                    required.append(param.name)

            result.append({
                "type": "function",
                "function": {
                    "name": defn.name,
                    "description": defn.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return result

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name. Returns result or error as text."""
        tool = self.get(name)
        if tool is None:
            available = list(self._tools.keys())
            return f"Error: Unknown tool '{name}'. Available tools: {available}"
        try:
            logger.info(f"Calling tool '{name}' with args: {arguments}")
            result = await tool.execute(**arguments)
            logger.info(f"Tool '{name}' returned: {result}")
            return result
        except Exception as e:
            logger.exception(f"Tool '{name}' raised unexpected error")
            return f"Error executing tool '{name}': {e}"
