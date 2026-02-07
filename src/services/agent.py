import logging
from collections.abc import AsyncIterator
from typing import Any

from src.config.schema import AgentConfig
from src.core.session import Message, Session
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentError(Exception):
    pass


class AgentService:
    """Agentic LLM loop with streaming and tool calling via Ollama."""

    def __init__(self, config: AgentConfig, tool_registry: ToolRegistry) -> None:
        self._config = config
        self._tools = tool_registry
        self._client: Any = None

    async def start(self) -> None:
        from ollama import AsyncClient

        self._client = AsyncClient(host=self._config.base_url)
        logger.info(f"Agent initialized with model: {self._config.model}")

    async def stop(self) -> None:
        self._client = None

    async def run(self, user_text: str, session: Session) -> AsyncIterator[str]:
        """
        Run the agent loop. Yields text chunks as they stream.
        Tool calls are handled internally and invisible to the caller.
        """
        if self._client is None:
            raise AgentError("Agent not started")

        session.add_message(Message(role="user", content=user_text))

        ollama_tools = self._tools.to_ollama_tools() or None

        for tool_round in range(self._config.max_tool_rounds + 1):
            full_text = ""
            tool_calls: list[dict[str, Any]] = []

            try:
                stream = await self._client.chat(
                    model=self._config.model,
                    messages=session.get_ollama_messages(),
                    tools=ollama_tools,
                    stream=self._config.stream,
                    options={"temperature": self._config.temperature},
                )
            except Exception as e:
                raise AgentError(f"Ollama chat failed: {e}") from e

            if self._config.stream:
                async for chunk in stream:
                    if chunk.message.content:
                        full_text += chunk.message.content
                        yield chunk.message.content
                    if chunk.message.tool_calls:
                        for tc in chunk.message.tool_calls:
                            tool_calls.append({
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            })
            else:
                # Non-streaming fallback
                response = stream
                if response.message.content:
                    full_text = response.message.content
                    yield full_text
                if response.message.tool_calls:
                    for tc in response.message.tool_calls:
                        tool_calls.append({
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        })

            # Record assistant message
            session.add_message(Message(
                role="assistant",
                content=full_text,
                tool_calls=tool_calls if tool_calls else None,
            ))

            if not tool_calls:
                break

            # Execute tool calls and feed results back
            logger.info(f"Tool round {tool_round + 1}: {len(tool_calls)} call(s)")
            for tc in tool_calls:
                result = await self._tools.call_tool(tc["name"], tc["arguments"])
                session.add_message(Message(
                    role="tool",
                    content=result,
                    tool_name=tc["name"],
                ))
