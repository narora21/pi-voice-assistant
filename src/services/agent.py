import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from ollama import AsyncClient

from src.config.schema import AgentConfig
from src.core.session import Message, Session
from src.tools.registry import ToolRegistry
from src.util.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class AgentError(Exception):
    pass


class AgentService:
    """Agentic LLM loop with streaming and tool calling via Ollama."""

    def __init__(self, config: AgentConfig, tool_registry: ToolRegistry) -> None:
        self._config = config
        self._tools = tool_registry
        self._client: AsyncClient | None = None

    async def start(self) -> None:
        self._client = AsyncClient()
        logger.info(f"Agent initialized with model: {self._config.model}")
        await self._warmup()

    async def _warmup(self) -> None:
        """Send a request with the full system prompt and tools to prime the KV cache."""
        logger.info(f"Warming up model: {self._config.model}")
        system_prompt = PromptLoader.load_system_prompt(self._config, self._tools)
        start = time.monotonic()
        try:
            ollama_tools = self._tools.to_ollama_tools() or None
            await self._client.chat(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "hi"},
                ],
                tools=ollama_tools,
                options={"num_predict": 1},
            )
            elapsed = time.monotonic() - start
            logger.info(f"Model warmup completed in {elapsed:.1f}s")
        except Exception:
            logger.warning("Model warmup failed — first request may be slow", exc_info=True)

    async def stop(self) -> None:
        self._client = None

    async def run(self, user_text: str, session: Session) -> AsyncIterator[str]:
        """
        Run the agent loop. Yields text chunks as they stream.
        Tool calls are handled internally and invisible to the caller.
        """
        start = time.monotonic()
        elapsed = None

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
                    options={
                        "temperature": self._config.temperature,
                        "num_ctx": self._config.num_ctx,
                        "num_thread": self._config.num_thread,
                    },
                    stream=True,
                )
            except Exception as e:
                raise AgentError(f"Ollama chat failed: {e}") from e

            async for chunk in stream:
                if elapsed is None:
                    elapsed = time.monotonic() - start
                if chunk["message"]["content"]:
                    full_text += chunk["message"]["content"]
                    yield chunk["message"]["content"]

                if chunk["message"].get("tool_calls"):
                    for tc in chunk["message"]["tool_calls"]:
                        tool_calls.append(tc)
            
            logger.info(f"\nTime to first token (TTF): {elapsed} s")

            # Record assistant message
            session.add_message(Message(
                role="assistant",
                content=full_text,
                tool_calls=tool_calls if tool_calls else None,
            ))

            if not tool_calls:
                logger.info("No tool calls, ending agent stream")
                break

            # Execute tool calls and feed results back
            logger.info(f"Tool round {tool_round + 1}: {len(tool_calls)} call(s)")
            for tc in tool_calls:
                fn = tc["function"]
                result = await self._tools.call_tool(fn["name"], fn["arguments"])
                session.add_message(Message(
                    role="tool",
                    content=result,
                    tool_name=fn["name"],
                ))