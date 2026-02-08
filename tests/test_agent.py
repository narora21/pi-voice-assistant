from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.schema import AgentConfig, SessionConfig
from src.core.session import Session
from src.services.agent import AgentError, AgentService
from src.tools.base import ToolDefinition, ToolParameter
from src.tools.registry import ToolRegistry


def make_chunk(content: str = "", tool_calls: list[dict] | None = None) -> dict:
    """Create a mock chunk dict matching Ollama's streaming response format."""
    chunk = {
        "message": {
            "content": content,
            "tool_calls": tool_calls or [],
        }
    }
    return chunk


def make_tool_call(name: str, arguments: dict[str, Any]) -> dict:
    """Create a tool call dict matching Ollama's format."""
    return {
        "function": {
            "name": name,
            "arguments": arguments,
        }
    }


class EchoTool:
    """Simple test tool that echoes input."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Echo back the input",
            parameters=[
                ToolParameter(name="text", type="string", description="Text to echo"),
            ],
        )

    async def execute(self, **kwargs: Any) -> str:
        return f"Echo: {kwargs.get('text', '')}"


@pytest.fixture
def agent_config():
    return AgentConfig(
        model="test-model",
        max_tool_rounds=3,
        stream=True,
    )


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(EchoTool())
    return r


@pytest.fixture
def session():
    s = Session(SessionConfig(idle_timeout_seconds=30.0, max_history_messages=50))
    s.start("You are a test assistant.")
    return s


@pytest.fixture
def agent(agent_config, registry):
    return AgentService(agent_config, registry)


@pytest.mark.asyncio
async def test_run_not_started(agent, session):
    with pytest.raises(AgentError, match="not started"):
        async for _ in agent.run("hello", session):
            pass


@pytest.mark.asyncio
async def test_run_simple_text_response(agent, session):
    """Agent streams a simple text response with no tool calls."""
    mock_client = AsyncMock()

    async def mock_stream():
        yield make_chunk(content="Hello ")
        yield make_chunk(content="world!")

    mock_client.chat.return_value = mock_stream()
    agent._client = mock_client

    chunks = []
    async for chunk in agent.run("hi", session):
        chunks.append(chunk)

    assert chunks == ["Hello ", "world!"]
    # Session should have: system, user, assistant
    assert len(session.messages) == 3
    assert session.messages[1].role == "user"
    assert session.messages[1].content == "hi"
    assert session.messages[2].role == "assistant"
    assert session.messages[2].content == "Hello world!"


@pytest.mark.asyncio
async def test_run_with_tool_call(agent, session):
    """Agent makes a tool call, then responds with text."""
    mock_client = AsyncMock()

    call_count = 0

    async def mock_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: LLM requests a tool call
            async def stream1():
                yield make_chunk(
                    content="",
                    tool_calls=[make_tool_call("echo", {"text": "test"})],
                )
            return stream1()
        else:
            # Second call: LLM responds with text after seeing tool result
            async def stream2():
                yield make_chunk(content="Done!")
            return stream2()

    mock_client.chat = mock_chat
    agent._client = mock_client

    chunks = []
    async for chunk in agent.run("echo test", session):
        chunks.append(chunk)

    assert "Done!" in chunks
    # Session: system, user, assistant(tool_call), tool(result), assistant(text)
    assert len(session.messages) == 5
    assert session.messages[3].role == "tool"
    assert "Echo: test" in session.messages[3].content


@pytest.mark.asyncio
async def test_max_tool_rounds(agent, session):
    """Agent stops after max_tool_rounds even if LLM keeps requesting tools."""
    mock_client = AsyncMock()

    async def mock_chat(**kwargs):
        async def stream():
            yield make_chunk(
                content="calling tool",
                tool_calls=[make_tool_call("echo", {"text": "loop"})],
            )
        return stream()

    mock_client.chat = mock_chat
    agent._client = mock_client

    chunks = []
    async for chunk in agent.run("loop", session):
        chunks.append(chunk)

    # Should have done max_tool_rounds + 1 iterations (0 to max_tool_rounds inclusive)
    # Each yields "calling tool" text, so we get max_tool_rounds + 1 chunks
    assert len(chunks) == agent._config.max_tool_rounds + 1