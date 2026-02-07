import time
from dataclasses import dataclass, field
from typing import Any

from src.config.schema import SessionConfig


@dataclass
class Message:
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_name: str | None = None


class Session:
    """Conversation session with history and idle timeout."""

    def __init__(self, config: SessionConfig) -> None:
        self._config = config
        self._messages: list[Message] = []
        self._last_activity: float = 0.0
        self._active: bool = False

    def start(self, system_prompt: str) -> None:
        """Begin a new session with system prompt. Clears old history."""
        self._messages = [Message(role="system", content=system_prompt)]
        self._last_activity = time.monotonic()
        self._active = True

    def add_message(self, message: Message) -> None:
        """Append message and update idle timer."""
        self._messages.append(message)
        self._last_activity = time.monotonic()
        self._trim_history()

    def get_ollama_messages(self) -> list[dict[str, Any]]:
        """Convert messages to Ollama SDK format."""
        result: list[dict[str, Any]] = []
        for msg in self._messages:
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            result.append(entry)
        return result

    @property
    def is_active(self) -> bool:
        """True if session started and idle timeout has not expired."""
        if not self._active:
            return False
        elapsed = time.monotonic() - self._last_activity
        if elapsed >= self._config.idle_timeout_seconds:
            self._active = False
            return False
        return True

    def touch(self) -> None:
        """Reset the idle timer."""
        self._last_activity = time.monotonic()

    def end(self) -> None:
        """Explicitly end the session."""
        self._active = False

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def _trim_history(self) -> None:
        """Keep system prompt + last N messages if over limit."""
        max_msgs = self._config.max_history_messages
        if len(self._messages) <= max_msgs:
            return
        system = self._messages[0] if self._messages and self._messages[0].role == "system" else None
        if system:
            self._messages = [system] + self._messages[-(max_msgs - 1):]
        else:
            self._messages = self._messages[-max_msgs:]
