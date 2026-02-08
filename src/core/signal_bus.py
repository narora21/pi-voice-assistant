from typing import Any


class SignalBus:
    """Simple signal bus for decoupled communication between tools and the orchestrator."""

    def __init__(self) -> None:
        self._signals: dict[str, Any] = {}

    def emit(self, name: str, value: Any = True) -> None:
        """Emit a signal with an optional value."""
        self._signals[name] = value

    def poll(self, name: str) -> Any | None:
        """Check and consume a signal. Returns None if not set."""
        return self._signals.pop(name, None)
