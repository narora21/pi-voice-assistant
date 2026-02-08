from enum import Enum, auto


class AssistantState(Enum):
    WAITING = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()


TRANSITIONS: dict[AssistantState, set[AssistantState]] = {
    AssistantState.WAITING: {AssistantState.LISTENING},
    AssistantState.LISTENING: {AssistantState.TRANSCRIBING, AssistantState.WAITING},
    AssistantState.TRANSCRIBING: {AssistantState.THINKING, AssistantState.WAITING},
    AssistantState.THINKING: {AssistantState.LISTENING, AssistantState.WAITING},
}


def validate_transition(current: AssistantState, target: AssistantState) -> None:
    """Raise ValueError if the transition is not allowed."""
    allowed = TRANSITIONS.get(current, set())
    if target not in allowed:
        raise ValueError(
            f"Invalid transition: {current.name} -> {target.name}. "
            f"Allowed: {[s.name for s in allowed]}"
        )
