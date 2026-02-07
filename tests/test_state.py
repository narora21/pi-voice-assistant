import pytest

from src.core.state import AssistantState, validate_transition


class TestValidTransitions:
    def test_waiting_to_listening(self):
        validate_transition(AssistantState.WAITING, AssistantState.LISTENING)

    def test_listening_to_transcribing(self):
        validate_transition(AssistantState.LISTENING, AssistantState.TRANSCRIBING)

    def test_listening_to_waiting(self):
        validate_transition(AssistantState.LISTENING, AssistantState.WAITING)

    def test_transcribing_to_thinking(self):
        validate_transition(AssistantState.TRANSCRIBING, AssistantState.THINKING)

    def test_transcribing_to_waiting(self):
        validate_transition(AssistantState.TRANSCRIBING, AssistantState.WAITING)

    def test_thinking_to_speaking(self):
        validate_transition(AssistantState.THINKING, AssistantState.SPEAKING)

    def test_thinking_to_waiting(self):
        validate_transition(AssistantState.THINKING, AssistantState.WAITING)

    def test_speaking_to_waiting(self):
        validate_transition(AssistantState.SPEAKING, AssistantState.WAITING)

    def test_speaking_to_listening(self):
        validate_transition(AssistantState.SPEAKING, AssistantState.LISTENING)


class TestInvalidTransitions:
    def test_waiting_to_thinking(self):
        with pytest.raises(ValueError, match="Invalid transition"):
            validate_transition(AssistantState.WAITING, AssistantState.THINKING)

    def test_waiting_to_speaking(self):
        with pytest.raises(ValueError, match="Invalid transition"):
            validate_transition(AssistantState.WAITING, AssistantState.SPEAKING)

    def test_listening_to_speaking(self):
        with pytest.raises(ValueError, match="Invalid transition"):
            validate_transition(AssistantState.LISTENING, AssistantState.SPEAKING)

    def test_thinking_to_listening(self):
        with pytest.raises(ValueError, match="Invalid transition"):
            validate_transition(AssistantState.THINKING, AssistantState.LISTENING)
