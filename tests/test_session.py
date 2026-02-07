import time

import pytest

from src.config.schema import SessionConfig
from src.core.session import Message, Session


@pytest.fixture
def session():
    return Session(SessionConfig(idle_timeout_seconds=5.0, max_history_messages=10))


def test_start_creates_system_message(session):
    session.start("You are helpful.")
    assert len(session.messages) == 1
    assert session.messages[0].role == "system"
    assert session.messages[0].content == "You are helpful."


def test_start_clears_previous_history(session):
    session.start("prompt 1")
    session.add_message(Message(role="user", content="hello"))
    session.start("prompt 2")
    assert len(session.messages) == 1
    assert session.messages[0].content == "prompt 2"


def test_add_message(session):
    session.start("sys")
    session.add_message(Message(role="user", content="hi"))
    session.add_message(Message(role="assistant", content="hello"))
    assert len(session.messages) == 3


def test_is_active_before_start(session):
    assert not session.is_active


def test_is_active_after_start(session):
    session.start("sys")
    assert session.is_active


def test_is_active_after_timeout(session):
    session.start("sys")
    # Set last activity far in the past to simulate timeout
    session._last_activity = time.monotonic() - 10.0
    assert not session.is_active


def test_touch_resets_timer(session):
    session.start("sys")
    session.touch()
    assert session.is_active


def test_end_deactivates(session):
    session.start("sys")
    session.end()
    assert not session.is_active


def test_get_ollama_messages(session):
    session.start("sys")
    session.add_message(Message(role="user", content="hi"))
    msgs = session.get_ollama_messages()
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_get_ollama_messages_with_tool_calls(session):
    session.start("sys")
    session.add_message(Message(
        role="assistant",
        content="",
        tool_calls=[{"name": "test", "arguments": {}}],
    ))
    msgs = session.get_ollama_messages()
    assert msgs[1]["tool_calls"] == [{"name": "test", "arguments": {}}]


def test_history_trimming(session):
    session.start("sys")
    for i in range(15):
        session.add_message(Message(role="user", content=f"msg {i}"))
    # max is 10: system + 9 most recent
    assert len(session.messages) == 10
    assert session.messages[0].role == "system"
    assert session.messages[1].content == "msg 6"
