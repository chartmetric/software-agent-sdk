"""Tests for the get_agent_partial_response utility function.

These pin the case a timed-out delegation actually presents: an agent that never
called finish and never emitted a standalone assistant message, whose only
record of progress is the `thought` on each action it took.
"""

from openhands.sdk.conversation.response_utils import (
    get_agent_final_response,
    get_agent_partial_response,
)
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.sdk.tool.builtins.finish import FinishAction


def _action(thought: str, index: int = 0) -> ActionEvent:
    """An ordinary non-finish tool call carrying the agent's reasoning."""
    return ActionEvent(
        source="agent",
        thought=[TextContent(text=thought)] if thought else [],
        action=FinishAction(message="unused"),
        tool_name="grep",
        tool_call_id=f"call-{index}",
        tool_call=MessageToolCall(
            id=f"call-{index}", name="grep", arguments="{}", origin="completion"
        ),
        llm_response_id=f"response-{index}",
    )


def test_returns_thoughts_when_agent_never_finished():
    # Arrange: an investigation cut short -- no finish, no assistant message.
    events = [_action("Looked at the city page", 0), _action("Found the table", 1)]

    # Act
    result = get_agent_partial_response(events)

    # Assert: chronological, so the parent reads the trail forwards.
    assert result == "Looked at the city page\nFound the table"


def test_final_response_reader_returns_nothing_for_the_same_events():
    # Arrange: the exact defect -- the *final*-response reader cannot see thoughts.
    events = [_action("Looked at the city page", 0), _action("Found the table", 1)]

    # Act
    result = get_agent_final_response(events)

    # Assert
    assert result == ""


def test_keeps_only_the_most_recent_thoughts():
    # Arrange
    events = [_action(f"step {i}", i) for i in range(10)]

    # Act
    result = get_agent_partial_response(events, max_thoughts=3)

    # Assert: the newest three, still in order.
    assert result == "step 7\nstep 8\nstep 9"


def test_truncates_rather_than_returning_an_unbounded_blob():
    # Arrange
    events = [_action("x" * 500, 0)]

    # Act
    result = get_agent_partial_response(events, max_chars=100)

    # Assert
    assert result.endswith("...")
    assert len(result) <= 103


def test_ignores_actions_with_no_thought_and_non_agent_sources():
    # Arrange
    events = [
        _action("", 0),
        MessageEvent(
            source="user",
            llm_message=Message(role="user", content=[TextContent(text="do it")]),
        ),
        _action("the only reasoning recorded", 1),
    ]

    # Act
    result = get_agent_partial_response(events)

    # Assert
    assert result == "the only reasoning recorded"


def test_returns_empty_string_when_nothing_was_established():
    # Arrange
    events: list = []

    # Act
    result = get_agent_partial_response(events)

    # Assert
    assert result == ""
