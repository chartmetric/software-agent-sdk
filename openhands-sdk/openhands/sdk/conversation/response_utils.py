"""Utility functions for extracting agent responses from conversation events."""

from collections.abc import Sequence

from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.event.base import Event
from openhands.sdk.llm.message import content_to_str
from openhands.sdk.tool.builtins.finish import FinishAction, FinishTool


def get_agent_final_response(events: Sequence[Event]) -> str:
    """Extract the final response from the agent.

    An agent can end a conversation in two ways:
    1. By calling the finish tool
    2. By returning a text message with no tool calls

    Args:
        events: List of conversation events to search through.

    Returns:
        The final response message from the agent, or empty string if not found.
    """
    # Find the last finish action or message event from the agent
    for event in reversed(events):
        # Case 1: finish tool call
        if (
            isinstance(event, ActionEvent)
            and event.source == "agent"
            and event.tool_name == FinishTool.name
        ):
            # Extract message from finish tool call
            if event.action is not None and isinstance(event.action, FinishAction):
                return event.action.message
            else:
                break
        # Case 2: text message with no tool calls (MessageEvent)
        elif isinstance(event, MessageEvent) and event.source == "agent":
            text_parts = content_to_str(event.llm_message.content)
            return "".join(text_parts)
    return ""


def get_agent_partial_response(
    events: Sequence[Event],
    max_thoughts: int = 6,
    max_chars: int = 4000,
) -> str:
    """Extract what an agent had established when it stopped before finishing.

    `get_agent_final_response` answers only for an agent that *finished*: it
    looks for a finish call or a standalone assistant message. An agent stopped
    by a wall-clock budget has neither by construction -- it never called finish,
    and an investigating agent's prose rides in the `thought` field of its
    `ActionEvent`s rather than in `MessageEvent`s. Reading those thoughts is the
    only record of the ground it covered, so without this a timed-out delegation
    costs its parent the full budget and returns nothing.

    Args:
        events: List of conversation events to search through.
        max_thoughts: How many of the most recent thoughts to keep.
        max_chars: Cap on the returned text, so one verbose step cannot crowd
            out the rest.

    Returns:
        The agent's most recent thoughts in chronological order, or an empty
        string when it recorded none.
    """
    collected: list[str] = []
    used = 0
    for event in reversed(events):
        if len(collected) >= max_thoughts or used >= max_chars:
            break
        if not (isinstance(event, ActionEvent) and event.source == "agent"):
            continue
        text = " ".join(content_to_str(event.thought)).strip()
        if not text:
            continue
        remaining = max_chars - used
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "..."
        collected.append(text)
        used += len(text)
    return "\n".join(reversed(collected))
