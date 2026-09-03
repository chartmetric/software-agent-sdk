"""What a run stops paying for once it has read something.

A run is charged for its whole history on every model call, and most of that
history is output it has already acted on. This condenser keeps the head of an
old result and drops stale reasoning, deterministically and without a model
call, so a summary is only needed when trimming is not enough.
"""

from typing import cast
from unittest.mock import Mock, patch

from openhands.sdk.context.condenser import TrimOldContext
from openhands.sdk.context.condenser.trim_old_context_condenser import (
    KEEP_RECENT_EVENTS,
    MAX_OBSERVATION_CHARS,
    MIN_TRIM_YIELD_TOKENS,
)
from openhands.sdk.context.view import View
from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.llm import TextContent
from openhands.sdk.tool.client_tool import ClientToolObservation


def _observation(text: str) -> ObservationEvent:
    # A real observation model rather than a mock: the trim rebuilds it with
    # `model_copy`, and a mock answers a mock for every field -- a test that
    # passes on code that produces nothing usable.
    return ObservationEvent.model_construct(
        tool_name="grep",
        observation=ClientToolObservation(content=[TextContent(text=text)]),
    )


def _text(event) -> str:
    return "".join(block.text for block in event.observation.content)


def _condense(view: View, *, total_tokens: int) -> View:
    with patch(
        "openhands.sdk.context.condenser.trim_old_context_condenser."
        "get_total_token_count",
        return_value=total_tokens,
    ):
        return cast(
            View, TrimOldContext(max_tokens=60_000).condense(view, agent_llm=Mock())
        )


def test_a_view_inside_its_budget_is_returned_untouched() -> None:
    view = View(events=[_observation("x" * 30_000) for _ in range(20)])

    assert _condense(view, total_tokens=10_000) is view


def test_over_budget_the_old_results_are_cut_and_the_recent_ones_are_not() -> None:
    view = View(events=[_observation("x" * 30_000) for _ in range(20)])

    result = _condense(view, total_tokens=200_000)

    old = result.events[: len(result.events) - KEEP_RECENT_EVENTS]
    recent = result.events[len(result.events) - KEEP_RECENT_EVENTS :]
    assert all(len(_text(event)) < 30_000 for event in old)
    assert all(_text(event).startswith("x" * MAX_OBSERVATION_CHARS) for event in old)
    assert all(len(_text(event)) == 30_000 for event in recent)


def test_stale_reasoning_is_dropped_and_recent_reasoning_is_kept() -> None:
    """Providers that take reasoning back re-read every prior turn's thinking."""
    events: list[LLMConvertibleEvent] = [
        ActionEvent.model_construct(tool_name="grep", reasoning_content="y" * 40_000)
        for _ in range(20)
    ]

    result = _condense(View(events=events), total_tokens=200_000)

    old = [cast(ActionEvent, e) for e in result.events[:-KEEP_RECENT_EVENTS]]
    recent = [cast(ActionEvent, e) for e in result.events[-KEEP_RECENT_EVENTS:]]
    assert all(event.reasoning_content is None for event in old)
    assert all(event.reasoning_content == "y" * 40_000 for event in recent)


def test_a_trim_worth_little_is_not_taken() -> None:
    """Editing the prompt costs the cache from that point on, so it must pay."""
    view = View(events=[_observation("short") for _ in range(20)])

    assert _condense(view, total_tokens=200_000) is view
    assert MIN_TRIM_YIELD_TOKENS > 0
