"""Tests for pruning superseded browser observations from LLM messages.

The message shapes here mirror exactly what ``ObservationEvent.to_llm_message``
produces for browser tool results: ``role="tool"``, ``name=<tool name>``,
``tool_call_id``, and a content list of ``TextContent``/``ImageContent``.
"""

import json
from unittest.mock import patch

from openhands.sdk.agent.browser_context_pruning import (
    KEEP_RECENT_BROWSER_PAGE_SNAPSHOTS,
    KEEP_RECENT_BROWSER_SCREENSHOTS,
    STALE_STATE_TEXT_HEAD_CHARS,
    prune_stale_browser_observations,
)
from openhands.sdk.agent.utils import prepare_llm_messages
from openhands.sdk.context.view import View
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import ImageContent, Message, TextContent


_DATA_URL = "data:image/png;base64,aGVsbG8="

# The real payload, as `_browser_state_payload` renders it for every tool that
# answers with the page. The `interactive_elements` key is what marks a block as
# a superseded snapshot, so a fixture without it would exercise a shape
# production never sends.
_LONG_STATE_TEXT = json.dumps(
    {
        "url": "https://example.com/page",
        "title": "Example",
        "tabs": [{"url": "https://example.com/page", "title": "Example"}],
        "interactive_elements": [
            {"index": index, "tag": "button", "text": "Click me"}
            for index in range(400)
        ],
    },
    indent=2,
)

# What `browser_get_content` returns: long, but the answer the agent asked for
# rather than a view of the page that a later snapshot replaces.
_LONG_PAGE_CONTENT = "Dominic Fike is an American singer.\n" * 400


def _browser_state_message(
    call_id: str, with_screenshot: bool = True, tool_name: str = "browser_get_state"
) -> Message:
    content: list[TextContent | ImageContent] = [
        TextContent(text=_LONG_STATE_TEXT),
        TextContent(text="Screenshot saved to: /tmp/shot.png"),
    ]
    if with_screenshot:
        content.append(ImageContent(image_urls=[_DATA_URL]))
    return Message(
        role="tool",
        name=tool_name,
        tool_call_id=call_id,
        content=content,
    )


def _browser_click_message(call_id: str) -> Message:
    return Message(
        role="tool",
        name="browser_click",
        tool_call_id=call_id,
        content=[TextContent(text="Clicked element 42")],
    )


def _assistant_message() -> Message:
    return Message(role="assistant", content=[TextContent(text="Checking the page.")])


def _images_of(message: Message) -> list[ImageContent]:
    return [item for item in message.content if isinstance(item, ImageContent)]


def test_few_browser_messages_pass_through_unchanged():
    messages = [
        _assistant_message(),
        *(
            _browser_state_message(f"call_{i}")
            for i in range(KEEP_RECENT_BROWSER_SCREENSHOTS)
        ),
    ]

    result = prune_stale_browser_observations(messages)

    assert result is messages


def test_stale_screenshots_replaced_and_recent_kept():
    stale_count = 3
    messages = [
        _browser_state_message(f"call_{i}")
        for i in range(stale_count + KEEP_RECENT_BROWSER_SCREENSHOTS)
    ]

    result = prune_stale_browser_observations(messages)

    for message in result[:stale_count]:
        assert not _images_of(message)
        placeholder_texts = [
            item.text
            for item in message.content
            if isinstance(item, TextContent) and "Screenshot omitted" in item.text
        ]
        assert placeholder_texts
    for message in result[stale_count:]:
        assert _images_of(message)


def test_stale_state_text_truncated_with_placeholder():
    stale = _browser_state_message("call_0")
    messages = [
        stale,
        *(
            _browser_state_message(f"call_{i + 1}")
            for i in range(KEEP_RECENT_BROWSER_PAGE_SNAPSHOTS)
        ),
    ]

    result = prune_stale_browser_observations(messages)

    stale_texts = [
        item.text for item in result[0].content if isinstance(item, TextContent)
    ]
    truncated = next(text for text in stale_texts if "stale browser state" in text)
    assert truncated.startswith(_LONG_STATE_TEXT[:STALE_STATE_TEXT_HEAD_CHARS])
    assert "must not be used" in truncated
    # The short "Screenshot saved to" line survives untouched.
    assert "Screenshot saved to: /tmp/shot.png" in stale_texts
    # Recent snapshots keep their full state text.
    for message in result[1:]:
        texts = [item.text for item in message.content if isinstance(item, TextContent)]
        assert _LONG_STATE_TEXT in texts


def test_a_stale_snapshot_is_truncated_whatever_tool_returned_it():
    """The window counts snapshots, not `browser_get_state` calls.

    Every state-changing action answers with the page it produced, so the
    oldest snapshot here was returned by `browser_get_state` and the two that
    supersede it by `browser_scroll` and `browser_click`. Keying the rule on the
    tool name left the first one full-size for the rest of the conversation and
    exempted 73% of production browser snapshot text from pruning.
    """
    messages = [
        _browser_state_message("call_0", with_screenshot=False),
        _browser_state_message(
            "call_1", with_screenshot=False, tool_name="browser_scroll"
        ),
        _browser_state_message(
            "call_2", with_screenshot=False, tool_name="browser_click"
        ),
    ]

    result = prune_stale_browser_observations(messages)

    stale_texts = [
        item.text for item in result[0].content if isinstance(item, TextContent)
    ]
    assert any("stale browser state" in text for text in stale_texts)
    for message in result[1:]:
        texts = [item.text for item in message.content if isinstance(item, TextContent)]
        assert _LONG_STATE_TEXT in texts


def test_a_page_content_answer_is_never_truncated():
    """`browser_get_content` returns the answer, not a view a snapshot replaces.

    It is a `browser_*` tool message and its text is well over the truncation
    threshold, so only the snapshot marker keeps it whole.
    """
    content_answer = Message(
        role="tool",
        name="browser_get_content",
        tool_call_id="call_0",
        content=[TextContent(text=_LONG_PAGE_CONTENT)],
    )
    messages = [
        content_answer,
        *(
            _browser_state_message(f"call_{i + 1}", with_screenshot=False)
            for i in range(KEEP_RECENT_BROWSER_PAGE_SNAPSHOTS)
        ),
    ]

    result = prune_stale_browser_observations(messages)

    assert result[0].content[0].text == _LONG_PAGE_CONTENT


def test_user_images_and_other_tools_untouched():
    user_image = Message(
        role="user",
        content=[
            TextContent(text="Look at this"),
            ImageContent(image_urls=[_DATA_URL]),
        ],
    )
    other_tool = Message(
        role="tool",
        name="terminal",
        tool_call_id="call_t",
        content=[TextContent(text="x" * 100_000), ImageContent(image_urls=[_DATA_URL])],
    )
    messages = [
        user_image,
        other_tool,
        *(
            _browser_state_message(f"call_{i}")
            for i in range(KEEP_RECENT_BROWSER_SCREENSHOTS + 1)
        ),
    ]

    result = prune_stale_browser_observations(messages)

    assert result[0] is user_image
    assert result[1] is other_tool


def test_short_browser_results_are_never_truncated():
    stale_click = _browser_click_message("call_0")
    messages = [
        stale_click,
        *(
            _browser_state_message(f"call_{i + 1}")
            for i in range(KEEP_RECENT_BROWSER_SCREENSHOTS + 1)
        ),
    ]

    result = prune_stale_browser_observations(messages)

    assert result[0] is stale_click


def test_input_messages_are_not_mutated():
    messages = [
        _browser_state_message(f"call_{i}")
        for i in range(KEEP_RECENT_BROWSER_SCREENSHOTS + 1)
    ]

    prune_stale_browser_observations(messages)

    assert _images_of(messages[0])
    assert any(
        isinstance(item, TextContent) and item.text == _LONG_STATE_TEXT
        for item in messages[0].content
    )


def test_pruning_is_idempotent():
    messages = [
        _browser_state_message(f"call_{i}")
        for i in range(KEEP_RECENT_BROWSER_SCREENSHOTS + 2)
    ]

    once = prune_stale_browser_observations(messages)
    twice = prune_stale_browser_observations(once)

    assert [m.model_dump() for m in once] == [m.model_dump() for m in twice]


@patch("openhands.sdk.event.base.LLMConvertibleEvent.events_to_messages")
def test_prepare_llm_messages_applies_pruning(mock_events_to_messages):
    stale_count = 2
    built_messages = [
        _browser_state_message(f"call_{i}")
        for i in range(stale_count + KEEP_RECENT_BROWSER_SCREENSHOTS)
    ]
    mock_events_to_messages.return_value = built_messages
    view = View(
        events=[
            MessageEvent(
                source="user",
                llm_message=Message(role="user", content=[TextContent(text="hi")]),
            )
        ]
    )

    result = prepare_llm_messages(view)

    assert isinstance(result, list)
    for message in result[:stale_count]:
        assert not _images_of(message)
    for message in result[stale_count:]:
        assert _images_of(message)
