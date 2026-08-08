"""Prune superseded browser observations from the per-step LLM message list.

Browser observations are the largest recurring payloads in a conversation:
``browser_get_state`` returns an accessibility snapshot that can reach the
50,000-character truncation limit, and screenshot-bearing observations add a
full-page image on top. Every one of them stays in the conversation history
and is re-sent on each subsequent LLM request until a condensation folds it
away, so a burst of browser verification late in a session can balloon the
context by six figures of tokens and force an expensive condensation.

Superseded browser snapshots carry very little decision value: the tool
contract already forbids acting on element indices from anything but the
current ``browser_get_state`` output, and a newer snapshot of the same browser
replaces the older view of the page. This module rewrites the *stale* browser
tool messages (never the most recent ones) at message-build time:

- Screenshots: every ``ImageContent`` in a stale ``browser_*`` tool message is
  replaced with a short text placeholder. The "Screenshot saved to: <path>"
  line emitted by the tool is left untouched, so the pixels remain recoverable
  from disk.
- ``browser_get_state`` text: oversized snapshot text in a stale message is
  cut to a head excerpt plus a placeholder explaining that the snapshot was
  superseded and how to re-fetch the current state.

The pruning is a pure, deterministic function of the message list, so every
process/step rebuilds the identical prompt. A given message's content changes
at most once — when enough newer browser snapshots arrive to push it out of
the keep-window — which limits prompt-cache invalidation to a single,
tail-local break per new snapshot.

Events themselves are never modified; only the transient ``Message`` objects
built for the current LLM call are rewritten.
"""

from __future__ import annotations

from collections.abc import Sequence

from openhands.sdk.llm import ImageContent, Message, TextContent


# Number of most-recent screenshot-bearing browser tool messages whose images
# are kept intact. Two covers the common "compare before/after my change"
# verification pattern while dropping everything older.
KEEP_RECENT_BROWSER_SCREENSHOTS = 2

# Number of most-recent `browser_get_state` tool messages whose snapshot text
# is kept intact.
KEEP_RECENT_BROWSER_STATE_TEXTS = 2

# Head excerpt preserved from a stale `browser_get_state` snapshot so the page
# URL/title context survives, without the full element tree.
STALE_STATE_TEXT_HEAD_CHARS = 2_000

# Only text blocks larger than this are candidates for truncation. Short
# blocks (error headers, "Screenshot saved to:" lines, agent-context rule
# injections) pass through untouched.
_STATE_TEXT_TRUNCATION_THRESHOLD_CHARS = 4_000

_BROWSER_TOOL_NAME_PREFIX = "browser_"
_BROWSER_GET_STATE_TOOL_NAME = "browser_get_state"

_STALE_SCREENSHOT_PLACEHOLDER = (
    "[Screenshot omitted: this browser snapshot was superseded by a more "
    "recent one later in the conversation. If the observation includes a "
    '"Screenshot saved to" path, the image file is still on disk. Call '
    "browser_get_state with include_screenshot=true to capture the current "
    "page.]"
)


def _stale_state_text_placeholder(omitted_chars: int) -> str:
    return (
        f"\n[... {omitted_chars} characters of stale browser state omitted. "
        "This snapshot was superseded by a newer browser_get_state result "
        "later in the conversation; element indices above are stale and must "
        "not be used. Call browser_get_state again for the current page.]"
    )


def _is_browser_tool_message(message: Message) -> bool:
    return (
        message.role == "tool"
        and message.name is not None
        and message.name.startswith(_BROWSER_TOOL_NAME_PREFIX)
    )


def _strip_images(
    content: Sequence[TextContent | ImageContent],
) -> list[TextContent | ImageContent]:
    return [
        TextContent(text=_STALE_SCREENSHOT_PLACEHOLDER)
        if isinstance(item, ImageContent)
        else item
        for item in content
    ]


def _truncate_state_texts(
    content: Sequence[TextContent | ImageContent],
) -> list[TextContent | ImageContent]:
    rewritten: list[TextContent | ImageContent] = []
    for item in content:
        if (
            isinstance(item, TextContent)
            and len(item.text) > _STATE_TEXT_TRUNCATION_THRESHOLD_CHARS
        ):
            omitted = len(item.text) - STALE_STATE_TEXT_HEAD_CHARS
            rewritten.append(
                item.model_copy(
                    update={
                        "text": item.text[:STALE_STATE_TEXT_HEAD_CHARS]
                        + _stale_state_text_placeholder(omitted)
                    }
                )
            )
        else:
            rewritten.append(item)
    return rewritten


def prune_stale_browser_observations(messages: list[Message]) -> list[Message]:
    """Rewrite stale browser tool messages in a freshly built message list.

    Keeps the most recent ``KEEP_RECENT_BROWSER_SCREENSHOTS`` screenshot-bearing
    browser tool messages and the most recent ``KEEP_RECENT_BROWSER_STATE_TEXTS``
    ``browser_get_state`` messages fully intact; older ones have their images
    replaced with placeholders and their oversized snapshot text truncated.

    Only ``role="tool"`` messages whose tool name starts with ``browser_`` are
    ever touched — user-uploaded images and every other tool result pass
    through unchanged. Returns a new list; input messages are not mutated.
    """
    screenshot_indices: list[int] = []
    state_text_indices: list[int] = []
    for index, message in enumerate(messages):
        if not _is_browser_tool_message(message):
            continue
        if message.contains_image:
            screenshot_indices.append(index)
        if message.name == _BROWSER_GET_STATE_TOOL_NAME:
            state_text_indices.append(index)

    stale_screenshot_indices = set(
        screenshot_indices[
            : max(0, len(screenshot_indices) - KEEP_RECENT_BROWSER_SCREENSHOTS)
        ]
    )
    stale_state_text_indices = set(
        state_text_indices[
            : max(0, len(state_text_indices) - KEEP_RECENT_BROWSER_STATE_TEXTS)
        ]
    )
    if not stale_screenshot_indices and not stale_state_text_indices:
        return messages

    pruned: list[Message] = []
    for index, message in enumerate(messages):
        content: Sequence[TextContent | ImageContent] = message.content
        if index in stale_screenshot_indices:
            content = _strip_images(content)
        if index in stale_state_text_indices:
            content = _truncate_state_texts(content)
        if content is not message.content:
            message = message.model_copy(update={"content": content})
        pruned.append(message)
    return pruned
