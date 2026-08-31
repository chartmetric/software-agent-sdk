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
- Page snapshot text: an oversized snapshot in a stale message is cut to a head
  excerpt plus a placeholder explaining that the snapshot was superseded and how
  to re-fetch the current state.

A stale snapshot is recognised by its *shape*, not by the name of the tool that
returned it. That distinction is why this rule needed a second pass: it
originally truncated text only for ``browser_get_state``, which was right while
that was the only tool returning a page snapshot. Making every state-changing
action answer with the page it produced moved the identical payload onto
``browser_click``, ``browser_scroll``, ``browser_navigate``, ``browser_sequence``,
``browser_switch_tab`` and the rest, and this rule did not follow -- measured over
three days of production traffic, 75 MB of browser snapshot text reached the
models and only the 27% of it named ``browser_get_state`` was ever eligible for
truncation. Keying on the shape means a browser tool added later is covered the
day it ships.

The shape is the ``interactive_elements`` key every snapshot carries, because
they are all rendered by one ``_browser_state_payload``. Over those same three
days that key appeared in 1,782 of 1,782 oversized text blocks from the eight
snapshot-returning tools, and in 0 of 204 from ``browser_get_content`` -- whose
text is the answer the agent asked for rather than a view of the page that a
later snapshot replaces. Truncating that would destroy a result nothing
supersedes, which is why the marker gates the rewrite instead of the tool name.

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
from typing import TypeGuard

from openhands.sdk.llm import ImageContent, Message, TextContent


# Number of most-recent screenshot-bearing browser tool messages whose images
# are kept intact. Two covers the common "compare before/after my change"
# verification pattern while dropping everything older.
KEEP_RECENT_BROWSER_SCREENSHOTS = 2

# Number of most-recent page-snapshot browser tool messages whose snapshot text
# is kept intact, counted across every tool that returns one rather than per
# tool name -- the agent acts on the newest snapshot, and which call produced it
# does not change that.
KEEP_RECENT_BROWSER_PAGE_SNAPSHOTS = 2

# Head excerpt preserved from a stale snapshot so the page URL/title context --
# and, for an action that answered with the page it produced, that action's own
# result line -- survives without the full element tree.
STALE_STATE_TEXT_HEAD_CHARS = 2_000

# Only text blocks larger than this are candidates for truncation. Short
# blocks (error headers, "Screenshot saved to:" lines, agent-context rule
# injections) pass through untouched.
_STATE_TEXT_TRUNCATION_THRESHOLD_CHARS = 4_000

_BROWSER_TOOL_NAME_PREFIX = "browser_"

# Every page snapshot is rendered by one `_browser_state_payload`, so they all
# carry this key and nothing else a browser tool returns does. Matching it is
# what keeps a `browser_get_content` answer -- which no later snapshot replaces
# -- out of the rewrite.
_PAGE_SNAPSHOT_MARKER = '"interactive_elements"'

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
        "This snapshot was superseded by a newer one later in the "
        "conversation; element indices above are stale and must not be used. "
        "Act on the most recent browser result, or call browser_get_state for "
        "the current page.]"
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


def _is_page_snapshot_text(
    item: TextContent | ImageContent,
) -> TypeGuard[TextContent]:
    """Whether this block is an oversized page snapshot a newer one replaces."""
    return (
        isinstance(item, TextContent)
        and len(item.text) > _STATE_TEXT_TRUNCATION_THRESHOLD_CHARS
        and _PAGE_SNAPSHOT_MARKER in item.text
    )


def _truncate_state_texts(
    content: Sequence[TextContent | ImageContent],
) -> list[TextContent | ImageContent]:
    rewritten: list[TextContent | ImageContent] = []
    for item in content:
        if _is_page_snapshot_text(item):
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
    browser tool messages and the most recent
    ``KEEP_RECENT_BROWSER_PAGE_SNAPSHOTS`` snapshot-bearing ones fully intact;
    older ones have their images replaced with placeholders and their oversized
    snapshot text truncated.

    Only ``role="tool"`` messages whose tool name starts with ``browser_`` are
    ever touched, and within those only blocks carrying a page snapshot — so a
    ``browser_get_content`` answer, a user-uploaded image and every other tool
    result pass through unchanged. Returns a new list; input messages are not
    mutated.
    """
    screenshot_indices: list[int] = []
    snapshot_indices: list[int] = []
    for index, message in enumerate(messages):
        if not _is_browser_tool_message(message):
            continue
        if message.contains_image:
            screenshot_indices.append(index)
        if any(_is_page_snapshot_text(item) for item in message.content):
            snapshot_indices.append(index)

    stale_screenshot_indices = set(
        screenshot_indices[
            : max(0, len(screenshot_indices) - KEEP_RECENT_BROWSER_SCREENSHOTS)
        ]
    )
    stale_snapshot_indices = set(
        snapshot_indices[
            : max(0, len(snapshot_indices) - KEEP_RECENT_BROWSER_PAGE_SNAPSHOTS)
        ]
    )
    if not stale_screenshot_indices and not stale_snapshot_indices:
        return messages

    pruned: list[Message] = []
    for index, message in enumerate(messages):
        content: Sequence[TextContent | ImageContent] = message.content
        if index in stale_screenshot_indices:
            content = _strip_images(content)
        if index in stale_snapshot_indices:
            content = _truncate_state_texts(content)
        if content is not message.content:
            message = message.model_copy(update={"content": content})
        pruned.append(message)
    return pruned
