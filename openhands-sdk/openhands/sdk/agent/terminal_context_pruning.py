"""Shrink superseded terminal output before a request is sent.

Terminal output is the largest thing an execution run puts in front of the
model and the only large one nothing bounded. Measured across 13 production
execution conversations with no forked ancestor: after browser pruning runs,
terminal observations are 1,020,569 of the 2,523,242 observation tokens the
model actually sees -- 40.4%, ahead of browser (33%) and code search (21%).
Every one of them is re-sent on every later step.

Why the browser rule is not reused as-is. A superseded browser snapshot is
stale by definition: the page moved on, and the element indices in it are
actively wrong. Terminal output is not -- a test failure from fifteen steps
ago is often the thing being fixed now. So this keeps far more of them whole
(ten against two) and, where it does truncate, keeps the **tail** as well as
the head. For a command it is the tail that carries the exit status, the
error and the summary line, while the head is usually setup noise; keeping
only a head, as the browser rule does, would discard exactly the part worth
reading.
"""

from collections.abc import Sequence

from openhands.sdk.llm import ImageContent, Message, TextContent


# Number of most-recent terminal tool messages kept whole. Ten rather than the
# browser rule's two because old command output stays relevant; measured, it
# still removes 60% of terminal tokens, against 77% at two and 49% at twenty.
KEEP_RECENT_TERMINAL_OUTPUTS = 10

# Kept from the start of a stale output: the command echo and the first lines
# of what it printed.
STALE_TERMINAL_HEAD_CHARS = 1_000

# Kept from the end: exit status, traceback, failure summary. Larger than the
# head on purpose.
STALE_TERMINAL_TAIL_CHARS = 2_000

# Only blocks with something to gain are touched, so short outputs pass
# through byte-identical.
_TERMINAL_TRUNCATION_THRESHOLD_CHARS = 6_000

_TERMINAL_TOOL_NAMES = frozenset({"terminal", "execute_bash", "bash"})


def _stale_terminal_placeholder(omitted_chars: int) -> str:
    return (
        f"\n[... {omitted_chars} characters of earlier terminal output omitted "
        "to keep the conversation readable. The head and tail of this command's "
        "output are shown; re-run the command if the middle matters.]\n"
    )


def _is_terminal_tool_message(message: Message) -> bool:
    return (
        message.role == "tool"
        and message.name is not None
        and message.name in _TERMINAL_TOOL_NAMES
    )


def _truncate_middle(
    content: Sequence[TextContent | ImageContent],
) -> list[TextContent | ImageContent]:
    rewritten: list[TextContent | ImageContent] = []
    for item in content:
        if (
            isinstance(item, TextContent)
            and len(item.text) > _TERMINAL_TRUNCATION_THRESHOLD_CHARS
        ):
            head = item.text[:STALE_TERMINAL_HEAD_CHARS]
            tail = item.text[-STALE_TERMINAL_TAIL_CHARS:]
            omitted = len(item.text) - len(head) - len(tail)
            rewritten.append(
                item.model_copy(
                    update={"text": head + _stale_terminal_placeholder(omitted) + tail}
                )
            )
        else:
            rewritten.append(item)
    return rewritten


def prune_stale_terminal_observations(messages: list[Message]) -> list[Message]:
    """Rewrite superseded terminal output in a freshly built message list.

    Keeps the most recent ``KEEP_RECENT_TERMINAL_OUTPUTS`` terminal messages
    untouched and replaces the middle of older large ones with a marker,
    preserving both ends. Returns a new list; the input is not mutated.
    """
    indices = [i for i, m in enumerate(messages) if _is_terminal_tool_message(m)]
    stale = set(indices[: max(0, len(indices) - KEEP_RECENT_TERMINAL_OUTPUTS)])
    if not stale:
        return messages

    pruned = list(messages)
    for index in stale:
        message = pruned[index]
        pruned[index] = message.model_copy(
            update={"content": _truncate_middle(message.content)}
        )
    return pruned
