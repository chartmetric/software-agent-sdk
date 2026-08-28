"""Superseded terminal output keeps both ends; recent output is untouched."""

from openhands.sdk.agent.terminal_context_pruning import (
    KEEP_RECENT_TERMINAL_OUTPUTS,
    STALE_TERMINAL_TAIL_CHARS,
    prune_stale_terminal_observations,
)
from openhands.sdk.llm import Message, TextContent


def _text_of(message: Message) -> str:
    """The message's single text block, narrowed for the type checker."""
    block = message.content[0]
    assert isinstance(block, TextContent)
    return block.text


def _terminal(text: str) -> Message:
    return Message(role="tool", name="terminal", content=[TextContent(text=text)])


def _output(marker: str) -> str:
    return f"$ pytest\n{marker}-head\n" + ("noise\n" * 3_000) + f"{marker}-FAILED"


def test_recent_output_is_left_byte_identical() -> None:
    messages = [_terminal(_output(str(i))) for i in range(KEEP_RECENT_TERMINAL_OUTPUTS)]

    pruned = prune_stale_terminal_observations(messages)

    assert [_text_of(m) for m in pruned] == [_text_of(m) for m in messages]


def test_a_superseded_output_keeps_its_tail_not_just_its_head() -> None:
    """The tail is where the exit status and the failure live.

    The browser rule keeps a head only, which for a command would discard
    exactly the part worth reading.
    """
    messages = [_terminal(_output("old"))] + [
        _terminal(_output(str(i))) for i in range(KEEP_RECENT_TERMINAL_OUTPUTS)
    ]

    pruned = prune_stale_terminal_observations(messages)

    stale = _text_of(pruned[0])
    assert len(stale) < len(_text_of(messages[0]))
    assert "$ pytest" in stale, "the command echo survives"
    assert "old-FAILED" in stale, "the failure at the end survives"
    assert "omitted" in stale
    assert len(stale) < STALE_TERMINAL_TAIL_CHARS * 4


def test_other_tools_are_not_touched() -> None:
    """Only terminal messages are in scope; a big search result is left alone."""
    search = Message(
        role="tool",
        name="search_organization_code",
        content=[TextContent(text="x" * 40_000)],
    )
    messages = [search] + [
        _terminal(_output(str(i))) for i in range(KEEP_RECENT_TERMINAL_OUTPUTS)
    ]

    pruned = prune_stale_terminal_observations(messages)

    assert _text_of(pruned[0]) == _text_of(search)
