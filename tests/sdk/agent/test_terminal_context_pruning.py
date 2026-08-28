"""Superseded terminal output keeps both ends; recent output is untouched."""

from openhands.sdk.agent.terminal_context_pruning import (
    KEEP_RECENT_TERMINAL_OUTPUTS,
    STALE_TERMINAL_TAIL_CHARS,
    prune_stale_terminal_observations,
)
from openhands.sdk.llm import Message, TextContent


def _terminal(text: str) -> Message:
    return Message(role="tool", name="terminal", content=[TextContent(text=text)])


def _output(marker: str) -> str:
    return f"$ pytest\n{marker}-head\n" + ("noise\n" * 3_000) + f"{marker}-FAILED"


def test_recent_output_is_left_byte_identical() -> None:
    messages = [_terminal(_output(str(i))) for i in range(KEEP_RECENT_TERMINAL_OUTPUTS)]

    pruned = prune_stale_terminal_observations(messages)

    assert [m.content[0].text for m in pruned] == [m.content[0].text for m in messages]


def test_a_superseded_output_keeps_its_tail_not_just_its_head() -> None:
    """The tail is where the exit status and the failure live.

    The browser rule keeps a head only, which for a command would discard
    exactly the part worth reading.
    """
    messages = [_terminal(_output("old"))] + [
        _terminal(_output(str(i))) for i in range(KEEP_RECENT_TERMINAL_OUTPUTS)
    ]

    pruned = prune_stale_terminal_observations(messages)

    stale = pruned[0].content[0].text
    assert len(stale) < len(messages[0].content[0].text)
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

    assert pruned[0].content[0].text == search.content[0].text
