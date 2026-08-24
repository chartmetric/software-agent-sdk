"""Every browser tool must lock the one browser session they all drive.

Without a shared key the executor falls back to a per-tool mutex named after the
tool, which serializes two `browser_get_state` calls and lets `browser_navigate`
run concurrently with `browser_get_state` on the same session. Measured over 7
days: 4 steps in 15,610 emitted two different browser tools in one step. The
failure mode is a screenshot of the wrong page, and screenshots are published as
evidence, so a rare wrong answer is worse than a common slow one.
"""

from __future__ import annotations

import inspect

import pytest

from openhands.tools.browser_use import definition as browser_definition
from openhands.tools.browser_use.definition import BrowserAction


def _tool_classes() -> list[type]:
    return [
        obj
        for _, obj in inspect.getmembers(browser_definition, inspect.isclass)
        if obj.__name__.startswith("Browser")
        and obj.__name__.endswith("Tool")
        and obj.__module__ == browser_definition.__name__
    ]


def test_every_browser_tool_declares_the_same_session_resource() -> None:
    """The keys must be equal across tools, not merely present on each.

    Equality is the whole property: distinct keys would lock each tool against
    itself and leave two different tools free to run together, which is exactly
    the state this replaced.
    """
    classes = _tool_classes()
    assert classes, "no browser tool definitions found"

    keysets = {}
    for cls in classes:
        resources = cls.declared_resources(cls, BrowserAction())  # type: ignore[arg-type]
        assert resources.declared, f"{cls.__name__} did not declare its resources"
        keysets[cls.__name__] = resources.keys

    # Deliberately not compared against an imported constant: that would fail
    # by ImportError before the fix, which proves a name exists rather than a
    # behaviour. The property is that the tools agree, and that they claim to.
    assert len(set(keysets.values())) == 1, keysets
    shared = next(iter(set(keysets.values())))
    assert shared, "tools agreed on having no lock at all, which is the bug"


@pytest.mark.parametrize(
    "left,right",
    [("BrowserNavigateTool", "BrowserGetStateTool")],
)
def test_the_two_tools_that_raced_now_share_a_lock(left: str, right: str) -> None:
    """The specific pair observed racing in production."""
    by_name = {cls.__name__: cls for cls in _tool_classes()}
    a = by_name[left].declared_resources(by_name[left], BrowserAction())  # type: ignore[arg-type]
    b = by_name[right].declared_resources(by_name[right], BrowserAction())  # type: ignore[arg-type]

    assert a.keys == b.keys
    # A shared key only serializes them if the executor is told to trust it.
    assert a.declared and b.declared
