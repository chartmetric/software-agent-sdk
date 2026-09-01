"""One call runs several browser steps, and says where it stopped.

Measured on conversation c3afdee0: 125 browser calls cost 105.9 seconds of
browser work and roughly 20.8 minutes of model round trips, in 24 runs of
consecutive browser-only steps. Collapsing those runs is worth ~101 steps; the
browser work inside them is worth ~1.8 minutes. So the sequence exists to remove
round trips, and everything here is about it staying as diagnosable as the calls
it replaces.
"""

from __future__ import annotations

from openhands.tools.browser_use.definition import (
    BrowserGetStateAction,
    BrowserNavigateAction,
    BrowserObservation,
    BrowserSequenceAction,
    BrowserSequenceStep,
)
from openhands.tools.browser_use.impl import BrowserToolExecutor


class _RecordingExecutor(BrowserToolExecutor):
    """Runs the real sequence logic against canned per-step results."""

    def __init__(self, results: list[BrowserObservation]) -> None:
        self.calls: list[object] = []
        self.automatic_state_suppressed: list[bool] = []
        self._results = list(results)
        self.full_output_save_dir = None

    def __call__(self, action, conversation=None):  # type: ignore[override]
        self.calls.append(action)
        self.automatic_state_suppressed.append(
            getattr(self, "_sequence_suppresses_automatic_state", False)
        )
        return self._results.pop(0)


def _ok(text: str, screenshot: str | None = None) -> BrowserObservation:
    observation = BrowserObservation.from_text(text=text)
    if screenshot:
        observation = observation.model_copy(update={"screenshot_data": screenshot})
    return observation


def _fail(text: str) -> BrowserObservation:
    return BrowserObservation.from_text(text=text, is_error=True)


def test_each_step_is_dispatched_as_the_action_its_own_tool_would_build() -> None:
    """A step must behave exactly as the standalone call it replaces.

    Validated by the concrete action class, not by the sequence's own schema, so
    a bad argument is rejected the same way and secret handling downstream still
    sees the type it expects.
    """
    executor = _RecordingExecutor([_ok("navigated"), _ok("state")])

    result = executor._run_sequence(
        BrowserSequenceAction(
            steps=[
                BrowserSequenceStep(
                    action="navigate", arguments={"url": "https://example.com"}
                ),
                BrowserSequenceStep(action="get_state", arguments={}),
            ]
        ),
        None,
    )

    assert isinstance(executor.calls[0], BrowserNavigateAction)
    assert executor.calls[0].url == "https://example.com"
    assert isinstance(executor.calls[1], BrowserGetStateAction)
    assert not result.is_error
    assert "navigated" in result.text and "state" in result.text


def test_a_failing_step_stops_the_run_and_names_what_was_not_run() -> None:
    """A batch that hides where it broke is worse than the calls it replaced."""
    executor = _RecordingExecutor([_ok("navigated"), _fail("element 4 not found")])

    result = executor._run_sequence(
        BrowserSequenceAction(
            steps=[
                BrowserSequenceStep(
                    action="navigate", arguments={"url": "https://example.com"}
                ),
                BrowserSequenceStep(action="click", arguments={"index": 4}),
                BrowserSequenceStep(action="get_state", arguments={}),
                BrowserSequenceStep(action="get_content", arguments={}),
            ]
        ),
        None,
    )

    assert result.is_error
    # Which step, by position and name.
    assert "step 2 (click)" in result.text
    assert "element 4 not found" in result.text
    # And that the rest did not silently not happen.
    assert "remaining 2 step(s) were not run" in result.text
    # It really did stop: get_state and get_content were never dispatched.
    assert len(executor.calls) == 2


def test_rejected_arguments_fail_that_step_without_touching_the_browser() -> None:
    """A malformed step is caught before it reaches the session."""
    executor = _RecordingExecutor([])

    result = executor._run_sequence(
        BrowserSequenceAction(
            steps=[
                BrowserSequenceStep(action="navigate", arguments={"not_a_url_field": 1})
            ]
        ),
        None,
    )

    assert result.is_error
    assert "step 1 (navigate)" in result.text
    assert executor.calls == []


def test_an_unknown_action_lists_the_ones_that_exist() -> None:
    """The refusal has to name the remedy, not only the condition."""
    executor = _RecordingExecutor([])

    result = executor._run_sequence(
        BrowserSequenceAction(
            steps=[BrowserSequenceStep(action="teleport", arguments={})]
        ),
        None,
    )

    assert result.is_error
    assert "unknown action" in result.text
    assert "navigate" in result.text and "get_state" in result.text


def test_the_screenshot_is_the_state_the_sequence_ended_on() -> None:
    """A sequence ends in one browser state; publishing any earlier frame would
    attach a page the run never finished on."""
    executor = _RecordingExecutor(
        [_ok("first", screenshot="AAA"), _ok("second", screenshot="BBB")]
    )

    result = executor._run_sequence(
        BrowserSequenceAction(
            steps=[
                BrowserSequenceStep(
                    action="navigate", arguments={"url": "https://example.com"}
                ),
                BrowserSequenceStep(action="get_state", arguments={}),
            ]
        ),
        None,
    )

    assert result.screenshot_data == "BBB"


def test_sequence_reads_one_final_state_instead_of_serializing_every_step() -> None:
    executor = _RecordingExecutor(
        [_ok("navigated"), _ok("clicked"), _ok("final state", screenshot="END")]
    )

    result = executor._run_sequence(
        BrowserSequenceAction(
            steps=[
                BrowserSequenceStep(
                    action="navigate", arguments={"url": "https://example.com"}
                ),
                BrowserSequenceStep(action="click", arguments={"index": 2}),
            ]
        ),
        None,
    )

    assert [type(action).__name__ for action in executor.calls] == [
        "BrowserNavigateAction",
        "BrowserClickAction",
        "BrowserGetStateAction",
    ]
    assert executor.automatic_state_suppressed == [True, True, False]
    assert result.screenshot_data == "END"
    assert "final state" in result.text
