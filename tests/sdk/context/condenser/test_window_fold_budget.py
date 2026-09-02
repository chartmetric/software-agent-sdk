"""The fold threshold follows the model's window, not a fixed event count."""

from unittest.mock import MagicMock

from openhands.sdk.context.condenser.llm_summarizing_condenser import (
    CONTEXT_WINDOW_FOLD_FRACTION,
    _window_fold_budget,
)
from openhands.sdk.llm import LLM


def _llm(window: int | None, output: int | None) -> LLM:
    llm = MagicMock(spec=LLM)
    llm.effective_max_input_tokens = window
    llm.effective_max_output_tokens = output
    return llm


def test_a_large_window_folds_late_and_a_small_one_early() -> None:
    """One event count cannot serve both, which is why the budget is derived.

    At roughly 761 tokens an event -- the production median -- the old default
    of 240 events folded near 183k, a fifth of a 922k window and well past a
    128k one.
    """
    large = _window_fold_budget(_llm(922_000, 128_000))
    small = _window_fold_budget(_llm(128_000, 4_096))

    assert large is not None
    assert small is not None
    assert large == int((922_000 - 128_000) * CONTEXT_WINDOW_FOLD_FRACTION)
    assert small == int((128_000 - 4_096) * CONTEXT_WINDOW_FOLD_FRACTION)
    assert small < 128_000, "a fold must happen before the window is full"


def test_the_output_allowance_is_reserved() -> None:
    """A prompt that fills the window leaves no room for the reply.

    The budget comes off what is left after the model's own output allowance,
    so a fold is triggered while the next turn still fits.
    """
    budget = _window_fold_budget(_llm(200_000, 100_000))

    assert budget is not None
    assert budget + 100_000 < 200_000


def test_an_unknown_window_derives_nothing() -> None:
    """Without a window there is no budget, and max_size stays the only trigger.

    That is the behaviour every caller had before a budget could be derived, so
    an unknown window must not start folding on a guess.
    """
    assert _window_fold_budget(None) is None
    assert _window_fold_budget(_llm(None, None)) is None
    assert _window_fold_budget(_llm(1_000, 4_000)) is None, "output exceeds window"
