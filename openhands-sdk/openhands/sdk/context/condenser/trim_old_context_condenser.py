"""Old tool output and old thinking, kept as what they were rather than as all.

Deterministic, and cheaper than a summary: it spends no model call and keeps
every event where it was, shortened. Chain it before a summarising condenser --
where trimming is enough to get back under budget, the summary never happens
and the detail it would have replaced is still there.

A run pays for its whole history on every model call it makes. Measured over
one production day (2026-09-03, 60 conversations, $38.79): the average call
carried 163,000 tokens whether the session made 31 calls or 156, against a
fixed floor of about 12,000 for the system prompt and tools. The rest is
observations -- a code search returns 14-37KB, a file read returns the file --
and every one of them is re-sent, whole, for the rest of the run.

The rule it applies: a run re-reads what it needs -- that is what its tools are
for -- so an old result is kept as evidence that it happened and what it was
about, rather than as its full text.

**Why this fires in blocks rather than on every call.** Trimming the oldest
observation each step would rewrite the prompt prefix each step, and the prefix
is what the provider's cache is keyed on -- production runs currently read
88-98% of their input from cache, so a rolling trim would trade a smaller
prompt for an uncached one and cost more than it saves. It therefore behaves
like a fold: nothing happens until the view crosses the same token budget the
condenser folds at, and then a block of old observations is trimmed at once.
The prefix changes once, the same way a summary changes it once.

It runs *before* the summarizing condenser in a pipeline. Where trimming is
enough to get back under budget, the summary -- a model call, and the loss of
the detail it replaces -- does not happen at all.
"""

from __future__ import annotations

from pydantic import Field

from openhands.sdk.context.condenser.base import PipelinableCondenserBase
from openhands.sdk.context.condenser.utils import get_total_token_count
from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import LLM, TextContent


# Reasoning fields, which the provider is sent again on every call.
#
# `send_reasoning_content` is on for the deepseek-v4-pro family, which is what
# this deployment runs for its general, exploration and coding roles, so every
# prior turn's full thinking is part of every later prompt. Measured on
# conversation 3c4a6071 (2026-09-03): 242KB of `reasoning_content` against
# 234KB of *all* tool output -- the largest single thing in the context, and
# larger than the results the run actually read.
#
# The recent ones earn their place; that is what interleaved thinking is for.
# The older ones are a transcript of how the run reached somewhere it has
# already arrived. The replayed copy of a conversation drops them outright for
# a different reason -- a provider binds them to the endpoint that made them --
# so nothing here is new about how they age.
REASONING_FIELDS = ("reasoning_content", "thinking_blocks", "responses_reasoning_item")

# What one superseded observation may still weigh.
#
# Measured against conversation efe3bd98 (2026-09-03, 391 calls, $7.26) by
# replaying its own events under this rule: keeping 4,000 characters of each
# old result -- the number the replayed copy of a conversation uses -- takes the
# average context from 179,000 tokens to 142,000, a fifth. Keeping 1,500 takes
# it to 119,000, a third, and 800 would take it to 105,000. The number of folds
# barely moves across that range (four to six), so this is bought with detail
# rather than with cache churn.
#
# 1,500 is where the trade stops being free: it is still the head of a table or
# the first lines of a file, enough to say what the result was and whether it is
# worth reading again, and the run can always read it again. Below that the
# entry stops carrying its own meaning.
MAX_OBSERVATION_CHARS = 1_500

# Observations near the end are the ones the next step is reasoning about, so
# they are never trimmed. Six covers a read-and-act cycle with room to spare.
KEEP_RECENT_EVENTS = 6

# How much a trim has to be worth before it is taken.
#
# This is what keeps the trim rare, and rare is what makes it pay. Editing an
# event inside the prompt invalidates the provider's cache from that event
# onward, so a trim is only free if it happens seldom: cutting one newly-aged
# observation every call would re-send the tail every call and spend more than
# the smaller context saves. Requiring a large yield batches the work into a
# few big folds over a run -- after one, it takes many calls to accumulate
# another 20,000 tokens of superseded output -- and each is worth many times
# what it costs.
MIN_TRIM_YIELD_TOKENS = 20_000
_CHARS_PER_TOKEN = 4

_TRIM_NOTE = (
    "\n\n[Older result trimmed to its first {kept:,} characters to keep this "
    "conversation inside its context budget. The full result is not gone from "
    "the record -- call the same tool again if you need the rest.]"
)


def _trimmed(text: str) -> str:
    return text[:MAX_OBSERVATION_CHARS] + _TRIM_NOTE.format(kept=MAX_OBSERVATION_CHARS)


class TrimOldContext(PipelinableCondenserBase):
    """Shorten superseded tool output once the view is over budget."""

    max_tokens: int = Field(default=60_000, gt=0)
    """The context this trim holds a run near. Above it, old results are cut.

    Well below the condenser's fold budget, and that is the point. What a call
    is billed for is mostly its *cached* prefix: at a provider price of a tenth
    for cached input, a 163,000-token context costs about 18,000 effective
    tokens a call against roughly 2,000 of genuinely new text, so the prefix is
    nine tenths of the bill and shrinking it is nearly all of the saving. A
    threshold set at the fold budget would never fire for the ordinary session,
    which sits at 90,000-140,000 and pays that tax on every call without ever
    being large enough to fold.

    A trim rewrites the prefix once, so it costs one uncached pass over what is
    left -- about 60,000 tokens, or eight calls' worth -- and then the run is
    back to paying a tenth of a much smaller number. It pays for itself within
    the first ten calls after it fires, and a run short enough that it never
    fires was never expensive.
    """

    def condense(self, view: View, agent_llm: LLM | None = None) -> View | Condensation:
        events = list(view.events)
        if agent_llm is None or len(events) <= KEEP_RECENT_EVENTS:
            return view
        if get_total_token_count(events, agent_llm) <= self.max_tokens:
            return view
        cutoff = len(events) - KEEP_RECENT_EVENTS
        if self._yield_of(events, cutoff) < MIN_TRIM_YIELD_TOKENS:
            # Above budget, but not by enough of *this* kind of weight to be
            # worth invalidating the cache for. The condenser behind this one
            # still folds if the view keeps growing.
            return view
        trimmed_events = []
        for index, event in enumerate(events):
            if index >= cutoff:
                trimmed_events.append(event)
                continue
            updates: dict = {
                field: None for field in REASONING_FIELDS if getattr(event, field, None)
            }
            observation = getattr(event, "observation", None)
            content = getattr(observation, "content", None)
            if observation is None or not isinstance(content, list):
                trimmed_events.append(
                    event.model_copy(update=updates) if updates else event
                )
                continue
            total = sum(
                len(block.text) for block in content if isinstance(block, TextContent)
            )
            if total <= MAX_OBSERVATION_CHARS:
                trimmed_events.append(
                    event.model_copy(update=updates) if updates else event
                )
                continue
            shortened = [
                block.model_copy(update={"text": _trimmed(block.text)})
                if isinstance(block, TextContent)
                else block
                for block in content
            ]
            updates["observation"] = observation.model_copy(
                update={"content": shortened}
            )
            trimmed_events.append(event.model_copy(update=updates))
        return View(
            events=trimmed_events,
            unhandled_condensation_request=view.unhandled_condensation_request,
        )

    @staticmethod
    def _yield_of(events: list, cutoff: int) -> int:
        """Tokens a trim would free, counted the way the trim would cut."""
        freed = 0
        for event in events[:cutoff]:
            freed += _reasoning_weight(event) // _CHARS_PER_TOKEN
            observation = getattr(event, "observation", None)
            content = getattr(observation, "content", None)
            if not isinstance(content, list):
                continue
            total = sum(
                len(block.text) for block in content if isinstance(block, TextContent)
            )
            if total > MAX_OBSERVATION_CHARS:
                freed += (total - MAX_OBSERVATION_CHARS) // _CHARS_PER_TOKEN
        return freed


def _reasoning_weight(event) -> int:
    """Characters of thinking this event would send again."""
    total = 0
    for field in REASONING_FIELDS:
        value = getattr(event, field, None)
        if not value:
            continue
        total += len(value) if isinstance(value, str) else len(str(value))
    return total
