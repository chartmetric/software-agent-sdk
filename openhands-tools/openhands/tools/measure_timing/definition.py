"""Time something the running system actually does, and return the numbers.

Pilot's sandbox agent answers "make this faster" from source: 180 terminal calls
and 0 timing commands in one production conversation, 32 and 0 in another after
a policy that spelled out the exact `curl -w '%{time_total}'` invocation. One
run announced "I completed the measurement and diagnosis" with nothing in it
that reported a duration.

This tool is not here because the shell cannot measure -- two controlled runs
against a service with a real bottleneck showed an agent with only terminal and
file_editor measuring it correctly, including finding the port itself and
comparing the slow route against a control route. It is here because a duration
deserves to come back structured rather than parsed out of curl output: the
first hit separated from the steady state, exit codes beside the numbers, a run
that never completed reported as blocked rather than as a large duration, and
the condition recorded so a later comparison is against the same thing.
"""

import asyncio
import statistics
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator
from rich.text import Text

from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.logger import get_logger
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


logger = get_logger(__name__)


class MeasurementBlocked(Exception):
    """The run did not complete, so there is no duration to report."""


MAX_REPEAT = 20
# More than a handful stops being a comparison and becomes a survey.
MAX_TARGETS = 6
# Below this the numbers do not single anything out, and saying they do is the
# attribution error this comparison exists to prevent.
DOMINANT_SHARE_PCT = 60.0
DEFAULT_REPEAT = 5
REQUEST_TIMEOUT_SECONDS = 60.0
COMMAND_TIMEOUT_SECONDS = 600.0
MAX_BODY_SAMPLE = 400
# A first run this many times the median of the rest is a cold start being
# measured, not the path. Observed in production: a dev server still compiling
# answered the request, so the number looked like a measurement and was not one.
COLD_FIRST_RUN_RATIO = 3.0
# ...and by at least this much in absolute terms, so ordinary jitter on a fast
# target does not read as a warm-up.
COLD_FIRST_RUN_FLOOR_MS = 50.0
# A sign-in wall answers instantly, so timing it produces a small, confident,
# and entirely false number for the page behind it. Measured: an auth-gated
# endpoint taking 745ms reported a 1ms median as a bare 302.
AUTH_STATUSES = frozenset({401, 403})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
AUTH_PATH_HINTS = ("login", "signin", "sign-in", "auth", "sso", "session", "account")

MEASURE_TIMING_DESCRIPTION = """Time something the running system actually does, and return the observed durations.

Use this whenever a conclusion depends on how long something takes: that a path is
slow, that it got faster, that a change reduced cost. Reading the source tells you
what work a path performs and never how long that work takes, so a diagnosis drawn
from source alone is not a measurement however plausible it reads.

Give `url` to time a request, or `command` to time anything the shell can run -- a
query, a build, a script, a test. Give `targets` instead when the question is
*which* of several things is the cost: a slow page is not a cause, so time the page
document and the requests it issues together and the shares say whether the server
or the page is the answer. Measuring is not attributing -- one number tells you
something is slow, and a comparison tells you what to change. If the target sits behind a sign-in, send the
session already established as a Cookie or Authorization header: a sign-in wall
answers immediately, so timing it yields a small confident number that describes
the wall and not the page. Time the request the page actually issues rather
than the page wrapping it. The
service's own port is in `/tmp/openhands-runtime-services/<id>.port` when the
runtime is managed here, so a loopback URL reaches it without going through the
preview proxy. An address outside the sandbox is timed the same way, with whatever
credential the session already established passed as a header.

Repeat enough times that one outlier cannot carry the result, and give
`reset_command` when the number depends on a starting condition -- a cold cache, a
cleared key -- so each run measures the same thing. Measure before the change and
again after, and report both."""  # noqa: E501


class TimingComparisonRow(BaseModel):
    """One target's result inside a comparison."""

    label: str
    median_ms: float
    share_pct: float
    note: str = ""


class TimingTarget(BaseModel):
    """One labelled thing to time, so several can be compared in one call."""

    label: str = Field(
        description=(
            "What this target is, in a few words -- 'page document', "
            "'artists API', 'render after data'. The labels are what make the "
            "comparison readable."
        )
    )
    url: str | None = Field(default=None, description="Absolute URL to request.")
    command: str | None = Field(
        default=None, description="Shell command to time instead of a request."
    )

    @model_validator(mode="after")
    def _one_target(self) -> "TimingTarget":
        if bool(self.url) == bool(self.command):
            raise ValueError(
                f"target {self.label!r}: give exactly one of `url` or `command`"
            )
        return self

    @property
    def target(self) -> str:
        return self.url or self.command or ""


class MeasureTimingAction(Action):
    """Request timings for one URL, repeated, optionally with a reset between runs."""

    url: str | None = Field(
        default=None,
        description=(
            "Absolute URL to request, for example "
            "http://localhost:12000/audience-research?ages=65%2B. Use the port from "
            "/tmp/openhands-runtime-services/<id>.port for a managed service. Give "
            "either this or `command`."
        ),
    )
    command: str | None = Field(
        default=None,
        description=(
            "Shell command to time instead of a request -- a query, a build, a test, "
            "a script. Its exit codes come back with the durations, because a command "
            "that failed fast is not a fast command."
        ),
    )
    method: str = Field(
        default="GET",
        description="HTTP method. GET unless the request being timed is not a GET.",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Headers to send, for example an Authorization or Cookie header carrying "
            "the session already established in the browser."
        ),
    )
    body: str | None = Field(
        default=None,
        description="Request body, when timing a method that sends one.",
    )
    repeat: int = Field(
        default=DEFAULT_REPEAT,
        description=(
            f"How many times to issue the request, 1 to {MAX_REPEAT}. More than one "
            "so a single outlier cannot carry the result."
        ),
    )
    reset_command: str | None = Field(
        default=None,
        description=(
            "Shell command run before each request to re-establish the starting "
            "condition, for example clearing a cache. Give this whenever the duration "
            "depends on that condition, or every run after the first measures a warm "
            "path instead."
        ),
    )
    targets: list[TimingTarget] = Field(
        default_factory=list,
        description=(
            "Two or more labelled things to time in one call, when the question is "
            "which of them is the cost. A slow page is not a cause: time the page "
            "document and the requests it issues, and the comparison says whether "
            "the server or the page is the answer. Bounded at "
            f"{MAX_TARGETS}. Use this instead of `url`/`command` when comparing."
        ),
    )
    condition: str | None = Field(
        default=None,
        description=(
            'What condition these numbers describe, in a few words -- "cold cache, '
            'ages=65+ filter". Recorded with the result so a later comparison is '
            "against the same thing."
        ),
    )

    @field_validator("repeat")
    @classmethod
    def _bounded_repeat(cls, value: int) -> int:
        return max(1, min(MAX_REPEAT, value))

    @model_validator(mode="after")
    def _one_target(self) -> "MeasureTimingAction":
        if self.targets:
            if self.url or self.command:
                raise ValueError(
                    "Give `targets` or a single `url`/`command`, not both."
                )
            if len(self.targets) < 2:
                raise ValueError(
                    "`targets` is for comparing, so it needs at least two. Use "
                    "`url` or `command` to time one thing."
                )
            if len(self.targets) > MAX_TARGETS:
                raise ValueError(f"At most {MAX_TARGETS} targets in one call.")
            return self
        if bool(self.url) == bool(self.command):
            raise ValueError(
                "Give exactly one of `url` or `command`, or two or more `targets` "
                "when the question is which of them is the cost."
            )
        return self

    @property
    def target(self) -> str:
        return self.url or self.command or ""

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Measuring ", style="bold")
        if self.url:
            content.append(f"{self.method} {self.url}\n")
        else:
            content.append(f"$ {self.command}\n")
        content.append(f"  {self.repeat}x")
        if self.reset_command:
            content.append(", resetting between runs")
        if self.condition:
            content.append(f" — {self.condition}")
        return content


class MeasureTimingObservation(Observation):
    """The observed durations, which are the evidence a timing claim needs."""

    target: str = Field(default="")
    target_kind: str = Field(default="request")
    condition: str | None = Field(default=None)
    durations_ms: list[float] = Field(default_factory=list)
    statuses: list[int] = Field(
        default_factory=list,
        description="HTTP status per run, or exit code per run for a command.",
    )
    median_ms: float = Field(default=0.0)
    min_ms: float = Field(default=0.0)
    max_ms: float = Field(default=0.0)
    reset_command: str | None = Field(default=None)
    reset_failures: int = Field(default=0)
    error: str | None = Field(default=None)
    body_sample: str | None = Field(default=None)
    first_hit_ms: float | None = Field(
        default=None,
        description=(
            "The very first request, which alone pays connection setup and any "
            "one-time work the target does. Excluded from the statistics below."
        ),
    )
    cold_first_run: bool = Field(
        default=False,
        description="The first hit dominated the steady state.",
    )
    comparison: list[TimingComparisonRow] = Field(
        default_factory=list,
        description=(
            "One entry per labelled target: label, median_ms, and its share of the "
            "total. Present only for a multi-target call."
        ),
    )
    auth_blocked_runs: int = Field(
        default=0,
        description="Runs answered by a sign-in wall rather than the target.",
    )
    auth_redirect_to: str = Field(
        default="", description="Where the sign-in wall sent the request."
    )

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        """The durations, as the LLM reads them.

        This override is load-bearing rather than cosmetic: the tool message is
        built from `to_llm_content`, so a subclass that carries its result in
        its own fields and leaves `content` empty sends an empty tool message.
        The prompt-cache path then indexes `content[-1]` and raises, which ends
        the whole conversation rather than just the observation.
        """
        return [TextContent(text=self.summary)]

    @property
    def summary(self) -> str:
        if self.error is not None:
            return (
                f"Could not time {self.target}: {self.error}. This is a blocked "
                "measurement, not a slow result -- do not report a duration."
            )
        if self.comparison:
            rows = sorted(self.comparison, key=lambda r: r.median_ms, reverse=True)
            lines = [f"condition: {self.condition or 'not stated'}"]
            for row in rows:
                line = (
                    f"  {row.label}: {row.median_ms:.0f}ms"
                    f"  ({row.share_pct:.0f}% of the measured total)"
                )
                if row.note:
                    line += f"  [{row.note}]"
                lines.append(line)
            top = rows[0]
            if len(rows) > 1 and top.share_pct >= DOMINANT_SHARE_PCT:
                lines.append(
                    f"  {top.label} is {top.share_pct:.0f}% of the total, so "
                    "that is where the cost is. Changing anything else leaves it in "
                    "place."
                )
            else:
                lines.append(
                    "  No single target dominates, so naming one of these as the "
                    "cause is not supported by these numbers."
                )
            return "\n".join(lines)
        if self.auth_blocked_runs and self.auth_blocked_runs == len(self.durations_ms):
            where = f" to {self.auth_redirect_to}" if self.auth_redirect_to else ""
            return (
                f"Did not measure {self.target}: every run was answered by a "
                f"sign-in wall{where} (status "
                f"{sorted(set(self.statuses))}). A sign-in wall answers "
                "immediately, so these durations describe the wall and not the "
                "page behind it -- do not report them as the page's latency. "
                "Send the session the browser already holds as a Cookie or "
                "Authorization header, or measure the request the page issues "
                "rather than the page."
            )
        runs = ", ".join(f"{value:.0f}ms" for value in self.durations_ms)
        label = "status codes" if self.target_kind == "request" else "exit codes"
        lines = [
            f"{self.target}",
            f"  condition: {self.condition or 'not stated'}",
            f"  runs: {runs}",
            f"  median {self.median_ms:.0f}ms  min {self.min_ms:.0f}ms  "
            f"max {self.max_ms:.0f}ms",
            f"  {label}: {sorted(set(self.statuses))}",
        ]
        if self.reset_command:
            lines.append(
                f"  reset before each run: {self.reset_command}"
                + (f" ({self.reset_failures} failed)" if self.reset_failures else "")
            )
        else:
            lines.append(
                "  no reset between runs, so every run after the first measured a "
                "warm path"
            )
        if self.body_sample:
            lines.append(f"  first run output began: {self.body_sample}")
        if self.auth_blocked_runs:
            lines.append(
                f"  {self.auth_blocked_runs} of {len(self.durations_ms)} runs hit "
                "a sign-in wall, so the figures above mix authenticated and "
                "unauthenticated responses"
            )
        if self.first_hit_ms is not None:
            lines.append(
                f"  first hit {self.first_hit_ms:.0f}ms, excluded from the "
                "figures above so connection setup is not counted as latency"
            )
        if self.cold_first_run:
            lines.append(
                f"  the first hit cost {self.first_hit_ms:.0f}ms against a "
                f"steady {self.median_ms:.0f}ms, so this target does one-time "
                "work -- a compile, a cache fill, a lazy import. Report the two "
                "as separate numbers; the median of both is neither."
            )
        return "\n".join(lines)

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.error is not None:
            content.append("measurement blocked: ", style="bold red")
            content.append(self.error)
            return content
        content.append(self.summary)
        return content


def _looks_like_auth_wall(status: int, location: str) -> bool:
    """Whether this response is a sign-in wall rather than the thing asked for."""
    if status in AUTH_STATUSES:
        return True
    if status not in REDIRECT_STATUSES:
        return False
    target = location.lower()
    return any(hint in target for hint in AUTH_PATH_HINTS)


def _first_hit_dominates(first_hit_ms: float | None, durations: list[float]) -> bool:
    """Whether the first hit cost far more than the steady state.

    True means the target does one-time work -- a compile, a cache fill, a lazy
    import -- and the two numbers describe different things.
    """
    if first_hit_ms is None or not durations:
        return False
    steady = statistics.median(durations)
    if steady <= 0:
        return False
    return (
        first_hit_ms >= steady * COLD_FIRST_RUN_RATIO
        and first_hit_ms - steady >= COLD_FIRST_RUN_FLOOR_MS
    )


class MeasureTimingExecutor(
    ToolExecutor[MeasureTimingAction, MeasureTimingObservation]
):
    def __init__(self, working_dir: str | None = None) -> None:
        self.working_dir = working_dir

    async def _reset(self, command: str) -> bool:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=self.working_dir,
        )
        try:
            return await asyncio.wait_for(process.wait(), timeout=30) == 0
        except TimeoutError:
            process.kill()
            return False

    async def _run_once(
        self, action: MeasureTimingAction, client: httpx.AsyncClient
    ) -> tuple[float, int, str, str]:
        """One timed run.

        Returns elapsed ms, result code, an output sample, and the redirect
        target if the response was one.
        """
        if action.url:
            started = time.perf_counter()
            response = await client.request(
                action.method.upper(),
                action.url,
                headers=action.headers or None,
                content=action.body,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            return (
                elapsed_ms,
                response.status_code,
                response.text[:MAX_BODY_SAMPLE],
                response.headers.get("location", ""),
            )

        started = time.perf_counter()
        process = await asyncio.create_subprocess_shell(
            action.command or "",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.working_dir,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=COMMAND_TIMEOUT_SECONDS
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise MeasurementBlocked(
                f"the command did not finish within {COMMAND_TIMEOUT_SECONDS:.0f}s"
            ) from None
        elapsed_ms = (time.perf_counter() - started) * 1000
        return (
            elapsed_ms,
            process.returncode if process.returncode is not None else -1,
            stdout.decode("utf-8", errors="replace")[:MAX_BODY_SAMPLE],
            "",
        )

    async def _measure_targets(
        self, action: MeasureTimingAction
    ) -> MeasureTimingObservation:
        """Time each labelled target and report their shares.

        A slow page is not a cause. Timing the page and the requests it issues in
        one call is what turns "it is slow" into "the server is 92% of it", which
        is the step that decides what to change. Run sequentially so the targets do
        not contend with each other and inflate the very numbers being compared.
        """
        rows: list[TimingComparisonRow] = []
        for target in action.targets:
            one = action.model_copy(
                update={
                    "targets": [],
                    "url": target.url,
                    "command": target.command,
                }
            )
            result = await self._measure(one)
            rows.append(
                TimingComparisonRow(
                    label=target.label,
                    median_ms=result.median_ms,
                    share_pct=0.0,
                    note=result.error
                    or (
                        "answered by a sign-in wall"
                        if result.auth_blocked_runs
                        and result.auth_blocked_runs == len(result.durations_ms)
                        else ""
                    ),
                )
            )
        total = sum(row.median_ms for row in rows) or 1.0
        for row in rows:
            row.share_pct = row.median_ms / total * 100.0
        return MeasureTimingObservation(
            target=", ".join(t.label for t in action.targets),
            target_kind="comparison",
            condition=action.condition,
            comparison=rows,
        )

    async def _measure(self, action: MeasureTimingAction) -> MeasureTimingObservation:
        durations: list[float] = []
        statuses: list[int] = []
        reset_failures = 0
        body_sample: str | None = None
        first_hit_ms: float | None = None
        auth_blocked = 0
        auth_location = ""
        target_kind = "request" if action.url else "command"

        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False
        ) as client:
            if action.url:
                # One request before the timed ones, so connection setup is not
                # charged to run one. Its duration is kept and reported on its
                # own -- it is the only run that pays first-hit cost, which is a
                # real number and not the steady state. A failure here is
                # ignored; the timed runs below report it properly.
                try:
                    first_hit_ms, _, _, _ = await self._run_once(action, client)
                except Exception:  # noqa: BLE001 - the first hit is best-effort
                    first_hit_ms = None
            for index in range(action.repeat):
                if action.reset_command and not await self._reset(action.reset_command):
                    reset_failures += 1
                try:
                    elapsed_ms, code, output, location = await self._run_once(
                        action, client
                    )
                except (
                    httpx.HTTPError,
                    ValueError,
                    MeasurementBlocked,
                    OSError,
                ) as exc:
                    detail = (
                        str(exc)
                        if isinstance(exc, MeasurementBlocked)
                        else f"{type(exc).__name__}: {exc}"
                    )
                    return MeasureTimingObservation(
                        target=action.target,
                        target_kind=target_kind,
                        condition=action.condition,
                        reset_command=action.reset_command,
                        error=detail,
                    )
                durations.append(elapsed_ms)
                statuses.append(code)
                if _looks_like_auth_wall(code, location):
                    auth_blocked += 1
                    auth_location = auth_location or location
                if index == 0:
                    body_sample = output.strip() or None

        return MeasureTimingObservation(
            target=action.target,
            target_kind=target_kind,
            condition=action.condition,
            first_hit_ms=first_hit_ms,
            auth_blocked_runs=auth_blocked,
            auth_redirect_to=auth_location,
            cold_first_run=_first_hit_dominates(first_hit_ms, durations),
            durations_ms=durations,
            statuses=statuses,
            median_ms=statistics.median(durations),
            min_ms=min(durations),
            max_ms=max(durations),
            reset_command=action.reset_command,
            reset_failures=reset_failures,
            body_sample=body_sample,
        )

    def __call__(
        self,
        action: MeasureTimingAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> MeasureTimingObservation:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        run = self._measure_targets if action.targets else self._measure
        if loop is None:
            return asyncio.run(run(action))
        # Called from inside a running loop: hand the work to a private one so the
        # blocking tool contract still holds.
        result: dict[str, MeasureTimingObservation] = {}

        def run_in_thread() -> None:
            result["value"] = asyncio.run(run(action))

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        thread.join()
        return result["value"]


class MeasureTimingTool(ToolDefinition[MeasureTimingAction, MeasureTimingObservation]):
    """Times a real request so a claim about duration has an observation behind it."""

    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["MeasureTimingTool"]:
        workspace = getattr(conv_state, "workspace", None)
        working_dir = getattr(workspace, "working_dir", None)
        return [
            cls(
                description=MEASURE_TIMING_DESCRIPTION,
                action_type=MeasureTimingAction,
                observation_type=MeasureTimingObservation,
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=MeasureTimingExecutor(working_dir=working_dir),
            )
        ]


register_tool(MeasureTimingTool.name, MeasureTimingTool)
