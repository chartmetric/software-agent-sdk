"""The tool has to return real durations, so it is tested against a real server."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import ValidationError

from openhands.sdk.llm import TextContent
from openhands.tools.measure_timing import (
    MeasureTimingAction,
    MeasureTimingExecutor,
)
from openhands.tools.measure_timing.definition import (
    COLD_FIRST_RUN_RATIO,
    MeasureTimingObservation,
    TimingComparisonRow,
    TimingTarget,
)


class _SlowHandler(BaseHTTPRequestHandler):
    delay_seconds = 0.05

    def do_GET(self):  # noqa: N802 - stdlib naming
        time.sleep(self.delay_seconds)
        payload = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Keep the test output clean."""
        return


@pytest.fixture
def slow_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/audience-research"
    server.shutdown()
    server.server_close()


def test_it_returns_a_duration_per_run_and_a_median(slow_server):
    """A claim about speed needs numbers, so the observation is numbers.

    The server sleeps 50ms per request, which is what the durations have to
    reflect -- a tool that returned a plausible summary without timing anything
    would be the failure it exists to prevent.
    """
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(url=slow_server, repeat=3, condition="cold cache, ages=65+")
    )

    assert observation.error is None
    assert len(observation.durations_ms) == 3
    assert observation.statuses == [200, 200, 200]
    # Every run must clear the server's own delay, and none should be absurd.
    assert all(45.0 < value < 5_000.0 for value in observation.durations_ms)
    assert observation.min_ms <= observation.median_ms <= observation.max_ms

    rendered = observation.summary
    assert "median" in rendered
    assert "cold cache, ages=65+" in rendered
    # Without a reset the agent is told what the later runs actually measured.
    assert "every run after the first measured a warm path" in rendered
    # Connection setup is charged to the first hit, not to run one.
    assert observation.first_hit_ms is not None
    assert "excluded from the figures above" in rendered


def test_a_reset_command_runs_before_each_request(slow_server, tmp_path):
    """A duration that depends on a starting condition needs that condition back.

    Without this the first run measures a cold path and the rest measure a warm
    one, and the median describes neither.
    """
    marker = tmp_path / "resets"
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(
            url=slow_server,
            repeat=3,
            reset_command=f"echo x >> {marker}",
            condition="cache cleared each run",
        )
    )

    assert observation.error is None
    assert observation.reset_failures == 0
    assert marker.read_text().count("x") == 3
    assert observation.reset_command is not None
    assert observation.reset_command in observation.summary


def test_a_failing_reset_is_counted_rather_than_hidden(slow_server):
    """A measurement whose condition was not re-established is not the condition
    it claims, so the count travels with the numbers."""
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(url=slow_server, repeat=2, reset_command="exit 3")
    )

    assert observation.error is None
    assert observation.reset_failures == 2
    assert "2 failed" in observation.summary


def test_an_unreachable_target_is_a_blocked_measurement_not_a_slow_one():
    """Reporting a duration for a request that never completed would be worse
    than reporting nothing."""
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(url="http://127.0.0.1:1/never-listening", repeat=2)
    )

    assert observation.error is not None
    assert observation.durations_ms == []
    assert "blocked measurement" in observation.summary
    assert "do not report a duration" in observation.summary


def test_repeat_is_bounded_so_one_call_cannot_run_away():
    assert MeasureTimingAction(url="http://x/", repeat=999).repeat == 20
    assert MeasureTimingAction(url="http://x/", repeat=0).repeat == 1


def test_a_shell_command_is_timed_the_same_way(tmp_path):
    """Not every duration is a request -- a query, a build, or a test is timed too."""
    observation = MeasureTimingExecutor(working_dir=str(tmp_path))(
        MeasureTimingAction(
            command="sleep 0.05 && echo done", repeat=2, condition="warm build cache"
        )
    )

    assert observation.error is None
    assert observation.target_kind == "command"
    assert len(observation.durations_ms) == 2
    assert all(45.0 < value < 5_000.0 for value in observation.durations_ms)
    assert observation.statuses == [0, 0]
    assert "exit codes" in observation.summary
    assert "done" in (observation.body_sample or "")


def test_a_failing_command_reports_its_exit_code_beside_the_duration():
    """A command that failed fast is not a fast command, so the code travels
    with the number."""
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(command="exit 7", repeat=2)
    )

    assert observation.error is None
    assert observation.statuses == [7, 7]
    assert "exit codes: [7]" in observation.summary


def test_exactly_one_target_is_required():
    """Timing nothing, or two things at once, is not a measurement."""
    with pytest.raises(ValidationError):
        MeasureTimingAction()
    with pytest.raises(ValidationError):
        MeasureTimingAction(url="http://x/", command="sleep 1")


class _ColdStartHandler(BaseHTTPRequestHandler):
    """Slow on the first request, fast afterwards -- a dev server compiling."""

    served = 0

    def do_GET(self):  # noqa: N802 - stdlib naming
        _ColdStartHandler.served += 1
        time.sleep(0.30 if _ColdStartHandler.served == 1 else 0.02)
        payload = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Keep the test output clean."""
        return


@pytest.fixture
def cold_start_server():
    _ColdStartHandler.served = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ColdStartHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/dashboard"
    server.shutdown()
    server.server_close()


def test_a_dominant_first_hit_is_reported_apart_from_the_steady_state(
    cold_start_server,
):
    """A number taken while the app was still coming up is not the path's number.

    This is the production shape: the agent hit a preview whose dev server was
    still compiling and reported the duration it got. The fix is not to hide the
    cold number but to stop averaging it into the steady one -- the median of a
    compile and four warm runs describes neither.
    """
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(url=cold_start_server, repeat=4)
    )

    assert observation.error is None
    assert observation.cold_first_run is True
    assert observation.first_hit_ms is not None
    # The cold cost is the first hit; the steady state excludes it.
    assert observation.first_hit_ms > observation.median_ms
    rendered = observation.summary
    assert "does one-time work" in rendered
    assert "the median of both is neither" in rendered


def test_a_steady_target_is_not_called_a_warm_up(slow_server):
    """The signal has to stay quiet when every run agrees, or it is noise."""
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(url=slow_server, repeat=4)
    )

    assert observation.cold_first_run is False
    assert "does one-time work" not in observation.summary


def test_connection_setup_is_not_charged_to_the_first_timed_run(slow_server):
    """The regression this pins was a false positive, not a missed one.

    Measured against a server that sleeps a flat 50ms: the first request on a
    fresh client took 274ms against 68ms for the rest, purely client and socket
    setup. On a ratio alone that reads as a cold app, so every fast endpoint
    would have been reported as doing one-time work.
    """
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(url=slow_server, repeat=4)
    )

    assert observation.cold_first_run is False
    # The timed runs agree with each other because none of them paid setup.
    assert observation.max_ms < observation.min_ms * COLD_FIRST_RUN_RATIO


@pytest.mark.parametrize(
    "action",
    [
        MeasureTimingAction(command="true", repeat=1),
        MeasureTimingAction(url="http://127.0.0.1:1/gone", repeat=1),
    ],
    ids=["measured", "blocked"],
)
def test_the_tool_message_is_never_empty(action):
    """The regression this pins killed a whole conversation, not one observation.

    The tool message is built from `to_llm_content`. An observation that keeps
    its result in its own fields and leaves `content` empty sends nothing, and
    the prompt-cache path then indexes `content[-1]` and raises -- so the run
    dies after the measurement succeeded.
    """
    observation = MeasureTimingExecutor()(action)

    llm_content = list(observation.to_llm_content)
    assert llm_content, "an empty tool message ends the conversation"
    last = llm_content[-1]
    assert isinstance(last, TextContent)
    assert last.text.strip()


class _AuthWallHandler(BaseHTTPRequestHandler):
    """302s to /login unless the session cookie is present, like the real thing."""

    COOKIE = "session=dev-session-token"

    def do_GET(self):  # noqa: N802 - stdlib naming
        if self.COOKIE not in (self.headers.get("Cookie") or ""):
            self.send_response(302)
            self.send_header("Location", "/login")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        time.sleep(0.05)
        payload = b'{"rows": []}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Keep the test output clean."""
        return


@pytest.fixture
def auth_walled_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AuthWallHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/report"
    server.shutdown()
    server.server_close()


def test_a_sign_in_wall_is_refused_rather_than_timed(auth_walled_server):
    """The worst failure this tool can produce is a fast, confident, false number.

    Measured against a page that really takes ~750ms: unauthenticated, every run
    is a 302 answered instantly, and reporting "median 1ms" would tell the agent
    the page is fast. This is the production shape -- the live run's own words
    were that the URL redirected to login, so it could not measure.
    """
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(url=auth_walled_server, repeat=3)
    )

    assert observation.auth_blocked_runs == 3
    rendered = observation.summary
    assert "Did not measure" in rendered
    assert "/login" in rendered
    assert "do not report them as the page's latency" in rendered
    # The remedy is named, because the session usually already exists.
    assert "Cookie or Authorization header" in rendered
    # And no median is offered as the page's latency.
    assert "median" not in rendered


def test_the_same_url_measures_once_the_session_is_supplied(auth_walled_server):
    """The credential turns a refusal into a real number, via the same call."""
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(
            url=auth_walled_server,
            repeat=3,
            headers={"Cookie": _AuthWallHandler.COOKIE},
            condition="authenticated session",
        )
    )

    assert observation.auth_blocked_runs == 0
    assert observation.statuses == [200, 200, 200]
    assert all(45.0 < value < 5_000.0 for value in observation.durations_ms)
    assert "median" in observation.summary


@pytest.fixture
def two_servers():
    """A slow side and a fast side, so attribution has something to get right."""
    servers = []

    def make(delay_seconds: float):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib naming
                time.sleep(delay_seconds)
                payload = b"ok"
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                """Keep the test output clean."""
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}/x"

    yield make
    for server in servers:
        server.shutdown()
        server.server_close()


def test_a_comparison_names_which_side_the_cost_is_on(two_servers):
    """A slow page is not a cause, and this is the step that turns it into one.

    The production failure this addresses: a run measured the artists request at
    18.463s, wrote that it was "a backend query/cache concern", and then changed
    the frontend -- the side it had just shown was not the cost. Measuring is not
    attributing, so the comparison does the attribution and says what follows from
    it.
    """
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(
            targets=[
                TimingTarget(label="artists API", url=two_servers(0.40)),
                TimingTarget(label="page document", url=two_servers(0.02)),
            ],
            repeat=3,
            condition="warm cache, ages=65+",
        )
    )

    assert observation.error is None
    assert observation.target_kind == "comparison"
    by_label = {row.label: row for row in observation.comparison}
    assert by_label["artists API"].share_pct > 80
    assert by_label["page document"].share_pct < 20

    rendered = observation.summary
    assert "artists API" in rendered
    assert "warm cache, ages=65+" in rendered
    assert "that is where the cost is" in rendered
    assert "Changing anything else leaves it in place" in rendered


def test_a_comparison_refuses_to_name_a_cause_it_cannot_see(two_servers):
    """Two similar numbers single nothing out, and saying they do is the error."""
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(
            targets=[
                TimingTarget(label="keywords API", url=two_servers(0.12)),
                TimingTarget(label="subreddits API", url=two_servers(0.10)),
            ],
            repeat=3,
        )
    )

    rendered = observation.summary
    assert "No single target dominates" in rendered
    assert "not supported by these numbers" in rendered
    assert "that is where the cost is" not in rendered


def test_a_comparison_needs_at_least_two_targets():
    """One target is a measurement, not a comparison."""
    with pytest.raises(ValidationError):
        MeasureTimingAction(targets=[TimingTarget(label="only", url="http://x/")])
    with pytest.raises(ValidationError):
        MeasureTimingAction(
            url="http://x/",
            targets=[
                TimingTarget(label="a", url="http://a/"),
                TimingTarget(label="b", url="http://b/"),
            ],
        )


def test_a_response_the_server_declined_is_not_reported_as_a_duration():
    """A 20-second 503 reads as a bottleneck and is an absence.

    Production, conversation 2a1e4ad5: a sandbox preview proxy waited its
    20-second cold-start window for a port nothing listened on, then returned
    its own 503. The tool recorded a duration and no error, because a 503 is a
    response, and the agent was handed 20056ms against 20064ms as a tidy 50/50
    comparison of two pages. Neither number was about a page.
    """
    from openhands.tools.measure_timing.definition import _unmeasured_note

    declined = MeasureTimingObservation(
        target="filtered",
        durations_ms=[20056.6, 20050.1, 20061.0],
        statuses=[503, 503, 503],
        median_ms=20056.6,
        min_ms=20050.1,
        max_ms=20061.0,
        unavailable_runs=3,
    )

    assert "Did not measure" in declined.summary
    assert "declined by the server" in declined.summary
    # The number itself must not be offered as the target's latency.
    assert "20057ms" not in declined.summary
    # A comparison row carries no statuses, so the note is where it says so.
    assert _unmeasured_note(declined) == "declined by the server (status [503])"


def test_a_comparison_where_nothing_answered_does_not_read_as_a_comparison():
    """Shares are computed over whatever came back, so they always look real.

    The closing line used to say no single target dominated -- true of two
    numbers that describe the same proxy timing itself out twice, and exactly
    the attribution the shares exist to prevent.
    """
    note = "declined by the server (status [503])"
    nothing_answered = MeasureTimingObservation(
        target="filtered, unfiltered",
        target_kind="comparison",
        condition="unauthenticated browser-equivalent document request",
        comparison=[
            TimingComparisonRow(
                label="filtered", median_ms=20056.6, share_pct=50.0, note=note
            ),
            TimingComparisonRow(
                label="unfiltered", median_ms=20064.5, share_pct=50.0, note=note
            ),
        ],
    )
    assert "None of these reached its target" in nothing_answered.summary
    assert "No single target dominates" not in nothing_answered.summary

    one_answered = MeasureTimingObservation(
        target="a, b",
        target_kind="comparison",
        condition="warm",
        comparison=[
            TimingComparisonRow(label="a", median_ms=900.0, share_pct=100.0),
            TimingComparisonRow(label="b", median_ms=20050.0, share_pct=0.0, note=note),
        ],
    )
    assert "1 of 2 targets did not answer" in one_answered.summary
    assert "Only one target answered" in one_answered.summary

    served = MeasureTimingObservation(
        target="a, b",
        target_kind="comparison",
        condition="warm",
        comparison=[
            TimingComparisonRow(label="a", median_ms=842.0, share_pct=73.9),
            TimingComparisonRow(label="b", median_ms=297.0, share_pct=26.1),
        ],
    )
    assert "a is 74% of the total" in served.summary


def test_the_durations_are_in_content_where_a_reader_finds_them(slow_server):
    """The observation carried its result only in fields nobody generic reads.

    `content` was empty, so the numbers existed for anything that knew to read
    `comparison` and `median_ms` and for nothing else. Two consequences, both
    observed: the LLM message was built from an empty list until `to_llm_content`
    was overridden to cover it, and the chat transcript showed a tool card with
    nothing in it -- a run that measured seven targets looked, to the person
    reading the session, like a run that had measured nothing.
    """
    observation = MeasureTimingExecutor()(
        MeasureTimingAction(
            targets=[
                TimingTarget(label="fast", url=slow_server),
                TimingTarget(label="also", url=slow_server),
            ],
            repeat=3,
            condition="warm, local",
        )
    )

    serialized = observation.model_dump(mode="json")
    assert len(serialized["content"]) == 1
    text = serialized["content"][0]["text"]
    # The same text the LLM is given, so the two readers cannot diverge.
    assert text == observation.summary
    assert "fast" in text and "also" in text
    assert "ms" in text
    assert "% of the measured total" in text
