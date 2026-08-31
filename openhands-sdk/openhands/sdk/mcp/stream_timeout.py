"""Make a severed MCP call fail as a transport error instead of hanging.

An MCP tool result arrives on the POST's SSE response stream, and the server
keeps that stream alive with a comment frame every
``MCP_STREAM_PING_INTERVAL_SECONDS`` (``sse_starlette``'s
``EventSourceResponse.DEFAULT_PING_INTERVAL``). A live call therefore never goes
more than that interval without bytes, whatever the tool is doing, while a call
whose server went away goes silent forever.

Nothing was reading that difference. The transport's httpx read timeout is
``mcp.shared._httpx_utils.MCP_DEFAULT_SSE_READ_TIMEOUT``, 300 seconds -- exactly
``MCPToolExecutor``'s own ``MCP_TOOL_TIMEOUT_SECONDS`` ceiling. The two fire at
the same moment, so the ceiling always won and the transport never got to say
the stream was dead: the agent was handed "the tool server may be unresponsive
or the operation is taking too long", which is a guess, five minutes after the
answer stopped being possible.

Measured in production over the three days to 2026-08-31: 25 calls lost this
way, 300 seconds each, 125 minutes total. They arrive in bursts -- four in 46
seconds across four conversations, four more in 52 seconds -- because one
release cutover severs every call in flight at once. The readiness wait is worst
hit at 16 of 490 calls (3.3%), being the one that blocks longest, but five
different tools appear, so this is about the connection and not about any tool.
One occurrence is settled end to end: the server logged its answer 13.7 seconds
in, two seconds into the replaced release's shutdown, and the agent still
recorded a 300-second timeout -- the answer was produced and could not be
delivered, which is why this has to be detected by the caller.

A read timeout of a few missed pings separates the two cases exactly, and costs
a legitimate long call nothing: the ping arrives whether or not the tool has
finished, so the gap between bytes never approaches the ceiling while the server
is alive. This is deliberately *not* a lower ``MCP_TOOL_TIMEOUT_SECONDS``: that
would bound every call by how long the slowest one may run, which is a budget
question, where this is a liveness question and has a fact on the wire to answer
it.
"""

import httpx

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# What the server sends while it is alive. `sse_starlette` pings every 15s by
# default and the MCP streamable-HTTP server takes that default, so this mirrors
# a value we do not set. Read from the library rather than written down, because
# a version that changes the interval must move this with it -- a ping slower
# than the timeout below would make every long call fail as if it were severed.
try:  # pragma: no cover - exercised by whichever branch the install provides
    from sse_starlette.sse import EventSourceResponse as _EventSourceResponse

    MCP_STREAM_PING_INTERVAL_SECONDS = float(_EventSourceResponse.DEFAULT_PING_INTERVAL)
except Exception:  # pragma: no cover - the client may run without the server dep
    MCP_STREAM_PING_INTERVAL_SECONDS = 15.0

# Three missed pings. One is not enough: a ping is written on a timer the server
# shares with everything else on its loop, so a single late frame is ordinary
# and must not read as a dead connection. Three is 45 seconds against a 300
# second ceiling -- the loss it converts is the whole of the difference.
MCP_STREAM_READ_TIMEOUT_SECONDS = 3 * MCP_STREAM_PING_INTERVAL_SECONDS

# Unchanged from MCP's own default. Only the read timeout is the liveness
# signal; connect, write and pool acquisition already fail loudly on their own.
MCP_CONNECT_TIMEOUT_SECONDS = 30.0


def stream_read_timeout() -> httpx.Timeout:
    """The httpx timeouts an MCP stream should be read with."""
    return httpx.Timeout(
        MCP_CONNECT_TIMEOUT_SECONDS,
        read=MCP_STREAM_READ_TIMEOUT_SECONDS,
    )


def _client_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
    **kwargs: object,
) -> httpx.AsyncClient:
    """Build the transport's httpx client with a liveness-sized read timeout.

    ``timeout`` is what fastmcp derived from ``read_timeout_seconds``, a
    per-request deadline in the MCP protocol. Its connect half is kept and its
    read half deliberately replaced: a caller asking for a long deadline is
    saying the tool may take that long, not that the connection may go silent
    that long, and the ping makes those two different questions.

    ``**kwargs`` because fastmcp passes ``follow_redirects`` and may add more; a
    factory that refused an unknown argument would break the transport rather
    than the timeout.
    """
    connect = timeout.connect if timeout is not None else MCP_CONNECT_TIMEOUT_SECONDS
    client_kwargs: dict[str, object] = {
        "follow_redirects": True,
        "timeout": httpx.Timeout(connect, read=MCP_STREAM_READ_TIMEOUT_SECONDS),
    }
    client_kwargs.update(
        {key: value for key, value in kwargs.items() if key != "follow_redirects"}
    )
    if headers is not None:
        client_kwargs["headers"] = headers
    if auth is not None:
        client_kwargs["auth"] = auth
    return httpx.AsyncClient(**client_kwargs)  # type: ignore[arg-type]


def apply_stream_read_timeout(transport: object) -> int:
    """Install the read timeout on every HTTP transport reachable from ``transport``.

    Returns how many it reached, so a caller can log it and a test can assert it
    rather than inferring the effect from the absence of a hang.

    A transport that already carries a factory is left alone: it was given one
    for a reason this module knows nothing about -- certificate verification is
    the case fastmcp itself uses -- and replacing it would trade a five-minute
    hang for a broken connection.
    """
    from fastmcp.client.transports import SSETransport, StreamableHttpTransport

    installed = 0
    for candidate in _reachable_transports(transport):
        if not isinstance(candidate, StreamableHttpTransport | SSETransport):
            continue
        if candidate.httpx_client_factory is not None:
            continue
        candidate.httpx_client_factory = _client_factory  # type: ignore[assignment]
        installed += 1
    return installed


def _reachable_transports(transport: object) -> list[object]:
    """``transport`` and any child transports it has already created.

    A config naming one server builds its transport eagerly, which is the shape
    every OpenHands runtime uses -- one server, with anything else proxied
    behind it. A config naming several builds a child per server inside
    ``connect_session`` instead, and those cannot be reached from here; they
    keep the 300 second default, which is what every server had before this
    existed.
    """
    found: list[object] = [transport]
    child = getattr(transport, "transport", None)
    if child is not None and child is not transport:
        found.extend(_reachable_transports(child))
    for item in getattr(transport, "_transports", ()) or ():
        if item is not transport and item not in found:
            found.append(item)
    return found
