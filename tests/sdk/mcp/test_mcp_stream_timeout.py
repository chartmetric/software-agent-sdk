"""A severed MCP stream has to fail as a transport error, not run out the ceiling."""

import socket
import time

import httpx
import pytest
from fastmcp.client.transports import StreamableHttpTransport

from openhands.sdk.mcp import stream_timeout
from openhands.sdk.mcp.client import MCPClient


def test_the_read_timeout_sits_between_the_ping_and_the_tool_ceiling() -> None:
    """The whole fix is that a live call and a dead one become distinguishable.

    Below the ping interval every long call would fail as if it were severed;
    at the tool ceiling the ceiling fires first and the transport never gets to
    report anything, which is the state this replaces.
    """
    from openhands.sdk.mcp.tool import MCP_TOOL_TIMEOUT_SECONDS

    assert (
        stream_timeout.MCP_STREAM_PING_INTERVAL_SECONDS
        < stream_timeout.MCP_STREAM_READ_TIMEOUT_SECONDS
        < MCP_TOOL_TIMEOUT_SECONDS
    )


def test_an_http_transport_reads_with_the_stream_timeout() -> None:
    """Arrange a bare transport, act by installing, assert what httpx will use."""
    transport = StreamableHttpTransport("https://example.invalid/mcp")

    installed = stream_timeout.apply_stream_read_timeout(transport)

    assert installed == 1
    assert transport.httpx_client_factory is not None
    client = transport.httpx_client_factory(headers=None, auth=None)
    assert client.timeout.read == stream_timeout.MCP_STREAM_READ_TIMEOUT_SECONDS
    assert client.timeout.connect == stream_timeout.MCP_CONNECT_TIMEOUT_SECONDS


def test_a_protocol_deadline_sets_how_long_a_tool_may_run_not_how_long_it_may_be_silent() -> (  # noqa: E501
    None
):
    """fastmcp turns `read_timeout_seconds` into an httpx read timeout as well.

    That conflates two questions. The deadline says the tool may take this
    long; the ping says the server is still there. Keep the caller's connect
    timeout, and keep answering liveness from the ping.
    """
    transport = StreamableHttpTransport("https://example.invalid/mcp")
    stream_timeout.apply_stream_read_timeout(transport)
    assert transport.httpx_client_factory is not None

    client = transport.httpx_client_factory(
        headers=None,
        auth=None,
        timeout=httpx.Timeout(12.0, read=600.0),
    )

    assert client.timeout.read == stream_timeout.MCP_STREAM_READ_TIMEOUT_SECONDS
    assert client.timeout.connect == 12.0


def test_a_transport_that_already_has_a_factory_is_left_alone() -> None:
    """It was given one for a reason this module knows nothing about."""

    def existing(**kwargs: object) -> httpx.AsyncClient:
        return httpx.AsyncClient()

    transport = StreamableHttpTransport("https://example.invalid/mcp")
    transport.httpx_client_factory = existing  # type: ignore[assignment]

    assert stream_timeout.apply_stream_read_timeout(transport) == 0
    assert transport.httpx_client_factory is existing


def test_the_client_installs_it_before_the_first_session() -> None:
    """The transport builds its httpx client on connect, so later is too late.

    The first session is the one that lists the tools, and it would otherwise
    read with the 300 second default.
    """
    client = MCPClient("https://example.invalid/mcp")

    factory = getattr(client.transport, "httpx_client_factory", None)
    assert factory is not None
    assert factory(headers=None, auth=None).timeout.read == (
        stream_timeout.MCP_STREAM_READ_TIMEOUT_SECONDS
    )


@pytest.mark.asyncio
async def test_a_server_that_accepts_and_then_says_nothing_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production failure, reduced: the connection is up and silent forever.

    A cutover leaves exactly this behind -- the request was accepted, the
    process that would answer is gone, and nothing on the wire says so. Before
    the read timeout the caller sat here until its own 300 second ceiling; the
    assertion is that it now ends as a transport error, and quickly.

    A listening socket nobody accepts from reproduces it exactly and with no
    moving parts: the kernel completes the handshake for the backlog, so the
    connect succeeds and the request is written, and no answer can ever come.
    """
    monkeypatch.setattr(stream_timeout, "MCP_STREAM_READ_TIMEOUT_SECONDS", 0.5)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        transport = StreamableHttpTransport(f"http://127.0.0.1:{port}/mcp")
        stream_timeout.apply_stream_read_timeout(transport)
        assert transport.httpx_client_factory is not None
        started = time.monotonic()
        async with transport.httpx_client_factory(headers=None, auth=None) as client:
            with pytest.raises(httpx.ReadTimeout):
                await client.post(f"http://127.0.0.1:{port}/mcp", json={})
        elapsed = time.monotonic() - started
    finally:
        listener.close()

    # Bounded by the read timeout rather than by the caller's ceiling. The
    # margin is for a loaded CI worker, not for a second attempt.
    assert elapsed < 5.0
