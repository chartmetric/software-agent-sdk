"""Reconnection after an MCP session dies without being closed.

A session that ends on its own -- a transient 5xx, a dropped socket, a server
restarted underneath the client -- never runs ``__aexit__``. fastmcp's
reentrancy counter is then still above zero while the session task is already
finished, and every later ``connect()`` refuses to start a fresh session. In
production one 502 during a deploy cutover disabled a conversation's MCP tools
for the following 15 minutes, so the publication tools it needed at the end of
the run were gone.

These tests drive the client into that exact state and require that it comes
back, rather than asserting that any particular text appears in the source.
"""

import asyncio
import socket
import threading
import time

import pytest
from fastmcp import FastMCP

from openhands.sdk.mcp import create_mcp_tools
from openhands.sdk.mcp.config import coerce_mcp_config
from openhands.sdk.mcp.exceptions import MCPError
from openhands.sdk.mcp.tool import MCPToolExecutor


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server() -> int:
    mcp = FastMCP("reconnect-test-server")

    @mcp.tool()
    def echo(message: str) -> str:
        """Echo a message."""
        return f"Echo: {message}"

    port = _find_free_port()

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            mcp.run_http_async(
                host="127.0.0.1",
                port=port,
                transport="http",
                show_banner=False,
                path="/mcp",
            )
        )

    threading.Thread(target=run, daemon=True).start()
    time.sleep(0.5)
    return port


def _config(port: int):
    return coerce_mcp_config(
        {"test": {"transport": "http", "url": f"http://127.0.0.1:{port}/mcp"}}
    )


def _kill_session_without_closing(client) -> None:
    """End the live session the way a server-side failure does.

    Stops the background session task without going through ``__aexit__``, so
    the reentrancy counter is left exactly as an unexpected disconnect leaves
    it. Nothing here reaches into the reconnect path itself.
    """

    async def stop():
        client._session_state.stop_event.set()
        await client._session_state.session_task

    client.call_async_from_sync(stop, timeout=10.0)


class TestReconnectAfterUnexpectedDisconnect:
    def test_tool_call_recovers_after_session_dies(self, live_server: int):
        with create_mcp_tools(_config(live_server), timeout=10.0) as tools:
            tool = next(t for t in tools if t.name == "echo")
            executor = tool.executor
            assert isinstance(executor, MCPToolExecutor)

            first = executor(tool.action_from_arguments({"message": "before"}))
            assert not first.is_error
            assert "before" in first.text

            _kill_session_without_closing(executor.client)
            assert not executor.client.is_connected()

            second = executor(tool.action_from_arguments({"message": "after"}))
            assert not second.is_error, second.text
            assert "after" in second.text

    def test_second_disconnect_also_recovers(self, live_server: int):
        """The reset must not be a one-shot: a run can lose the session twice."""
        with create_mcp_tools(_config(live_server), timeout=10.0) as tools:
            tool = next(t for t in tools if t.name == "echo")
            executor = tool.executor
            assert isinstance(executor, MCPToolExecutor)

            for attempt in range(2):
                _kill_session_without_closing(executor.client)
                result = executor(
                    tool.action_from_arguments({"message": f"round-{attempt}"})
                )
                assert not result.is_error, result.text
                assert f"round-{attempt}" in result.text


class _UnreachableClient:
    """A client whose reconnection always fails, wrapper over cause."""

    _closed = False

    def is_connected(self) -> bool:
        return False

    async def connect(self) -> None:
        raise MCPError("MCP Connection Failure") from RuntimeError(
            "Internal error: nesting counter should be 0 when starting new "
            "session, got 1"
        )


class TestReconnectFailureNamesItsCause:
    def test_observation_carries_the_underlying_error(self):
        executor = MCPToolExecutor(
            tool_name="publish_visual_artifact_file",
            client=_UnreachableClient(),  # type: ignore[arg-type]
            timeout=5.0,
        )
        from openhands.sdk.mcp.definition import MCPToolAction

        observation = asyncio.run(executor.call_tool(MCPToolAction(data={})))

        assert observation.is_error
        # The wrapper alone said only that a connection failed; the reason has
        # to travel with it or it exists nowhere the reader can see.
        assert "MCP Connection Failure" in observation.text
        assert "nesting counter" in observation.text
        assert "RuntimeError" in observation.text


class TestReconnectBudgetIsBounded:
    """A server that is genuinely gone must surface, not hang the turn.

    The retries exist for a deploy cutover, which is short. If the budget were
    unbounded -- or large -- a dead server would stall every tool call that met
    it, and the cost would land inside the turn rather than once per turn.
    """

    def test_a_dead_server_fails_within_the_retry_budget(self):
        from openhands.sdk.mcp.client import _RECONNECT_DELAYS_SECONDS, MCPClient

        # Checked before the timing assertions, and against literals rather
        # than against the constant itself. Without this a single immediate
        # retry -- the behaviour this replaced -- makes the elapsed floor below
        # `sum(...) * 0.8 == 0`, so the test passes while asserting nothing.
        assert len(_RECONNECT_DELAYS_SECONDS) >= 3
        assert sum(_RECONNECT_DELAYS_SECONDS) >= 3.0

        dead_port = _find_free_port()  # nothing is listening on it
        client = MCPClient(f"http://127.0.0.1:{dead_port}/mcp")

        started = time.monotonic()
        with pytest.raises(MCPError):
            asyncio.run(client.connect())
        elapsed = time.monotonic() - started

        # It really backed off rather than spinning through its attempts.
        assert elapsed >= 3.0
        assert elapsed >= sum(_RECONNECT_DELAYS_SECONDS) * 0.8
        # And it gave up. The ceiling is generous so a slow machine does not
        # fail the test, but it still fails if the retries became unbounded.
        assert elapsed < 30
