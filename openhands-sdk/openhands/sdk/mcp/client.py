"""Minimal sync helpers on top of fastmcp.Client, preserving original behavior."""

import asyncio
import inspect
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from fastmcp import Client as AsyncMCPClient

from openhands.sdk.logger import get_logger
from openhands.sdk.mcp.exceptions import MCPError
from openhands.sdk.utils.async_executor import AsyncExecutor


if TYPE_CHECKING:
    from openhands.sdk.mcp.tool import MCPToolDefinition


logger = get_logger(__name__)

# Delay before each reconnection attempt. The first is immediate: a session that
# died of the counter bug above is ready to be replaced at once, and waiting
# would only slow the common case. The rest are sized to one specific event, a
# deploy cutover -- Caddy repoints to an already-health-checked slot and the
# previous one stops, so the gap is short and bounded, but not instantaneous. A
# single immediate retry samples the one moment most likely to still be bad.
#
# Do not grow these. This is not a general retry policy: it is paid inside a tool
# call, on every call that meets a dead session rather than once per turn, and it
# competes with the turn's own budget. A server that is genuinely gone has to
# surface as an error the agent can report, not hang the turn waiting for it.
_RECONNECT_DELAYS_SECONDS = (0.0, 0.25, 1.0, 3.5)


ToolsReconciledCallback = Callable[
    ["MCPClient", Sequence["MCPToolDefinition"]],
    None,
]


class MCPClient(AsyncMCPClient):
    """MCP client with sync helpers and lifecycle management.

    Extends fastmcp.Client with:
      - call_async_from_sync(awaitable_or_fn, *args, timeout=None, **kwargs)
      - call_sync_from_async(fn, *args, **kwargs)  # await this from async code

    After create_mcp_tools() populates it, use as a sync context manager:

        with create_mcp_tools(config) as client:
            for tool in client.tools:
                # use tool
        # Connection automatically closed

    Or manage lifecycle manually by calling sync_close() when done.
    """

    _executor: AsyncExecutor
    _closed: bool
    _tools: "list[MCPToolDefinition]"
    _tools_reconciled_callback: ToolsReconciledCallback | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = AsyncExecutor()
        self._closed = False
        self._tools = []
        self._tools_reconciled_callback = None

    @property
    def tools(self) -> "list[MCPToolDefinition]":
        """The MCP tools using this client connection (returns a copy)."""
        return list(self._tools)

    async def connect(self) -> None:
        """Establish connection to the MCP server.

        A session that dies on its own -- a transient 5xx from the server, a
        dropped socket, a server restarted underneath us -- never runs
        ``__aexit__``, so fastmcp's reentrancy counter is still above zero
        while its session task has already finished. The next ``_connect()``
        sees it must start a fresh session, finds the counter is not zero, and
        refuses with ``RuntimeError("Internal error: nesting counter should be
        0 ...")``. Nothing ever decrements it, so every later reconnect fails
        the same way: one blip disables every MCP tool for the remainder of the
        conversation. Measured in production, a single 502 during a deploy
        cutover cost the run its publication tools for the following 15
        minutes, across 11 consecutive calls.

        So a failed entry is followed by a forced disconnect -- which is what
        resets the counter -- and a bounded series of retries, before the
        failure is reported.
        """
        try:
            await self.__aenter__()
            return
        except RuntimeError as exc:
            last_exc: BaseException = exc

        logger.info(
            "MCP connect failed (%s); resetting session state and retrying.", last_exc
        )
        for delay in _RECONNECT_DELAYS_SECONDS:
            if delay:
                await asyncio.sleep(delay)
            try:
                await self._disconnect(force=True)
            except Exception as disconnect_exc:
                # Awaiting the already-finished session task re-raises whatever
                # killed it. That is the reason we are reconnecting, not a new
                # problem, and the counter has already been reset by the time
                # it is raised.
                logger.debug(
                    "Ignoring failure from the dead MCP session during reset: %s",
                    disconnect_exc,
                )
            try:
                await self.__aenter__()
                return
            except Exception as retry_exc:
                last_exc = retry_exc
        raise MCPError("MCP Connection Failure") from last_exc

    def call_async_from_sync(
        self,
        awaitable_or_fn: Callable[..., Any] | Any,
        *args,
        timeout: float,
        **kwargs,
    ) -> Any:
        """
        Run a coroutine or async function on this client's loop from sync code.

        Usage:
            mcp.call_async_from_sync(async_fn, arg1, kw=...)
            mcp.call_async_from_sync(coro)
        """
        return self._executor.run_async(
            awaitable_or_fn, *args, timeout=timeout, **kwargs
        )

    async def call_sync_from_async(
        self, fn: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """
        Await running a blocking function in the default threadpool from async code.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def sync_close(self) -> None:
        """
        Synchronously close the MCP client and cleanup resources.

        This will attempt to call the async close() method if available,
        then shutdown the background event loop. Safe to call multiple times.
        """
        if self._closed:
            return

        # Best-effort: try async close if parent provides it
        if hasattr(self, "close") and inspect.iscoroutinefunction(self.close):
            try:
                self._executor.run_async(self.close, timeout=10.0)
            except Exception:
                pass  # Ignore close errors during cleanup

        # Always cleanup the executor
        self._executor.close()
        self._closed = True

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.sync_close()
        except Exception:
            pass  # Ignore cleanup errors during deletion

    # Sync context manager support
    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.sync_close()

    # Iteration support for tools
    def __iter__(self) -> "Iterator[MCPToolDefinition]":
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __getitem__(self, index: int) -> "MCPToolDefinition":
        return self._tools[index]
