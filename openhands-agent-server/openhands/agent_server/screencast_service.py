"""Live CDP screencast streaming for passive browser viewers.

Bridges the shared browser tool's `Page.startScreencast` frames (fired on
the browser's dedicated background thread — see
`openhands.tools.browser_use.impl.BrowserToolExecutor`) to WebSocket
subscribers on the agent-server's own asyncio loop.

Unlike `DesktopService`, this needs neither VNC nor a headed browser: CDP
screencast captures the page's render pipeline directly, so it works with
the default headless browser configuration.
"""

import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from openhands.agent_server.pub_sub import PubSub, Subscriber
from openhands.sdk.logger import get_logger
from openhands.tools.browser_use.definition import BrowserToolSet


logger = get_logger(__name__)

# Screencast viewers are a small, deliberately-bounded audience (a live
# "watch the agent work" panel, not a broadcast feature).
_MAX_SCREENCAST_SUBSCRIBERS = 10

ControlOwner = Literal["agent", "human"]


@dataclass
class ScreencastService:
    """Reference-counts WebSocket viewers and starts/stops the underlying
    CDP screencast accordingly: streaming only runs while at least one
    viewer is connected.

    Also brokers exclusive human takeover of the browser: at most one
    connected subscriber may hold the control lock at a time (recorded here
    as `_controlling_subscriber_id`), and that ownership is mirrored onto the
    shared `BrowserToolExecutor` (`_control_owner` there) so the agent's own
    browser_* tool calls are blocked while a human is driving -- including
    from subagents, which reuse that same executor instance.
    """

    _pub_sub: PubSub[dict[str, Any]] = field(
        default_factory=lambda: PubSub[dict[str, Any]](
            max_subscribers=_MAX_SCREENCAST_SUBSCRIBERS
        ),
        init=False,
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _active: bool = field(default=False, init=False)
    _control_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _control_owner: ControlOwner = field(default="agent", init=False)
    _controlling_subscriber_id: UUID | None = field(default=None, init=False)

    async def subscribe(self, subscriber: Subscriber[dict[str, Any]]) -> UUID:
        """Register a viewer and ensure the screencast is running.

        Raises:
            MaxSubscribersError: if the viewer limit has been reached.
        """
        subscriber_id = self._pub_sub.subscribe(subscriber)
        try:
            await self._ensure_started()
        except Exception:
            self._pub_sub.unsubscribe(subscriber_id)
            raise
        return subscriber_id

    async def unsubscribe(self, subscriber_id: UUID) -> None:
        """Remove a viewer, stopping the screencast once none remain.

        If the disconnecting viewer was holding the control lock, release it
        first -- a dropped connection must never leave the browser
        permanently locked to a human who is no longer there.
        """
        if self._controlling_subscriber_id == subscriber_id:
            await self.release_control(subscriber_id)
        self._pub_sub.unsubscribe(subscriber_id)
        if not self._pub_sub.subscriber_ids():
            await self._ensure_stopped()

    @property
    def control_owner(self) -> ControlOwner:
        """Who currently controls the browser: "agent" or "human"."""
        return self._control_owner

    def is_controller(self, subscriber_id: UUID) -> bool:
        """Whether `subscriber_id` currently holds the control lock."""
        return self._controlling_subscriber_id == subscriber_id

    async def take_control(self, subscriber_id: UUID) -> bool:
        """Acquire exclusive control for `subscriber_id`.

        Only one subscriber may control the browser at a time -- the first
        to successfully call this becomes the controller. A request from a
        second subscriber while someone else already controls is an
        unauthorized interaction and is silently ignored (returns False).
        """
        async with self._control_lock:
            if self._controlling_subscriber_id is not None:
                if self._controlling_subscriber_id == subscriber_id:
                    return True  # already the controller; idempotent
                return False
            self._controlling_subscriber_id = subscriber_id
            self._control_owner = "human"
            executor = BrowserToolSet.get_or_create_shared_executor()
            await asyncio.to_thread(executor.set_control_owner, "human")
            return True

    async def release_control(self, subscriber_id: UUID) -> bool:
        """Release control held by `subscriber_id`, restoring agent access.

        A release request from anyone other than the current controller is
        an unauthorized interaction and is silently ignored (returns False).
        """
        async with self._control_lock:
            if self._controlling_subscriber_id != subscriber_id:
                return False
            self._controlling_subscriber_id = None
            self._control_owner = "agent"
            executor = BrowserToolSet.get_or_create_shared_executor()
            await asyncio.to_thread(executor.set_control_owner, "agent")
            return True

    async def dispatch_input(
        self, subscriber_id: UUID, kind: str, payload: dict[str, Any]
    ) -> None:
        """Forward a mouse/key event to the browser, if `subscriber_id`
        currently owns the control lock. Input from any other subscriber
        (including one that never took control) is silently ignored."""
        if not self.is_controller(subscriber_id):
            return
        executor = BrowserToolSet.get_or_create_shared_executor()
        await asyncio.to_thread(executor.dispatch_screencast_input, kind, payload)

    async def _ensure_started(self) -> None:
        async with self._lock:
            if self._active:
                return

            loop = asyncio.get_running_loop()
            executor = BrowserToolSet.get_or_create_shared_executor()

            def _on_frame(data: str, metadata: dict[str, Any]) -> None:
                # Invoked on the browser's dedicated background thread
                # (see impl.BrowserToolExecutor.start_screencast) — hop back
                # onto the agent-server's own loop to publish.
                future = asyncio.run_coroutine_threadsafe(
                    self._pub_sub(
                        {"type": "frame", "data": data, "metadata": metadata}
                    ),
                    loop,
                )
                future.add_done_callback(_log_publish_failure)

            started = await asyncio.to_thread(executor.start_screencast, _on_frame)
            self._active = bool(started)
            if not self._active:
                logger.warning("Screencast failed to start")

    async def _ensure_stopped(self) -> None:
        async with self._lock:
            if not self._active:
                return
            # Re-check under the lock: a new subscriber may have joined
            # between the caller's subscriber_ids() check and here.
            if self._pub_sub.subscriber_ids():
                return

            executor = BrowserToolSet.get_or_create_shared_executor()
            await asyncio.to_thread(executor.stop_screencast)
            self._active = False

    async def close(self) -> None:
        """Stop the screencast and release all subscribers (server shutdown)."""
        await self._pub_sub.close()
        await self._ensure_stopped()

    async def __aenter__(self) -> "ScreencastService":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()


def _log_publish_failure(future: "concurrent.futures.Future[None]") -> None:
    if future.cancelled():
        return
    error = future.exception()
    if error is not None:
        logger.debug(f"Failed to publish screencast frame: {error}")


_screencast_service: ScreencastService | None = None


def get_default_screencast_service() -> ScreencastService:
    """Get the default screencast service instance."""
    global _screencast_service
    if _screencast_service is None:
        _screencast_service = ScreencastService()
    return _screencast_service
