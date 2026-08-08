"""CDP screencast session management for live browser viewing.

Error Handling Policy
======================
Screencast is a secondary, passive-viewing feature that must never block or
break primary browser operations (navigation, clicking, typing, etc.).

1. **Caller-facing operations** (start, stop): return a bool indicating
   success and log at WARNING for unexpected errors.
2. **Frame handling**: never let a failure in the caller's `on_frame`
   callback, or in acknowledging a frame, propagate into CDP's own event
   dispatch loop. Log at DEBUG and continue.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from browser_use.utils import create_task_with_error_handling

from openhands.sdk import get_logger


if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession
    from cdp_use.cdp.input.commands import (
        DispatchKeyEventParameters,
        DispatchMouseEventParameters,
    )
    from cdp_use.cdp.page.events import ScreencastFrameEvent


logger = get_logger(__name__)

# CDP's own event-type vocabulary for Input.dispatchMouseEvent / dispatchKeyEvent.
# The WS protocol layer (agent-server sockets.py) maps its wire-level action
# names ("down"/"up"/"move"/"wheel", "down"/"up"/"char") onto these before
# calling dispatch_mouse/dispatch_key, so this module only ever speaks CDP's
# own vocabulary -- consistent with start()/stop() using CDP's own field names
# directly rather than inventing a parallel one.
_SUPPORTED_MOUSE_TYPES = frozenset(
    {"mousePressed", "mouseReleased", "mouseMoved", "mouseWheel"}
)
_SUPPORTED_KEY_TYPES = frozenset({"keyDown", "keyUp", "char"})


# =============================================================================
# ScreencastSession
# =============================================================================


@dataclass
class ScreencastSession:
    """Streams CDP `Page.startScreencast` frames from the shared browser
    session to a caller-supplied callback.

    This mirrors `RecordingSession`'s structure but has no buffering/storage
    of its own — each frame is handed to `on_frame` as soon as it arrives and
    acknowledged immediately, since a live viewer cares about freshness, not
    completeness.
    """

    _on_frame: Callable[[str, dict[str, Any]], None] | None = field(
        default=None, repr=False
    )
    _on_cursor: Callable[[dict[str, Any]], None] | None = field(
        default=None, repr=False
    )
    _browser_session: BrowserSession | None = field(default=None, repr=False)
    _input_client: Any | None = field(default=None, repr=False)
    _original_dispatch_mouse: Any | None = field(default=None, repr=False)
    _current_session_id: str | None = None
    _is_active: bool = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    def _handle_frame(
        self, event: ScreencastFrameEvent, session_id: str | None
    ) -> None:
        """Synchronous CDP event handler for `Page.screencastFrame`.

        Only processes frames from the session we intend to stream, to guard
        against a race where a just-stopped session's frames arrive late.
        """
        if self._current_session_id and session_id != self._current_session_id:
            return

        if self._on_frame is not None:
            try:
                self._on_frame(event["data"], dict(event.get("metadata", {})))
            except Exception as e:
                logger.debug(f"Screencast on_frame callback failed: {e}")

        if self._browser_session is not None:
            create_task_with_error_handling(
                self._ack_frame(event, session_id),
                name="ack_screencast_frame",
                logger_instance=logger,
                suppress_exceptions=True,
            )

    async def _ack_frame(
        self, event: ScreencastFrameEvent, session_id: str | None
    ) -> None:
        """Acknowledge a frame so CDP keeps sending new ones.

        Uses the root client (not the routed cdp_session), matching
        browser-use's own RecordingWatchdog._ack_screencast_frame.
        """
        if self._browser_session is None:
            return
        try:
            await self._browser_session.cdp_client.send.Page.screencastFrameAck(
                params={"sessionId": event["sessionId"]}, session_id=session_id
            )
        except Exception as e:
            logger.debug(f"Failed to acknowledge screencast frame: {e}")

    async def start(
        self,
        browser_session: BrowserSession,
        on_frame: Callable[[str, dict[str, Any]], None],
        on_cursor: Callable[[dict[str, Any]], None] | None = None,
        format: str = "jpeg",  # noqa: A002
        quality: int = 80,
        max_width: int = 1280,
        max_height: int = 800,
        every_nth_frame: int = 1,
    ) -> bool:
        """Start streaming screencast frames for the current CDP target.

        When `on_cursor` is provided, every mouse event dispatched through the
        browser's root CDP client while the screencast is active — the agent's
        own tool-driven clicks as well as human-takeover input — is reported to
        it as ``{"x": float, "y": float, "event": "<CDP mouse type>"}`` in CDP
        viewport coordinates, so viewers can render a live cursor over frames
        that (being page-render captures) contain no cursor of their own.
        `on_cursor` is invoked on the browser's dedicated background thread,
        like `on_frame`.

        Returns True on success, False on failure (never raises — see
        module Error Handling Policy).
        """
        try:
            cdp_session = await browser_session.get_or_create_cdp_session()

            self._browser_session = browser_session
            self._on_frame = on_frame
            self._on_cursor = on_cursor
            self._current_session_id = cdp_session.session_id

            browser_session.cdp_client.register.Page.screencastFrame(self._handle_frame)
            await cdp_session.cdp_client.send.Page.startScreencast(
                params={
                    "format": format,
                    "quality": quality,
                    "maxWidth": max_width,
                    "maxHeight": max_height,
                    "everyNthFrame": every_nth_frame,
                },
                session_id=cdp_session.session_id,
            )
            if on_cursor is not None:
                self._install_input_observer(browser_session)
            self._is_active = True
            logger.info("Screencast started")
            return True
        except Exception as e:
            logger.warning(f"Failed to start screencast: {e}")
            self._is_active = False
            return False

    def _install_input_observer(self, browser_session: BrowserSession) -> None:
        """Wrap the root CDP client's `Input.dispatchMouseEvent` so every
        mouse event — browser-use's own action watchdogs and human-takeover
        `dispatch_mouse` alike route through this one bound method — also
        reports its position to `_on_cursor`. The wrapper always forwards to
        the original first; cursor reporting failures never affect dispatch
        (see module Error Handling Policy). `_reset_state` restores the
        original method."""
        input_client = browser_session.cdp_client.send.Input
        original = input_client.dispatchMouseEvent

        async def _observed_dispatch_mouse(
            params: Any, session_id: str | None = None
        ) -> Any:
            result = await original(params, session_id=session_id)
            self._notify_cursor(params)
            return result

        input_client.dispatchMouseEvent = _observed_dispatch_mouse
        self._input_client = input_client
        self._original_dispatch_mouse = original

    def _notify_cursor(self, params: Any) -> None:
        """Report a dispatched mouse event's position to `_on_cursor`.

        Never raises — cursor reporting is a cosmetic side channel of an
        active screencast (see module Error Handling Policy)."""
        if self._on_cursor is None:
            return
        try:
            event_type = params.get("type")
            x, y = params.get("x"), params.get("y")
            if event_type not in _SUPPORTED_MOUSE_TYPES:
                return
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return
            self._on_cursor({"x": x, "y": y, "event": event_type})
        except Exception as e:
            logger.debug(f"Screencast on_cursor callback failed: {e}")

    async def stop(self) -> bool:
        """Stop streaming screencast frames.

        Returns True on success (or if already stopped), False if the CDP
        call itself failed (state is cleared regardless — see module Error
        Handling Policy).
        """
        if not self._is_active or self._browser_session is None:
            self._reset_state()
            return True

        try:
            await self._browser_session.cdp_client.send.Page.stopScreencast(
                session_id=self._current_session_id
            )
            logger.info("Screencast stopped")
            return True
        except Exception as e:
            logger.debug(f"Failed to stop screencast cleanly: {e}")
            return False
        finally:
            self._reset_state()

    async def dispatch_mouse(
        self,
        type: str,  # noqa: A002
        x: float,
        y: float,
        button: str = "left",
        click_count: int = 1,
        delta_x: float = 0,
        delta_y: float = 0,
    ) -> None:
        """Dispatch a mouse event to the current CDP target for human takeover.

        `type` must be one of the CDP-native mouse event types (mousePressed,
        mouseReleased, mouseMoved, mouseWheel); anything else is silently
        ignored. Never raises -- input dispatch is a secondary feature of an
        active screencast and a failure here must not tear down the session
        (see module Error Handling Policy).
        """
        if not self._is_active or self._browser_session is None:
            return
        if type not in _SUPPORTED_MOUSE_TYPES:
            logger.debug(f"Ignoring unsupported mouse event type: {type}")
            return

        try:
            params: dict[str, Any] = {
                "type": type,
                "x": x,
                "y": y,
                "button": button,
                "clickCount": click_count,
            }
            if type == "mouseWheel":
                params["deltaX"] = delta_x
                params["deltaY"] = delta_y
            await self._browser_session.cdp_client.send.Input.dispatchMouseEvent(
                params=cast("DispatchMouseEventParameters", params),
                session_id=self._current_session_id,
            )
        except Exception as e:
            logger.debug(f"Failed to dispatch mouse event: {e}")

    async def dispatch_key(
        self,
        type: str,  # noqa: A002
        key: str,
        code: str,
        text: str | None,
    ) -> None:
        """Dispatch a keyboard event to the current CDP target for human takeover.

        `type` must be one of the CDP-native key event types (keyDown, keyUp,
        char); anything else is silently ignored. Never raises -- see
        dispatch_mouse and the module Error Handling Policy.
        """
        if not self._is_active or self._browser_session is None:
            return
        if type not in _SUPPORTED_KEY_TYPES:
            logger.debug(f"Ignoring unsupported key event type: {type}")
            return

        try:
            params: dict[str, Any] = {"type": type, "key": key, "code": code}
            if text is not None:
                params["text"] = text
            await self._browser_session.cdp_client.send.Input.dispatchKeyEvent(
                params=cast("DispatchKeyEventParameters", params),
                session_id=self._current_session_id,
            )
        except Exception as e:
            logger.debug(f"Failed to dispatch key event: {e}")

    def _reset_state(self) -> None:
        if self._input_client is not None and self._original_dispatch_mouse is not None:
            self._input_client.dispatchMouseEvent = self._original_dispatch_mouse
        self._input_client = None
        self._original_dispatch_mouse = None
        self._is_active = False
        self._current_session_id = None
        self._on_frame = None
        self._on_cursor = None
        self._browser_session = None
