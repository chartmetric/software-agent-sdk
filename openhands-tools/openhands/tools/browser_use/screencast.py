from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import CDPSession

from openhands.sdk import get_logger


logger = get_logger(__name__)
_SUPPORTED_MOUSE_TYPES = frozenset(
    {"mousePressed", "mouseReleased", "mouseMoved", "mouseWheel"}
)
_SUPPORTED_KEY_TYPES = frozenset({"keyDown", "keyUp", "char"})


@dataclass
class ScreencastSession:
    """Live screencast and takeover input on one Playwright CDP session."""

    _cdp: CDPSession | None = field(default=None, repr=False)
    _on_frame: Callable[[str, dict[str, Any]], None] | None = field(
        default=None, repr=False
    )
    _on_cursor: Callable[[dict[str, Any]], None] | None = field(
        default=None, repr=False
    )
    _is_active: bool = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    async def start(
        self,
        cdp: CDPSession,
        on_frame: Callable[[str, dict[str, Any]], None],
        on_cursor: Callable[[dict[str, Any]], None] | None = None,
        format: str = "jpeg",  # noqa: A002
        quality: int = 80,
        max_width: int = 1280,
        max_height: int = 800,
        every_nth_frame: int = 1,
    ) -> bool:
        try:
            self._cdp = cdp
            self._on_frame = on_frame
            self._on_cursor = on_cursor
            cdp.on("Page.screencastFrame", self._handle_frame)
            await cdp.send(
                "Page.startScreencast",
                {
                    "format": format,
                    "quality": quality,
                    "maxWidth": max_width,
                    "maxHeight": max_height,
                    "everyNthFrame": every_nth_frame,
                },
            )
            self._is_active = True
            return True
        except Exception:
            logger.warning("Failed to start screencast", exc_info=True)
            self._reset()
            return False

    def _handle_frame(self, event: dict[str, Any]) -> None:
        if self._on_frame is not None:
            try:
                self._on_frame(event["data"], dict(event.get("metadata", {})))
            except Exception:
                logger.debug("Screencast frame callback failed", exc_info=True)
        if self._cdp is not None:
            asyncio.create_task(self._ack(event))

    async def _ack(self, event: dict[str, Any]) -> None:
        if self._cdp is None:
            return
        try:
            await self._cdp.send(
                "Page.screencastFrameAck", {"sessionId": event["sessionId"]}
            )
        except Exception:
            logger.debug("Screencast frame acknowledgement failed", exc_info=True)

    async def stop(self) -> bool:
        if not self._is_active or self._cdp is None:
            self._reset()
            return True
        cdp = self._cdp
        try:
            await cdp.send("Page.stopScreencast")
            return True
        except Exception:
            logger.debug("Failed to stop screencast", exc_info=True)
            return False
        finally:
            cdp.remove_listener("Page.screencastFrame", self._handle_frame)
            self._reset()

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
        if (
            not self._is_active
            or self._cdp is None
            or type not in _SUPPORTED_MOUSE_TYPES
        ):
            return
        params: dict[str, Any] = {
            "type": type,
            "x": x,
            "y": y,
            "button": button,
            "clickCount": click_count,
        }
        if type == "mouseWheel":
            params.update({"deltaX": delta_x, "deltaY": delta_y})
        try:
            await self._cdp.send("Input.dispatchMouseEvent", params)
            self._notify_cursor(params)
        except Exception:
            logger.debug("Screencast mouse dispatch failed", exc_info=True)

    async def dispatch_key(
        self,
        type: str,  # noqa: A002
        key: str,
        code: str,
        text: str | None,
        windows_virtual_key_code: int | None = None,
    ) -> None:
        if not self._is_active or self._cdp is None or type not in _SUPPORTED_KEY_TYPES:
            return
        params: dict[str, Any] = {"type": type, "key": key, "code": code}
        if text is not None:
            params["text"] = text
        if windows_virtual_key_code is not None:
            params["windowsVirtualKeyCode"] = windows_virtual_key_code
            params["nativeVirtualKeyCode"] = windows_virtual_key_code
        try:
            await self._cdp.send("Input.dispatchKeyEvent", params)
        except Exception:
            logger.debug("Screencast key dispatch failed", exc_info=True)

    def notify_agent_cursor(self, x: float, y: float, event: str) -> None:
        self._notify_cursor({"x": x, "y": y, "type": event})

    def _notify_cursor(self, params: dict[str, Any]) -> None:
        if self._on_cursor is None:
            return
        try:
            self._on_cursor(
                {"x": params["x"], "y": params["y"], "event": params["type"]}
            )
        except Exception:
            logger.debug("Screencast cursor callback failed", exc_info=True)

    def _reset(self) -> None:
        self._cdp = None
        self._on_frame = None
        self._on_cursor = None
        self._is_active = False
