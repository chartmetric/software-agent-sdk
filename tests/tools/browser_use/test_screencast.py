"""Tests for CDP screencast session management.

These tests verify that:
1. start() wires the CDP screencast calls with the expected parameters
2. Frames are always acknowledged, even if the caller's callback raises
3. Frames from a stale (already-stopped) session are ignored
4. stop() clears state even when the underlying CDP call fails
"""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from cdp_use.cdp.page.events import ScreencastFrameEvent

from openhands.tools.browser_use.screencast import ScreencastSession


def _make_frame_event(
    data: str = "base64==", metadata: dict[str, Any] | None = None, session_id: int = 1
) -> ScreencastFrameEvent:
    """Build a CDP ScreencastFrameEvent for tests without hand-filling every
    ScreencastFrameMetadata field the type checker demands but the code
    under test never reads."""
    return cast(
        ScreencastFrameEvent,
        {"data": data, "metadata": metadata or {}, "sessionId": session_id},
    )


@pytest.fixture
def mock_cdp_session():
    cdp_session = MagicMock()
    cdp_session.session_id = "test-session-id"
    cdp_session.cdp_client = MagicMock()
    cdp_session.cdp_client.send = MagicMock()
    cdp_session.cdp_client.send.Page = MagicMock()
    cdp_session.cdp_client.send.Page.startScreencast = AsyncMock()
    return cdp_session


@pytest.fixture
def mock_browser_session(mock_cdp_session):
    browser_session = MagicMock()
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
    browser_session.cdp_client = MagicMock()
    browser_session.cdp_client.register = MagicMock()
    browser_session.cdp_client.register.Page = MagicMock()
    browser_session.cdp_client.send = MagicMock()
    browser_session.cdp_client.send.Page = MagicMock()
    browser_session.cdp_client.send.Page.stopScreencast = AsyncMock()
    browser_session.cdp_client.send.Page.screencastFrameAck = AsyncMock()
    browser_session.cdp_client.send.Input = MagicMock()
    browser_session.cdp_client.send.Input.dispatchMouseEvent = AsyncMock()
    browser_session.cdp_client.send.Input.dispatchKeyEvent = AsyncMock()
    return browser_session


@pytest.fixture
async def started_session(mock_browser_session) -> ScreencastSession:
    """A ScreencastSession that has already start()ed against
    mock_browser_session, for dispatch_mouse/dispatch_key tests that only
    care about behavior once a session is active."""
    session = ScreencastSession()
    await session.start(mock_browser_session, MagicMock())
    return session


class TestScreencastStart:
    @pytest.mark.asyncio
    async def test_start_sends_expected_cdp_params(
        self, mock_browser_session, mock_cdp_session
    ):
        session = ScreencastSession()
        on_frame = MagicMock()

        result = await session.start(
            mock_browser_session,
            on_frame,
            format="jpeg",
            quality=70,
            max_width=1024,
            max_height=768,
            every_nth_frame=2,
        )

        assert result is True
        assert session.is_active is True
        mock_cdp_session.cdp_client.send.Page.startScreencast.assert_awaited_once_with(
            params={
                "format": "jpeg",
                "quality": 70,
                "maxWidth": 1024,
                "maxHeight": 768,
                "everyNthFrame": 2,
            },
            session_id="test-session-id",
        )

    @pytest.mark.asyncio
    async def test_start_registers_frame_handler_on_root_client(
        self, mock_browser_session
    ):
        session = ScreencastSession()
        await session.start(mock_browser_session, MagicMock())

        mock_browser_session.cdp_client.register.Page.screencastFrame.assert_called_once_with(
            session._handle_frame
        )

    @pytest.mark.asyncio
    async def test_start_returns_false_and_never_raises_on_cdp_failure(
        self, mock_browser_session, mock_cdp_session
    ):
        mock_cdp_session.cdp_client.send.Page.startScreencast = AsyncMock(
            side_effect=RuntimeError("CDP target crashed")
        )

        session = ScreencastSession()
        result = await session.start(mock_browser_session, MagicMock())

        assert result is False
        assert session.is_active is False


class TestScreencastFrameHandling:
    @pytest.mark.asyncio
    async def test_frame_is_forwarded_to_on_frame_callback(self, mock_browser_session):
        session = ScreencastSession()
        received: list[tuple[str, dict]] = []
        await session.start(
            mock_browser_session, lambda data, meta: received.append((data, meta))
        )

        session._handle_frame(
            _make_frame_event(metadata={"deviceWidth": 800}),
            "test-session-id",
        )

        assert received == [("base64==", {"deviceWidth": 800})]

    @pytest.mark.asyncio
    async def test_handle_frame_does_not_raise_if_callback_raises(
        self, mock_browser_session
    ):
        session = ScreencastSession()

        def raising_callback(data, meta):
            raise ValueError("boom")

        await session.start(mock_browser_session, raising_callback)

        # Must not propagate the callback's exception into CDP's dispatch loop.
        session._handle_frame(_make_frame_event(session_id=42), "test-session-id")

    @pytest.mark.asyncio
    async def test_ack_frame_uses_root_client_and_frame_session_id(
        self, mock_browser_session
    ):
        session = ScreencastSession()
        await session.start(mock_browser_session, MagicMock())

        await session._ack_frame(_make_frame_event(session_id=42), "test-session-id")

        mock_browser_session.cdp_client.send.Page.screencastFrameAck.assert_awaited_once_with(
            params={"sessionId": 42}, session_id="test-session-id"
        )

    @pytest.mark.asyncio
    async def test_frame_from_stale_session_is_ignored(self, mock_browser_session):
        session = ScreencastSession()
        received: list[str] = []
        await session.start(
            mock_browser_session, lambda data, meta: received.append(data)
        )

        # Simulate a frame arriving from a target we've since moved away from.
        session._handle_frame(_make_frame_event(data="stale"), "old-session-id")

        assert received == []


class TestScreencastStop:
    @pytest.mark.asyncio
    async def test_stop_sends_stop_screencast_with_current_session_id(
        self, mock_browser_session
    ):
        session = ScreencastSession()
        await session.start(mock_browser_session, MagicMock())

        result = await session.stop()

        assert result is True
        assert session.is_active is False
        mock_browser_session.cdp_client.send.Page.stopScreencast.assert_awaited_once_with(
            session_id="test-session-id"
        )

    @pytest.mark.asyncio
    async def test_stop_when_never_started_is_a_noop_success(self):
        session = ScreencastSession()
        assert await session.stop() is True
        assert session.is_active is False

    @pytest.mark.asyncio
    async def test_stop_clears_state_even_if_cdp_call_fails(self, mock_browser_session):
        mock_browser_session.cdp_client.send.Page.stopScreencast = AsyncMock(
            side_effect=RuntimeError("already gone")
        )
        session = ScreencastSession()
        await session.start(mock_browser_session, MagicMock())

        result = await session.stop()

        assert result is False
        assert session.is_active is False
        assert session._current_session_id is None


class TestScreencastDispatchMouse:
    @pytest.mark.asyncio
    async def test_dispatch_mouse_sends_expected_cdp_params(
        self, started_session, mock_browser_session
    ):
        await started_session.dispatch_mouse(
            "mousePressed", x=12.5, y=34.0, button="left", click_count=2
        )

        mock_browser_session.cdp_client.send.Input.dispatchMouseEvent.assert_awaited_once_with(
            params={
                "type": "mousePressed",
                "x": 12.5,
                "y": 34.0,
                "button": "left",
                "clickCount": 2,
            },
            session_id="test-session-id",
        )

    @pytest.mark.asyncio
    async def test_dispatch_mouse_wheel_includes_deltas(
        self, started_session, mock_browser_session
    ):
        await started_session.dispatch_mouse(
            "mouseWheel", x=1, y=2, delta_x=10, delta_y=-5
        )

        mock_browser_session.cdp_client.send.Input.dispatchMouseEvent.assert_awaited_once_with(
            params={
                "type": "mouseWheel",
                "x": 1,
                "y": 2,
                "button": "left",
                "clickCount": 1,
                "deltaX": 10,
                "deltaY": -5,
            },
            session_id="test-session-id",
        )

    @pytest.mark.asyncio
    async def test_dispatch_mouse_routes_through_current_session_id(
        self, started_session, mock_browser_session
    ):
        started_session._current_session_id = "some-other-target"

        await started_session.dispatch_mouse("mouseMoved", x=0, y=0)

        assert (
            mock_browser_session.cdp_client.send.Input.dispatchMouseEvent.await_args.kwargs[
                "session_id"
            ]
            == "some-other-target"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cdp_type", ["mousePressed", "mouseReleased", "mouseMoved", "mouseWheel"]
    )
    async def test_dispatch_mouse_supports_all_documented_types(
        self, started_session, mock_browser_session, cdp_type
    ):
        await started_session.dispatch_mouse(cdp_type, x=0, y=0)

        mock_browser_session.cdp_client.send.Input.dispatchMouseEvent.assert_awaited_once()
        assert (
            mock_browser_session.cdp_client.send.Input.dispatchMouseEvent.await_args.kwargs[
                "params"
            ]["type"]
            == cdp_type
        )

    @pytest.mark.asyncio
    async def test_dispatch_mouse_ignores_unsupported_type(
        self, started_session, mock_browser_session
    ):
        await started_session.dispatch_mouse("mouseWheelBogus", x=0, y=0)

        mock_browser_session.cdp_client.send.Input.dispatchMouseEvent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_mouse_noop_when_not_active(self, mock_browser_session):
        session = ScreencastSession()  # never started

        await session.dispatch_mouse("mousePressed", x=0, y=0)

        mock_browser_session.cdp_client.send.Input.dispatchMouseEvent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_mouse_never_raises_on_cdp_failure(
        self, started_session, mock_browser_session
    ):
        mock_browser_session.cdp_client.send.Input.dispatchMouseEvent = AsyncMock(
            side_effect=RuntimeError("CDP target crashed")
        )

        # Must not raise.
        await started_session.dispatch_mouse("mousePressed", x=0, y=0)


class TestScreencastDispatchKey:
    @pytest.mark.asyncio
    async def test_dispatch_key_sends_expected_cdp_params(
        self, started_session, mock_browser_session
    ):
        await started_session.dispatch_key("keyDown", key="a", code="KeyA", text="a")

        mock_browser_session.cdp_client.send.Input.dispatchKeyEvent.assert_awaited_once_with(
            params={"type": "keyDown", "key": "a", "code": "KeyA", "text": "a"},
            session_id="test-session-id",
        )

    @pytest.mark.asyncio
    async def test_dispatch_key_omits_text_when_none(
        self, started_session, mock_browser_session
    ):
        await started_session.dispatch_key("keyUp", key="a", code="KeyA", text=None)

        mock_browser_session.cdp_client.send.Input.dispatchKeyEvent.assert_awaited_once_with(
            params={"type": "keyUp", "key": "a", "code": "KeyA"},
            session_id="test-session-id",
        )

    @pytest.mark.asyncio
    async def test_dispatch_key_sends_virtual_key_code_on_both_fields(
        self, started_session, mock_browser_session
    ):
        """Chrome resolves editing commands from the virtual key code, not
        from `key`: a Backspace dispatched without one fires the page's
        keydown handlers and deletes nothing, so a human taking over the
        screencast could type but never correct a typo."""
        await started_session.dispatch_key(
            "keyDown",
            key="Backspace",
            code="Backspace",
            text=None,
            windows_virtual_key_code=8,
        )

        mock_browser_session.cdp_client.send.Input.dispatchKeyEvent.assert_awaited_once_with(
            params={
                "type": "keyDown",
                "key": "Backspace",
                "code": "Backspace",
                "windowsVirtualKeyCode": 8,
                "nativeVirtualKeyCode": 8,
            },
            session_id="test-session-id",
        )

    @pytest.mark.asyncio
    async def test_dispatch_key_omits_virtual_key_code_when_none(
        self, started_session, mock_browser_session
    ):
        """A client that predates the field keeps the previous wire shape
        rather than being sent a zero, which Chrome would read as a real
        (and wrong) key."""
        await started_session.dispatch_key(
            "keyDown", key="a", code="KeyA", text="a", windows_virtual_key_code=None
        )

        dispatch = mock_browser_session.cdp_client.send.Input.dispatchKeyEvent
        params = dispatch.await_args.kwargs["params"]
        assert "windowsVirtualKeyCode" not in params
        assert "nativeVirtualKeyCode" not in params

    @pytest.mark.asyncio
    async def test_dispatch_key_routes_through_current_session_id(
        self, started_session, mock_browser_session
    ):
        started_session._current_session_id = "some-other-target"

        await started_session.dispatch_key("char", key="a", code="KeyA", text="a")

        assert (
            mock_browser_session.cdp_client.send.Input.dispatchKeyEvent.await_args.kwargs[
                "session_id"
            ]
            == "some-other-target"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cdp_type", ["keyDown", "keyUp", "char"])
    async def test_dispatch_key_supports_all_documented_types(
        self, started_session, mock_browser_session, cdp_type
    ):
        await started_session.dispatch_key(cdp_type, key="a", code="KeyA", text=None)

        mock_browser_session.cdp_client.send.Input.dispatchKeyEvent.assert_awaited_once()
        assert (
            mock_browser_session.cdp_client.send.Input.dispatchKeyEvent.await_args.kwargs[
                "params"
            ]["type"]
            == cdp_type
        )

    @pytest.mark.asyncio
    async def test_dispatch_key_ignores_unsupported_type(
        self, started_session, mock_browser_session
    ):
        await started_session.dispatch_key(
            "rawKeyDown", key="a", code="KeyA", text=None
        )

        mock_browser_session.cdp_client.send.Input.dispatchKeyEvent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_key_noop_when_not_active(self, mock_browser_session):
        session = ScreencastSession()  # never started

        await session.dispatch_key("keyDown", key="a", code="KeyA", text=None)

        mock_browser_session.cdp_client.send.Input.dispatchKeyEvent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_key_never_raises_on_cdp_failure(
        self, started_session, mock_browser_session
    ):
        mock_browser_session.cdp_client.send.Input.dispatchKeyEvent = AsyncMock(
            side_effect=RuntimeError("CDP target crashed")
        )

        # Must not raise.
        await started_session.dispatch_key("keyDown", key="a", code="KeyA", text=None)


class TestScreencastCursorReporting:
    """Cursor reporting: with `on_cursor` supplied, every mouse event
    dispatched through the root CDP client is forwarded unchanged and its
    position reported; stop() restores the original dispatch method."""

    @pytest.mark.asyncio
    async def test_dispatched_mouse_event_is_forwarded_and_reported(
        self, mock_browser_session
    ):
        original = mock_browser_session.cdp_client.send.Input.dispatchMouseEvent
        cursor_events: list[dict[str, Any]] = []
        session = ScreencastSession()
        assert await session.start(
            mock_browser_session, MagicMock(), on_cursor=cursor_events.append
        )

        wrapped = mock_browser_session.cdp_client.send.Input.dispatchMouseEvent
        assert wrapped is not original
        params = {"type": "mousePressed", "x": 10.0, "y": 20.0, "button": "left"}
        await wrapped(params, session_id="s1")

        original.assert_awaited_once_with(params, session_id="s1")
        assert cursor_events == [{"x": 10.0, "y": 20.0, "event": "mousePressed"}]

    @pytest.mark.asyncio
    async def test_stop_restores_original_dispatch_method(self, mock_browser_session):
        original = mock_browser_session.cdp_client.send.Input.dispatchMouseEvent
        session = ScreencastSession()
        await session.start(mock_browser_session, MagicMock(), on_cursor=MagicMock())

        await session.stop()

        assert mock_browser_session.cdp_client.send.Input.dispatchMouseEvent is original

    @pytest.mark.asyncio
    async def test_without_on_cursor_dispatch_method_is_untouched(
        self, mock_browser_session
    ):
        original = mock_browser_session.cdp_client.send.Input.dispatchMouseEvent
        session = ScreencastSession()
        await session.start(mock_browser_session, MagicMock())

        assert mock_browser_session.cdp_client.send.Input.dispatchMouseEvent is original

    @pytest.mark.asyncio
    async def test_cursor_callback_failure_does_not_break_dispatch(
        self, mock_browser_session
    ):
        original = mock_browser_session.cdp_client.send.Input.dispatchMouseEvent
        session = ScreencastSession()
        await session.start(
            mock_browser_session,
            MagicMock(),
            on_cursor=MagicMock(side_effect=RuntimeError("viewer went away")),
        )

        wrapped = mock_browser_session.cdp_client.send.Input.dispatchMouseEvent
        # Must not raise, and must still forward to the original.
        await wrapped({"type": "mouseMoved", "x": 1, "y": 2})

        original.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_malformed_or_unsupported_events_are_not_reported(
        self, mock_browser_session
    ):
        cursor_events: list[dict[str, Any]] = []
        session = ScreencastSession()
        await session.start(
            mock_browser_session, MagicMock(), on_cursor=cursor_events.append
        )

        wrapped = mock_browser_session.cdp_client.send.Input.dispatchMouseEvent
        await wrapped({"type": "unsupportedType", "x": 1, "y": 2})
        await wrapped({"type": "mousePressed", "x": "not-a-number", "y": 2})

        assert cursor_events == []
