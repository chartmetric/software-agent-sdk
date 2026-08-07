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
    return browser_session


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
