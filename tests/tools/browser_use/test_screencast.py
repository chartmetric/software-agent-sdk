import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from openhands.tools.browser_use.screencast import ScreencastSession


@pytest.fixture
def cdp():
    session = MagicMock()
    session.send = AsyncMock()
    session.on = MagicMock()
    session.remove_listener = MagicMock()
    return session


@pytest.mark.asyncio
async def test_start_uses_the_playwright_cdp_session(cdp):
    screencast = ScreencastSession()

    assert await screencast.start(cdp, MagicMock(), max_width=1920) is True

    cdp.on.assert_called_once_with("Page.screencastFrame", screencast._handle_frame)
    cdp.send.assert_awaited_once_with(
        "Page.startScreencast",
        {
            "format": "jpeg",
            "quality": 80,
            "maxWidth": 1920,
            "maxHeight": 800,
            "everyNthFrame": 1,
        },
    )
    assert screencast.is_active is True


@pytest.mark.asyncio
async def test_frame_is_forwarded_and_acknowledged(cdp):
    received = []
    screencast = ScreencastSession()
    await screencast.start(
        cdp, lambda data, metadata: received.append((data, metadata))
    )
    cdp.send.reset_mock()

    screencast._handle_frame(
        {"data": "jpeg", "metadata": {"pageScaleFactor": 1}, "sessionId": 9}
    )
    await asyncio.sleep(0)

    assert received == [("jpeg", {"pageScaleFactor": 1})]
    cdp.send.assert_awaited_once_with("Page.screencastFrameAck", {"sessionId": 9})


@pytest.mark.asyncio
async def test_stop_detaches_the_listener_even_when_cdp_fails(cdp):
    screencast = ScreencastSession()
    await screencast.start(cdp, MagicMock())
    cdp.send.side_effect = RuntimeError("target closed")

    assert await screencast.stop() is False

    cdp.remove_listener.assert_called_once_with(
        "Page.screencastFrame", screencast._handle_frame
    )
    assert screencast.is_active is False


@pytest.mark.asyncio
async def test_takeover_mouse_dispatch_reports_the_cursor(cdp):
    cursor = []
    screencast = ScreencastSession()
    await screencast.start(cdp, MagicMock(), on_cursor=cursor.append)
    cdp.send.reset_mock()

    await screencast.dispatch_mouse("mouseMoved", 10, 20)

    cdp.send.assert_awaited_once_with(
        "Input.dispatchMouseEvent",
        {
            "type": "mouseMoved",
            "x": 10,
            "y": 20,
            "button": "left",
            "clickCount": 1,
        },
    )
    assert cursor == [{"x": 10, "y": 20, "event": "mouseMoved"}]


@pytest.mark.asyncio
async def test_takeover_key_dispatch_preserves_editing_key_codes(cdp):
    screencast = ScreencastSession()
    await screencast.start(cdp, MagicMock())
    cdp.send.reset_mock()

    await screencast.dispatch_key("keyDown", "Backspace", "Backspace", None, 8)

    cdp.send.assert_awaited_once_with(
        "Input.dispatchKeyEvent",
        {
            "type": "keyDown",
            "key": "Backspace",
            "code": "Backspace",
            "windowsVirtualKeyCode": 8,
            "nativeVirtualKeyCode": 8,
        },
    )


@pytest.mark.asyncio
async def test_unsupported_input_is_ignored(cdp):
    screencast = ScreencastSession()
    await screencast.start(cdp, MagicMock())
    cdp.send.reset_mock()

    await screencast.dispatch_mouse("tap", 1, 2)
    await screencast.dispatch_key("paste", "v", "KeyV", "v")

    cdp.send.assert_not_awaited()
