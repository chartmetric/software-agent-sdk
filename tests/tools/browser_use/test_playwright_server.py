import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.tools.browser_use.playwright_server import PlaywrightBrowserServer


@pytest.fixture
def playwright_runtime():
    page = MagicMock()
    page.url = "about:blank"
    page.goto = AsyncMock()
    page.evaluate = AsyncMock()
    page.screenshot = AsyncMock(return_value=b"jpeg-bytes")
    page.close = AsyncMock()
    page.is_closed.return_value = False

    context = MagicMock()
    context.pages = []
    context.new_page = AsyncMock(return_value=page)
    context.add_init_script = AsyncMock()
    context.route = AsyncMock()
    context.close = AsyncMock()

    browser = MagicMock()
    browser.is_connected.return_value = True
    browser.new_context = AsyncMock(return_value=context)
    browser.close = AsyncMock()

    chromium = MagicMock()
    chromium.launch = AsyncMock(return_value=browser)
    runtime = SimpleNamespace(chromium=chromium, stop=AsyncMock())
    starter = MagicMock()
    starter.start = AsyncMock(return_value=runtime)
    return starter, runtime, browser, context, page


@pytest.mark.asyncio
async def test_playwright_server_launches_one_persistent_browser(playwright_runtime):
    starter, _, browser, context, _ = playwright_runtime
    server = PlaywrightBrowserServer(session_timeout_minutes=30)

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(
            headless=True,
            executable_path="/usr/bin/chromium",
            chromium_sandbox=True,
            window_size={"width": 1440, "height": 900},
        )
        await server.start(
            headless=True,
            executable_path="/usr/bin/chromium",
            chromium_sandbox=True,
            window_size={"width": 1440, "height": 900},
        )

    starter.start.assert_awaited_once()
    browser.new_context.assert_awaited_once_with(
        viewport={"width": 1440, "height": 900}
    )
    assert context.new_page.await_count == 1
    assert server.is_live is True


@pytest.mark.asyncio
async def test_playwright_navigation_does_not_wait_for_network_idle(
    playwright_runtime,
):
    starter, _, _, _, page = playwright_runtime
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(headless=True, executable_path="/usr/bin/chromium")
        result = await server.navigate("http://127.0.0.1:3000/dashboard")

    page.goto.assert_awaited_once_with(
        "http://127.0.0.1:3000/dashboard",
        wait_until="domcontentloaded",
    )
    assert "127.0.0.1:3000/dashboard" in result


@pytest.mark.asyncio
async def test_playwright_server_closes_context_browser_and_runtime(
    playwright_runtime,
):
    starter, runtime, browser, context, _ = playwright_runtime
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(headless=True, executable_path="/usr/bin/chromium")
        await server.close()

    context.close.assert_awaited_once()
    browser.close.assert_awaited_once()
    runtime.stop.assert_awaited_once()
    assert server.is_live is False


@pytest.mark.asyncio
async def test_browser_state_is_one_dom_read_plus_optional_screenshot(
    playwright_runtime,
):
    starter, _, _, _, page = playwright_runtime
    page.evaluate.return_value = {
        "url": "http://127.0.0.1:3000/dashboard",
        "title": "Dashboard",
        "tabs": [],
        "interactive_elements": [{"index": 0, "tag": "button", "text": "Save"}],
        "viewport": {"width": 1280, "height": 800},
        "page": {"width": 1280, "height": 1600},
        "scroll": {"x": 0, "y": 0},
        "pages_above": 0,
        "pages_below": 1,
        "semantic_outline": {"items": [], "total": 0, "truncated": False},
    }
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(headless=True, executable_path="/usr/bin/chromium")
        state = json.loads(await server.get_browser_state(include_screenshot=True))

    page.evaluate.assert_awaited_once()
    page.screenshot.assert_awaited_once_with(type="jpeg", quality=75)
    assert state["interactive_elements"][0]["text"] == "Save"
    assert state["screenshot"] == base64.b64encode(b"jpeg-bytes").decode()


@pytest.mark.asyncio
async def test_click_targets_the_index_from_the_latest_state(playwright_runtime):
    starter, _, _, _, page = playwright_runtime
    page.evaluate.return_value = {
        "url": "http://127.0.0.1:3000/dashboard",
        "title": "Dashboard",
        "tabs": [],
        "interactive_elements": [{"index": 0, "tag": "button", "text": "Save"}],
        "viewport": {"width": 1280, "height": 800},
        "page": {"width": 1280, "height": 800},
        "scroll": {"x": 0, "y": 0},
        "pages_above": 0,
        "pages_below": 0,
        "semantic_outline": {"items": [], "total": 0, "truncated": False},
    }
    locator = MagicMock()
    locator.count = AsyncMock(return_value=1)
    locator.click = AsyncMock()
    locator.bounding_box = AsyncMock(return_value=None)
    page.locator.return_value = locator
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(headless=True, executable_path="/usr/bin/chromium")
        await server.get_browser_state()
        await server.click(0)

    page.locator.assert_called_with('[data-oh-browser-index="0"]')
    locator.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_viewport_changes_the_page_without_replacing_its_session(
    playwright_runtime,
):
    starter, _, browser, _, page = playwright_runtime
    page.set_viewport_size = AsyncMock()
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(headless=True, executable_path="/usr/bin/chromium")
        result = await server.set_viewport(390, 844)

    page.set_viewport_size.assert_awaited_once_with({"width": 390, "height": 844})
    browser.new_context.assert_awaited_once()
    assert result == "Viewport set to 390x844"


@pytest.mark.asyncio
async def test_secret_input_is_filled_without_echoing_its_value(playwright_runtime):
    starter, _, _, _, page = playwright_runtime
    locator = MagicMock()
    locator.fill = AsyncMock()
    page.locator.return_value = locator
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(headless=True, executable_path="/usr/bin/chromium")
        result = await server.type_text(2, "private-password", secret=True)

    locator.fill.assert_awaited_once_with("private-password")
    assert result == "Typed <secret> into element 2"
    assert "private-password" not in result


@pytest.mark.asyncio
async def test_scroll_to_text_is_one_dom_operation(playwright_runtime):
    starter, _, _, _, page = playwright_runtime
    page.evaluate.return_value = "Noteworthy Insights"
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(headless=True, executable_path="/usr/bin/chromium")
        result = await server.scroll_to_text("Noteworthy Insights")

    page.evaluate.assert_awaited_once()
    assert page.evaluate.await_args.args[1] == "Noteworthy Insights"
    assert "block: 'center'" in page.evaluate.await_args.args[0]
    assert result == "Scrolled to 'Noteworthy Insights'"


@pytest.mark.asyncio
async def test_allowed_domains_guard_top_level_redirects(playwright_runtime):
    starter, _, _, context, _ = playwright_runtime
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(
            headless=True,
            executable_path="/usr/bin/chromium",
            allowed_domains=["preview.example.com"],
        )

    context.route.assert_awaited_once_with("**/*", server._guard_route)
    route = MagicMock()
    route.abort = AsyncMock()
    route.continue_ = AsyncMock()
    request = MagicMock()
    request.url = "https://evil.example/redirect"
    request.is_navigation_request.return_value = True
    request.frame.parent_frame = None

    await server._guard_route(route, request)

    route.abort.assert_awaited_once_with("blockedbyclient")
    route.continue_.assert_not_awaited()


@pytest.mark.asyncio
async def test_switching_pages_rebinds_the_cdp_target(playwright_runtime):
    starter, _, _, context, _ = playwright_runtime
    first_cdp = MagicMock()
    first_cdp.detach = AsyncMock()
    second_cdp = MagicMock()
    second_cdp.detach = AsyncMock()
    context.new_cdp_session = AsyncMock(side_effect=[first_cdp, second_cdp])
    next_page = MagicMock()
    next_page.is_closed.return_value = False
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(headless=True, executable_path="/usr/bin/chromium")
        assert await server.cdp_session() is first_cdp
        await server._activate_page(next_page)
        assert await server.cdp_session() is second_cdp

    first_cdp.detach.assert_awaited_once()
    assert context.new_cdp_session.await_count == 2


@pytest.mark.asyncio
async def test_content_is_bounded_and_names_the_continuation(playwright_runtime):
    starter, _, _, _, page = playwright_runtime
    body = MagicMock()
    body.inner_text = AsyncMock(return_value="x" * 40_000)
    page.locator.return_value = body
    page.url = "https://preview.example.com/report"
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(headless=True, executable_path="/usr/bin/chromium")
        content = await server.get_content(False, 100)

    assert len(content) < 31_000
    assert "start_from_char=30100" in content
    assert "https://preview.example.com/report" in content
