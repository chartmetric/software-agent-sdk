import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.tools.browser_use.impl import BrowserToolExecutor
from openhands.tools.browser_use.playwright_server import (
    PlaywrightBrowserServer,
    StaleElementError,
)


@pytest.mark.asyncio
async def test_browser_state_excludes_zero_area_targets():
    executable = BrowserToolExecutor.check_chromium_available()
    if executable is None:
        pytest.skip("Chromium is not installed")
    server = PlaywrightBrowserServer()
    try:
        await server.start(headless=True, executable_path=executable)
        page = await server.get_current_page()
        await page.set_content(
            "<button onclick=\"this.textContent='Opened'\">Open user</button>"
            '<ol tabindex="-1" style="width:300px;height:0;margin:0"></ol>'
            '<button style="width:0;height:0;padding:0;border:0;overflow:hidden">'
            "Collapsed</button>"
        )

        state = json.loads(await server.get_browser_state(include_screenshot=False))

        assert [item["text"] for item in state["interactive_elements"]] == ["Open user"]
        await server.click(state["interactive_elements"][0]["index"])
        assert await page.get_by_role("button", name="Opened").count() == 1
    finally:
        await server.close()


@pytest.fixture
def playwright_runtime():
    page = MagicMock()
    page.url = "about:blank"
    page.goto = AsyncMock()
    page.wait_for_function = AsyncMock()
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
async def test_scroll_to_text_is_one_dom_operation_then_a_settled_read(
    playwright_runtime,
):
    """One DOM operation finds and scrolls; what follows is only reading the
    scroll position until it holds still. 79f2fa2b (2026-09-11): the frame was
    taken while `scroll-behavior: smooth` was still moving the page, and the
    section came out 200px below where it was sent."""
    starter, _, _, _, page = playwright_runtime
    # The finder, then centre-offset/page-height readings: one that still
    # moved (the page grew above the target), one that rests at the centre.
    page.evaluate = AsyncMock(
        side_effect=["Noteworthy Insights", [300, 5000], [0, 5400], [0, 5400]]
    )
    page.wait_for_timeout = AsyncMock()
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(headless=True, executable_path="/usr/bin/chromium")
        result = await server.scroll_to_text("Noteworthy Insights")

    scroll_call = page.evaluate.await_args_list[0]
    assert scroll_call.args[1] == "Noteworthy Insights"
    assert "block: 'center'" in scroll_call.args[0]
    assert "behavior: 'instant'" in scroll_call.args[0]
    # Three re-centring reads: 300px off, at the centre while the page still
    # grew, at the centre with the height unchanged.
    assert page.evaluate.await_count == 4
    assert page.wait_for_timeout.await_count == 2
    assert "behavior: 'instant'" in page.evaluate.await_args_list[-1].args[0]
    assert result == "Scrolled to 'Noteworthy Insights'"


@pytest.mark.asyncio
async def test_the_browser_is_launched_without_smooth_scrolling(playwright_runtime):
    starter, _, browser, _, _ = playwright_runtime
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(headless=True, executable_path="/usr/bin/chromium")

    chromium = starter.start.return_value.chromium
    assert "--disable-smooth-scrolling" in chromium.launch.await_args.kwargs["args"]


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
async def test_stable_frame_wait_reads_pixels_directly_from_cdp(playwright_runtime):
    starter, _, _, context, page = playwright_runtime
    cdp = MagicMock()
    cdp.send = AsyncMock(
        side_effect=[
            {"data": "drawing"},
            {"data": "settled"},
            {"data": "settled"},
        ]
    )
    context.new_cdp_session = AsyncMock(return_value=cdp)
    server = PlaywrightBrowserServer()

    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(headless=True, executable_path="/usr/bin/chromium")
        settled = await server.wait_for_stable_frame(sample_count=4, interval_seconds=0)

    assert settled is True
    assert cdp.send.await_count == 3
    cdp.send.assert_awaited_with(
        "Page.captureScreenshot",
        {"format": "jpeg", "quality": 75, "fromSurface": True},
    )
    page.screenshot.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_navigation_policy_applies_to_context_requests(playwright_runtime):
    starter, _, _, context, _ = playwright_runtime
    policy = AsyncMock(side_effect=ValueError("Blocked by policy"))
    server = PlaywrightBrowserServer()
    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(
            headless=True,
            executable_path="/usr/bin/chromium",
            navigation_policy=policy,
        )
    context.route.assert_awaited_once_with("**/*", server._guard_route)
    request = MagicMock()
    request.url = "http://preview.example.com:8000/private"
    request.is_navigation_request.return_value = True
    request.frame.parent_frame = None
    route = MagicMock(abort=AsyncMock(), continue_=AsyncMock())
    await server._guard_route(route, request)
    policy.assert_awaited_once_with(request.url)
    route.abort.assert_awaited_once_with("blockedbyclient")
    route.continue_.assert_not_awaited()


@pytest.mark.asyncio
async def test_browser_metadata_reads_current_page(playwright_runtime):
    _, _, _, _, page = playwright_runtime
    page.url = "https://preview.example.com/result"
    page.title = AsyncMock(return_value="Result")
    page.locator.return_value.inner_text = AsyncMock(return_value="Rendered page")
    server = PlaywrightBrowserServer()
    server._page = page
    assert await server.browser_metadata() == {
        "url": page.url,
        "title": "Result",
        "text": "Rendered page",
    }


@pytest.mark.asyncio
async def test_navigation_policy_rejects_urls_without_network_requests(
    playwright_runtime,
):
    starter, _, _, _, page = playwright_runtime
    server = PlaywrightBrowserServer()
    policy = AsyncMock(side_effect=ValueError("HTTPS required"))
    with patch(
        "openhands.tools.browser_use.playwright_server.async_playwright",
        return_value=starter,
    ):
        await server.start(
            headless=True,
            executable_path="/usr/bin/chromium",
            navigation_policy=policy,
        )
    with pytest.raises(ValueError, match="HTTPS required"):
        await server.navigate("data:text/html,hello")
    page.goto.assert_not_awaited()


@pytest.mark.asyncio
async def test_registered_values_mask_later_captures_and_metadata(playwright_runtime):
    _, _, _, _, page = playwright_runtime
    secret = "account@example.test"
    page.evaluate.return_value = {"title": secret, "interactive_elements": []}
    page.title = AsyncMock(return_value=f"Profile {secret}")
    elements = page.locator.return_value
    elements.inner_text = AsyncMock(return_value=f"Signed in as {secret}")
    elements.evaluate_all = AsyncMock(return_value=[["Profile"], [secret], [secret]])
    server = PlaywrightBrowserServer()
    server._page = page
    server.set_sensitive_values([secret])
    server.set_sensitive_values(["second-secret"])

    state = json.loads(await server.get_browser_state(include_screenshot=True))
    assert secret not in json.dumps(state)
    page.screenshot.assert_awaited_once_with(
        type="jpeg",
        quality=75,
        mask=[elements.nth(1), elements.nth(2)],
        mask_color="#000000",
    )
    metadata = await server.browser_metadata()
    assert secret not in json.dumps(metadata)
    assert "Signed in as <secret>" == metadata["text"]


@pytest.mark.asyncio
async def test_typing_secret_registers_it_for_later_captures(playwright_runtime):
    _, _, _, _, page = playwright_runtime
    page.locator.return_value.fill = AsyncMock()
    page.evaluate.return_value = {"title": "registered-secret"}
    server = PlaywrightBrowserServer()
    server._page = page
    await server.type_text(0, "registered-secret", secret=True)
    assert "registered-secret" not in await server.get_browser_state()


@pytest.mark.asyncio
async def test_a_page_reading_spends_no_characters_on_padding(playwright_runtime):
    """The state the agent reads is compact JSON, not an indented document.

    Every browser observation is cut at 50,000 characters before it reaches the
    model, so indentation is not merely paid for -- it is paid for by dropping
    the tail of the page. The same object compact measured 0.61-0.62x its
    indented size on real production snapshots.
    """
    starter, _, _, _, page = playwright_runtime
    page.evaluate.return_value = {
        "url": "http://127.0.0.1:3000/dashboard",
        "title": "Dashboard",
        "tabs": [],
        "interactive_elements": [
            {"index": index, "tag": "button", "text": f"Row {index}"}
            for index in range(40)
        ],
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
        serialized = await server.get_browser_state(include_screenshot=False)

    # The page survives the compaction intact; only the whitespace is gone.
    assert json.loads(serialized) == page.evaluate.return_value
    assert "\n" not in serialized
    assert len(serialized) < 0.7 * len(json.dumps(page.evaluate.return_value, indent=2))


@pytest.mark.asyncio
async def test_a_plain_control_carries_no_empty_or_false_fields():
    """An element's absent attributes are absent from its reading.

    A plain <button> has no role, no type and no aria-label, and is not
    disabled; spelling all four out repeats about 40 characters per element on
    a page that may carry 100 of them, and the model reads a missing field and
    an empty one the same way.
    """
    executable = BrowserToolExecutor.check_chromium_available()
    if executable is None:
        pytest.skip("Chromium is not installed")
    server = PlaywrightBrowserServer()
    try:
        await server.start(headless=True, executable_path=executable)
        page = await server.get_current_page()
        await page.set_content(
            '<button>Save</button><input type="text" aria-label="Search" disabled>'
        )

        serialized = await server.get_browser_state(include_screenshot=False)
        elements = json.loads(serialized)["interactive_elements"]

        plain = next(item for item in elements if item.get("text") == "Save")
        assert "disabled" not in plain
        assert "role" not in plain
        assert "type" not in plain
        assert "name" not in plain
        # index 0 is a value, not an absence, and has to survive the same pass.
        assert plain["index"] == 0

        # What is actually set still comes through.
        search = next(item for item in elements if item.get("name") == "Search")
        assert search["disabled"] is True
        assert search["type"] == "text"
    finally:
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("page", "expected"),
    [
        pytest.param(
            "<div id='host'><button onclick=\"document.title='clicked'\">Go</button>"
            "</div>"
            "<script>window.rerender = () => {"
            "  const fresh = document.createElement('button');"
            "  fresh.textContent = 'Go';"
            "  document.getElementById('host').replaceChildren(fresh);"
            "}</script>",
            "no longer on the page",
            id="replaced by a re-render",
        ),
        pytest.param(
            "<button onclick=\"document.title='clicked'\">Go</button>"
            "<script>window.rerender = () => {"
            "  const sheet = document.createElement('div');"
            "  sheet.setAttribute('aria-label', 'Cookie banner');"
            "  sheet.style.cssText = 'position:fixed;inset:0;background:#fff';"
            "  document.body.append(sheet);"
            "}</script>",
            "'Cookie banner'",
            id="covered by an overlay",
        ),
        pytest.param(
            "<button onclick=\"document.title='clicked'\">Go</button>"
            "<script>window.rerender = () => {"
            "  document.querySelector('button').disabled = true;"
            "}</script>",
            "is disabled",
            id="disabled while the state was read",
        ),
    ],
)
async def test_an_action_on_an_element_the_page_no_longer_offers_refuses_at_once(
    page, expected
):
    """The page moves between the state read and the click, as a live app does.

    Each of these used to be the same 30s wait on a selector and then a
    Playwright line naming no remedy: on Chartmetric Pilot production over the
    week to 2026-09-18, 48 of 1,141 browser calls died that way with the action
    never made. The answer is now immediate and says what to do next.
    """
    executable = BrowserToolExecutor.check_chromium_available()
    if executable is None:
        pytest.skip("Chromium is not installed")
    server = PlaywrightBrowserServer()
    try:
        await server.start(headless=True, executable_path=executable)
        live = await server.get_current_page()
        await live.set_content(page)
        state = json.loads(await server.get_browser_state(include_screenshot=False))
        index = next(
            item["index"]
            for item in state["interactive_elements"]
            if item.get("text") == "Go"
        )

        await live.evaluate("window.rerender()")

        started = asyncio.get_running_loop().time()
        with pytest.raises(StaleElementError) as refusal:
            await server.click(index)
        assert asyncio.get_running_loop().time() - started < 5, "answered at once"
        assert expected in str(refusal.value)
        # Every refusal names what to do next.
        assert "read" in str(refusal.value).lower()
        assert await live.title() != "clicked", "the click was not made"
    finally:
        await server.close()
