from __future__ import annotations

import base64
import fnmatch
import json
from typing import Any, cast
from urllib.parse import urlparse
from uuid import uuid4

from playwright.async_api import (
    Browser,
    BrowserContext,
    CDPSession,
    Error as PlaywrightError,
    Page,
    Playwright,
    Request,
    Route,
    StorageState,
    TimeoutError as PlaywrightTimeoutError,
    ViewportSize,
    async_playwright,
)

from openhands.tools.browser_use.recording import RecordingSession
from openhands.tools.browser_use.screencast import ScreencastSession
from openhands.tools.browser_use.semantic import FIND_VISIBLE_TEXT_SCRIPT


_INDEX_ATTRIBUTE = "data-oh-browser-index"
_STATE_SCRIPT = r"""
() => {
  const INDEX = 'data-oh-browser-index';
  const LIMIT = 100;
  const rendered = (element) => {
    if (element.getClientRects().length === 0) return false;
    for (let node = element; node; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (style.visibility === 'hidden' || style.display === 'none') return false;
      if (Number.parseFloat(style.opacity || '1') <= 0.05) return false;
    }
    return true;
  };
  document.querySelectorAll(`[${INDEX}]`).forEach((element) => {
    element.removeAttribute(INDEX);
  });
  const selector = [
    'a[href]', 'button', 'input', 'textarea', 'select',
    '[role="button"]', '[role="link"]', '[tabindex]'
  ].join(',');
  const candidates = Array.from(document.querySelectorAll(selector))
    .filter(rendered).slice(0, LIMIT);
  const interactive = candidates.map((element, index) => {
    element.setAttribute(INDEX, String(index));
    const rect = element.getBoundingClientRect();
    return {
      index,
      tag: element.tagName.toLowerCase(),
      role: (element.getAttribute('role') || '').slice(0, 80),
      type: (element.getAttribute('type') || '').slice(0, 80),
      name: (element.getAttribute('aria-label') ||
        element.getAttribute('placeholder') || '').slice(0, 240),
      text: (element.innerText || element.value || '')
        .trim().replace(/\s+/g, ' ').slice(0, 240),
      disabled: Boolean(element.disabled),
      x: Math.round(rect.x),
      y: Math.round(rect.y),
    };
  });
  const outlineSelector = [
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', '[role="heading"]',
    'main', 'nav', 'aside', 'section', 'article', 'form',
    '[role="main"]', '[role="navigation"]', '[role="complementary"]',
    '[role="region"]', '[role="form"]'
  ].join(',');
  const outline = [];
  for (const element of document.querySelectorAll(outlineSelector)) {
    if (!rendered(element)) continue;
    const rect = element.getBoundingClientRect();
    const tag = element.tagName.toLowerCase();
    const role = (element.getAttribute('role') || tag).toLowerCase();
    const stableId = (element.id || '').slice(0, 120);
    const name = (element.getAttribute('aria-label') || element.innerText || '')
      .trim().replace(/\s+/g, ' ').slice(0, 160);
    if (!name && !stableId && !['main', 'nav', 'aside'].includes(tag)) continue;
    outline.push({
      kind: /^h[1-6]$/.test(tag) || role === 'heading' ? 'heading' : 'landmark',
      tag,
      role: tag === 'nav' && role === 'nav' ? 'navigation' : role,
      name: name || stableId || role,
      id: stableId,
      y: Math.round(rect.top + scrollY),
      location: rect.bottom < 0
        ? 'above' : rect.top > innerHeight ? 'below' : 'viewport',
    });
    if (outline.length === 80) break;
  }
  const root = document.documentElement;
  const body = document.body;
  const pageWidth = Math.max(root.scrollWidth, body ? body.scrollWidth : 0);
  const pageHeight = Math.max(root.scrollHeight, body ? body.scrollHeight : 0);
  const below = Math.max(pageHeight - (scrollY + innerHeight), 0);
  return {
    url: location.href,
    title: document.title,
    tabs: [],
    interactive_elements: interactive,
    viewport: {width: innerWidth, height: innerHeight},
    page: {width: pageWidth, height: pageHeight},
    scroll: {x: scrollX, y: scrollY},
    pages_above: innerHeight ? Math.round(scrollY / innerHeight * 10) / 10 : 0,
    pages_below: innerHeight ? Math.round(below / innerHeight * 10) / 10 : 0,
    semantic_outline: {
      items: outline,
      total: outline.length,
      truncated: outline.length === 80,
    },
  };
}
"""


class PlaywrightBrowserServer:
    """One persistent Playwright Chromium session shared by browser tools."""

    def __init__(self, session_timeout_minutes: int = 30) -> None:
        self.session_timeout_minutes = session_timeout_minutes
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._pages: dict[str, Page] = {}
        self._allowed_domains: tuple[str, ...] = ()
        self._inject_scripts: list[str] = []
        self._cdp_session: CDPSession | None = None
        self._cdp_page: Page | None = None
        self._recording_session: RecordingSession | None = None
        self._screencast_session: ScreencastSession | None = None
        self._screencast_request: tuple[Any, dict[str, Any]] | None = None

    @property
    def is_live(self) -> bool:
        return self._browser is not None and self._browser.is_connected()

    @property
    def _is_recording(self) -> bool:
        return bool(self._recording_session and self._recording_session.is_active)

    @property
    def browser_session(self) -> PlaywrightBrowserServer:
        return self

    async def start(
        self,
        *,
        headless: bool,
        executable_path: str,
        chromium_sandbox: bool = False,
        window_size: ViewportSize | None = None,
        allowed_domains: list[str] | None = None,
        **_: Any,
    ) -> None:
        if self.is_live:
            return
        self._playwright = await async_playwright().start()
        self._allowed_domains = tuple(allowed_domains or ())
        launch_args = ["--disable-dev-shm-usage"]
        if window_size is not None:
            launch_args.append(
                f"--window-size={window_size['width']},{window_size['height']}"
            )
        self._browser = await self._playwright.chromium.launch(
            headless=headless,
            executable_path=executable_path,
            chromium_sandbox=chromium_sandbox,
            args=launch_args,
        )
        self._context = await self._browser.new_context(
            viewport=window_size or {"width": 1280, "height": 800}
        )
        for script in self._inject_scripts:
            await self._context.add_init_script(script=script)
        if self._allowed_domains:
            await self._context.route("**/*", self._guard_route)
        self._context.on("page", self._register_page)
        self._page = await self._context.new_page()
        self._register_page(self._page)

    async def navigate(self, url: str, new_tab: bool = False) -> str:
        self._validate_url(url)
        page = await self._new_page() if new_tab else self._require_page()
        await page.goto(url, wait_until="domcontentloaded")
        await self._wait_for_meaningful_page(page)
        return f"Navigated to {url}"

    async def go_back(self) -> str:
        page = self._require_page()
        await page.go_back(wait_until="domcontentloaded")
        return f"Navigated back to {page.url}"

    async def get_browser_state(self, include_screenshot: bool = False) -> str:
        page = self._require_page()
        state = await page.evaluate(_STATE_SCRIPT)
        if not isinstance(state, dict):
            raise RuntimeError("Browser state response was invalid")
        if include_screenshot:
            screenshot = await page.screenshot(type="jpeg", quality=75)
            state["screenshot"] = base64.b64encode(screenshot).decode()
        return json.dumps(state, indent=2)

    async def click(self, index: int, new_tab: bool = False) -> str:
        locator = self._indexed_locator(index)
        box = await locator.bounding_box()
        if new_tab:
            page = self._require_page()
            try:
                async with page.expect_popup(timeout=2000) as popup:
                    await locator.click()
                await self._activate_page(await popup.value)
            except PlaywrightTimeoutError:
                # The click already happened. A target that chose same-tab
                # navigation is still a successful click.
                pass
        else:
            await locator.click()
        await self._wait_for_meaningful_page(self._require_page())
        if box is not None and self._screencast_session is not None:
            self._screencast_session.notify_agent_cursor(
                box["x"] + box["width"] / 2,
                box["y"] + box["height"] / 2,
                "mouseReleased",
            )
        return f"Clicked element {index}"

    async def _wait_for_meaningful_page(self, page: Page) -> None:
        try:
            await page.wait_for_function(
                """
                () => Boolean(
                  document.body?.innerText.trim() ||
                  document.querySelector(
                    'a[href], button, input, textarea, select, canvas, [role]'
                  )
                )
                """,
                timeout=2000,
            )
        except PlaywrightTimeoutError:
            pass

    async def type_text(self, index: int, text: str, *, secret: bool = False) -> str:
        locator = self._indexed_locator(index)
        await locator.fill(text)
        value = "<secret>" if secret else repr(text)
        return f"Typed {value} into element {index}"

    async def scroll(self, direction: str = "down") -> str:
        page = self._require_page()
        viewport = page.viewport_size or {"width": 1280, "height": 800}
        if direction not in {"up", "down"}:
            raise ValueError("Scroll direction must be 'up' or 'down'")
        delta = viewport["height"] * (1 if direction == "down" else -1)
        await page.mouse.wheel(0, delta)
        return f"Scrolled {direction}"

    async def scroll_to_text(self, text: str) -> str:
        page = self._require_page()
        found = await page.evaluate(
            """
            (wanted) => {
              const exactId = document.getElementById(wanted);
              const rendered = (element) => {
                if (element.getClientRects().length === 0) return false;
                for (let node = element; node; node = node.parentElement) {
                  const style = getComputedStyle(node);
                  if (style.visibility === 'hidden' || style.display === 'none') {
                    return false;
                  }
                }
                return true;
              };
              const candidates = exactId ? [exactId] : Array.from(
                document.querySelectorAll('body *')
              ).filter((element) =>
                (element.innerText || '').toLowerCase().includes(wanted.toLowerCase())
              );
              const target = candidates.find(rendered);
              if (!target) return false;
              target.scrollIntoView({block: 'center', inline: 'nearest'});
              return (target.innerText || '').trim() || target.id || wanted;
            }
            """,
            text,
        )
        if not found:
            return (
                f"No element on the page shows {text!r}. It may not have loaded "
                "yet, may be behind a tab, or may be on another page. Read "
                "browser_get_content before concluding it is absent."
            )
        return f"Scrolled to {found!r}"

    async def find_visible_text(self, text: str, max_results: int = 10) -> str:
        page = self._require_page()
        result = await page.evaluate(
            FIND_VISIBLE_TEXT_SCRIPT, {"needle": text, "limit": max_results}
        )
        return json.dumps(result, indent=2)

    async def set_viewport(self, width: int, height: int) -> str:
        page = self._require_page()
        await page.set_viewport_size({"width": width, "height": height})
        return f"Viewport set to {width}x{height}"

    async def get_storage(self) -> str:
        context = self._require_context()
        state = cast(dict[str, Any], await context.storage_state(indexed_db=True))
        page = self._require_page()
        try:
            origin, session_storage = await page.evaluate(
                """
                () => [location.origin, Object.entries(sessionStorage).map(
                  ([name, value]) => ({name, value})
                )]
                """
            )
        except PlaywrightError:
            return json.dumps(state, indent=2)
        origins = state.setdefault("origins", [])
        stored_origin = next(
            (candidate for candidate in origins if candidate.get("origin") == origin),
            None,
        )
        if stored_origin is None:
            stored_origin = {"origin": origin, "localStorage": []}
            origins.append(stored_origin)
        stored_origin["sessionStorage"] = session_storage
        return json.dumps(state, indent=2)

    async def set_storage(self, storage_state: dict[str, Any]) -> str:
        context = self._require_context()
        playwright_state = cast(
            StorageState,
            {
                "cookies": storage_state.get("cookies", []),
                "origins": [
                    {
                        "origin": origin["origin"],
                        "localStorage": origin.get("localStorage", []),
                    }
                    for origin in storage_state.get("origins", [])
                    if origin.get("origin")
                ],
            },
        )
        await context.set_storage_state(playwright_state)
        page = self._require_page()
        current_origin = await page.evaluate("location.origin")
        for origin in storage_state.get("origins", []):
            if origin.get("origin") != current_origin:
                continue
            await page.evaluate(
                """
                (items) => {
                  sessionStorage.clear();
                  for (const item of items) {
                    sessionStorage.setItem(item.name || item.key, item.value);
                  }
                }
                """,
                origin.get("sessionStorage", []),
            )
        return "Browser storage updated successfully"

    async def get_current_page(self) -> Page:
        return self._require_page()

    async def list_tabs(self) -> str:
        self._sync_pages()
        tabs = [
            {"id": tab_id, "url": page.url, "active": page is self._page}
            for tab_id, page in self._pages.items()
        ]
        return json.dumps(tabs, indent=2)

    async def switch_tab(self, tab_id: str) -> str:
        self._sync_pages()
        page = self._pages.get(tab_id)
        if page is None:
            raise ValueError(f"Tab {tab_id!r} was not found")
        await self._activate_page(page)
        await page.bring_to_front()
        return f"Switched to tab {tab_id}"

    async def close_tab(self, tab_id: str) -> str:
        self._sync_pages()
        page = self._pages.get(tab_id)
        if page is None:
            raise ValueError(f"Tab {tab_id!r} was not found")
        await page.close()
        self._pages.pop(tab_id, None)
        if page is self._page:
            next_page = next(iter(self._pages.values()), None)
            if next_page is not None:
                await self._activate_page(next_page)
            else:
                self._page = None
        return f"Closed tab {tab_id}"

    async def get_content(self, extract_links: bool, start_from_char: int) -> str:
        page = self._require_page()
        content = await page.locator("body").inner_text()
        if extract_links:
            links = await page.locator("a[href]").evaluate_all(
                r"""
                (elements) => elements.slice(0, 200).map((element) => ({
                  text: (element.innerText || '').trim().replace(/\s+/g, ' '),
                  href: element.href,
                }))
                """
            )
            if links:
                rendered = "\n".join(
                    f"- [{link['text'] or link['href']}]({link['href']})"
                    for link in links
                )
                content = f"{content}\n\nLinks:\n{rendered}"
        if start_from_char >= len(content) and content:
            return (
                f"start_from_char ({start_from_char}) exceeds content length "
                f"({len(content)})."
            )
        limit = 30_000
        end = min(start_from_char + limit, len(content))
        chunk = content[start_from_char:end]
        continuation = (
            f" Truncated; use start_from_char={end} to continue."
            if end < len(content)
            else ""
        )
        return (
            f"<url>\n{page.url}\n</url>\n"
            f"<content_stats>\nVisible text characters: {len(content)}."
            f"{continuation}\n</content_stats>\n"
            f"<webpage_content>\n{chunk}\n</webpage_content>"
        )

    def set_inject_scripts(self, scripts: list[str]) -> None:
        self._inject_scripts = list(scripts)

    async def inject_scripts(self) -> None:
        context = self._require_context()
        for script in self._inject_scripts:
            await context.add_init_script(script=script)

    async def cdp_session(self) -> CDPSession:
        page = self._require_page()
        if self._cdp_session is None or self._cdp_page is not page:
            self._cdp_session = await self._require_context().new_cdp_session(page)
            self._cdp_page = page
        return self._cdp_session

    async def start_recording(self, output_dir: str | None = None) -> str:
        if self._recording_session is None:
            self._recording_session = RecordingSession(output_dir=output_dir)
        return await self._recording_session.start(
            self._require_context(), self._require_page
        )

    async def stop_recording(self) -> str:
        if self._recording_session is None:
            return "Error: Not recording. Call browser_start_recording first."
        result = await self._recording_session.stop()
        self._recording_session.reset()
        return result

    async def flush_recording_events(self) -> int:
        if self._recording_session is None:
            return 0
        return await self._recording_session.flush_events()

    async def restart_recording_on_new_page(self) -> None:
        if self._recording_session is not None:
            await self._recording_session.restart_on_new_page()

    async def start_screencast(self, on_frame, **kwargs: Any) -> bool:
        self._screencast_request = (on_frame, dict(kwargs))
        if self._screencast_session is not None:
            await self._screencast_session.stop()
        self._screencast_session = ScreencastSession()
        return await self._screencast_session.start(
            await self.cdp_session(), on_frame, **kwargs
        )

    async def stop_screencast(self, *, preserve_request: bool = False) -> bool:
        if not preserve_request:
            self._screencast_request = None
        if self._screencast_session is None:
            return True
        result = await self._screencast_session.stop()
        self._screencast_session = None
        return result

    async def dispatch_screencast_mouse(self, **kwargs: Any) -> None:
        if self._screencast_session is not None:
            await self._screencast_session.dispatch_mouse(**kwargs)

    async def dispatch_screencast_key(self, **kwargs: Any) -> None:
        if self._screencast_session is not None:
            await self._screencast_session.dispatch_key(**kwargs)

    async def close(self) -> None:
        await self.stop_screencast()
        if self._recording_session is not None and self._recording_session.is_active:
            await self._recording_session.stop()
        self._recording_session = None
        context, browser, playwright = self._context, self._browser, self._playwright
        self._page = None
        self._pages.clear()
        self._cdp_session = None
        self._cdp_page = None
        self._context = None
        self._browser = None
        self._playwright = None
        if context is not None:
            await context.close()
        if browser is not None:
            await browser.close()
        if playwright is not None:
            await playwright.stop()

    def _require_page(self) -> Page:
        if self._page is None or self._page.is_closed():
            raise RuntimeError("Browser session is not initialized")
        return self._page

    def _require_context(self) -> BrowserContext:
        if self._context is None:
            raise RuntimeError("Browser session is not initialized")
        return self._context

    def _indexed_locator(self, index: int):
        page = self._require_page()
        locator = page.locator(f'[{_INDEX_ATTRIBUTE}="{index}"]')
        return locator

    async def _new_page(self) -> Page:
        page = await self._require_context().new_page()
        await self._activate_page(page)
        return page

    async def _activate_page(self, page: Page) -> None:
        if page is self._page:
            return
        request = self._screencast_request
        if self._screencast_session is not None:
            await self.stop_screencast(preserve_request=True)
        if self._cdp_session is not None:
            try:
                await self._cdp_session.detach()
            except PlaywrightError:
                pass
        self._cdp_session = None
        self._cdp_page = None
        self._page = page
        self._register_page(page)
        if request is not None:
            await self.start_screencast(request[0], **request[1])

    def _register_page(self, page: Page) -> None:
        if any(candidate is page for candidate in self._pages.values()):
            return
        self._pages[f"tab-{uuid4().hex[:12]}"] = page

    def _sync_pages(self) -> None:
        context = self._require_context()
        live_pages = [page for page in context.pages if not page.is_closed()]
        self._pages = {
            tab_id: page
            for tab_id, page in self._pages.items()
            if any(candidate is page for candidate in live_pages)
        }
        for page in live_pages:
            self._register_page(page)

    def _validate_url(self, url: str) -> None:
        if not self._allowed_domains:
            return
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if any(
            fnmatch.fnmatch(hostname, pattern) or hostname == pattern.removeprefix("*.")
            for pattern in self._allowed_domains
        ):
            return
        raise ValueError(f"Navigation to {hostname!r} is not allowed")

    async def _guard_route(self, route: Route, request: Request) -> None:
        if request.is_navigation_request() and request.frame.parent_frame is None:
            try:
                self._validate_url(request.url)
            except ValueError:
                await route.abort("blockedbyclient")
                return
        await route.continue_()
