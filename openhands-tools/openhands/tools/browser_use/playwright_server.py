from __future__ import annotations

import asyncio
import base64
import fnmatch
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast
from urllib.parse import urlparse
from uuid import uuid4

from playwright.async_api import (
    Browser,
    BrowserContext,
    CDPSession,
    Error as PlaywrightError,
    Locator,
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


class StaleElementError(RuntimeError):
    """The element an action names is not the one the page offered.

    Raised instead of letting a selector wait out its timeout on a node the
    page has replaced. Carries what changed and what to do about it.
    """


_STATE_SCRIPT = r"""
() => {
  const INDEX = 'data-oh-browser-index';
  // An empty string, a false flag and a null read the same to the model as
  // the field not being there, and on a 100-element page they are the bulk
  // of what repeats: `disabled` alone measured 429-500 characters per
  // snapshot on real pages, before counting the empty `role`, `type` and
  // `name` of every plain <a> and <button>. Dropping them costs no
  // information and leaves that much more of the page inside the
  // observation's 50,000-character ceiling.
  const present = (entry) => {
    const kept = {};
    for (const [key, value] of Object.entries(entry)) {
      if (value === '' || value === false || value === null) continue;
      kept[key] = value;
    }
    return kept;
  };
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
    .filter((element) => {
      const rect = element.getBoundingClientRect();
      return rendered(element) && rect.width > 0 && rect.height > 0;
    }).slice(0, LIMIT);
  // What the read hands the next action: the element itself, and what it
  // looked like when it was offered. An attribute alone cannot answer
  // "is this still the thing the model was shown" -- a re-render replaces
  // the node and takes the attribute with it, and the selector that goes
  // looking for it waits out its timeout before saying anything. Held this
  // way, an action asks the question in one evaluate. From jev-ultrafast's
  // `snapshot.js` (`cache`, `cache.guard`); see vendor/jev-ultrafast/.
  const cache = window.__ohBrowserElements ||= {};
  cache.elements = new Map();
  cache.guards = new Map();
  cache.rendered = rendered;
  // What the element *is*, not what it currently reads. Holding the node
  // itself already settles identity, so a label that re-renders -- a count in
  // a tab, "Follow" becoming "Following", a spinner in a button -- is the same
  // control and must still be clickable. What this catches is the node being
  // kept and repurposed: a different tag, role, accessible name or href under
  // the number the model was given. jev's own guard carries the value and the
  // surrounding text too, because its loop re-observes after every step and
  // pays nothing for a re-read; here a refusal costs the run a model call, so
  // it is spent only on an element that has become something else.
  cache.guard = (element) => [
    element.tagName.toLowerCase(),
    (element.getAttribute('role') || '').slice(0, 80),
    (element.getAttribute('aria-label') ||
      element.getAttribute('placeholder') || '').slice(0, 240),
    (element.getAttribute('href') || '').slice(0, 240),
  ];
  cache.label = (element) => (element.innerText || element.value || '')
    .trim().replace(/\s+/g, ' ').slice(0, 80);
  const interactive = candidates.map((element, index) => {
    element.setAttribute(INDEX, String(index));
    cache.elements.set(index, element);
    cache.guards.set(index, cache.guard(element));
    const rect = element.getBoundingClientRect();
    return present({
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
    });
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
    outline.push(present({
      kind: /^h[1-6]$/.test(tag) || role === 'heading' ? 'heading' : 'landmark',
      tag,
      role: tag === 'nav' && role === 'nav' ? 'navigation' : role,
      name: name || stableId || role,
      id: stableId,
      y: Math.round(rect.top + scrollY),
      location: rect.bottom < 0
        ? 'above' : rect.top > innerHeight ? 'below' : 'viewport',
    }));
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


# Asked of the page before an action touches an element the model chose from a
# state it read earlier: is this still that element, and can it be reached?
# Every answer is one evaluate and returns at once, where the selector it
# replaces waited out its timeout to say the same thing. Adapted from
# jev-ultrafast (`fresh`, and the hit test in `browser_operation`); see
# vendor/jev-ultrafast/.
_ELEMENT_GUARD_SCRIPT = r"""
(index) => {
  const cache = window.__ohBrowserElements;
  // No state has been read since this page loaded, so there is nothing the
  // index could have been taken from and nothing to compare against.
  if (!cache || !cache.elements) return {status: 'unread'};
  const element = cache.elements.get(index);
  if (!element || !element.isConnected || !cache.rendered(element)) {
    return {status: 'gone'};
  }
  const then = cache.guards.get(index);
  const now = cache.guard(element);
  if (JSON.stringify(now) !== JSON.stringify(then)) {
    return {status: 'changed', was: then[2] || then[1] || then[0],
            is: cache.label(element) || now[2] || now[1] || now[0]};
  }
  if (element.matches(':disabled') ||
      element.closest('[aria-disabled="true"],[inert]')) {
    return {status: 'disabled'};
  }
  const rect = element.getBoundingClientRect();
  const x = rect.x + rect.width / 2, y = rect.y + rect.height / 2;
  if (!rect.width || !rect.height) return {status: 'gone'};
  if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) {
    return {status: 'offscreen', x: Math.round(x), y: Math.round(y)};
  }
  const at = document.elementFromPoint(x, y);
  if (at && !element.contains(at) && !at.contains(element)) {
    const covering = (at.getAttribute('aria-label') || at.innerText || at.tagName)
      .trim().replace(/\s+/g, ' ').slice(0, 80);
    return {status: 'covered', by: covering};
  }
  return {status: 'ok', x, y};
}
"""


# When nothing on the page shows the text a scroll was asked for, the page is
# walked to the bottom a screen at a time so anything deferred mounts, looking
# again after each step. On Pilot 4625ca37 (2026-09-11) three of four looks
# for a section that mounts on scroll answered "no element shows", and the
# run spent twenty calls scrolling and reading source between them.
SCROLL_TO_TEXT_WALK_STEPS = 40
SCROLL_TO_TEXT_WALK_SETTLE_MS = 250
# After a scroll, the page is read only once it has stopped moving. Every
# Chartmetric page sets `scroll-behavior: smooth`, so `scrollIntoView` starts
# an animation and returns; the frame the SDK takes for the observation then
# shows the section short of where it was sent (Pilot 79f2fa2b, 2026-09-11:
# the Noteworthy Insights card 200px below centre; 30463b33 the same day
# landed centred by timing alone). `behavior: 'instant'` and
# `--disable-smooth-scrolling` remove both sources; the settle wait covers a
# page that animates its own scroll position.
SCROLL_SETTLE_POLL_MS = 50
SCROLL_SETTLE_MAX_MS = 1000
_SCROLL_POSITION_SCRIPT = "() => [scrollX, scrollY]"
# How far from the viewport's centre the scrolled-to element may rest.
SCROLL_CENTRE_TOLERANCE_PX = 4

# The element that shows `wanted`: the deepest holder of the text, and among
# those the one whose own text is shortest -- the label itself, not a paragraph
# that mentions it and not the page's outermost container, which also "shows"
# it (cef12908, 2026-09-10: `to_text` jumped to the top of the page).
_FIND_TEXT_TARGET_JS = """
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
  const needle = wanted.toLowerCase();
  const holders = exactId ? [exactId] : Array.from(
    document.querySelectorAll('body *')
  ).filter((element) =>
    rendered(element)
    && (element.innerText || '').toLowerCase().includes(needle)
  );
  const deepest = holders.filter(
    (element) => !holders.some(
      (other) => other !== element && element.contains(other)
    )
  );
  deepest.sort((a, b) =>
    (a.innerText || '').length - (b.innerText || '').length
  );
  const target = deepest[0];
"""

# Scroll the target to the centre and report where it rests: its offset from
# the viewport's centre, and the document's height. A page that mounts sections
# lazily grows *above* the target after the jump -- on Chartmetric's artist
# page the panels above mount when scrolled past, so a section sent to the
# centre from the top of the page rested 180-390px low (79f2fa2b, 1ab57f9a,
# 2026-09-11) while the same scroll from nearby landed centred (30463b33).
# Called until the offset is inside tolerance and the height has stopped moving.
_RECENTRE_TEXT_TARGET_SCRIPT = (
    "(wanted) => {"
    + _FIND_TEXT_TARGET_JS
    + """
  if (!target) return null;
  target.scrollIntoView({block: 'center', inline: 'nearest', behavior: 'instant'});
  const rect = target.getBoundingClientRect();
  return [
    Math.round(rect.top + rect.height / 2 - innerHeight / 2),
    document.documentElement.scrollHeight,
  ];
}"""
)

_MOUNT_WALK_SCRIPT = """
() => {
  scrollBy(0, Math.round(innerHeight * 0.9));
  return scrollY + innerHeight >= document.documentElement.scrollHeight - 2;
}
"""


# Every page reading the agent gets is JSON, and `indent=2` spends about four
# characters of padding on every one of its values. Measured on three real
# production snapshots (Pilot 2026-09-12, 86-100 interactive elements each),
# the same object compact is 0.61-0.62x the indented size -- and the size is
# not free: `BrowserObservation.to_llm_content` cuts every observation at
# `DEFAULT_TEXT_CONTENT_LIMIT` (50,000 characters), which browser observations
# were hitting on 29 of 36, 6 of 6 and 5 of 5 calls in the two runs sampled.
# Under that ceiling the padding is not merely paid for, it is paid for by
# throwing away the tail of the page, so compacting buys roughly 40% more of
# the page inside the same cap. No information is dropped: it is the same
# object, printed without the whitespace.
def _dump(payload: object) -> str:
    """Serialize a browser payload without indentation padding."""
    return json.dumps(payload, separators=(",", ":"))


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
        self._navigation_policy: Callable[[str], Awaitable[None]] | None = None
        self._sensitive_values: tuple[str, ...] = ()
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
        navigation_policy: Callable[[str], Awaitable[None]] | None = None,
        **_: Any,
    ) -> None:
        if self.is_live:
            return
        self._playwright = await async_playwright().start()
        self._allowed_domains = tuple(allowed_domains or ())
        self._navigation_policy = navigation_policy
        # Chromium animates wheel scrolls by default, and a frame taken right
        # after `mouse.wheel` catches the page mid-animation; the site's own
        # `scroll-behavior: smooth` does the same to `scrollIntoView`. Both
        # off: an automation wants the page where it asked for it.
        launch_args = ["--disable-dev-shm-usage", "--disable-smooth-scrolling"]
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
        if self._allowed_domains or self._navigation_policy is not None:
            await self._context.route("**/*", self._guard_route)
        self._context.on("page", self._register_page)
        self._page = await self._context.new_page()
        self._register_page(self._page)

    def set_sensitive_values(self, values: Sequence[str]) -> None:
        """Register cumulative in-memory redactions for this browser session."""
        self._sensitive_values = tuple(
            sorted(
                set(self._sensitive_values).union(value for value in values if value),
                key=len,
                reverse=True,
            )
        )

    def mask_sensitive_text(self, text: str) -> str:
        for value in self._sensitive_values:
            text = text.replace(value, "<secret>")
        return text

    def _mask_sensitive_state(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.mask_sensitive_text(value)
        if isinstance(value, dict):
            return {
                key: self._mask_sensitive_state(item) for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._mask_sensitive_state(item) for item in value]
        return value

    async def _screenshot_masks(self, page: Page) -> list[Locator]:
        elements = page.locator("body, body *")
        contents = await elements.evaluate_all(
            """(elements) => elements.map(element => [
                typeof element.value === 'string' ? element.value : '',
                ...Array.from(element.childNodes)
                  .filter(node => node.nodeType === Node.TEXT_NODE)
                  .map(node => node.textContent || '')
            ])"""
        )
        return [
            elements.nth(index)
            for index, parts in enumerate(contents)
            if any(
                secret in part for secret in self._sensitive_values for part in parts
            )
        ]

    async def browser_metadata(self) -> dict[str, str]:
        """Read the active page identity and at most 12,000 rendered characters."""
        page = self._require_page()
        return {
            "url": self.mask_sensitive_text(page.url),
            "title": self.mask_sensitive_text(await page.title()),
            "text": self.mask_sensitive_text(
                await page.locator("body").inner_text(timeout=5000)
            )[:12000],
        }

    async def navigate(self, url: str, new_tab: bool = False) -> str:
        await self._validate_navigation(url)
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
        state = self._mask_sensitive_state(state)
        if include_screenshot:
            if self._sensitive_values:
                screenshot = await page.screenshot(
                    type="jpeg",
                    quality=75,
                    mask=await self._screenshot_masks(page),
                    mask_color="#000000",
                )
            else:
                screenshot = await page.screenshot(type="jpeg", quality=75)
            state["screenshot"] = base64.b64encode(screenshot).decode()
        return _dump(state)

    async def _require_live_element(self, index: int, verb: str) -> None:
        """Refuse an action on an element the page no longer offers.

        Says so in the time one evaluate takes. The selector this stands in
        front of waits `ACTION_TIMEOUT_MS` for a node that a re-render has
        already replaced, and then reports a Playwright line that names no
        remedy: measured on Chartmetric Pilot production over the week to
        2026-09-18, 48 of 1,141 browser calls died that way, 30s each, the
        action never made. Every refusal here names what to do next, which is
        almost always to read the page again.
        """
        page = self._require_page()
        verdict = await page.evaluate(_ELEMENT_GUARD_SCRIPT, index)
        status = verdict.get("status") if isinstance(verdict, dict) else None
        if status in (None, "ok", "unread"):
            # `unread` is a caller that has an index from somewhere other than
            # this page's state read. It is not this check's business to
            # refuse it; the locator still has its say.
            return
        remedy = "Read the page again with browser_get_state and use its indexes."
        if status == "gone":
            raise StaleElementError(
                f"Element {index} is no longer on the page: it was replaced or "
                f"removed after the state that offered it. {remedy}"
            )
        if status == "changed":
            raise StaleElementError(
                f"Element {index} is now {verdict.get('is')!r}, not "
                f"{verdict.get('was')!r} -- the page re-rendered and this index "
                f"belongs to something else. {remedy}"
            )
        if status == "disabled":
            raise StaleElementError(
                f"Element {index} is disabled, so {verb} would do nothing. "
                "Satisfy what the page is waiting for, then read it again."
            )
        if status == "offscreen":
            raise StaleElementError(
                f"Element {index} has moved out of the viewport since the state "
                "that offered it. Scroll to it and read the page again."
            )
        if status == "covered":
            raise StaleElementError(
                f"Element {index} is behind {verdict.get('by')!r}, which would "
                f"take the {verb} instead. Deal with what is in front of it "
                "-- a dialog, a cookie banner, an overlay -- and read the page again."
            )

    async def click(self, index: int, new_tab: bool = False) -> str:
        await self._require_live_element(index, "a click")
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
        await self._require_live_element(index, "typing")
        locator = self._indexed_locator(index)
        if secret:
            self.set_sensitive_values([text])
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
        await self._settle_scroll(page)
        return f"Scrolled {direction}"

    async def _settle_scroll(self, page: Page) -> None:
        """Return once the scroll position has held still for one poll."""
        previous = await page.evaluate(_SCROLL_POSITION_SCRIPT)
        waited = 0
        while waited < SCROLL_SETTLE_MAX_MS:
            await page.wait_for_timeout(SCROLL_SETTLE_POLL_MS)
            waited += SCROLL_SETTLE_POLL_MS
            current = await page.evaluate(_SCROLL_POSITION_SCRIPT)
            if current == previous:
                return
            previous = current

    async def scroll_to_text(self, text: str) -> str:
        page = self._require_page()
        found = await self._scroll_to_text_once(page, text)
        if not found:
            # Not on the page yet: walk it a screen at a time so a deferred
            # section mounts, and look again after each step.
            for _ in range(SCROLL_TO_TEXT_WALK_STEPS):
                at_bottom = await page.evaluate(_MOUNT_WALK_SCRIPT)
                await page.wait_for_timeout(SCROLL_TO_TEXT_WALK_SETTLE_MS)
                found = await self._scroll_to_text_once(page, text)
                if found or at_bottom:
                    break
        if not found:
            return (
                f"No element on the page shows {text!r}. It may not have loaded "
                "yet, may be behind a tab, or may be on another page. Read "
                "browser_get_content before concluding it is absent."
            )
        await self._hold_target_at_centre(page, text)
        return f"Scrolled to {found!r}"

    async def _scroll_to_text_once(self, page: Page, text: str):
        return await page.evaluate(
            "(wanted) => {"
            + _FIND_TEXT_TARGET_JS
            + """
              if (!target) return false;
              target.scrollIntoView({
                block: 'center', inline: 'nearest', behavior: 'instant'
              });
              return (target.innerText || '').trim() || target.id || wanted;
            }""",
            text,
        )

    async def _hold_target_at_centre(self, page: Page, text: str) -> None:
        """Re-centre the target until it rests there and the page stops growing.

        One instant scroll is not the end of a scroll on a page that mounts
        content on scroll: the jump fires the scroll handlers, sections above
        the target mount, and the target moves down by their height while the
        scroll position stays. So the target is sent to the centre again after
        every poll until it is within tolerance and the document height has
        held still for one poll, bounded by the same budget the settle uses.
        """
        previous: list | None = None
        waited = 0
        while True:
            reading = await page.evaluate(_RECENTRE_TEXT_TARGET_SCRIPT, text)
            if reading is None:
                return
            offset, height = reading
            if (
                previous is not None
                and abs(offset) <= SCROLL_CENTRE_TOLERANCE_PX
                and height == previous[1]
            ):
                return
            if waited >= SCROLL_SETTLE_MAX_MS:
                return
            previous = reading
            await page.wait_for_timeout(SCROLL_SETTLE_POLL_MS)
            waited += SCROLL_SETTLE_POLL_MS

    async def find_visible_text(self, text: str, max_results: int = 10) -> str:
        page = self._require_page()
        result = await page.evaluate(
            FIND_VISIBLE_TEXT_SCRIPT, {"needle": text, "limit": max_results}
        )
        return _dump(result)

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
            return _dump(state)
        origins = state.setdefault("origins", [])
        stored_origin = next(
            (candidate for candidate in origins if candidate.get("origin") == origin),
            None,
        )
        if stored_origin is None:
            stored_origin = {"origin": origin, "localStorage": []}
            origins.append(stored_origin)
        stored_origin["sessionStorage"] = session_storage
        return _dump(state)

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
        return _dump(tabs)

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

    async def wait_for_stable_frame(
        self, sample_count: int = 4, interval_seconds: float = 0.4
    ) -> bool:
        """Wait until two compositor captures match, within a fixed budget."""
        cdp = await self.cdp_session()
        previous: str | None = None
        for sample in range(sample_count):
            frame = await cdp.send(
                "Page.captureScreenshot",
                {"format": "jpeg", "quality": 75, "fromSurface": True},
            )
            current = frame.get("data")
            if not isinstance(current, str):
                raise RuntimeError("CDP screenshot response was invalid")
            if current == previous:
                return True
            previous = current
            if sample + 1 < sample_count:
                await asyncio.sleep(interval_seconds)
        return False

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

    async def _validate_navigation(self, url: str) -> None:
        self._validate_url(url)
        if self._navigation_policy is not None:
            await self._navigation_policy(url)

    async def _guard_route(self, route: Route, request: Request) -> None:
        if request.is_navigation_request() and request.frame.parent_frame is None:
            try:
                await self._validate_navigation(request.url)
            except Exception:
                await route.abort("blockedbyclient")
                return
        await route.continue_()
