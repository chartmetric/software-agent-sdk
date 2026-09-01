from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from playwright.async_api import BrowserContext, Page

from openhands.sdk import get_logger
from openhands.tools.browser_use.event_storage import EventStorage


logger = get_logger(__name__)
_JS_DIR = Path(__file__).parent / "js"


@dataclass
class RecordingConfig:
    flush_interval_seconds: float = 5.0
    rrweb_load_timeout_ms: int = 10000
    cdn_url: str = "https://unpkg.com/rrweb@2.0.0-alpha.17/dist/rrweb.umd.cjs"


DEFAULT_CONFIG = RecordingConfig()


@lru_cache(maxsize=16)
def _load_js_file(filename: str) -> str:
    return (_JS_DIR / filename).read_text()


def get_rrweb_loader_js(cdn_url: str) -> str:
    return _load_js_file("rrweb-loader.js").replace("{{CDN_URL}}", cdn_url)


@dataclass
class RecordingSession:
    """rrweb recording driven directly through the active Playwright page."""

    output_dir: str | None = None
    config: RecordingConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    _storage: EventStorage = field(default_factory=EventStorage, repr=False)
    _events: list[dict] = field(default_factory=list)
    _flush_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _is_recording: bool = False
    _scripts_injected: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _page_provider: Callable[[], Page] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._storage.output_dir = self.output_dir

    @property
    def is_active(self) -> bool:
        return self._is_recording

    @property
    def session_dir(self) -> str | None:
        return self._storage.session_dir

    @property
    def total_events(self) -> int:
        return self._storage.total_events

    @property
    def file_count(self) -> int:
        return self._storage.file_count

    @property
    def events(self) -> list[dict]:
        return self._events

    async def start(
        self,
        context: BrowserContext,
        page_provider: Callable[[], Page],
    ) -> str:
        if self._is_recording:
            return "Already recording"
        self._page_provider = page_provider
        loader = get_rrweb_loader_js(self.config.cdn_url)
        if not self._scripts_injected:
            await context.add_init_script(script=loader)
            self._scripts_injected = True
        page = page_provider()
        await page.evaluate(loader)
        self._storage.reset()
        self._storage.output_dir = self.output_dir
        self._storage.create_session_subfolder()
        self._events = []
        ready = await asyncio.wait_for(
            page.evaluate(_load_js_file("wait-for-rrweb.js")),
            timeout=self.config.rrweb_load_timeout_ms / 1000,
        )
        if not isinstance(ready, dict) or not ready.get("success"):
            return "Error: Unable to start recording because rrweb did not load"
        status = await page.evaluate(_load_js_file("start-recording.js"))
        if not isinstance(status, dict) or status.get("status") not in {
            "started",
            "already_recording",
        }:
            return f"Error: Unable to start recording: {status}"
        self._is_recording = True
        await page.evaluate("window.__rrweb_should_record = true")
        self._flush_task = asyncio.create_task(self._periodic_flush())
        return "Recording started"

    async def flush_events(self) -> int:
        if not self._is_recording or self._page_provider is None:
            return 0
        try:
            raw = await self._page_provider().evaluate(_load_js_file("flush-events.js"))
            data = json.loads(raw) if isinstance(raw, str) else raw
            events = data.get("events", []) if isinstance(data, dict) else []
            if events:
                async with self._lock:
                    self._events.extend(events)
            return len(events)
        except Exception:
            logger.debug("Recording event flush failed", exc_info=True)
            return 0

    async def restart_on_new_page(self) -> None:
        if not self._is_recording or self._page_provider is None:
            return
        try:
            page = self._page_provider()
            ready = await asyncio.wait_for(
                page.evaluate(_load_js_file("wait-for-rrweb.js")),
                timeout=self.config.rrweb_load_timeout_ms / 1000,
            )
            if isinstance(ready, dict) and ready.get("success"):
                await page.evaluate(_load_js_file("start-recording-simple.js"))
                await page.evaluate("window.__rrweb_should_record = true")
        except Exception:
            logger.debug("Recording restart failed", exc_info=True)

    async def stop(self) -> str:
        if not self._is_recording or self._page_provider is None:
            return "Error: Not recording. Call browser_start_recording first."
        self._is_recording = False
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        try:
            raw = await self._page_provider().evaluate(
                _load_js_file("stop-recording.js")
            )
            data = json.loads(raw) if isinstance(raw, str) else raw
            events = data.get("events", []) if isinstance(data, dict) else []
            async with self._lock:
                self._events.extend(events)
                if self._events:
                    saved = self._storage.save_events(self._events)
                    if saved:
                        self._events = []
            total_events = self._storage.total_events
            total_files = self._storage.file_count
            result = (
                f"Recording stopped. Captured {total_events} events "
                f"in {total_files} file(s)."
            )
            if self._storage.session_dir:
                result += f" Saved to: {self._storage.session_dir}"
            return result
        finally:
            try:
                await self._page_provider().evaluate(
                    "window.__rrweb_should_record = false"
                )
            except Exception:
                logger.debug("Could not clear rrweb recording flag", exc_info=True)

    async def _periodic_flush(self) -> None:
        while self._is_recording:
            await asyncio.sleep(self.config.flush_interval_seconds)
            await self.flush_events()
            async with self._lock:
                if self._events:
                    saved = self._storage.save_events(self._events)
                    if saved:
                        self._events = []

    def reset(self) -> None:
        self._events = []
        self._is_recording = False
        self._storage.reset()
        self._flush_task = None
        self._page_provider = None
