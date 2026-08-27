from collections.abc import Callable
from typing import Any

from browser_use.browser.events import TypeTextEvent
from browser_use.dom.markdown_extractor import extract_clean_markdown

from openhands.sdk import get_logger
from openhands.tools.browser_use.logging_fix import LogSafeBrowserUseServer
from openhands.tools.browser_use.recording import RecordingSession
from openhands.tools.browser_use.screencast import ScreencastSession


logger = get_logger(__name__)


# =============================================================================
# CustomBrowserUseServer Class
# =============================================================================


class CustomBrowserUseServer(LogSafeBrowserUseServer):
    """
    Custom BrowserUseServer with a new tool for extracting web
    page's content in markdown.
    """

    def __init__(self, session_timeout_minutes: int = 10):
        super().__init__(session_timeout_minutes=session_timeout_minutes)
        # Scripts to inject into every new document (before page scripts run)
        self._inject_scripts: list[str] = []
        # Script identifiers returned by CDP (for cleanup if needed)
        self._injected_script_ids: list[str] = []
        # Recording session - encapsulates all recording state and logic
        self._recording_session: RecordingSession | None = None
        # Screencast session - live CDP frame streaming for passive viewers
        self._screencast_session: ScreencastSession | None = None

    @property
    def _is_recording(self) -> bool:
        """Check if recording is currently active."""
        return self._recording_session is not None and self._recording_session.is_active

    async def _cleanup_screencast(self) -> None:
        """Stop any active screencast. Should be called when the browser
        session is being closed."""
        if self._screencast_session is None:
            return
        try:
            await self._screencast_session.stop()
        except Exception as e:
            logger.debug(f"Screencast cleanup error (non-fatal): {e}")
        finally:
            self._screencast_session = None

    async def _cleanup_recording(self) -> None:
        """Cleanup recording session resources.

        Stops any active recording, saves remaining events, and releases resources.
        Should be called when the browser session is being closed.
        """
        if self._recording_session is None:
            return

        try:
            # Stop recording if active to save any remaining events
            if self._recording_session.is_active and self.browser_session:
                await self._recording_session.stop(self.browser_session)
            else:
                # Just reset if not active or no browser session
                self._recording_session.reset()
        except Exception as e:
            logger.debug(f"Recording cleanup error (non-fatal): {e}")
        finally:
            self._recording_session = None

    async def _close_browser(self) -> str:
        """Close the browser session and cleanup recording/screencast resources."""
        await self._cleanup_recording()
        await self._cleanup_screencast()
        return await super()._close_browser()

    async def _close_session(self, session_id: str) -> str:
        """Close a specific browser session and cleanup recording/screencast
        if needed."""
        # Cleanup recording/screencast if closing the current session
        if self.browser_session and self.browser_session.id == session_id:
            await self._cleanup_recording()
            await self._cleanup_screencast()
        return await super()._close_session(session_id)

    async def _close_all_sessions(self) -> str:
        """Close all active browser sessions and cleanup recording/screencast
        resources."""
        await self._cleanup_recording()
        await self._cleanup_screencast()
        return await super()._close_all_sessions()

    def set_inject_scripts(self, scripts: list[str]) -> None:
        """Set scripts to be injected into every new document.

        Args:
            scripts: List of JavaScript code strings to inject.
                     Each script will be evaluated before page scripts run.
        """
        self._inject_scripts = scripts

    async def _set_viewport(self, width: int, height: int) -> str:
        """Render the page already open at another width.

        A responsive layout is the one thing no control on the page can reveal:
        a product ships a theme toggle, but nothing in it makes the window
        narrow. Without this the browser could only ever render one width, so a
        mobile layout was unobservable rather than merely unverified.

        Applied to the session already signed in, because a fresh one would have
        to log in again before it could show the surface under review.
        """
        if not self.browser_session:
            return "Error: No browser session active"

        page = await self.browser_session.get_current_page()
        if page is None:
            return "Error: No page is open to resize"

        await page.set_viewport_size(width, height)
        return f"Viewport set to {width}x{height}"

    async def _type_secret_text(self, index: int, text: str) -> str:
        if not self.browser_session:
            return "Error: No browser session active"

        element = await self.browser_session.get_dom_element_by_index(index)
        if not element:
            return f"Element with index {index} not found"

        event = self.browser_session.event_bus.dispatch(
            TypeTextEvent(
                node=element,
                text=text,
                is_sensitive=True,
                sensitive_key_name="secret",
            )
        )
        await event
        return f"Typed <secret> into element {index}"

    async def _inject_scripts_to_session(self) -> None:
        """Inject configured user scripts into the browser session using CDP.

        Uses Page.addScriptToEvaluateOnNewDocument to inject scripts that
        will run on every new document before the page's scripts execute.
        Note: rrweb scripts are injected lazily when recording starts.
        """
        if not self.browser_session or not self._inject_scripts:
            return

        try:
            cdp_session = await self.browser_session.get_or_create_cdp_session()
            cdp_client = cdp_session.cdp_client

            for script in self._inject_scripts:
                result = await cdp_client.send.Page.addScriptToEvaluateOnNewDocument(
                    params={"source": script, "runImmediately": True},
                    session_id=cdp_session.session_id,
                )
                script_id = result.get("identifier")
                if script_id:
                    self._injected_script_ids.append(script_id)
                    logger.debug(f"Injected script with identifier: {script_id}")

            num_scripts = len(self._inject_scripts)
            logger.info(f"Injected {num_scripts} user script(s) into browser session")
        except Exception as e:
            logger.warning(f"Failed to inject scripts: {e}")

    async def _flush_recording_events(self) -> int:
        """Flush recording events from browser to Python storage.

        Returns the number of events flushed.
        """
        if not self.browser_session or not self._recording_session:
            return 0
        return await self._recording_session.flush_events(self.browser_session)

    async def _restart_recording_on_new_page(self) -> None:
        """Restart recording on a new page after navigation."""
        if not self.browser_session or not self._recording_session:
            return
        await self._recording_session.restart_on_new_page(self.browser_session)

    async def _start_recording(self, output_dir: str | None = None) -> str:
        """Start rrweb session recording.

        Recording persists across page navigations - events are periodically flushed
        to timestamped JSON files in a session subfolder.

        Each recording session creates a new subfolder under output_dir with format:
        {output_dir}/recording-{timestamp}/

        Args:
            output_dir: Root directory for recording files. If provided, a timestamped
                subfolder will be created for this recording session.
        """
        if not self.browser_session:
            return "Error: No browser session active"

        # Create a new recording session with output_dir
        self._recording_session = RecordingSession(output_dir=output_dir)
        return await self._recording_session.start(self.browser_session)

    async def _stop_recording(self) -> str:
        """Stop rrweb recording and save remaining events.

        Events are saved to the directory configured at start_recording time.

        Returns:
            A summary message with the save directory and file count.
        """
        if not self.browser_session:
            return "Error: No browser session active"

        if not self._recording_session or not self._recording_session.is_active:
            return "Error: Not recording. Call browser_start_recording first."

        result = await self._recording_session.stop(self.browser_session)
        # Reset the session after stopping
        self._recording_session.reset()
        return result

    async def _start_screencast(
        self, on_frame: Callable[[str, dict[str, Any]], None], **kwargs
    ) -> bool:
        """Start streaming CDP screencast frames for the current session.

        `on_frame` is invoked synchronously on every `Page.screencastFrame`
        event with `(base64_jpeg_data, metadata_dict)`. Returns True if the
        screencast started, False otherwise (never raises).
        """
        if not self.browser_session:
            return False

        self._screencast_session = ScreencastSession()
        return await self._screencast_session.start(
            self.browser_session, on_frame, **kwargs
        )

    async def _stop_screencast(self) -> bool:
        """Stop streaming CDP screencast frames, if active."""
        if not self._screencast_session:
            return True
        result = await self._screencast_session.stop()
        self._screencast_session = None
        return result

    async def _dispatch_screencast_mouse(self, **kwargs) -> None:
        """Dispatch a mouse event for human screencast takeover, if a
        screencast session is active. No-op (never raises) otherwise --
        see `ScreencastSession.dispatch_mouse`'s Error Handling Policy."""
        if not self._screencast_session:
            return
        await self._screencast_session.dispatch_mouse(**kwargs)

    async def _dispatch_screencast_key(self, **kwargs) -> None:
        """Dispatch a keyboard event for human screencast takeover, if a
        screencast session is active. No-op (never raises) otherwise --
        see `ScreencastSession.dispatch_key`'s Error Handling Policy."""
        if not self._screencast_session:
            return
        await self._screencast_session.dispatch_key(**kwargs)

    async def _get_browser_state(self, include_screenshot: bool = False) -> str:
        """The upstream state, plus where on the page it was taken.

        Upstream `browser_use` 0.11.9 returns url, title, tabs and the
        interactive elements, and nothing about the document those elements were
        read from. That makes an absence unfalsifiable: the elements it lists
        are the ones the DOM serializer reached, a reader cannot tell a short
        page from the top of a long one, and the tool's own description says
        `all interactive elements`. Measured across two production runs on
        2026-08-27, nine `browser_get_state` observations carried no scroll
        position and no document height, and both runs concluded that a
        component was not on a page they had only seen the top of.

        `BrowserStateSummary` has carried `page_info` the whole time -- 0.11.9
        simply never serialises it, and 0.11.13 does. This adds it here so the
        answer does not depend on which of those the image resolved, and so a
        later upgrade changes nothing about what the agent reads.

        `pages_below` is the number the agent actually acts on. A count of
        pixels invites arithmetic against a viewport height it has to find
        elsewhere; "1.8 screens below" is a decision.
        """
        import json

        state_json = await super()._get_browser_state(include_screenshot)
        try:
            state = json.loads(state_json)
        except (TypeError, ValueError):
            # An error string rather than a state document. Upstream returns one
            # when there is no session, and it is not this method's to rewrite.
            return state_json
        if not isinstance(state, dict) or "scroll" in state:
            return state_json

        session = self.browser_session
        page_info = getattr(
            getattr(session, "_cached_browser_state_summary", None), "page_info", None
        )
        if page_info is None:
            return state_json

        viewport_height = getattr(page_info, "viewport_height", 0) or 0
        page_height = getattr(page_info, "page_height", 0) or 0
        scroll_y = getattr(page_info, "scroll_y", 0) or 0
        state["viewport"] = {
            "width": getattr(page_info, "viewport_width", None),
            "height": viewport_height,
        }
        state["page"] = {
            "width": getattr(page_info, "page_width", None),
            "height": page_height,
        }
        state["scroll"] = {"x": getattr(page_info, "scroll_x", None), "y": scroll_y}
        if viewport_height:
            below = max(page_height - (scroll_y + viewport_height), 0)
            state["pages_above"] = round(scroll_y / viewport_height, 1)
            state["pages_below"] = round(below / viewport_height, 1)
        return json.dumps(state, indent=2)

    async def _get_storage(self) -> str:
        """Get browser storage (cookies, local storage, session storage)."""
        import json

        if not self.browser_session:
            return "Error: No browser session active"

        try:
            # Use the private method from BrowserSession to get storage state
            # This returns a dict with 'cookies' and 'origins'
            # (localStorage/sessionStorage)
            storage_state = await self.browser_session._cdp_get_storage_state()
            return json.dumps(storage_state, indent=2)
        except Exception as e:
            logger.exception("Error getting storage state", exc_info=e)
            return f"Error getting storage state: {str(e)}"

    async def _set_storage(self, storage_state: dict) -> str:
        """Set browser storage (cookies, local storage, session storage)."""
        if not self.browser_session:
            return "Error: No browser session active"

        try:
            # 1. Set cookies
            cookies = storage_state.get("cookies", [])
            if cookies:
                await self.browser_session._cdp_set_cookies(cookies)

            # 2. Set local/session storage
            origins = storage_state.get("origins", [])
            if origins:
                cdp_session = await self.browser_session.get_or_create_cdp_session()

                # Enable DOMStorage
                await cdp_session.cdp_client.send.DOMStorage.enable(
                    session_id=cdp_session.session_id
                )

                try:
                    for origin_data in origins:
                        origin = origin_data.get("origin")
                        if not origin:
                            continue

                        dom_storage = cdp_session.cdp_client.send.DOMStorage

                        # Set localStorage
                        for item in origin_data.get("localStorage", []):
                            key = item.get("key") or item.get("name")
                            if not key:
                                continue
                            await dom_storage.setDOMStorageItem(
                                params={
                                    "storageId": {
                                        "securityOrigin": origin,
                                        "isLocalStorage": True,
                                    },
                                    "key": key,
                                    "value": item["value"],
                                },
                                session_id=cdp_session.session_id,
                            )

                        # Set sessionStorage
                        for item in origin_data.get("sessionStorage", []):
                            key = item.get("key") or item.get("name")
                            if not key:
                                continue
                            await dom_storage.setDOMStorageItem(
                                params={
                                    "storageId": {
                                        "securityOrigin": origin,
                                        "isLocalStorage": False,
                                    },
                                    "key": key,
                                    "value": item["value"],
                                },
                                session_id=cdp_session.session_id,
                            )
                finally:
                    # Disable DOMStorage
                    await cdp_session.cdp_client.send.DOMStorage.disable(
                        session_id=cdp_session.session_id
                    )

            return "Storage set successfully"
        except Exception as e:
            logger.exception("Error setting storage state", exc_info=e)
            return f"Error setting storage state: {str(e)}"

    async def _get_content(self, extract_links=False, start_from_char: int = 0) -> str:
        MAX_CHAR_LIMIT = 30000

        if not self.browser_session:
            return "Error: No browser session active"

        # Extract clean markdown using the new method
        try:
            content, content_stats = await extract_clean_markdown(
                browser_session=self.browser_session, extract_links=extract_links
            )
        except Exception as e:
            logger.exception(
                "Error extracting clean markdown", exc_info=e, stack_info=True
            )
            return f"Could not extract clean markdown: {type(e).__name__}"

        # Original content length for processing
        final_filtered_length = content_stats["final_filtered_chars"]

        if start_from_char > 0:
            if start_from_char >= len(content):
                return f"start_from_char ({start_from_char}) exceeds content length ({len(content)}). Content has {final_filtered_length} characters after filtering."  # noqa: E501

            content = content[start_from_char:]
            content_stats["started_from_char"] = start_from_char

        # Smart truncation with context preservation
        truncated = False
        if len(content) > MAX_CHAR_LIMIT:
            # Try to truncate at a natural break point (paragraph, sentence)
            truncate_at = MAX_CHAR_LIMIT

            # Look for paragraph break within last 500 chars of limit
            paragraph_break = content.rfind(
                "\n\n", MAX_CHAR_LIMIT - 500, MAX_CHAR_LIMIT
            )
            if paragraph_break > 0:
                truncate_at = paragraph_break
            else:
                # Look for sentence break within last 200 chars of limit
                sentence_break = content.rfind(
                    ".", MAX_CHAR_LIMIT - 200, MAX_CHAR_LIMIT
                )
                if sentence_break > 0:
                    truncate_at = sentence_break + 1

            content = content[:truncate_at]
            truncated = True
            next_start = (start_from_char or 0) + truncate_at
            content_stats["truncated_at_char"] = truncate_at
            content_stats["next_start_char"] = next_start

        # Add content statistics to the result
        original_html_length = content_stats["original_html_chars"]
        initial_markdown_length = content_stats["initial_markdown_chars"]
        chars_filtered = content_stats["filtered_chars_removed"]

        stats_summary = (
            f"Content processed: {original_html_length:,}"
            + f" HTML chars → {initial_markdown_length:,}"
            + f" initial markdown → {final_filtered_length:,} filtered markdown"
        )
        if start_from_char > 0:
            stats_summary += f" (started from char {start_from_char:,})"
        if truncated:
            stats_summary += f" → {len(content):,} final chars (truncated, use start_from_char={content_stats['next_start_char']} to continue)"  # noqa: E501
        elif chars_filtered > 0:
            stats_summary += f" (filtered {chars_filtered:,} chars of noise)"

        prompt = f"""<content_stats>
{stats_summary}
</content_stats>

<webpage_content>
{content}
</webpage_content>"""
        current_url = await self.browser_session.get_current_page_url()

        return f"""<url>
{current_url}
</url>
<content>
{prompt}
</content>"""
