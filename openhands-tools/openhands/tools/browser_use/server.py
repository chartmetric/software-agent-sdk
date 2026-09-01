from openhands.tools.browser_use.playwright_server import PlaywrightBrowserServer


class CustomBrowserUseServer(PlaywrightBrowserServer):
    """Backward-compatible name for the Playwright browser engine."""

    async def _init_browser_session(self, **config) -> None:
        await self.start(**config)

    async def _close_all_sessions(self) -> None:
        await self.close()

    async def _close_browser(self) -> str:
        await self.close()
        return "Browser closed"

    async def _navigate(self, url: str, new_tab: bool = False) -> str:
        return await self.navigate(url, new_tab)

    async def _go_back(self) -> str:
        return await self.go_back()

    async def _click(self, index: int, new_tab: bool = False) -> str:
        return await self.click(index, new_tab)

    async def _type_text(self, index: int, text: str) -> str:
        return await self.type_text(index, text)

    async def _type_secret_text(self, index: int, text: str) -> str:
        return await self.type_text(index, text, secret=True)

    async def _get_browser_state(self, include_screenshot: bool = False) -> str:
        return await self.get_browser_state(include_screenshot)

    async def _scroll(self, direction: str) -> str:
        return await self.scroll(direction)

    async def _scroll_to_text(self, text: str) -> str:
        return await self.scroll_to_text(text)

    async def _find_visible_text(self, text: str, max_results: int) -> str:
        return await self.find_visible_text(text, max_results)

    async def _set_viewport(self, width: int, height: int) -> str:
        return await self.set_viewport(width, height)

    async def _get_storage(self) -> str:
        return await self.get_storage()

    async def _set_storage(self, storage_state: dict) -> str:
        return await self.set_storage(storage_state)

    async def _list_tabs(self) -> str:
        return await self.list_tabs()

    async def _switch_tab(self, tab_id: str) -> str:
        return await self.switch_tab(tab_id)

    async def _close_tab(self, tab_id: str) -> str:
        return await self.close_tab(tab_id)

    async def _get_content(
        self, extract_links: bool = False, start_from_char: int = 0
    ) -> str:
        return await self.get_content(extract_links, start_from_char)

    async def _inject_scripts_to_session(self) -> None:
        await self.inject_scripts()

    async def _start_recording(self, output_dir: str | None = None) -> str:
        return await self.start_recording(output_dir)

    async def _stop_recording(self) -> str:
        return await self.stop_recording()

    async def _flush_recording_events(self) -> int:
        return await self.flush_recording_events()

    async def _restart_recording_on_new_page(self) -> None:
        await self.restart_recording_on_new_page()

    async def _start_screencast(self, on_frame, **kwargs) -> bool:
        return await self.start_screencast(on_frame, **kwargs)

    async def _stop_screencast(self) -> bool:
        return await self.stop_screencast()

    async def _dispatch_screencast_mouse(self, **kwargs) -> None:
        await self.dispatch_screencast_mouse(**kwargs)

    async def _dispatch_screencast_key(self, **kwargs) -> None:
        await self.dispatch_screencast_key(**kwargs)
