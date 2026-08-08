"""Service which preloads chromium."""

from __future__ import annotations

import asyncio

from openhands.agent_server.config import get_default_config
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.schema import Action
from openhands.sdk.tool.tool import create_action_type_with_risk
from openhands.sdk.utils.models import get_known_concrete_subclasses


_logger = get_logger(__name__)


class ToolPreloadService:
    """Service which preloads tools / chromium reducing time to
    start first conversation"""

    running: bool = False
    _warm_up_task: asyncio.Task[None] | None = None

    async def start(self) -> bool:
        """Preload tools"""

        # Skip if already running
        if self.running:
            return True

        self.running = True
        try:
            from openhands.tools.browser_use.impl import BrowserToolExecutor

            # Constructing the executor makes the Chromium binary available and
            # imports the browser stack, but does not launch a browser: that
            # still happens lazily on the first browser action. Launch and tear
            # down a session once, in the background, so Chromium and its shared
            # libraries are warm in the OS cache and the first conversation's
            # launch is fast. It runs off the startup path (fire-and-forget) so
            # a slow or failed launch never delays server readiness.
            executor = BrowserToolExecutor()
            self._warm_up_task = asyncio.create_task(executor.warm_up())

            # Pre-creating all these classes prevents processing which costs
            # significant time per tool on the first conversation invocation.
            for action_type in get_known_concrete_subclasses(Action):
                create_action_type_with_risk(action_type)

            _logger.debug(f"Loaded {BrowserToolExecutor}")
            return True
        except Exception:
            _logger.exception("Error preloading chromium")
            return False

    async def stop(self) -> None:
        """Stop the tool preload process."""
        self.running = False
        task = self._warm_up_task
        self._warm_up_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    def is_running(self) -> bool:
        """Check if tool preload is running."""
        return self.running


_tool_preload_service: ToolPreloadService | None = None


def get_tool_preload_service() -> ToolPreloadService | None:
    """Get the tool preload service instance if preload is enabled."""
    global _tool_preload_service
    config = get_default_config()

    if not config.preload_tools:
        _logger.info("Tool preload is disabled in configuration")
        return None

    if _tool_preload_service is None:
        _tool_preload_service = ToolPreloadService()
    return _tool_preload_service
