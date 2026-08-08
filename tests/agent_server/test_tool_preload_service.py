"""Tests for the tool-preload service's background Chromium warm-up."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.agent_server.tool_preload_service import ToolPreloadService


@pytest.mark.asyncio
async def test_start_schedules_a_background_browser_warm_up():
    """start() constructs the executor and warms Chromium off the startup path.

    The warm-up must run as a background task so a slow or failed browser launch
    never delays server readiness.
    """
    executor = MagicMock()
    executor.warm_up = AsyncMock()

    service = ToolPreloadService()
    with patch(
        "openhands.tools.browser_use.impl.BrowserToolExecutor",
        return_value=executor,
    ):
        assert await service.start() is True
        assert service._warm_up_task is not None
        # start() returns without waiting on the launch; drain the task here.
        await service._warm_up_task

    executor.warm_up.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_cancels_an_unfinished_warm_up():
    started = __import__("asyncio").Event()

    async def _never_finishes():
        started.set()
        await __import__("asyncio").sleep(3600)

    executor = MagicMock()
    executor.warm_up = _never_finishes

    service = ToolPreloadService()
    with patch(
        "openhands.tools.browser_use.impl.BrowserToolExecutor",
        return_value=executor,
    ):
        await service.start()
        await started.wait()
        task = service._warm_up_task
        assert task is not None and not task.done()

        await service.stop()

    assert task.cancelled() or task.done()
    assert service._warm_up_task is None


@pytest.mark.asyncio
async def test_start_is_idempotent_while_running():
    service = ToolPreloadService()
    service.running = True
    # Already running: start() is a no-op and schedules no new warm-up.
    assert await service.start() is True
    assert service._warm_up_task is None
