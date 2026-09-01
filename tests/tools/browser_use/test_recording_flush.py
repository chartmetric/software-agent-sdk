import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from openhands.tools.browser_use.recording import RecordingConfig, RecordingSession


@pytest.fixture
def runtime(tmp_path):
    context = MagicMock()
    context.add_init_script = AsyncMock()
    page = MagicMock()
    page.evaluate = AsyncMock(
        side_effect=[None, {"success": True}, {"status": "started"}, None]
    )
    session = RecordingSession(
        output_dir=str(tmp_path),
        config=RecordingConfig(flush_interval_seconds=60),
    )
    return session, context, page


@pytest.mark.asyncio
async def test_start_installs_one_context_script_and_records_current_page(runtime):
    session, context, page = runtime

    result = await session.start(context, lambda: page)

    assert result == "Recording started"
    assert session.is_active is True
    context.add_init_script.assert_awaited_once()
    assert page.evaluate.await_count == 4
    assert session.session_dir is not None

    session._flush_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await session._flush_task


@pytest.mark.asyncio
async def test_flush_collects_events_without_blocking_page_actions(tmp_path):
    page = MagicMock()
    page.evaluate = AsyncMock(
        return_value=json.dumps({"events": [{"type": 3, "timestamp": 1}]})
    )
    session = RecordingSession(output_dir=str(tmp_path))
    session._is_recording = True
    session._page_provider = lambda: page

    assert await session.flush_events() == 1
    assert session.events == [{"type": 3, "timestamp": 1}]


@pytest.mark.asyncio
async def test_stop_persists_buffered_and_final_events(tmp_path):
    page = MagicMock()
    page.evaluate = AsyncMock(
        side_effect=[
            json.dumps({"events": [{"type": 3, "timestamp": 2}]}),
            None,
        ]
    )
    session = RecordingSession(output_dir=str(tmp_path))
    session._storage.create_session_subfolder()
    session._is_recording = True
    session._page_provider = lambda: page
    session._events = [{"type": 3, "timestamp": 1}]

    result = await session.stop()

    assert "2 events" in result
    assert "1 file(s)" in result
    saved = list(tmp_path.glob("recording-*/*.json"))
    assert len(saved) == 1
    assert len(json.loads(saved[0].read_text())) == 2


@pytest.mark.asyncio
async def test_stop_without_start_is_a_clear_error():
    assert "Not recording" in await RecordingSession().stop()


@pytest.mark.asyncio
async def test_recording_failure_does_not_raise(runtime):
    session, context, page = runtime
    page.evaluate.side_effect = RuntimeError("page closed")

    with pytest.raises(RuntimeError, match="page closed"):
        await session.start(context, lambda: page)


def test_reset_clears_runtime_state(runtime):
    session, _, page = runtime
    session._is_recording = True
    session._page_provider = lambda: page
    session._events = [{"type": 1}]

    session.reset()

    assert session.is_active is False
    assert session.events == []
    assert session._page_provider is None
