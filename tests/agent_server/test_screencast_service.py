"""Tests for ScreencastService's viewer reference-counting.

These tests verify that:
1. The first subscriber starts the underlying CDP screencast exactly once,
   even under concurrent subscribe calls.
2. The last unsubscribe stops it.
3. A subscribe that races with an in-flight "last unsubscribe" does not get
   its screencast stopped out from under it.
4. MaxSubscribersError propagates without leaving a phantom subscription.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from openhands.agent_server.pub_sub import MaxSubscribersError, Subscriber
from openhands.agent_server.screencast_service import ScreencastService


class _FakeSubscriber(Subscriber[dict]):
    async def __call__(self, message: dict) -> None:
        pass


def _make_executor(start_result: bool = True) -> MagicMock:
    executor = MagicMock()
    executor.start_screencast = MagicMock(return_value=start_result)
    executor.stop_screencast = MagicMock(return_value=True)
    return executor


@pytest.fixture
def patched_shared_executor():
    executor = _make_executor()
    with patch(
        "openhands.agent_server.screencast_service.BrowserToolSet"
        ".get_or_create_shared_executor",
        return_value=executor,
    ):
        yield executor


class TestSubscribeStartsScreencastOnce:
    @pytest.mark.asyncio
    async def test_first_subscriber_starts_screencast(self, patched_shared_executor):
        service = ScreencastService()
        await service.subscribe(_FakeSubscriber())

        patched_shared_executor.start_screencast.assert_called_once()

    @pytest.mark.asyncio
    async def test_second_subscriber_does_not_start_again(
        self, patched_shared_executor
    ):
        service = ScreencastService()
        await service.subscribe(_FakeSubscriber())
        await service.subscribe(_FakeSubscriber())

        patched_shared_executor.start_screencast.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_subscribes_start_screencast_exactly_once(
        self, patched_shared_executor
    ):
        service = ScreencastService()
        await asyncio.gather(*[service.subscribe(_FakeSubscriber()) for _ in range(5)])

        patched_shared_executor.start_screencast.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_raises_when_at_capacity(self, patched_shared_executor):
        service = ScreencastService()
        service._pub_sub.max_subscribers = 1
        await service.subscribe(_FakeSubscriber())

        with pytest.raises(MaxSubscribersError):
            await service.subscribe(_FakeSubscriber())

        # Only the one successful subscriber should be tracked.
        assert len(service._pub_sub.subscriber_ids()) == 1


class TestUnsubscribeStopsScreencast:
    @pytest.mark.asyncio
    async def test_last_unsubscribe_stops_screencast(self, patched_shared_executor):
        service = ScreencastService()
        sub_id = await service.subscribe(_FakeSubscriber())

        await service.unsubscribe(sub_id)

        patched_shared_executor.stop_screencast.assert_called_once()
        assert service._active is False

    @pytest.mark.asyncio
    async def test_unsubscribe_with_remaining_viewers_does_not_stop(
        self, patched_shared_executor
    ):
        service = ScreencastService()
        sub_id_1 = await service.subscribe(_FakeSubscriber())
        await service.subscribe(_FakeSubscriber())

        await service.unsubscribe(sub_id_1)

        patched_shared_executor.stop_screencast.assert_not_called()
        assert service._active is True

    @pytest.mark.asyncio
    async def test_new_subscriber_arriving_during_stop_self_heals_to_active(
        self, patched_shared_executor
    ):
        """Regression test for a reference-counting race: a new viewer
        subscribing while the previous last-viewer's unsubscribe is mid-stop
        cannot leave the screencast permanently off. The shared lock
        serializes the two operations, so the worst case is a stop
        immediately followed by a restart (a brief, self-healing blip) —
        never a stopped screencast with a viewer attached. Eliminating even
        that blip would need a stop debounce, which is an explicit
        out-of-scope fast-follow (see plan)."""
        service = ScreencastService()
        sub_id = await service.subscribe(_FakeSubscriber())

        async def unsubscribe_then_resubscribe():
            await service.unsubscribe(sub_id)

        async def resubscribe():
            await asyncio.sleep(0)
            await service.subscribe(_FakeSubscriber())

        await asyncio.gather(unsubscribe_then_resubscribe(), resubscribe())

        # The invariant that matters: a connected viewer never ends up with
        # a permanently-stopped screencast.
        assert service._active is True
        assert len(service._pub_sub.subscriber_ids()) == 1


class TestStartFailure:
    @pytest.mark.asyncio
    async def test_failed_start_does_not_mark_active(self):
        executor = _make_executor(start_result=False)
        with patch(
            "openhands.agent_server.screencast_service.BrowserToolSet"
            ".get_or_create_shared_executor",
            return_value=executor,
        ):
            service = ScreencastService()
            await service.subscribe(_FakeSubscriber())

            assert service._active is False
