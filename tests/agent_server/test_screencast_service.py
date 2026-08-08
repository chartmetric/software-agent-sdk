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
from uuid import uuid4

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
    executor.set_control_owner = MagicMock()
    executor.dispatch_screencast_input = MagicMock()
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


class TestControlOwnerStateMachine:
    """agent -> human -> agent, and the authorization rules around it."""

    @pytest.mark.asyncio
    async def test_default_owner_is_agent(self, patched_shared_executor):
        service = ScreencastService()

        assert service.control_owner == "agent"

    @pytest.mark.asyncio
    async def test_human_can_acquire_control(self, patched_shared_executor):
        service = ScreencastService()
        viewer = uuid4()

        acquired = await service.take_control(viewer)

        assert acquired is True
        assert service.control_owner == "human"
        assert service.is_controller(viewer) is True
        patched_shared_executor.set_control_owner.assert_called_once_with("human")

    @pytest.mark.asyncio
    async def test_agent_browser_actions_blocked_while_human_controls(
        self, patched_shared_executor
    ):
        """The actual blocking happens inside BrowserToolExecutor._execute_action
        (see test_browser_executor.py) by reading `_control_owner`, which this
        service sets via `executor.set_control_owner`. Verify the propagation
        this service is responsible for."""
        service = ScreencastService()
        viewer = uuid4()

        await service.take_control(viewer)

        patched_shared_executor.set_control_owner.assert_called_once_with("human")

    @pytest.mark.asyncio
    async def test_second_subscriber_cannot_take_control_from_first(
        self, patched_shared_executor
    ):
        service = ScreencastService()
        first, second = uuid4(), uuid4()
        await service.take_control(first)
        patched_shared_executor.set_control_owner.reset_mock()

        acquired = await service.take_control(second)

        assert acquired is False
        assert service.is_controller(first) is True
        assert service.is_controller(second) is False
        patched_shared_executor.set_control_owner.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_controller_input_is_ignored(self, patched_shared_executor):
        service = ScreencastService()
        controller, bystander = uuid4(), uuid4()
        await service.take_control(controller)

        await service.dispatch_input(bystander, "mouse", {"type": "mouseMoved"})

        patched_shared_executor.dispatch_screencast_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_controller_input_is_forwarded(self, patched_shared_executor):
        service = ScreencastService()
        controller = uuid4()
        await service.take_control(controller)

        payload = {"type": "mouseMoved", "x": 1, "y": 2}
        await service.dispatch_input(controller, "mouse", payload)

        patched_shared_executor.dispatch_screencast_input.assert_called_once_with(
            "mouse", payload
        )

    @pytest.mark.asyncio
    async def test_input_before_taking_control_is_ignored(
        self, patched_shared_executor
    ):
        service = ScreencastService()
        viewer = uuid4()

        await service.dispatch_input(viewer, "mouse", {"type": "mouseMoved"})

        patched_shared_executor.dispatch_screencast_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_release_by_non_controller_is_ignored(self, patched_shared_executor):
        service = ScreencastService()
        controller, bystander = uuid4(), uuid4()
        await service.take_control(controller)

        released = await service.release_control(bystander)

        assert released is False
        assert service.control_owner == "human"
        assert service.is_controller(controller) is True

    @pytest.mark.asyncio
    async def test_releasing_control_restores_agent_access(
        self, patched_shared_executor
    ):
        service = ScreencastService()
        viewer = uuid4()
        await service.take_control(viewer)
        patched_shared_executor.set_control_owner.reset_mock()

        released = await service.release_control(viewer)

        assert released is True
        assert service.control_owner == "agent"
        assert service.is_controller(viewer) is False
        patched_shared_executor.set_control_owner.assert_called_once_with("agent")

    @pytest.mark.asyncio
    async def test_input_ignored_after_release(self, patched_shared_executor):
        service = ScreencastService()
        viewer = uuid4()
        await service.take_control(viewer)
        await service.release_control(viewer)

        await service.dispatch_input(viewer, "mouse", {"type": "mouseMoved"})

        patched_shared_executor.dispatch_screencast_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnecting_controller_releases_control(
        self, patched_shared_executor
    ):
        """A dropped connection must never leave the browser permanently
        locked to a human who is no longer there."""
        service = ScreencastService()
        sub_id = await service.subscribe(_FakeSubscriber())
        await service.take_control(sub_id)
        assert service.control_owner == "human"

        await service.unsubscribe(sub_id)

        assert service.control_owner == "agent"
        assert service.is_controller(sub_id) is False

    @pytest.mark.asyncio
    async def test_disconnecting_non_controller_does_not_release_control(
        self, patched_shared_executor
    ):
        service = ScreencastService()
        controller = uuid4()
        bystander_sub_id = await service.subscribe(_FakeSubscriber())
        await service.take_control(controller)

        await service.unsubscribe(bystander_sub_id)

        assert service.control_owner == "human"
        assert service.is_controller(controller) is True


class TestCursorPublication:
    @pytest.mark.asyncio
    async def test_cursor_events_are_published_to_subscribers(
        self, patched_shared_executor
    ):
        received: list[dict] = []

        class _Recorder(Subscriber[dict]):
            async def __call__(self, message: dict) -> None:
                received.append(message)

        service = ScreencastService()
        await service.subscribe(_Recorder())

        # start_screencast must be wired with an on_cursor callback...
        _, kwargs = patched_shared_executor.start_screencast.call_args
        on_cursor = kwargs["on_cursor"]

        # ...which, invoked from the browser's background thread, publishes a
        # typed cursor message onto the service's pub/sub.
        await asyncio.to_thread(
            on_cursor, {"x": 10.0, "y": 20.0, "event": "mousePressed"}
        )
        for _ in range(10):
            if received:
                break
            await asyncio.sleep(0.01)

        assert received == [
            {"type": "cursor", "x": 10.0, "y": 20.0, "event": "mousePressed"}
        ]
