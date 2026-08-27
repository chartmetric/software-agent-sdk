"""Tests for the `/sockets/screencast` WebSocket route.

Mirrors `test_bash_events_socket_uses_app_state_bash_event_service` in
`test_sockets_service_getters.py`: the handler is invoked directly with a
mocked WebSocket rather than through a real handshake, since auth
(`_accept_authenticated_websocket`) and the app.state-first lookup pattern
are already covered generically for the sibling `/bash-events` route.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import WebSocketDisconnect

import openhands.agent_server.sockets as sockets_mod
from openhands.agent_server.config import Config
from openhands.agent_server.pub_sub import MaxSubscribersError
from openhands.agent_server.screencast_service import ScreencastService
from openhands.agent_server.sockets import _get_screencast_service, screencast_socket


def _make_ws(state: dict[str, object] | None = None) -> MagicMock:
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    ws.headers = {}
    ws.app.state = SimpleNamespace(**(state or {}))
    return ws


class TestGetScreencastService:
    def test_prefers_app_state(self):
        service = MagicMock(spec=ScreencastService)
        ws = _make_ws({"screencast_service": service})
        assert _get_screencast_service(ws) is service

    def test_falls_back_to_module_singleton(self):
        ws = _make_ws()
        assert _get_screencast_service(ws) is sockets_mod.screencast_service

    def test_ignores_wrong_type(self):
        for bogus in (None, "not a service", 42):
            ws = _make_ws({"screencast_service": bogus})
            assert isinstance(_get_screencast_service(ws), ScreencastService)


class TestScreencastSocketRoute:
    @pytest.mark.asyncio
    async def test_uses_app_state_screencast_service(self):
        per_app_service = MagicMock(spec=ScreencastService)
        per_app_service.subscribe = AsyncMock(return_value=uuid4())
        per_app_service.unsubscribe = AsyncMock()

        ws = _make_ws(
            {
                "screencast_service": per_app_service,
                "config": Config(session_api_keys=[]),
            }
        )

        await screencast_socket(ws, session_api_key=None)

        per_app_service.subscribe.assert_called_once()
        per_app_service.unsubscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_closes_with_1013_when_subscriber_limit_reached(self):
        per_app_service = MagicMock(spec=ScreencastService)
        per_app_service.subscribe = AsyncMock(side_effect=MaxSubscribersError())

        ws = _make_ws(
            {
                "screencast_service": per_app_service,
                "config": Config(session_api_keys=[]),
            }
        )

        await screencast_socket(ws, session_api_key=None)

        ws.close.assert_awaited_once_with(
            code=1013, reason="Too many screencast viewers"
        )

    @pytest.mark.asyncio
    async def test_redundant_auth_frame_is_ignored_not_treated_as_error(self):
        per_app_service = MagicMock(spec=ScreencastService)
        per_app_service.subscribe = AsyncMock(return_value=uuid4())
        per_app_service.unsubscribe = AsyncMock()

        ws = _make_ws(
            {
                "screencast_service": per_app_service,
                "config": Config(session_api_keys=[]),
            }
        )
        ws.receive_json = AsyncMock(
            side_effect=[
                {"type": "auth", "session_api_key": "x"},
                WebSocketDisconnect(),
            ]
        )

        await screencast_socket(ws, session_api_key=None)

        per_app_service.subscribe.assert_called_once()
        per_app_service.unsubscribe.assert_called_once()


class TestScreencastSubscriberQueue:
    @pytest.mark.asyncio
    async def test_stale_frames_are_skipped_without_evicting_cursor_updates(self):
        """The per-type latest-wins queue: while the sender is blocked on a
        slow client, newer frames replace the pending frame, but a pending
        cursor update survives the frame churn (and vice versa)."""
        from openhands.agent_server.sockets import _ScreencastWebSocketSubscriber

        ws = _make_ws()
        gate = asyncio.Event()
        sent: list[dict] = []

        async def _slow_send(message: dict) -> None:
            sent.append(message)
            await gate.wait()

        ws.send_json = AsyncMock(side_effect=_slow_send)
        subscriber = _ScreencastWebSocketSubscriber(ws)
        try:
            frame_old = {"type": "frame", "data": "first", "metadata": {}}
            frame_stale = {"type": "frame", "data": "stale", "metadata": {}}
            cursor = {"type": "cursor", "x": 1, "y": 2, "event": "mousePressed"}
            frame_new = {"type": "frame", "data": "newest", "metadata": {}}

            await subscriber(frame_old)
            # Let the sender task pick up frame_old and block inside send_json.
            for _ in range(3):
                await asyncio.sleep(0)

            await subscriber(frame_stale)
            await subscriber(cursor)
            await subscriber(frame_new)  # replaces frame_stale, not the cursor

            gate.set()
            for _ in range(10):
                if len(sent) >= 3:
                    break
                await asyncio.sleep(0.01)

            assert sent == [frame_old, frame_new, cursor]
        finally:
            await subscriber.close()


def _make_gated_dispatch_service():
    """A service whose dispatch_input records calls and blocks on a gate,
    letting tests hold the pump's worker mid-dispatch deterministically."""
    service = MagicMock(spec=ScreencastService)
    gate = asyncio.Event()
    calls: list[tuple[str, dict]] = []

    async def _dispatch(subscriber_id, kind, payload):
        calls.append((kind, payload))
        await gate.wait()

    service.dispatch_input = AsyncMock(side_effect=_dispatch)
    return service, gate, calls


async def _wait_until(predicate, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.01)


class TestScreencastInputPump:
    @pytest.mark.asyncio
    async def test_pending_moves_coalesce_to_the_latest_position(self):
        from openhands.agent_server.sockets import _ScreencastInputPump

        service, gate, calls = _make_gated_dispatch_service()
        pump = _ScreencastInputPump(service, uuid4())
        try:
            pump.enqueue("mouse", {"type": "mouseMoved", "x": 1, "y": 1})
            await _wait_until(lambda: len(calls) == 1)

            # Queued behind the blocked dispatch: only the newest move survives.
            pump.enqueue("mouse", {"type": "mouseMoved", "x": 2, "y": 2})
            pump.enqueue("mouse", {"type": "mouseMoved", "x": 3, "y": 3})

            gate.set()
            await _wait_until(lambda: len(calls) == 2)

            assert [(kind, p["x"], p["y"]) for kind, p in calls] == [
                ("mouse", 1, 1),
                ("mouse", 3, 3),
            ]
        finally:
            await pump.close()

    @pytest.mark.asyncio
    async def test_pending_wheels_merge_their_deltas(self):
        from openhands.agent_server.sockets import _ScreencastInputPump

        service, gate, calls = _make_gated_dispatch_service()
        pump = _ScreencastInputPump(service, uuid4())
        try:
            pump.enqueue("mouse", {"type": "mouseWheel", "x": 1, "y": 1, "delta_y": 1})
            await _wait_until(lambda: len(calls) == 1)

            pump.enqueue("mouse", {"type": "mouseWheel", "x": 2, "y": 2, "delta_y": 2})
            pump.enqueue("mouse", {"type": "mouseWheel", "x": 3, "y": 3, "delta_y": 3})

            gate.set()
            await _wait_until(lambda: len(calls) == 2)

            merged = calls[1][1]
            assert (merged["x"], merged["y"], merged["delta_y"]) == (3, 3, 5)
        finally:
            await pump.close()

    @pytest.mark.asyncio
    async def test_presses_are_never_coalesced_and_order_is_preserved(self):
        from openhands.agent_server.sockets import _ScreencastInputPump

        service, gate, calls = _make_gated_dispatch_service()
        pump = _ScreencastInputPump(service, uuid4())
        try:
            pump.enqueue("mouse", {"type": "mouseMoved", "x": 1, "y": 1})
            await _wait_until(lambda: len(calls) == 1)

            pump.enqueue("mouse", {"type": "mouseMoved", "x": 2, "y": 2})
            pump.enqueue("mouse", {"type": "mousePressed", "x": 2, "y": 2})
            pump.enqueue("mouse", {"type": "mouseMoved", "x": 3, "y": 3})

            gate.set()
            await _wait_until(lambda: len(calls) == 4)

            assert [p["type"] for _, p in calls] == [
                "mouseMoved",
                "mouseMoved",
                "mousePressed",
                "mouseMoved",
            ]
        finally:
            await pump.close()

    @pytest.mark.asyncio
    async def test_route_forwards_input_through_the_pump(self):
        service = MagicMock(spec=ScreencastService)
        service.subscribe = AsyncMock(return_value=uuid4())
        service.unsubscribe = AsyncMock()
        dispatched: list[tuple[str, dict]] = []

        async def _dispatch(subscriber_id, kind, payload):
            dispatched.append((kind, payload))

        service.dispatch_input = AsyncMock(side_effect=_dispatch)

        ws = _make_ws({"screencast_service": service})
        receive_count = {"n": 0}

        async def _receive():
            receive_count["n"] += 1
            if receive_count["n"] == 1:
                return {
                    "type": "mouse",
                    "action": "down",
                    "x": 5,
                    "y": 6,
                }
            # Keep the connection open until the pump has drained the input,
            # then disconnect.
            await _wait_until(lambda: dispatched)
            raise WebSocketDisconnect()

        ws.receive_json = AsyncMock(side_effect=_receive)

        await screencast_socket(ws, session_api_key=None)

        assert len(dispatched) == 1
        kind, payload = dispatched[0]
        assert kind == "mouse"
        assert (payload["type"], payload["x"], payload["y"]) == (
            "mousePressed",
            5,
            6,
        )


class TestScreencastKeyMessages:
    """Chrome runs a keyboard event's editing command off the virtual key
    code, never off `key`: measured against a real CDP target, a Backspace
    with no code deletes nothing while the same event carrying 8 deletes the
    character. The wire protocol therefore has to carry it end to end."""

    @staticmethod
    async def _enqueued(message: dict) -> list[tuple[str, dict]]:
        enqueued: list[tuple[str, dict]] = []
        pump = MagicMock()
        pump.enqueue = MagicMock(
            side_effect=lambda kind, payload: enqueued.append((kind, payload))
        )
        await sockets_mod._handle_screencast_client_message(
            MagicMock(spec=ScreencastService),
            uuid4(),
            message,
            MagicMock(),
            pump,
        )
        return enqueued

    @pytest.mark.asyncio
    async def test_key_code_reaches_the_dispatcher(self):
        enqueued = await self._enqueued(
            {
                "type": "key",
                "action": "down",
                "key": "Backspace",
                "code": "Backspace",
                "keyCode": 8,
            }
        )

        assert len(enqueued) == 1
        kind, payload = enqueued[0]
        assert kind == "key"
        assert payload["type"] == "keyDown"
        assert payload["windows_virtual_key_code"] == 8

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bogus", [None, "8", 8.5, True])
    async def test_non_integer_key_code_is_dropped_not_forwarded(self, bogus):
        """A bad value must not reach CDP as a real key: `True` is an int in
        Python and would arrive as virtual key 1."""
        enqueued = await self._enqueued(
            {
                "type": "key",
                "action": "down",
                "key": "a",
                "code": "KeyA",
                "keyCode": bogus,
            }
        )

        assert enqueued[0][1]["windows_virtual_key_code"] is None

    @pytest.mark.asyncio
    async def test_client_that_sends_no_key_code_is_still_forwarded(self):
        """Rolling out the frontend and the sandbox image is not atomic, so
        the older client keeps working rather than losing keyboard input."""
        enqueued = await self._enqueued(
            {"type": "key", "action": "char", "key": "a", "code": "KeyA", "text": "a"}
        )

        assert enqueued[0][1]["windows_virtual_key_code"] is None
        assert enqueued[0][1]["text"] == "a"


class TestBinaryFrameDelivery:
    @pytest.mark.asyncio
    async def test_opted_in_viewer_receives_framed_bytes(self):
        import base64
        import json

        from openhands.agent_server.sockets import _ScreencastWebSocketSubscriber

        ws = _make_ws()
        ws.send_bytes = AsyncMock()
        subscriber = _ScreencastWebSocketSubscriber(ws)
        try:
            subscriber.set_binary_frames(True)
            jpeg = b"\xff\xd8jpeg-bytes"
            metadata = {"deviceWidth": 1280, "deviceHeight": 800}
            await subscriber(
                {
                    "type": "frame",
                    "data": base64.b64encode(jpeg).decode("ascii"),
                    "metadata": metadata,
                }
            )
            await _wait_until(lambda: ws.send_bytes.await_count == 1)

            payload = ws.send_bytes.await_args.args[0]
            metadata_len = int.from_bytes(payload[:4], "big")
            assert json.loads(payload[4 : 4 + metadata_len]) == metadata
            assert payload[4 + metadata_len :] == jpeg
            ws.send_json.assert_not_awaited()

            # Non-frame messages stay JSON text even when opted in.
            cursor = {"type": "cursor", "x": 1, "y": 2, "event": "mousePressed"}
            await subscriber(cursor)
            await _wait_until(lambda: ws.send_json.await_count == 1)
            ws.send_json.assert_awaited_once_with(cursor)
        finally:
            await subscriber.close()

    @pytest.mark.asyncio
    async def test_default_delivery_remains_json(self):
        from openhands.agent_server.sockets import _ScreencastWebSocketSubscriber

        ws = _make_ws()
        ws.send_bytes = AsyncMock()
        subscriber = _ScreencastWebSocketSubscriber(ws)
        try:
            frame = {"type": "frame", "data": "AAAA", "metadata": {}}
            await subscriber(frame)
            await _wait_until(lambda: ws.send_json.await_count == 1)

            ws.send_json.assert_awaited_once_with(frame)
            ws.send_bytes.assert_not_awaited()
        finally:
            await subscriber.close()
