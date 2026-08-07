"""Tests for the `/sockets/screencast` WebSocket route.

Mirrors `test_bash_events_socket_uses_app_state_bash_event_service` in
`test_sockets_service_getters.py`: the handler is invoked directly with a
mocked WebSocket rather than through a real handshake, since auth
(`_accept_authenticated_websocket`) and the app.state-first lookup pattern
are already covered generically for the sibling `/bash-events` route.
"""

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
