import asyncio
from unittest.mock import AsyncMock

import pytest

from openhands.agent_server import served_app_service
from openhands.agent_server.served_app_service import ServedAppService, _DiscoveredApp


@pytest.mark.asyncio
async def test_reconcile_prioritizes_web_apps_and_assigns_worker_ports(monkeypatch):
    monkeypatch.setattr(served_app_service, "_listening_ports", lambda: {4000, 3001})

    async def probe(port: int):
        return _DiscoveredApp(port=port, kind="web" if port == 3001 else "api")

    async def start_server(handler, host, port):
        return _FakeServer()

    monkeypatch.setattr(served_app_service, "_probe_http", probe)
    monkeypatch.setattr(asyncio, "start_server", start_server)
    service = ServedAppService()

    await service._reconcile()

    assert [app.model_dump() for app in service.list_apps()] == [
        {
            "name": "Web App",
            "port": 3001,
            "worker_name": "WORKER_1",
            "worker_port": 8011,
            "kind": "web",
        },
        {
            "name": "API",
            "port": 4000,
            "worker_name": "WORKER_2",
            "worker_port": 8012,
            "kind": "api",
        },
    ]


@pytest.mark.asyncio
async def test_reconcile_retries_worker_bridge_after_bind_failure(monkeypatch):
    monkeypatch.setattr(served_app_service, "_listening_ports", lambda: {3001})

    async def probe(port: int):
        return _DiscoveredApp(port=port, kind="web")

    server = _FakeServer()
    start_server = AsyncMock(side_effect=[OSError, server])
    monkeypatch.setattr(served_app_service, "_probe_http", probe)
    monkeypatch.setattr(asyncio, "start_server", start_server)
    service = ServedAppService()

    await service._reconcile()
    await service._reconcile()

    assert start_server.await_count == 2
    assert service._servers == {8011: server}


class _FakeServer:
    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass
