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


def test_configured_worker_ports_follow_the_injected_environment(monkeypatch):
    """The relay must listen where the control plane routes each worker name.

    Hardcoding a different pair made an app bound directly to a worker port get
    relayed onto a second one, so a single web app was reported twice and the
    duplicate pointed at a port nothing served.
    """
    monkeypatch.setenv("WORKER_1", "12000")
    monkeypatch.setenv("WORKER_2", "12001")

    assert served_app_service._configured_worker_ports() == (12000, 12001)


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"WORKER_1": "12000"},
        {"WORKER_1": "12000", "WORKER_2": "not-a-port"},
        {"WORKER_1": "0", "WORKER_2": "70000"},
    ],
    ids=["absent", "incomplete", "unparsable", "out-of-range"],
)
def test_configured_worker_ports_fall_back_when_unusable(monkeypatch, environment):
    for name in ("WORKER_1", "WORKER_2"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert (
        served_app_service._configured_worker_ports()
        == served_app_service.DEFAULT_WORKER_PORTS
    )


def test_worker_ports_are_reserved_from_discovery():
    """A repository service handed a worker port as its PORT collided with the
    relay that was already bound there, so the managed Preview never started."""
    assert set(served_app_service.WORKER_PORTS) <= served_app_service.RESERVED_PORTS
    assert 60002 in served_app_service.RESERVED_PORTS
