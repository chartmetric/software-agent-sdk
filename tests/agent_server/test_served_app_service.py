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


def test_worker_ports_stay_discoverable():
    """A service bound to a worker port is an app, not a port to skip.

    Reserving the worker ports hid such an app from discovery, and the freed
    worker was then handed to a *different* app -- which was relayed onto the
    port the hidden one was already serving, so the panel offered one app under
    the other's name. Verified against real sockets: with apps on 12000 and 3000
    and WORKER_1/WORKER_2 = 12000/12001, reserving them discovered only 3000 and
    mapped 12000 to it.
    """
    assert not (
        set(served_app_service.WORKER_PORTS) & served_app_service.RESERVED_PORTS
    )


def test_sandbox_service_ports_are_reserved_from_discovery():
    """The desktop stream answers HTTP, so discovery listed it as a served app."""
    for port in (60000, 60001, 60002):
        assert port in served_app_service.RESERVED_PORTS


def test_an_app_on_a_worker_port_keeps_it_and_the_next_app_takes_the_other(
    monkeypatch,
):
    """The mapping the App panel is proxied to, for the two placements that exist.

    An app already sitting on a worker port must map to itself with no relay; any
    other app takes a still-free worker. Getting this wrong routed a worker to a
    port nothing served.
    """
    monkeypatch.setattr(served_app_service, "WORKER_PORTS", (12000, 12001))
    monkeypatch.setattr(served_app_service, "_listening_ports", lambda: {12000, 3000})

    async def probe(port):
        return _DiscoveredApp(port=port, kind="web")

    started = []

    async def start_server(handler, host, port):
        started.append(port)
        return AsyncMock()

    monkeypatch.setattr(served_app_service, "_probe_http", probe)
    monkeypatch.setattr(asyncio, "start_server", start_server)

    service = ServedAppService()
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        service._reconcile()
    )

    assert service._targets == {12000: 12000, 12001: 3000}
    assert started == [12001]
