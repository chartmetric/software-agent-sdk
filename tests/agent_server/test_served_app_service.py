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


@pytest.mark.asyncio
async def test_the_browsers_devtools_listener_is_not_a_served_app(monkeypatch):
    """The agent's Chromium debugger answers text/html, but it is not an app.

    Treating it as one handed it a public worker, displaced the repository
    preview from its slot, and mirrored the agent's own browser to whoever
    opened that worker in the App panel.
    """
    monkeypatch.setattr(served_app_service, "_listening_ports", lambda: {3000, 39001})

    async def probe(port: int):
        return _DiscoveredApp(port=port, kind="web")

    async def probe_devtools(port: int):
        return port == 39001

    async def start_server(handler, host, port):
        return _FakeServer()

    monkeypatch.setattr(served_app_service, "_probe_http", probe)
    monkeypatch.setattr(served_app_service, "_probe_devtools", probe_devtools)
    monkeypatch.setattr(asyncio, "start_server", start_server)
    service = ServedAppService()

    await service._reconcile()

    assert [(app.name, app.port) for app in service.list_apps()] == [("Web App", 3000)]


@pytest.mark.asyncio
async def test_an_unconfirmed_devtools_probe_is_not_cached_as_a_verdict(monkeypatch):
    """A timed-out confirmation says nothing; the next cycle must re-ask.

    Caching it as "not the debugger" would offer the mirror forever after one
    busy moment; caching it as "the debugger" would hide a real app forever.
    """
    monkeypatch.setattr(served_app_service, "_listening_ports", lambda: {39001})

    async def probe(port: int):
        return _DiscoveredApp(port=port, kind="web")

    verdicts = iter([None, True])

    async def probe_devtools(port: int):
        return next(verdicts)

    async def start_server(handler, host, port):
        return _FakeServer()

    monkeypatch.setattr(served_app_service, "_probe_http", probe)
    monkeypatch.setattr(served_app_service, "_probe_devtools", probe_devtools)
    monkeypatch.setattr(asyncio, "start_server", start_server)
    service = ServedAppService()

    await service._reconcile()
    assert [app.port for app in service.list_apps()] == [39001]

    await service._reconcile()
    assert service.list_apps() == []


@pytest.mark.asyncio
async def test_an_app_keeps_its_worker_when_another_listener_appears(monkeypatch):
    """Worker assignments must not reshuffle because discovery found one more.

    Rebuilding the mapping from scratch every cycle moved the same application
    between public worker URLs whenever a listener appeared or vanished, so the
    App panel's address changed and the preview reloaded mid-conversation.
    """
    listeners = {3000}
    monkeypatch.setattr(served_app_service, "_listening_ports", lambda: set(listeners))

    async def probe(port: int):
        return _DiscoveredApp(port=port, kind="web")

    async def probe_devtools(port: int):
        return False

    async def start_server(handler, host, port):
        return _FakeServer()

    monkeypatch.setattr(served_app_service, "_probe_http", probe)
    monkeypatch.setattr(served_app_service, "_probe_devtools", probe_devtools)
    monkeypatch.setattr(asyncio, "start_server", start_server)
    service = ServedAppService()

    await service._reconcile()
    assert service._targets == {8011: 3000}

    # A second listener sorts ahead of the first by port; the first still
    # keeps its worker and the newcomer takes the free one.
    listeners.add(2999)
    await service._reconcile()

    assert service._targets == {8011: 3000, 8012: 2999}


@pytest.mark.asyncio
async def test_a_silent_but_listening_app_survives_a_few_probe_misses(monkeypatch):
    """A dev server stalled past the probe timeout is busy, not gone.

    Dropping it on one missed probe tore down its relay and blanked the App
    panel every time the repository compiled; only an app that stays silent for
    several cycles -- or stops listening -- is really gone.
    """
    monkeypatch.setattr(served_app_service, "_listening_ports", lambda: {3000})

    answers = iter([_DiscoveredApp(port=3000, kind="web"), None, None, None])

    async def probe(port: int):
        return next(answers)

    async def probe_devtools(port: int):
        return False

    async def start_server(handler, host, port):
        return _FakeServer()

    monkeypatch.setattr(served_app_service, "_probe_http", probe)
    monkeypatch.setattr(served_app_service, "_probe_devtools", probe_devtools)
    monkeypatch.setattr(asyncio, "start_server", start_server)
    service = ServedAppService()

    await service._reconcile()
    assert [app.port for app in service.list_apps()] == [3000]

    # Two consecutive misses keep the app; the third drops it.
    await service._reconcile()
    assert [app.port for app in service.list_apps()] == [3000]
    await service._reconcile()
    assert [app.port for app in service.list_apps()] == [3000]
    await service._reconcile()
    assert service.list_apps() == []


@pytest.mark.asyncio
async def test_a_port_that_stops_listening_is_dropped_immediately(monkeypatch):
    """Probe-miss tolerance is for silence, not for a closed port."""
    listeners = {3000}
    monkeypatch.setattr(served_app_service, "_listening_ports", lambda: set(listeners))

    async def probe(port: int):
        return _DiscoveredApp(port=port, kind="web")

    async def probe_devtools(port: int):
        return False

    async def start_server(handler, host, port):
        return _FakeServer()

    monkeypatch.setattr(served_app_service, "_probe_http", probe)
    monkeypatch.setattr(served_app_service, "_probe_devtools", probe_devtools)
    monkeypatch.setattr(asyncio, "start_server", start_server)
    service = ServedAppService()

    await service._reconcile()
    assert [app.port for app in service.list_apps()] == [3000]

    listeners.clear()
    await service._reconcile()

    assert service.list_apps() == []


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


def test_launcher_relay_targets_parses_the_marker(tmp_path, monkeypatch):
    """The relay argv tail is ``<public> <upstream> oh-runtime-port-proxy-<public>``."""
    relay = tmp_path / "101"
    relay.mkdir()
    (relay / "cmdline").write_bytes(
        b"python\x00-c\x00import asyncio...\x0012000\x003000"
        b"\x00oh-runtime-port-proxy-12000\x00"
    )
    unrelated = tmp_path / "102"
    unrelated.mkdir()
    (unrelated / "cmdline").write_bytes(b"node\x00server.js\x00")
    mismatched = tmp_path / "103"
    mismatched.mkdir()
    (mismatched / "cmdline").write_bytes(
        b"python\x00-c\x00code\x0012001\x003001\x00oh-runtime-port-proxy-9999\x00"
    )
    (tmp_path / "not-a-pid").mkdir()
    monkeypatch.setattr(served_app_service, "PROC_DIR", tmp_path)

    assert served_app_service._launcher_relay_targets() == {12000: 3000}


@pytest.mark.asyncio
async def test_the_launchers_own_relay_and_its_upstream_are_one_app(monkeypatch):
    """A launcher relay on a worker port is an alias, not a second app.

    The launcher parks its own relay on WORKER_1 and forwards it to the dev
    server. Discovery counted the relay as one app and the dev server as
    another, so the panel offered the same application on both workers.
    """
    monkeypatch.setattr(served_app_service, "WORKER_PORTS", (12000, 12001))
    monkeypatch.setattr(served_app_service, "_listening_ports", lambda: {12000, 3000})
    monkeypatch.setattr(
        served_app_service, "_launcher_relay_targets", lambda: {12000: 3000}
    )

    probed = []

    async def probe(port):
        probed.append(port)
        return _DiscoveredApp(port=port, kind="web")

    async def probe_devtools(port):
        return False

    started = []

    async def start_server(handler, host, port):
        started.append(port)
        return AsyncMock()

    monkeypatch.setattr(served_app_service, "_probe_http", probe)
    monkeypatch.setattr(served_app_service, "_probe_devtools", probe_devtools)
    monkeypatch.setattr(asyncio, "start_server", start_server)

    service = ServedAppService()
    await service._reconcile()

    assert probed == [3000]
    assert service._targets == {12000: 3000}
    # The launcher relay owns the worker socket; no bind attempt.
    assert started == []
    assert [app.model_dump() for app in service.list_apps()] == [
        {
            "name": "Web App",
            "port": 3000,
            "worker_name": "WORKER_1",
            "worker_port": 12000,
            "kind": "web",
        },
    ]


@pytest.mark.asyncio
async def test_the_worker_is_taken_over_when_the_launcher_relay_exits(monkeypatch):
    """The mapping survives the relay exit and the service binds the port."""
    monkeypatch.setattr(served_app_service, "WORKER_PORTS", (12000, 12001))
    monkeypatch.setattr(served_app_service, "_listening_ports", lambda: {12000, 3000})
    relays = {12000: 3000}
    monkeypatch.setattr(
        served_app_service, "_launcher_relay_targets", lambda: dict(relays)
    )

    async def probe(port):
        return _DiscoveredApp(port=port, kind="web")

    async def probe_devtools(port):
        return False

    started = []

    async def start_server(handler, host, port):
        started.append(port)
        return AsyncMock()

    monkeypatch.setattr(served_app_service, "_probe_http", probe)
    monkeypatch.setattr(served_app_service, "_probe_devtools", probe_devtools)
    monkeypatch.setattr(asyncio, "start_server", start_server)

    service = ServedAppService()
    await service._reconcile()
    assert started == []

    relays.clear()
    monkeypatch.setattr(served_app_service, "_listening_ports", lambda: {3000})
    await service._reconcile()

    assert service._targets == {12000: 3000}
    assert started == [12000]
