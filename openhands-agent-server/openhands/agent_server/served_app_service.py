import asyncio
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

DEFAULT_WORKER_PORTS = (8011, 8012)


def _configured_worker_ports() -> tuple[int, ...]:
    """The ports the control plane publishes this sandbox's workers on.

    They are injected as WORKER_1/WORKER_2 and the proxy routes each name to the
    port it names. Relaying on a different pair meant an app that bound a worker
    port directly was also relayed onto another one, so one web app was
    discovered twice and reported as two apps -- the second routed to a port
    nothing listened on. It also left the real worker ports out of the reserved
    set, so a repository service handed one of them as its PORT could collide
    with a relay. Fall back to the historical pair when nothing is injected.
    """
    ports: list[int] = []
    for name in ("WORKER_1", "WORKER_2"):
        raw = os.environ.get(name, "").strip()
        if not raw.isdigit():
            continue
        port = int(raw)
        if 0 < port < 65536:
            ports.append(port)
    return tuple(ports) if len(ports) == 2 else DEFAULT_WORKER_PORTS


WORKER_PORTS = _configured_worker_ports()
DISCOVERY_INTERVAL_SECONDS = 2
PROBE_TIMEOUT_SECONDS = 1
# A busy dev server (Turbopack compiling, SSR under load) can stall its accept
# queue past the probe timeout without being gone. Dropping the app on a single
# missed probe tore down its worker relay and made the App panel blink; the app
# is only really gone when its port stops listening or it stays silent this many
# cycles in a row.
MAX_CONSECUTIVE_PROBE_MISSES = 3
# 60002 is the desktop stream; it was absent, so the desktop was discovered as a
# served app of its own.
#
# The worker ports are deliberately NOT reserved. A repository service handed one
# of them as its PORT is a real served app that happens to already sit where the
# control plane routes that worker, and `_reconcile` handles exactly that by
# binding no relay for it. Reserving them instead hid such an app from discovery
# and then let a *different* app be relayed onto the port it was already serving,
# so the panel offered one app under the other's name.
RESERVED_PORTS = frozenset({22, 8000, 8001, 8002, 60000, 60001, 60002})
PROC_NET_PATHS = (Path("/proc/net/tcp"), Path("/proc/net/tcp6"))
PROC_DIR = Path("/proc")


class ServedApp(BaseModel):
    name: str
    port: int
    worker_name: str
    worker_port: int
    kind: str


@dataclass(frozen=True)
class _DiscoveredApp:
    port: int
    kind: str


def _listening_ports() -> set[int]:
    ports: set[int] = set()
    for path in PROC_NET_PATHS:
        if not path.exists():
            continue
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "0A":
                continue
            ports.add(int(fields[1].rsplit(":", 1)[1], 16))
    return ports - RESERVED_PORTS


async def _probe_http(port: int) -> _DiscoveredApp | None:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port),
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        writer.write(
            (
                f"GET / HTTP/1.1\r\nHost: localhost:{port}\r\nConnection: close\r\n\r\n"
            ).encode()
        )
        await writer.drain()
        response = await asyncio.wait_for(
            reader.read(8192), timeout=PROBE_TIMEOUT_SECONDS
        )
        writer.close()
        await writer.wait_closed()
    except (OSError, TimeoutError):
        return None

    if not response.startswith(b"HTTP/"):
        return None
    headers = response.partition(b"\r\n\r\n")[0].lower()
    kind = "web" if b"content-type: text/html" in headers else "api"
    return _DiscoveredApp(port=port, kind=kind)


async def _probe_devtools(port: int) -> bool | None:
    """Whether this listener is a Chrome DevTools (CDP) endpoint.

    The browser tool's Chromium opens a remote-debugging port whose root
    answers ``200 text/html`` -- exactly the sniff that classifies a served app
    as ``web``. Treating it as an app handed it a public worker, displaced the
    repository preview from its slot, and mirrored the agent's own browser to
    anyone who opened that worker. CDP's version document is unmistakable, so
    ask for it directly.

    ``None`` means the probe could not complete and says nothing either way;
    the caller must not cache it as a verdict.
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port),
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        writer.write(
            (
                f"GET /json/version HTTP/1.1\r\nHost: localhost:{port}\r\n"
                "Connection: close\r\n\r\n"
            ).encode()
        )
        await writer.drain()
        response = await asyncio.wait_for(
            reader.read(8192), timeout=PROBE_TIMEOUT_SECONDS
        )
        writer.close()
        await writer.wait_closed()
    except (OSError, TimeoutError):
        return None
    if not response.startswith(b"HTTP/"):
        return False
    status_line = response.partition(b"\r\n")[0]
    if b" 200" not in status_line:
        return False
    return b"webSocketDebuggerUrl" in response


class ServedAppService:
    def __init__(self) -> None:
        self._apps: dict[int, _DiscoveredApp] = {}
        self._targets: dict[int, int] = {}
        self._servers: dict[int, asyncio.Server] = {}
        self._probe_misses: dict[int, int] = {}
        self._devtools_verdicts: dict[int, bool] = {}
        self._discovery_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self._reconcile()
        self._discovery_task = asyncio.create_task(self._discovery_loop())

    async def stop(self) -> None:
        if self._discovery_task:
            self._discovery_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._discovery_task
        for server in self._servers.values():
            server.close()
        await asyncio.gather(
            *(server.wait_closed() for server in self._servers.values())
        )
        self._servers.clear()
        self._apps.clear()
        self._targets.clear()
        self._probe_misses.clear()
        self._devtools_verdicts.clear()

    def list_apps(self) -> list[ServedApp]:
        apps = []
        assigned_apps = [
            self._apps[target_port]
            for target_port in self._targets.values()
            if target_port in self._apps
        ]
        kind_totals = {
            kind: sum(app.kind == kind for app in assigned_apps)
            for kind in ("web", "api")
        }
        kind_indexes = {"web": 0, "api": 0}
        for worker_port, target_port in sorted(self._targets.items()):
            app = self._apps.get(target_port)
            if app is None:
                continue
            kind_indexes[app.kind] += 1
            base_name = "Web App" if app.kind == "web" else "API"
            name = (
                base_name
                if kind_totals[app.kind] == 1
                else f"{base_name} {kind_indexes[app.kind]}"
            )
            worker_number = WORKER_PORTS.index(worker_port) + 1
            apps.append(
                ServedApp(
                    name=name,
                    port=target_port,
                    worker_name=f"WORKER_{worker_number}",
                    worker_port=worker_port,
                    kind=app.kind,
                )
            )
        return apps

    async def _discovery_loop(self) -> None:
        while True:
            await asyncio.sleep(DISCOVERY_INTERVAL_SECONDS)
            try:
                await self._reconcile()
            except Exception:
                logger.warning("Failed to discover served apps", exc_info=True)

    async def _reconcile(self) -> None:
        listening_ports = _listening_ports() - set(self._servers)
        ordered_ports = sorted(listening_ports)
        probes = await asyncio.gather(*(_probe_http(port) for port in ordered_ports))

        apps: dict[int, _DiscoveredApp] = {}
        for port, probed in zip(ordered_ports, probes, strict=True):
            if probed is None:
                # Still listening but silent within the timeout: a busy event
                # loop, not a stopped app. Keep what we knew for a few cycles
                # so the worker mapping (and the App panel) does not flap.
                known = self._apps.get(port)
                if known is None:
                    continue
                misses = self._probe_misses.get(port, 0) + 1
                self._probe_misses[port] = misses
                if misses < MAX_CONSECUTIVE_PROBE_MISSES:
                    apps[port] = known
                continue
            self._probe_misses.pop(port, None)
            apps[port] = probed

        # The agent's browser debugger answers text/html on its root, so it
        # reads as a web app; confirm and exclude it before it can take a
        # public worker. Verdicts are cached per listening port -- Chromium
        # picks a fresh port each launch, so a port's nature never changes
        # while it stays open.
        for port, app in list(apps.items()):
            if app.kind != "web":
                continue
            verdict = self._devtools_verdicts.get(port)
            if verdict is None:
                verdict = await _probe_devtools(port)
                if verdict is not None:
                    self._devtools_verdicts[port] = verdict
            if verdict:
                del apps[port]
        for port in list(self._devtools_verdicts):
            if port not in listening_ports:
                del self._devtools_verdicts[port]
        for port in list(self._probe_misses):
            if port not in listening_ports:
                del self._probe_misses[port]

        discovered = sorted(
            apps.values(), key=lambda app: (app.kind != "web", app.port)
        )

        targets: dict[int, int] = {
            app.port: app.port for app in discovered if app.port in WORKER_PORTS
        }
        # An app keeps the worker it already holds. Rebuilding the mapping from
        # scratch every cycle let an appearing or vanishing listener shuffle
        # every assignment, so the same application moved between public worker
        # URLs from one discovery pass to the next and the App panel reloaded.
        assigned = set(targets.values())
        for worker_port, target_port in self._targets.items():
            if worker_port in targets or target_port in assigned:
                continue
            if target_port not in apps or target_port in WORKER_PORTS:
                continue
            targets[worker_port] = target_port
            assigned.add(target_port)
        available_workers = [port for port in WORKER_PORTS if port not in targets]
        unassigned_apps = [
            app
            for app in discovered
            if app.port not in WORKER_PORTS and app.port not in assigned
        ]
        targets.update(
            zip(
                available_workers,
                (app.port for app in unassigned_apps),
                strict=False,
            )
        )
        for worker_port in set(self._servers) - set(targets):
            server = self._servers.pop(worker_port)
            server.close()
            await server.wait_closed()

        self._apps, self._targets = apps, targets
        for worker_port, target_port in targets.items():
            if worker_port in self._servers or worker_port == target_port:
                continue
            try:
                self._servers[worker_port] = await asyncio.start_server(
                    lambda reader, writer, port=worker_port: self._relay(
                        port, reader, writer
                    ),
                    "0.0.0.0",
                    worker_port,
                )
            except OSError:
                logger.info(
                    "Worker port %d is already owned by a repository service",
                    worker_port,
                )

    async def _relay(
        self,
        worker_port: int,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        target_port = self._targets.get(worker_port)
        if target_port is None:
            client_writer.close()
            await client_writer.wait_closed()
            return
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                "127.0.0.1", target_port
            )
        except OSError:
            client_writer.close()
            await client_writer.wait_closed()
            return

        async def copy(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                while data := await reader.read(64 * 1024):
                    writer.write(data)
                    await writer.drain()
            except (ConnectionError, asyncio.CancelledError):
                pass
            finally:
                writer.close()

        await asyncio.gather(
            copy(client_reader, upstream_writer),
            copy(upstream_reader, client_writer),
        )


_served_app_service = ServedAppService()


def get_served_app_service() -> ServedAppService:
    return _served_app_service
