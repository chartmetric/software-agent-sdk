import asyncio
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

WORKER_PORTS = (8011, 8012)
DISCOVERY_INTERVAL_SECONDS = 2
PROBE_TIMEOUT_SECONDS = 1
RESERVED_PORTS = frozenset({22, 8000, 8001, 8002, 60000, 60001})
PROC_NET_PATHS = (Path("/proc/net/tcp"), Path("/proc/net/tcp6"))


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


class ServedAppService:
    def __init__(self) -> None:
        self._apps: dict[int, _DiscoveredApp] = {}
        self._targets: dict[int, int] = {}
        self._servers: dict[int, asyncio.Server] = {}
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
        probes = await asyncio.gather(*(_probe_http(port) for port in listening_ports))
        discovered = [app for app in probes if app is not None]
        discovered.sort(key=lambda app: (app.kind != "web", app.port))
        apps = {app.port: app for app in discovered}

        targets: dict[int, int] = {
            app.port: app.port for app in discovered if app.port in WORKER_PORTS
        }
        available_workers = [port for port in WORKER_PORTS if port not in targets]
        indirect_apps = [app for app in discovered if app.port not in WORKER_PORTS]
        targets.update(
            zip(
                available_workers,
                (app.port for app in indirect_apps),
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
