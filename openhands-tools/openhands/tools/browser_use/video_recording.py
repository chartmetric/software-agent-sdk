"""Visible desktop video recording for browser QA evidence."""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import os
import shutil
import signal
from datetime import UTC, datetime
from pathlib import Path


VIDEO_OUTPUT_DIR = "browser_videos"
VIDEO_START_ATTEMPTS = 2
VIDEO_START_SETTLE_SECONDS = 0.75
VIDEO_START_ERROR_MAX_CHARS = 500


def _active_top_level_window_id(display_name: str) -> int:
    """Return the X11 frame containing the currently focused browser widget."""
    library_name = ctypes.util.find_library("X11")
    if library_name is None:
        raise RuntimeError("libX11 is unavailable")

    x11 = ctypes.CDLL(library_name)
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XGetInputFocus.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int),
    ]
    x11.XQueryTree.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
        ctypes.POINTER(ctypes.c_uint),
    ]
    x11.XQueryTree.restype = ctypes.c_int
    x11.XFree.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]

    display = x11.XOpenDisplay(display_name.encode())
    if not display:
        raise RuntimeError(f"could not open X11 display {display_name}")

    try:
        focused = ctypes.c_ulong()
        revert_to = ctypes.c_int()
        x11.XGetInputFocus(display, ctypes.byref(focused), ctypes.byref(revert_to))
        root_window = x11.XDefaultRootWindow(display)
        window = focused.value
        if window in (0, 1, root_window):
            raise RuntimeError("X11 has no focused window")

        while True:
            returned_root = ctypes.c_ulong()
            parent = ctypes.c_ulong()
            children = ctypes.POINTER(ctypes.c_ulong)()
            child_count = ctypes.c_uint()
            status = x11.XQueryTree(
                display,
                window,
                ctypes.byref(returned_root),
                ctypes.byref(parent),
                ctypes.byref(children),
                ctypes.byref(child_count),
            )
            if children:
                x11.XFree(children)
            if status == 0:
                raise RuntimeError("could not inspect the focused X11 window")
            if parent.value == root_window:
                return window
            if parent.value in (0, window):
                raise RuntimeError("focused X11 window has no top-level frame")
            window = parent.value
    finally:
        x11.XCloseDisplay(display)


class BrowserVideoRecorder:
    """Record the visible headed browser window to WebM."""

    def __init__(self, output_root: str | None) -> None:
        self._output_root = output_root
        self._process: asyncio.subprocess.Process | None = None
        self._output_path: Path | None = None

    @property
    def is_recording(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> str:
        if self.is_recording:
            assert self._output_path is not None
            return f"Error: Video recording is already active at {self._output_path}"

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return "Error: ffmpeg is required for browser video recording"

        display = os.getenv("DISPLAY")
        if display is None:
            return "Error: DISPLAY is required for visible browser video recording"

        try:
            window_id = _active_top_level_window_id(display)
        except RuntimeError as exc:
            return f"Error: Could not identify the visible browser window: {exc}"

        output_root = (
            Path(self._output_root)
            if self._output_root is not None
            else Path.cwd() / ".agent_tmp" / "browser_observations"
        )
        output_dir = output_root / VIDEO_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self._output_path = (output_dir / f"browser-{timestamp}.webm").resolve()

        last_returncode: int | None = None
        last_error = ""
        for attempt in range(VIDEO_START_ATTEMPTS):
            self._process = await asyncio.create_subprocess_exec(
                ffmpeg,
                "-hide_banner",
                "-nostats",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "x11grab",
                "-framerate",
                "15",
                "-window_id",
                str(window_id),
                "-i",
                f"{display}.0",
                "-an",
                "-c:v",
                "libvpx-vp9",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
                "-b:v",
                "1M",
                str(self._output_path),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env=self._external_process_env(),
            )
            await asyncio.sleep(VIDEO_START_SETTLE_SECONDS)
            if self._process.returncode is None:
                return f"Browser video recording started: {self._output_path}"

            last_returncode = self._process.returncode
            _, stderr = await self._process.communicate()
            last_error = self._format_start_error(stderr)
            self._process = None
            self._output_path.unlink(missing_ok=True)
            if attempt + 1 < VIDEO_START_ATTEMPTS:
                await asyncio.sleep(VIDEO_START_SETTLE_SECONDS)

        self._reset()
        detail = f": {last_error}" if last_error else ""
        return (
            f"Error: ffmpeg exited before recording started ({last_returncode}){detail}"
        )

    async def stop(self) -> str:
        if not self.is_recording:
            return "Error: Browser video recording is not active"

        assert self._process is not None
        assert self._output_path is not None
        process = self._process
        output_path = self._output_path
        process.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
        self._reset()

        if not output_path.is_file() or output_path.stat().st_size == 0:
            return "Error: Browser video recording did not produce a file"
        return f"Browser video recording saved: {output_path}"

    @staticmethod
    def _format_start_error(stderr: bytes | None) -> str:
        if not stderr:
            return ""
        detail = " ".join(stderr.decode(errors="replace").split())
        return detail[-VIDEO_START_ERROR_MAX_CHARS:]

    @staticmethod
    def _external_process_env() -> dict[str, str]:
        """Restore the dynamic-library path before launching system binaries.

        PyInstaller prepends its extraction directory to ``LD_LIBRARY_PATH`` so
        the bundled agent server can load its own shared libraries. A system
        ffmpeg inherits that path otherwise and can load an older bundled
        libstdc++ instead of the host version it was linked against.
        """
        environment = os.environ.copy()
        original_library_path = environment.get("LD_LIBRARY_PATH_ORIG")
        if original_library_path is None:
            environment.pop("LD_LIBRARY_PATH", None)
        else:
            environment["LD_LIBRARY_PATH"] = original_library_path
        return environment

    def _reset(self) -> None:
        self._process = None
        self._output_path = None
