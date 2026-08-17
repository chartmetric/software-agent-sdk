"""Browser tool executor implementation using browser-use MCP server wrapper."""

from __future__ import annotations

import builtins
import functools
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeVar


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation

from openhands.sdk.logger import DEBUG, get_logger
from openhands.sdk.tool import ToolExecutor
from openhands.sdk.utils import sanitized_env
from openhands.sdk.utils.async_executor import AsyncExecutor
from openhands.tools.browser_use.definition import (
    BROWSER_RECORDING_OUTPUT_DIR,
    BrowserAction,
    BrowserFillFormAction,
    BrowserFormField,
    BrowserGetSecretAction,
    BrowserObservation,
    BrowserTypeAction,
)
from openhands.tools.browser_use.server import CustomBrowserUseServer
from openhands.tools.browser_use.video_recording import BrowserVideoRecorder
from openhands.tools.utils.timeout import (
    TimeoutError as ToolTimeoutError,
    run_with_timeout,
)


F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])


def recording_aware(
    func: Callable[..., Coroutine[Any, Any, Any]],
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Decorator that handles recording flush before/after navigation operations.

    This decorator:
    1. Flushes recording events before the operation (to preserve them)
    2. Executes the operation
    3. Restarts recording on the new page if recording was active

    Error Handling Policy (see recording.py module docstring for full details):
    - Recording is a secondary feature that should never block browser operations
    - AttributeError: silent pass (recording not initialized - expected)
    - Other exceptions: log at DEBUG, don't interrupt navigation
    """

    @functools.wraps(func)
    async def wrapper(self: BrowserToolExecutor, *args: Any, **kwargs: Any) -> Any:
        is_recording = self._server._is_recording
        if is_recording:
            try:
                await self._server._flush_recording_events()
            except AttributeError:
                # Recording not initialized - expected, silent pass
                pass
            except Exception as e:
                # Internal operation: log at DEBUG, don't interrupt navigation
                logger.debug(f"Recording flush before {func.__name__} skipped: {e}")

        result = await func(self, *args, **kwargs)

        if is_recording:
            try:
                await self._server._restart_recording_on_new_page()
            except AttributeError:
                # Recording not initialized - expected, silent pass
                pass
            except Exception as e:
                # Internal operation: log at DEBUG, don't interrupt navigation
                logger.debug(f"Recording restart after {func.__name__} skipped: {e}")

        return result

    return wrapper


# Suppress browser-use logging for cleaner integration
if DEBUG:
    logging.getLogger("browser_use").setLevel(logging.DEBUG)
else:
    logging.getLogger("browser_use").setLevel(logging.WARNING)

logger = get_logger(__name__)

DEFAULT_BROWSER_ACTION_TIMEOUT_SECONDS: Final[float] = 300.0
# After this many consecutive failures, reset the browser session
# (assumes the browser has crashed or become unrecoverable).
MAX_CONSECUTIVE_FAILURES: Final[int] = 3
# Shorter timeout used after a failure to avoid long cascading waits
# against a dead browser.
DEGRADED_TIMEOUT_SECONDS: Final[float] = 30.0
MASK_SAFE_VIDEO_OUTPUT_ROOT: Final[str] = str(Path(tempfile.gettempdir()) / "ohva")


def _current_platform(platform: str | None = None) -> str:
    return sys.platform if platform is None else platform


def _windows_browser_install_paths() -> list[Path]:
    roots = [
        os.environ.get("PROGRAMFILES", "C:\\Program Files"),
        os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"),
        os.environ.get("LOCALAPPDATA"),
    ]
    browsers = [
        ("Google", "Chrome", "Application", "chrome.exe"),
        ("Microsoft", "Edge", "Application", "msedge.exe"),
        ("Chromium", "Application", "chrome.exe"),
    ]

    paths: list[Path] = []
    for root in roots:
        if root is None:
            continue
        for parts in browsers:
            paths.append(Path(root).joinpath(*parts))
    return paths


def _standard_chromium_paths(platform: str | None = None) -> list[Path]:
    match _current_platform(platform):
        case "darwin":
            return [
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
                Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            ]
        case "win32":
            return _windows_browser_install_paths()
        case _:
            return [
                Path("/usr/bin/google-chrome"),
                Path("/usr/bin/google-chrome-stable"),
                Path("/usr/bin/chromium"),
                Path("/usr/bin/chromium-browser"),
                Path("/usr/bin/microsoft-edge"),
                Path("/usr/bin/microsoft-edge-stable"),
            ]


def _playwright_cache_dirs(platform: str | None = None) -> list[Path]:
    match _current_platform(platform):
        case "darwin":
            return [Path.home() / "Library" / "Caches" / "ms-playwright"]
        case "win32":
            if local_app_data := os.environ.get("LOCALAPPDATA"):
                return [Path(local_app_data) / "ms-playwright"]
            return [Path.home() / "AppData" / "Local" / "ms-playwright"]
        case _:
            return [Path.home() / ".cache" / "ms-playwright"]


def _playwright_chromium_paths(
    chromium_dir: Path,
    platform: str | None = None,
) -> list[Path]:
    match _current_platform(platform):
        case "darwin":
            return [
                chromium_dir
                / "chrome-mac-arm64"
                / "Google Chrome for Testing.app"
                / "Contents"
                / "MacOS"
                / "Google Chrome for Testing",
                chromium_dir
                / "chrome-mac"
                / "Google Chrome for Testing.app"
                / "Contents"
                / "MacOS"
                / "Google Chrome for Testing",
                chromium_dir
                / "chrome-mac"
                / "Chromium.app"
                / "Contents"
                / "MacOS"
                / "Chromium",
            ]
        case "win32":
            return [
                chromium_dir / "chrome-win64" / "chrome.exe",
                chromium_dir / "chrome-win" / "chrome.exe",
            ]
        case _:
            return [
                chromium_dir / "chrome-linux64" / "chrome",
                chromium_dir / "chrome-linux" / "chrome",
            ]


def _path_binary_candidates(platform: str | None = None) -> tuple[str, ...]:
    if _current_platform(platform) == "win32":
        return ("chrome", "msedge", "chromium")
    return (
        "google-chrome",
        "chrome",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
    )


def _format_browser_operation_error(
    error: BaseException, timeout_seconds: float | None = None
) -> str:
    if error_detail := str(error).strip():
        pass
    elif isinstance(error, builtins.TimeoutError):
        error_detail = (
            f"Operation timed out after {int(timeout_seconds)} seconds"
            if timeout_seconds is not None
            else "Operation timed out"
        )
    else:
        error_detail = error.__class__.__name__
    return f"Browser operation failed: {error_detail}"


def _install_chromium() -> bool:
    """Attempt to install Chromium via uvx playwright install."""
    try:
        # Check if uvx is available
        if not shutil.which("uvx"):
            logger.warning("uvx not found - cannot auto-install Chromium")
            return False

        logger.info("Attempting to install Chromium via uvx...")
        result = subprocess.run(
            ["uvx", "playwright", "install", "chromium", "--with-deps", "--no-shell"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout for installation
            env=sanitized_env(),
        )

        if result.returncode == 0:
            logger.info("Chromium installation completed successfully")
            return True
        else:
            logger.error(f"Chromium installation failed: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.error(f"Error during Chromium installation: {e}")
        return False


def _get_chromium_error_message() -> str:
    """Get the error message for when Chromium is not available."""
    return (
        "Chromium is required for browser operations but is not installed.\n\n"
        "To install Chromium, run one of the following commands:\n"
        "  1. Using uvx (recommended): uvx playwright install chromium "
        "--with-deps --no-shell\n"
        "  2. Using pip: pip install playwright && playwright install chromium\n"
        "  3. Using system package manager:\n"
        "     - Ubuntu/Debian: sudo apt install chromium-browser\n"
        "     - macOS: brew install chromium\n"
        "     - Windows: winget install Chromium.Chromium\n\n"
        "After installation, restart your application to use the browser tool."
    )


class BrowserToolExecutor(ToolExecutor[BrowserAction, BrowserObservation]):
    """Executor that wraps browser-use MCP server for OpenHands integration."""

    _server: CustomBrowserUseServer
    _config: dict[str, Any]
    _initialized: bool
    _async_executor: AsyncExecutor
    _cleanup_initiated: bool
    _close_lock: threading.Lock
    _action_timeout_seconds: float
    _control_owner: str

    @staticmethod
    @functools.cache
    def check_chromium_available() -> str | None:
        """Check if a Chromium/Chrome binary is available.

        Returns:
            Path to Chromium binary if found, None otherwise
        """
        # Check standard installation paths (prefer full Chrome installs)
        for path in _standard_chromium_paths():
            if path.exists():
                return str(path)

        # Check Playwright-installed Chromium (preferred over PATH lookups
        # because PATH binaries like homebrew chromium may lack CDP support)
        for playwright_cache in _playwright_cache_dirs():
            if playwright_cache.exists():
                chromium_dirs = list(playwright_cache.glob("chromium-*"))
                for chromium_dir in chromium_dirs:
                    for path in _playwright_chromium_paths(chromium_dir):
                        if path.exists():
                            return str(path)

        # Fallback: check PATH for any chromium-based binary
        for binary in _path_binary_candidates():
            if path := shutil.which(binary):
                return path

        return None

    def _ensure_chromium_available(self) -> str:
        """Ensure Chromium is available for browser operations.

        Raises:
            Exception: If Chromium is not available
        """
        if path := self.check_chromium_available():
            logger.info(f"Chromium is available for browser operations at {path}")
            return path

        # Chromium not available - provide clear installation instructions
        raise Exception(_get_chromium_error_message())

    def __init__(
        self,
        headless: bool = True,
        allowed_domains: list[str] | None = None,
        session_timeout_minutes: int = 30,
        init_timeout_seconds: int = 30,
        action_timeout_seconds: float = DEFAULT_BROWSER_ACTION_TIMEOUT_SECONDS,
        full_output_save_dir: str | None = None,
        inject_scripts: list[str] | None = None,
        **config,
    ):
        """Initialize BrowserToolExecutor with timeout protection.

        Args:
            headless: Whether to run browser in headless mode
            allowed_domains: List of allowed domains for browser operations
            session_timeout_minutes: Browser session timeout in minutes
            init_timeout_seconds: Timeout for browser initialization in seconds
            action_timeout_seconds: Timeout for each browser action in seconds
            full_output_save_dir: Absolute path to directory to save full output
                logs and files, used when truncation is needed.
            inject_scripts: List of JavaScript code strings to inject into every
                new document. Scripts are injected via CDP's
                Page.addScriptToEvaluateOnNewDocument and run before page scripts.
                Useful for injecting recording tools like rrweb.
            **config: Additional configuration options
        """

        self._close_lock = threading.Lock()

        def init_logic():
            nonlocal headless
            executable_path = self._ensure_chromium_available()
            self._server = CustomBrowserUseServer(
                session_timeout_minutes=session_timeout_minutes,
            )
            if os.getenv("OH_ENABLE_VNC", "false").lower() in {"true", "1", "yes"}:
                headless = False  # Force headless off if VNC is enabled
                window_width, window_height = (
                    int(dimension)
                    for dimension in os.getenv("VNC_GEOMETRY", "1280x800").split(
                        "x", maxsplit=1
                    )
                )
                config["window_size"] = {
                    "width": window_width,
                    "height": window_height,
                }
                logger.info("VNC is enabled - running browser in non-headless mode")

            # Configure scripts to inject
            if inject_scripts:
                self._server.set_inject_scripts(inject_scripts)

            # Chromium refuses to run as root with sandboxing enabled.
            # Disable the sandbox when running as root so CHROME_DOCKER_ARGS
            # (--no-sandbox, --disable-setuid-sandbox, etc.) are applied.
            # SECURITY: Running Chrome as root without a sandbox is risky
            # - a compromised browser has full root access. Use only in
            # controlled environments.
            getuid = getattr(os, "getuid", None)
            running_as_root = getuid is not None and getuid() == 0
            if running_as_root:
                logger.warning(
                    "Running as root - disabling Chromium sandbox "
                    "(required for root). This reduces security isolation."
                )

            self._config = {
                "headless": headless,
                "allowed_domains": allowed_domains or [],
                "executable_path": executable_path,
                "chromium_sandbox": not running_as_root,
                **config,
            }

        try:
            run_with_timeout(init_logic, init_timeout_seconds)
        except ToolTimeoutError:
            raise Exception(
                f"Browser tool initialization timed out after {init_timeout_seconds}s"
            )

        if action_timeout_seconds <= 0:
            raise ValueError("action_timeout_seconds must be greater than 0")

        self.full_output_save_dir: str | None = full_output_save_dir
        self._initialized = False
        self._async_executor = AsyncExecutor()
        self._cleanup_initiated = False
        self._action_timeout_seconds = action_timeout_seconds
        self._consecutive_failures = 0
        # Screencast human-takeover lock. Lives on this shared, process-wide
        # executor (not on ScreencastService) so it also blocks subagents,
        # which reuse the same executor instance via
        # BrowserToolSet.get_or_create_shared_executor().
        self._control_owner = "agent"
        # Keep the path returned to the model independent of repository and
        # workspace names. Those names can legitimately equal a configured
        # secret value, which would redact the path and make publication fail.
        self._video_recorder = BrowserVideoRecorder(MASK_SAFE_VIDEO_OUTPUT_ROOT)

    def __call__(
        self,
        action: BrowserAction,
        conversation: LocalConversation | None = None,
    ):
        """Submit an action to run in the background loop and wait for result."""
        # Use a shorter timeout on the last retry before a reset would trigger,
        # to avoid long cascading waits against a dead browser.
        effective_timeout = (
            DEGRADED_TIMEOUT_SECONDS
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES - 1
            else self._action_timeout_seconds
        )

        secret_text: str | None = None
        runtime_secret_values: dict[int, str] = {}
        mask_text: Callable[[str], str] | None = None
        if conversation is not None:
            secret_registry = conversation.state.secret_registry
            mask_text = secret_registry.mask_secrets_in_output
            if isinstance(action, BrowserGetSecretAction):
                secret_text = secret_registry.get_secret_value(
                    action.secret_name, action.json_field
                )
                if secret_text is None:
                    return BrowserObservation.from_text(
                        text="Unable to resolve the registered browser secret",
                        is_error=True,
                        full_output_save_dir=self.full_output_save_dir,
                    )
                return BrowserObservation.from_secret(
                    secret_text,
                    full_output_save_dir=self.full_output_save_dir,
                )
            if isinstance(action, BrowserTypeAction):
                if action.secret_name is not None:
                    secret_text = secret_registry.get_secret_value(
                        action.secret_name, action.json_field
                    )
                    if secret_text is None:
                        return BrowserObservation.from_text(
                            text="Unable to resolve the registered browser secret",
                            is_error=True,
                            full_output_save_dir=self.full_output_save_dir,
                        )
                elif action.text is not None:
                    masked_text = mask_text(action.text)
                    if masked_text != action.text:
                        secret_text = action.text
            if isinstance(action, BrowserFillFormAction):
                for position, field in enumerate(action.fields):
                    if field.secret_name is not None:
                        value = secret_registry.get_secret_value(
                            field.secret_name, field.json_field
                        )
                        if value is None:
                            return BrowserObservation.from_text(
                                text="Unable to resolve a registered browser secret",
                                is_error=True,
                                full_output_save_dir=self.full_output_save_dir,
                            )
                        runtime_secret_values[position] = value
                    elif field.text is not None:
                        masked_text = mask_text(field.text)
                        if masked_text != field.text:
                            runtime_secret_values[position] = field.text
                if action.include_screenshot and runtime_secret_values:
                    return BrowserObservation.from_text(
                        text=(
                            "A form containing registered secret values cannot include "
                            "a screenshot. Submit without one, verify that the "
                            "credential fields are no longer visible, then use a "
                            "separate "
                            "browser_get_state call if visual evidence is required."
                        ),
                        is_error=True,
                        full_output_save_dir=self.full_output_save_dir,
                    )
        elif isinstance(action, (BrowserGetSecretAction, BrowserTypeAction)) and (
            isinstance(action, BrowserGetSecretAction) or action.secret_name is not None
        ):
            return BrowserObservation.from_text(
                text="Unable to resolve the registered browser secret",
                is_error=True,
                full_output_save_dir=self.full_output_save_dir,
            )

        try:
            if mask_text is None:
                result = self._async_executor.run_async(
                    self._execute_action,
                    action,
                    timeout=effective_timeout,
                )
            else:
                result = self._async_executor.run_async(
                    self._execute_action,
                    action,
                    secret_text,
                    mask_text,
                    runtime_secret_values,
                    timeout=effective_timeout,
                )
        except builtins.TimeoutError as error:
            # Timeouts indicate the browser may be dead/hung — track them
            # for crash detection. Regular action errors (invalid selector,
            # missing element) are NOT counted since those are normal agent
            # mistakes, not browser crashes.
            error_text = _format_browser_operation_error(
                error, timeout_seconds=effective_timeout
            )
            if mask_text is not None:
                error_text = mask_text(error_text)
            return self._handle_timeout_failure(error_text)

        self._consecutive_failures = 0
        return self._mask_observation(result, mask_text)

    def _mask_observation(
        self,
        observation: BrowserObservation,
        mask_text: Callable[[str], str] | None,
    ) -> BrowserObservation:
        if mask_text is None or not observation.text:
            return observation
        masked_text = mask_text(observation.text)
        if masked_text == observation.text:
            return observation
        data = observation.model_dump(exclude={"content", "full_output_save_dir"})
        return BrowserObservation.from_text(
            text=masked_text,
            full_output_save_dir=self.full_output_save_dir,
            **data,
        )

    def _handle_timeout_failure(self, error_text: str) -> BrowserObservation:
        """Track consecutive timeout failures and reset session if needed."""
        self._consecutive_failures += 1
        logger.debug(
            "Browser timeout failure %d/%d",
            self._consecutive_failures,
            MAX_CONSECUTIVE_FAILURES,
        )

        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "Browser appears crashed (%d consecutive failures). "
                "Resetting session for automatic recovery.",
                self._consecutive_failures,
            )
            # Best-effort cleanup of the old browser process/session.
            # If the browser truly crashed this will fail fast; if it's
            # wedged this avoids leaking the process.
            try:
                self._async_executor.run_async(self.cleanup, timeout=5.0)
            except Exception as e:
                logger.debug(
                    "Cleanup during session reset failed "
                    "(expected if browser crashed): %s",
                    e,
                )
            self._initialized = False
            self._consecutive_failures = 0
            error_text = (
                f"{error_text}\n\n"
                "The browser session has been reset after multiple consecutive "
                "failures (possible crash). The browser will be restarted on "
                "the next action. Please retry your action."
            )

        return BrowserObservation.from_text(
            text=error_text,
            is_error=True,
            full_output_save_dir=self.full_output_save_dir,
        )

    async def _execute_action(
        self,
        action,
        secret_text: str | None = None,
        mask_text: Callable[[str], str] | None = None,
        runtime_secret_values: dict[int, str] | None = None,
    ):
        """Execute browser action asynchronously."""
        from openhands.tools.browser_use.definition import (
            BrowserClickAction,
            BrowserCloseTabAction,
            BrowserFillFormAction,
            BrowserGetContentAction,
            BrowserGetStateAction,
            BrowserGetStorageAction,
            BrowserGoBackAction,
            BrowserListTabsAction,
            BrowserNavigateAction,
            BrowserObservation,
            BrowserScrollAction,
            BrowserSetStorageAction,
            BrowserStartRecordingAction,
            BrowserStartVideoRecordingAction,
            BrowserStopRecordingAction,
            BrowserStopVideoRecordingAction,
            BrowserSwitchTabAction,
        )

        if self._control_owner == "human":
            return BrowserObservation.from_text(
                text=(
                    "Browser control is currently held by a human user via the "
                    "live screencast view. Wait for them to return control "
                    "before using the browser tool again."
                ),
                is_error=True,
                full_output_save_dir=self.full_output_save_dir,
            )

        try:
            result = ""
            # Route to appropriate method based on action type
            if isinstance(action, BrowserNavigateAction):
                result = await self.navigate(action.url, action.new_tab)
            elif isinstance(action, BrowserClickAction):
                result = await self.click(action.index, action.new_tab)
            elif isinstance(action, BrowserTypeAction):
                if secret_text is not None:
                    result = await self.type_secret_text(action.index, secret_text)
                else:
                    if action.text is None:
                        raise ValueError("Browser text was not provided")
                    result = await self.type_text(action.index, action.text)
            elif isinstance(action, BrowserFillFormAction):
                return await self.fill_form(
                    action.fields,
                    action.submit_index,
                    action.include_screenshot,
                    runtime_secret_values or {},
                )
            elif isinstance(action, BrowserGetStateAction):
                return await self.get_state(action.include_screenshot)
            elif isinstance(action, BrowserGetStorageAction):
                result = await self.get_storage()
            elif isinstance(action, BrowserSetStorageAction):
                result = await self.set_storage(action.storage_state)
            elif isinstance(action, BrowserGetContentAction):
                result = await self.get_content(
                    action.extract_links, action.start_from_char
                )
            elif isinstance(action, BrowserScrollAction):
                result = await self.scroll(action.direction)
            elif isinstance(action, BrowserGoBackAction):
                result = await self.go_back()
            elif isinstance(action, BrowserListTabsAction):
                result = await self.list_tabs()
            elif isinstance(action, BrowserSwitchTabAction):
                result = await self.switch_tab(action.tab_id)
            elif isinstance(action, BrowserCloseTabAction):
                result = await self.close_tab(action.tab_id)
            elif isinstance(action, BrowserStartRecordingAction):
                result = await self.start_recording()
            elif isinstance(action, BrowserStopRecordingAction):
                result = await self.stop_recording()
            elif isinstance(action, BrowserStartVideoRecordingAction):
                result = await self.start_video_recording()
            elif isinstance(action, BrowserStopVideoRecordingAction):
                result = await self.stop_video_recording()
            else:
                error_msg = f"Unsupported action type: {type(action)}"
                return BrowserObservation.from_text(
                    text=error_msg,
                    is_error=True,
                    full_output_save_dir=self.full_output_save_dir,
                )

            return BrowserObservation.from_text(
                text=result,
                is_error=False,
                full_output_save_dir=self.full_output_save_dir,
            )
        except Exception as error:
            error_msg = _format_browser_operation_error(error)
            if mask_text is not None:
                error_msg = mask_text(error_msg)
                logging.error(error_msg)
            else:
                logging.error(error_msg, exc_info=True)
            return BrowserObservation.from_text(
                text=error_msg,
                is_error=True,
                full_output_save_dir=self.full_output_save_dir,
            )

    async def _ensure_initialized(self):
        """Ensure browser session is initialized."""
        if not self._initialized:
            # Initialize browser session with our config
            await self._server._init_browser_session(**self._config)
            # Inject any configured user scripts after session is ready
            # Note: rrweb scripts are injected lazily when recording starts
            await self._server._inject_scripts_to_session()
            self._initialized = True

    async def warm_up(self) -> None:
        """Launch and tear down a browser session once to warm the OS caches.

        The first real browser action otherwise pays the cost of launching
        Chromium cold -- reading the binary and its shared libraries off disk,
        loading fonts. Doing it once ahead of time (e.g. from the tool-preload
        service at server startup) leaves those pages resident in the OS cache,
        so the first conversation's launch is fast. The session is torn down
        immediately afterwards, so this holds no browser process open.

        Best-effort: any failure is swallowed, leaving the browser to launch
        lazily on first use exactly as before.
        """
        try:
            await self._ensure_initialized()
        except Exception:
            logger.debug("Browser warm-up initialization failed", exc_info=True)
        finally:
            try:
                await self.cleanup()
            except Exception:
                logger.debug("Browser warm-up cleanup failed", exc_info=True)

    # Navigation & Browser Control Methods
    @recording_aware
    async def navigate(self, url: str, new_tab: bool = False) -> str:
        """Navigate to a URL."""
        await self._ensure_initialized()
        return await self._server._navigate(url, new_tab)

    @recording_aware
    async def go_back(self) -> str:
        """Go back in browser history."""
        await self._ensure_initialized()
        return await self._server._go_back()

    # Page Interaction
    @recording_aware
    async def click(self, index: int, new_tab: bool = False) -> str:
        """Click an element by index."""
        await self._ensure_initialized()
        return await self._server._click(index, new_tab)

    async def type_text(self, index: int, text: str) -> str:
        """Type text into an element."""
        await self._ensure_initialized()
        return await self._server._type_text(index, text)

    async def type_secret_text(self, index: int, text: str) -> str:
        await self._ensure_initialized()
        return await self._server._type_secret_text(index, text)

    async def fill_form(
        self,
        fields: list[BrowserFormField],
        submit_index: int | None,
        include_screenshot: bool,
        runtime_secret_values: dict[int, str],
    ) -> BrowserObservation:
        """Fill one stable form and return only its final browser state."""
        for position, field in enumerate(fields):
            value = runtime_secret_values.get(position)
            if value is not None:
                await self.type_secret_text(field.index, value)
            elif field.text is not None:
                await self.type_text(field.index, field.text)
            else:
                raise ValueError(f"Form field {position} has no input value")
        if submit_index is not None:
            await self.click(submit_index)
        return await self.get_state(include_screenshot)

    async def scroll(self, direction: str = "down") -> str:
        """Scroll the page."""
        await self._ensure_initialized()
        return await self._server._scroll(direction)

    async def get_state(self, include_screenshot: bool = False):
        """Get current browser state with interactive elements."""
        from openhands.tools.browser_use.definition import BrowserObservation

        await self._ensure_initialized()
        result_json = await self._server._get_browser_state(include_screenshot)

        if include_screenshot:
            try:
                result_data = json.loads(result_json)
                screenshot_data = result_data.pop("screenshot", None)

                # Return clean JSON + separate screenshot data
                clean_json = json.dumps(result_data, indent=2)
                return BrowserObservation.from_text(
                    text=clean_json,
                    is_error=False,
                    screenshot_data=screenshot_data,
                    full_output_save_dir=self.full_output_save_dir,
                )
            except json.JSONDecodeError:
                # If JSON parsing fails, return as-is
                pass

        return BrowserObservation.from_text(
            text=result_json,
            is_error=False,
            full_output_save_dir=self.full_output_save_dir,
        )

    async def get_storage(self) -> str:
        """Get browser storage (cookies, local storage, session storage)."""
        await self._ensure_initialized()
        return await self._server._get_storage()

    async def set_storage(self, storage_state: dict) -> str:
        """Set browser storage (cookies, local storage, session storage)."""
        await self._ensure_initialized()
        return await self._server._set_storage(storage_state)

    # Tab Management
    async def list_tabs(self) -> str:
        """List all open tabs."""
        await self._ensure_initialized()
        return await self._server._list_tabs()

    async def switch_tab(self, tab_id: str) -> str:
        """Switch to a different tab."""
        await self._ensure_initialized()
        return await self._server._switch_tab(tab_id)

    async def close_tab(self, tab_id: str) -> str:
        """Close a specific tab."""
        await self._ensure_initialized()
        return await self._server._close_tab(tab_id)

    # Content Extraction
    async def get_content(self, extract_links: bool, start_from_char: int) -> str:
        """Extract page content, optionally with links."""
        await self._ensure_initialized()
        return await self._server._get_content(
            extract_links=extract_links, start_from_char=start_from_char
        )

    # Session Recording
    async def start_recording(self) -> str:
        """Start recording the browser session using rrweb.

        Recording events are periodically flushed to timestamped JSON files
        in a session subfolder under BROWSER_RECORDING_OUTPUT_DIR.
        Events are flushed every 5 seconds.
        """
        await self._ensure_initialized()
        return await self._server._start_recording(
            output_dir=BROWSER_RECORDING_OUTPUT_DIR
        )

    async def stop_recording(self) -> str:
        """Stop recording and save remaining events to file.

        Stops the periodic flush, collects any remaining events, and saves
        them to a final numbered JSON file. Returns a summary message with
        the total events and file count.
        """
        await self._ensure_initialized()
        return await self._server._stop_recording()

    async def start_video_recording(self) -> str:
        """Start recording the visible browser desktop to WebM."""
        await self._ensure_initialized()
        return await self._video_recorder.start()

    async def stop_video_recording(self) -> str:
        """Stop the visible browser recording and finalize the WebM file."""
        return await self._video_recorder.stop()

    # Live CDP Screencast
    #
    # Unlike the methods above, these are plain *synchronous* entry points,
    # not agent-invocable BrowserActions: nothing about "someone is watching
    # a live view" belongs in the LLM's tool surface. They mirror the
    # sync-to-async bridge `__call__` itself uses (`_async_executor.run_async`)
    # so infrastructure code on another thread/loop (e.g. the agent-server's
    # screencast WebSocket service) can drive them the same way
    # `DesktopService.navigate()` drives `BrowserNavigateAction` — via
    # `asyncio.to_thread(executor.start_screencast, on_frame)`.
    def start_screencast(
        self, on_frame: Callable[[str, dict[str, Any]], None], **kwargs: Any
    ) -> bool:
        """Start CDP screencast streaming on the shared browser session.

        `on_frame` is invoked on the browser's dedicated background thread —
        callers that need to reach another event loop must hop via
        `asyncio.run_coroutine_threadsafe` themselves.
        """

        async def _do_start() -> bool:
            await self._ensure_initialized()
            return await self._server._start_screencast(on_frame, **kwargs)

        return self._async_executor.run_async(_do_start)

    def stop_screencast(self) -> bool:
        """Stop CDP screencast streaming, if active."""
        if not self._initialized:
            return True
        return self._async_executor.run_async(self._server._stop_screencast)

    def dispatch_screencast_input(self, kind: str, payload: dict[str, Any]) -> None:
        """Dispatch a mouse/key event for human screencast takeover.

        `kind` is "mouse" or "key"; anything else is ignored. No-op if the
        browser session hasn't been initialized yet. Same threading contract
        as start_screencast/stop_screencast.
        """
        if not self._initialized:
            return

        async def _do_dispatch() -> None:
            if kind == "mouse":
                await self._server._dispatch_screencast_mouse(**payload)
            elif kind == "key":
                await self._server._dispatch_screencast_key(**payload)

        self._async_executor.run_async(_do_dispatch)

    def set_control_owner(self, owner: str) -> None:
        """Set who currently controls the browser: "agent" or "human".

        Called by the agent-server's screencast take_control/release_control
        handling (via ScreencastService). This lock lives on the shared,
        process-wide executor rather than on ScreencastService itself so it
        also blocks subagents, which reuse this same executor instance via
        BrowserToolSet.get_or_create_shared_executor().
        """
        self._control_owner = owner

    def is_human_controlling(self) -> bool:
        """Whether a human currently holds the screencast takeover lock."""
        return self._control_owner == "human"

    async def close_browser(self) -> str:
        """Close the browser session."""
        if self._initialized:
            result = await self._server._close_browser()
            self._initialized = False
            return result
        return "No browser session to close"

    async def cleanup(self):
        """Cleanup browser resources."""
        try:
            if self._video_recorder.is_recording:
                await self._video_recorder.stop()
            # Use _close_all_sessions instead of close_browser because it calls
            # session.kill() which properly stops the event bus and drains
            # pending events (including BrowserKillEvent that terminates the
            # Chromium subprocess). close_browser() alone dispatches
            # BrowserKillEvent fire-and-forget and returns before it's processed,
            # which can leave the browser process alive.
            if hasattr(self._server, "_close_all_sessions"):
                await self._server._close_all_sessions()
            else:
                await self.close_browser()
        except Exception as e:
            logger.warning(f"Error during browser cleanup: {e}")

    def close(self):
        """Close the browser executor and cleanup resources."""
        with self._close_lock:
            shared_close_lock_acquired = self._detach_shared_executor_for_close()
            if self._cleanup_initiated:
                if shared_close_lock_acquired:
                    self._release_shared_executor_creation_lock()
                return
            self._cleanup_initiated = True
            try:
                # Run cleanup in the async executor with a shorter timeout
                self._async_executor.run_async(self.cleanup, timeout=30.0)
            except Exception as e:
                logger.warning(f"Error during browser cleanup: {e}")
            finally:
                try:
                    # Always close the async executor
                    self._async_executor.close()
                finally:
                    if shared_close_lock_acquired:
                        self._release_shared_executor_creation_lock()
                    else:
                        self._release_shared_executor_reference()

    def _detach_shared_executor_for_close(self) -> bool:
        from openhands.tools.browser_use.definition import BrowserToolSet

        if BrowserToolSet._shared_executor is not self:
            return False

        BrowserToolSet._shared_executor_creation_lock.acquire()
        with BrowserToolSet._shared_executor_lock:
            if BrowserToolSet._shared_executor is self:
                BrowserToolSet._shared_executor = None
                return True

        BrowserToolSet._shared_executor_creation_lock.release()
        return False

    @staticmethod
    def _release_shared_executor_creation_lock() -> None:
        from openhands.tools.browser_use.definition import BrowserToolSet

        BrowserToolSet._shared_executor_creation_lock.release()

    def _release_shared_executor_reference(self):
        # Avoid taking the shared executor lock for ordinary/stale executors.
        # __del__ can run while BrowserToolSet.create() is creating a new shared
        # executor; a stale executor finalizer trying to acquire the same lock can
        # deadlock that create path, especially on Windows.
        from openhands.tools.browser_use.definition import BrowserToolSet

        if BrowserToolSet._shared_executor is not self:
            return
        with BrowserToolSet._shared_executor_lock:
            if BrowserToolSet._shared_executor is self:
                BrowserToolSet._shared_executor = None

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass  # Ignore cleanup errors during deletion
