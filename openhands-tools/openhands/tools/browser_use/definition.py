"""Browser-use tool implementation for web automation."""

import base64
import hashlib
import logging
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Self

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from rich.text import Text

from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)
from openhands.sdk.utils import DEFAULT_TEXT_CONTENT_LIMIT, maybe_truncate


_logger = logging.getLogger(__name__)

# Lazy import to avoid hanging during module import
if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState
    from openhands.tools.browser_use.impl import BrowserToolExecutor


# Directory where browser session recordings are saved
BROWSER_RECORDING_OUTPUT_DIR = os.path.join(".agent_tmp", "browser_observations")

# Mapping of base64 prefixes to MIME types for image detection
BASE64_IMAGE_PREFIXES = {
    "/9j/": "image/jpeg",
    "iVBORw0KGgo": "image/png",
    "R0lGODlh": "image/gif",
    "UklGR": "image/webp",
}


def detect_image_mime_type(base64_data: str) -> str:
    """Detect MIME type from base64-encoded image data.

    Args:
        base64_data: Base64-encoded image data

    Returns:
        Detected MIME type, defaults to "image/png" if not detected
    """
    for prefix, mime_type in BASE64_IMAGE_PREFIXES.items():
        if base64_data.startswith(prefix):
            return mime_type
    return "image/png"


class BrowserObservation(Observation):
    """Base observation for browser operations."""

    _runtime_secret_value: str | None = PrivateAttr(default=None)

    screenshot_data: str | None = Field(
        default=None, description="Base64 screenshot data if available"
    )
    full_output_save_dir: str | None = Field(
        default=None,
        description="Directory where full output files are saved",
    )

    @classmethod
    def from_secret(cls, value: str, **kwargs) -> Self:
        observation = cls.from_text(text="<secret-hidden>", **kwargs)
        observation._runtime_secret_value = value
        return observation

    @property
    def visualize(self) -> Text:
        if self._runtime_secret_value is not None:
            return Text(self.text)
        return super().visualize

    def _save_screenshot(self, base64_data: str, save_dir: str) -> str | None:
        try:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)

            mime_type = detect_image_mime_type(base64_data)
            ext = mime_type.split("/")[-1]
            if ext == "jpeg":
                ext = "jpg"

            # Generate hash for filename
            content_hash = hashlib.sha256(base64_data.encode("utf-8")).hexdigest()[:8]
            filename = f"browser_screenshot_{content_hash}.{ext}"
            file_path = save_dir_path / filename

            if not file_path.exists():
                image_data = base64.b64decode(base64_data)
                file_path.write_bytes(image_data)

            return str(file_path)
        except Exception:
            return None

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        llm_content: list[TextContent | ImageContent] = []

        # If is_error is true, prepend error message
        if self.is_error:
            llm_content.append(TextContent(text=self.ERROR_MESSAGE_HEADER))

        # Get text content and truncate if needed
        content_text = self._runtime_secret_value or self.text
        if content_text:
            llm_content.append(
                TextContent(
                    text=maybe_truncate(
                        content=content_text,
                        truncate_after=DEFAULT_TEXT_CONTENT_LIMIT,
                        save_dir=self.full_output_save_dir,
                        tool_prefix="browser",
                    )
                )
            )

        if self.screenshot_data:
            mime_type = detect_image_mime_type(self.screenshot_data)

            # Save screenshot if directory is available
            if self.full_output_save_dir:
                saved_path = self._save_screenshot(
                    self.screenshot_data, self.full_output_save_dir
                )
                if saved_path:
                    llm_content.append(
                        TextContent(text=f"Screenshot saved to: {saved_path}")
                    )

            # Convert base64 to data URL format for ImageContent
            data_url = f"data:{mime_type};base64,{self.screenshot_data}"
            llm_content.append(ImageContent(image_urls=[data_url]))

        return llm_content


# ============================================
# Base Browser Action
# ============================================
class BrowserAction(Action):
    """Base class for all browser actions.

    This base class serves as the parent for all browser-related actions,
    enabling proper type hierarchy and eliminating the need for union types.
    """

    pass


# Every browser tool drives the same one browser, so they all lock this.
#
# Without it each tool fell back to the executor's per-tool mutex, keyed on the
# tool's own name -- which serializes two `browser_get_state` calls and lets
# `browser_navigate` run *concurrently* with `browser_get_state`, on one session.
# The failure mode is a capture of the wrong page, and captures are what feed
# published evidence. Measured over 7 days: 4 steps in 15,610 emitted two
# different browser tools in one step -- rare, and not a risk worth carrying,
# because a wrong screenshot does not announce itself.
#
# Declaring it is also what lets an execution agent raise
# `tool_concurrency_limit` above 1 at all: the ceiling was held at 1 precisely
# because this key did not exist.
BROWSER_SESSION_RESOURCE = "browser:session"


class _SharesOneBrowserSession:
    """Mixin declaring that a tool drives the single shared browser session.

    Mixed into every browser tool definition rather than set per tool, so a tool
    added later cannot quietly opt out of the lock by forgetting to override.
    """

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=(BROWSER_SESSION_RESOURCE,), declared=True)


# ============================================
# `go_to_url`
# ============================================
class BrowserNavigateAction(BrowserAction):
    """Schema for browser navigation."""

    url: str = Field(description="The URL to navigate to")
    new_tab: bool = Field(
        default=False, description="Whether to open in a new tab. Default: False"
    )


BROWSER_NAVIGATE_DESCRIPTION = """Navigate to a URL in the browser.

This tool allows you to navigate to any web page. You can optionally open the URL in a new tab.

Parameters:
- url: The URL to navigate to (required)
- new_tab: Whether to open in a new tab (optional, default: False)

Examples:
- Navigate to Google: url="https://www.google.com"
- Open GitHub in new tab: url="https://github.com", new_tab=True

Returns the resulting page state, so a following browser_get_state call is
redundant: read the state in this tool's own result.
"""  # noqa: E501


class BrowserNavigateTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserNavigateAction, BrowserObservation],
):
    """Tool for browser navigation."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_NAVIGATE_DESCRIPTION,
                action_type=BrowserNavigateAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_navigate",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_click`
# ============================================
class BrowserClickAction(BrowserAction):
    """Schema for clicking elements."""

    index: int = Field(
        ge=0, description="The index of the element to click (from browser_get_state)"
    )
    new_tab: bool = Field(
        default=False,
        description="Whether to open any resulting navigation in a new tab. Default: False",  # noqa: E501
    )


BROWSER_CLICK_DESCRIPTION = """Click an element on the page by its index.

Use this tool to click on interactive elements like buttons, links, or form controls. 
The index comes from the browser_get_state tool output.

Returns the resulting page state, so a following browser_get_state call is
redundant: read the state in this tool's own result and pick the next index from
it.

Parameters:
- index: The index of the element to click (from browser_get_state)
- new_tab: Whether to open any resulting navigation in a new tab (optional)

Important: Only use indices that appear in the most recent browser state you were
given, whether that came from browser_get_state or from the previous action.
"""  # noqa: E501


class BrowserClickTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserClickAction, BrowserObservation],
):
    """Tool for clicking browser elements."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_CLICK_DESCRIPTION,
                action_type=BrowserClickAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_click",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_type`
# ============================================
class BrowserTypeAction(BrowserAction):
    """Schema for typing text into elements."""

    index: int = Field(
        ge=0, description="The index of the input element (from browser_get_state)"
    )
    text: str | None = Field(default=None, description="Literal text to type")
    secret_name: str | None = Field(
        default=None,
        min_length=1,
        description="Registered secret name whose value should be typed",
    )
    json_field: str | None = Field(
        default=None,
        min_length=1,
        description="Optional top-level JSON field to type from the registered secret",
    )

    @model_validator(mode="after")
    def validate_input_source(self):
        if (self.text is None) == (self.secret_name is None):
            raise ValueError("Provide exactly one of text or secret_name")
        if self.json_field is not None and self.secret_name is None:
            raise ValueError("json_field requires secret_name")
        return self


BROWSER_TYPE_DESCRIPTION = """Type text into an input field.

Use this tool to enter text into form fields, search boxes, or other text input elements.
The index comes from the browser_get_state tool output. For credentials, either use a
value returned by browser_get_secret or reference a registered secret directly.

Parameters:
- index: The index of the input element (from browser_get_state)
- text: Literal text to type, including a value returned by browser_get_secret
- secret_name: Registered secret name whose value should be typed
- json_field: Optional top-level JSON string field to type from the registered secret

Important: Only use indices that appear in your current browser_get_state output.

Returns the resulting page state, so a following browser_get_state call is
redundant: read the state in this tool's own result.
"""  # noqa: E501


class BrowserTypeTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserTypeAction, BrowserObservation],
):
    """Tool for typing text into browser elements."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_TYPE_DESCRIPTION,
                action_type=BrowserTypeAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_type",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_fill_form`
# ============================================
class BrowserFormField(BaseModel):
    """One input field filled as part of a browser form transaction."""

    index: int = Field(
        ge=0, description="The index of the input element (from browser_get_state)"
    )
    text: str | None = Field(default=None, description="Literal text to type")
    secret_name: str | None = Field(
        default=None,
        min_length=1,
        description="Registered secret name whose value should be typed",
    )
    json_field: str | None = Field(
        default=None,
        min_length=1,
        description="Optional top-level JSON field to type from the registered secret",
    )

    @model_validator(mode="after")
    def validate_input_source(self):
        if (self.text is None) == (self.secret_name is None):
            raise ValueError("Provide exactly one of text or secret_name")
        if self.json_field is not None and self.secret_name is None:
            raise ValueError("json_field requires secret_name")
        return self


class BrowserFillFormAction(BrowserAction):
    """Fill multiple current-page inputs and optionally submit the form."""

    fields: list[BrowserFormField] = Field(min_length=1, max_length=20)
    submit_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional button index to click after all fields are filled",
    )
    include_screenshot: bool = Field(
        default=False,
        description="Whether the final page state should include a screenshot",
    )

    @model_validator(mode="after")
    def validate_indices(self):
        indices = [field.index for field in self.fields]
        if len(indices) != len(set(indices)):
            raise ValueError("Each form field index must be unique")
        if self.submit_index in indices:
            raise ValueError("submit_index must not also be a form field index")
        return self


BROWSER_FILL_FORM_DESCRIPTION = """Fill multiple fields from one browser_get_state
result, optionally click one submit button, and return the final page state.

Use this instead of separate browser_type and browser_click calls when the current page
already shows every field and the optional submit control. Each field accepts literal
text or a registered secret reference. Registered secret values never enter model
context.

Only use indices from the latest browser_get_state result. The submit click always runs
last. Screenshots are omitted unless include_screenshot is true. A batch containing
registered secret values cannot include a screenshot; capture visual evidence only after
the credential fields are no longer visible.
"""


class BrowserFillFormTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserFillFormAction, BrowserObservation],
):
    """Tool for filling and optionally submitting one current-page form."""

    # Browser tools do not consume arbitrary MCP metadata. Narrowing this
    # inherited field keeps their public OpenAPI component strongly typed.
    meta: dict[str, str] | None = None

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_FILL_FORM_DESCRIPTION,
                action_type=BrowserFillFormAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_fill_form",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_get_secret`
# ============================================
class BrowserGetSecretAction(BrowserAction):
    """Schema for retrieving a registered secret for browser use."""

    secret_name: str = Field(min_length=1, description="Registered secret name")
    json_field: str | None = Field(
        default=None,
        min_length=1,
        description="Optional top-level JSON string field to retrieve",
    )


BROWSER_GET_SECRET_DESCRIPTION = """Retrieve a registered secret for browser input.

The raw value is returned only to the live model context. Use the returned value in a
subsequent browser_type call. Persisted events and logs mask the value.

Parameters:
- secret_name: Registered secret name
- json_field: Optional top-level JSON string field to retrieve
"""


class BrowserGetSecretTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserGetSecretAction, BrowserObservation],
):
    """Tool for retrieving a registered secret at runtime."""

    # Browser tools do not consume arbitrary MCP metadata. Narrowing this
    # inherited field keeps their public OpenAPI component strongly typed.
    meta: dict[str, str] | None = None

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_GET_SECRET_DESCRIPTION,
                action_type=BrowserGetSecretAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_get_secret",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_get_state`
# ============================================
class BrowserGetStateAction(BrowserAction):
    """Schema for getting browser state."""

    include_screenshot: bool = Field(
        default=False,
        description="Whether to include a screenshot of the current page. Default: False",  # noqa: E501
    )


BROWSER_GET_STATE_DESCRIPTION = """Get the page's interactive elements and where on the page you are reading them.

Returns the numbered interactive elements you can click or type into, and with
them `scroll`, `page`, `pages_above` and `pages_below`. Use it frequently.

What it returns is not the whole page, and the two ways it is partial are the
ways an absence gets called wrongly:

- Only interactive elements. Headings, labels, body copy and anything else that
  is not clickable or typeable are absent from this result whatever the page
  shows. Use browser_get_content to read text.
- Only what is reachable at the current scroll position. `pages_below` says how
  many more screens are under you; scroll and read again before concluding that
  something is not on the page.

Parameters:
- include_screenshot: Whether to include a screenshot (optional, default: False)
"""  # noqa: E501


class BrowserGetStateTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserGetStateAction, BrowserObservation],
):
    """Tool for getting browser state."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_GET_STATE_DESCRIPTION,
                action_type=BrowserGetStateAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_get_state",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_get_content`
# ============================================
class BrowserGetContentAction(BrowserAction):
    """Schema for getting page content in markdown."""

    extract_links: bool = Field(
        default=False,
        description="Whether to include links in the content (default: False)",
    )
    start_from_char: int = Field(
        default=0,
        ge=0,
        description="Character index to start from in the page content (default: 0)",
    )


BROWSER_GET_CONTENT_DESCRIPTION = """Extract the main content of the current page in clean markdown format. It has been filtered to remove noise and advertising content.

If the content was truncated and you need more information, use start_from_char parameter to continue from where truncation occurred.
"""  # noqa: E501


class BrowserGetContentTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserGetContentAction, BrowserObservation],
):
    """Tool for getting page content in markdown."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_GET_CONTENT_DESCRIPTION,
                action_type=BrowserGetContentAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_get_content",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_scroll`
# ============================================
class BrowserScrollAction(BrowserAction):
    """Schema for scrolling the page."""

    direction: Literal["up", "down"] = Field(
        default="down",
        description="Direction to scroll. Options: 'up', 'down'. Default: 'down'",
    )


BROWSER_SCROLL_DESCRIPTION = """Scroll the page up or down.

Use this tool to scroll through page content when elements are not visible or when you need
to see more content.

Parameters:
- direction: Direction to scroll - "up" or "down" (optional, default: "down")

Returns the resulting page state, so a following browser_get_state call is
redundant: read the state in this tool's own result.
"""  # noqa: E501


class BrowserScrollTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserScrollAction, BrowserObservation],
):
    """Tool for scrolling the browser page."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_SCROLL_DESCRIPTION,
                action_type=BrowserScrollAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_scroll",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_set_viewport`
# ============================================
class BrowserSetViewportAction(BrowserAction):
    """Schema for re-rendering the open page at another viewport size."""

    width: int = Field(
        ge=320,
        le=3840,
        description="Viewport width in CSS pixels. A phone is about 390.",
    )
    height: int = Field(
        ge=400,
        le=2160,
        description="Viewport height in CSS pixels. A phone is about 844.",
    )


BROWSER_SET_VIEWPORT_DESCRIPTION = """Re-render the page already open at a different viewport size.

Use this to see a responsive layout the way a narrower device renders it, for example
390x844 for a phone. The page keeps its signed-in session, so this costs a resize rather
than another login.

Width is the one rendering condition no control on the page can supply. A theme is not set
here: a product that offers dark mode ships a toggle, so switch it by clicking it.

Parameters:
- width: viewport width in CSS pixels
- height: viewport height in CSS pixels
"""  # noqa: E501


class BrowserSetViewportTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserSetViewportAction, BrowserObservation],
):
    """Tool for re-rendering the open page at another viewport size."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_SET_VIEWPORT_DESCRIPTION,
                action_type=BrowserSetViewportAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_set_viewport",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_go_back`
# ============================================
class BrowserGoBackAction(BrowserAction):
    """Schema for going back in browser history."""

    pass


BROWSER_GO_BACK_DESCRIPTION = """Go back to the previous page in browser history.

Use this tool to navigate back to the previously visited page, similar to clicking the 
browser's back button.

Returns the resulting page state, so a following browser_get_state call is
redundant: read the state in this tool's own result.
"""  # noqa: E501


class BrowserGoBackTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserGoBackAction, BrowserObservation],
):
    """Tool for going back in browser history."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_GO_BACK_DESCRIPTION,
                action_type=BrowserGoBackAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_go_back",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_list_tabs`
# ============================================
class BrowserListTabsAction(BrowserAction):
    """Schema for listing browser tabs."""

    pass


BROWSER_LIST_TABS_DESCRIPTION = """List all open browser tabs.

This tool shows all currently open tabs with their IDs, titles, and URLs. Use the tab IDs
with browser_switch_tab or browser_close_tab.
"""  # noqa: E501


class BrowserListTabsTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserListTabsAction, BrowserObservation],
):
    """Tool for listing browser tabs."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_LIST_TABS_DESCRIPTION,
                action_type=BrowserListTabsAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_list_tabs",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_switch_tab`
# ============================================
class BrowserSwitchTabAction(BrowserAction):
    """Schema for switching browser tabs."""

    tab_id: str = Field(
        description="4 Character Tab ID of the tab to switch"
        + " to (from browser_list_tabs)"
    )


BROWSER_SWITCH_TAB_DESCRIPTION = """Switch to a different browser tab.

Use this tool to switch between open tabs. Get the tab_id from browser_list_tabs.

Parameters:
- tab_id: 4 Character Tab ID of the tab to switch to

Returns the resulting page state, so a following browser_get_state call is
redundant: read the state in this tool's own result.
"""


class BrowserSwitchTabTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserSwitchTabAction, BrowserObservation],
):
    """Tool for switching browser tabs."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_SWITCH_TAB_DESCRIPTION,
                action_type=BrowserSwitchTabAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_switch_tab",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_close_tab`
# ============================================
class BrowserCloseTabAction(BrowserAction):
    """Schema for closing browser tabs."""

    tab_id: str = Field(
        description="4 Character Tab ID of the tab to close (from browser_list_tabs)"
    )


BROWSER_CLOSE_TAB_DESCRIPTION = """Close a specific browser tab.

Use this tool to close tabs you no longer need. Get the tab_id from browser_list_tabs.

Parameters:
- tab_id: 4 Character Tab ID of the tab to close

Returns the resulting page state, so a following browser_get_state call is
redundant: read the state in this tool's own result.
"""


class BrowserCloseTabTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserCloseTabAction, BrowserObservation],
):
    """Tool for closing browser tabs."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_CLOSE_TAB_DESCRIPTION,
                action_type=BrowserCloseTabAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_close_tab",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_get_storage`
# ============================================
class BrowserGetStorageAction(BrowserAction):
    """Schema for getting browser storage (cookies, local storage, session storage)."""

    pass


BROWSER_GET_STORAGE_DESCRIPTION = """Get browser storage data including cookies,
local storage, and session storage.

This tool extracts all cookies and storage data from the current browser session.
Useful for debugging, session management, or extracting authentication tokens.
"""


class BrowserGetStorageTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserGetStorageAction, BrowserObservation],
):
    """Tool for getting browser storage."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_GET_STORAGE_DESCRIPTION,
                action_type=BrowserGetStorageAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_get_storage",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_set_storage`
# ============================================
class BrowserSetStorageAction(BrowserAction):
    """Schema for setting browser storage (cookies, local storage, session storage)."""

    storage_state: dict = Field(
        description="Storage state dictionary containing 'cookies' and 'origins' (from browser_get_storage)"  # noqa: E501
    )


BROWSER_SET_STORAGE_DESCRIPTION = """Set browser storage data including cookies,
local storage, and session storage.

This tool allows you to restore or set the browser's storage state. You can use the
output from browser_get_storage to restore a previous session.

Parameters:
- storage_state: A dictionary containing 'cookies' and 'origins'.
  - cookies: List of cookie objects
  - origins: List of origin objects containing 'localStorage' and 'sessionStorage'
"""


class BrowserSetStorageTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserSetStorageAction, BrowserObservation],
):
    """Tool for setting browser storage."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_SET_STORAGE_DESCRIPTION,
                action_type=BrowserSetStorageAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_set_storage",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_start_recording`
# ============================================
class BrowserStartRecordingAction(BrowserAction):
    """Schema for starting browser session recording."""

    pass


BROWSER_START_RECORDING_DESCRIPTION = f"""Start recording the browser session.

This tool starts recording all browser interactions using rrweb. The recording
captures DOM mutations, mouse movements, clicks, scrolls, and other user interactions.

Output Location: {BROWSER_RECORDING_OUTPUT_DIR}/recording-<timestamp>/
Format: Recording events are saved as numbered JSON files (1.json, 2.json, etc.)
containing rrweb event arrays. Events are flushed every 5 seconds or when they
exceed 1 MB. These files can be replayed using rrweb-player.

Call browser_stop_recording to stop recording and save any remaining events.

Note: Recording persists across page navigations - the recording will automatically
restart on new pages.
"""


class BrowserStartRecordingTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserStartRecordingAction, BrowserObservation],
):
    """Tool for starting browser session recording."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_START_RECORDING_DESCRIPTION,
                action_type=BrowserStartRecordingAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_start_recording",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_stop_recording`
# ============================================
class BrowserStopRecordingAction(BrowserAction):
    """Schema for stopping browser session recording."""

    pass


BROWSER_STOP_RECORDING_DESCRIPTION = f"""Stop recording the browser session.

This tool stops the current recording session and saves any remaining events to disk.

Output Location: {BROWSER_RECORDING_OUTPUT_DIR}/recording-<timestamp>/
Format: Events are saved as numbered JSON files (1.json, 2.json, etc.) containing
rrweb event arrays. These files can be replayed using rrweb-player to visualize
the recorded session.

Returns a summary message with the total event count, file count, and save directory.
"""


class BrowserStopRecordingTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserStopRecordingAction, BrowserObservation],
):
    """Tool for stopping browser session recording."""

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_STOP_RECORDING_DESCRIPTION,
                action_type=BrowserStopRecordingAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_stop_recording",
                    # Modifies state: stops recording, flushes events to disk
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_start_video_recording`
# ============================================
class BrowserStartVideoRecordingAction(BrowserAction):
    """Schema for starting visible browser video recording."""

    pass


BROWSER_START_VIDEO_RECORDING_DESCRIPTION = """Start recording the visible browser
window to a WebM video file.

The browser must run in headed mode on an X11 display. Use this before exercising a
user-visible interaction that needs video evidence, then call
browser_stop_video_recording to finalize the file.
"""


class BrowserStartVideoRecordingTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserStartVideoRecordingAction, BrowserObservation],
):
    """Tool for starting encoded browser video recording."""

    meta: dict[str, str] | None = None

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_START_VIDEO_RECORDING_DESCRIPTION,
                action_type=BrowserStartVideoRecordingAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_start_video_recording",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_stop_video_recording`
# ============================================
class BrowserStopVideoRecordingAction(BrowserAction):
    """Schema for stopping visible browser video recording."""

    pass


BROWSER_STOP_VIDEO_RECORDING_DESCRIPTION = """Stop the visible browser recording and
finalize its WebM file.

The result includes the absolute sandbox path. Pass that path to an artifact publishing
tool when the recording should be attached to a session or pull request.
"""


class BrowserStopVideoRecordingTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserStopVideoRecordingAction, BrowserObservation],
):
    """Tool for stopping encoded browser video recording."""

    meta: dict[str, str] | None = None

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_STOP_VIDEO_RECORDING_DESCRIPTION,
                action_type=BrowserStopVideoRecordingAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_stop_video_recording",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# ============================================
# `browser_sequence`
# ============================================
def _sequence_step_actions() -> dict[str, type[BrowserAction]]:
    """Step name to the action class the single-action tool of that name uses.

    Built from the same classes those tools use, so a step is validated by
    exactly the schema its standalone equivalent would have applied. Recording
    and storage actions are deliberately absent: they bracket a whole flow
    rather than sit inside one, and `get_secret` is absent because a secret
    should be resolved in its own observable call.
    """
    return {
        "navigate": BrowserNavigateAction,
        "click": BrowserClickAction,
        "type": BrowserTypeAction,
        "get_state": BrowserGetStateAction,
        "get_content": BrowserGetContentAction,
        "scroll": BrowserScrollAction,
        "go_back": BrowserGoBackAction,
        "list_tabs": BrowserListTabsAction,
        "switch_tab": BrowserSwitchTabAction,
    }


class BrowserSequenceStep(BaseModel):
    """One browser interaction inside a sequence."""

    action: str = Field(
        description=(
            "Which browser action to run. One of: "
            "navigate, click, type, get_state, get_content, scroll, "
            "go_back, list_tabs, switch_tab. Use browser_fill_form on its own: "
            "its fields are objects, which a step's flat arguments cannot carry."
        )
    )
    arguments: dict[str, str | int | bool | None] = Field(
        default_factory=dict,
        description=(
            "Arguments for that action, exactly as the single-action tool of the "
            'same name takes them. For example navigate takes {"url": ...}, '
            'click takes {"index": ...}.'
        ),
    )


class BrowserSequenceAction(BrowserAction):
    """Schema for running several browser interactions in one call."""

    steps: list[BrowserSequenceStep] = Field(
        min_length=1,
        max_length=20,
        description="The interactions to run, in order.",
    )


BROWSER_SEQUENCE_DESCRIPTION = """Run several browser interactions in one call, in order.

Use this whenever you already know the next few browser steps: navigating and then reading
the page, or clicking through a form and reading the result. Each step is the same action
the single-action browser tool of that name performs, with the same arguments.

The sequence stops at the first step that fails, and tells you which one and what the
remaining steps were, so a failure is as diagnosable as it would have been on its own.
The returned state is the state after the last step that ran.

Parameters:
- steps: list of {action, arguments}, in order (required, 1-20 steps)

Example - open a page and read it:
  steps=[{"action": "navigate", "arguments": {"url": "https://example.com/reports"}},
         {"action": "get_state", "arguments": {}}]

Example - filter and read the result:
  steps=[{"action": "click", "arguments": {"index": 4}},
         {"action": "click", "arguments": {"index": 11}},
         {"action": "get_state", "arguments": {}}]

Prefer one sequence over the same steps issued one at a time: each separate call costs a
full model round trip, and the browser work itself is milliseconds.
"""  # noqa: E501


class BrowserSequenceTool(
    _SharesOneBrowserSession,
    ToolDefinition[BrowserSequenceAction, BrowserObservation],
):
    """Tool for running a batch of browser interactions in one call."""

    # Browser tools do not consume arbitrary MCP metadata. Narrowing this
    # inherited field keeps their public OpenAPI component strongly typed.
    meta: dict[str, str] | None = None

    @classmethod
    def create(cls, executor: "BrowserToolExecutor") -> Sequence[Self]:
        return [
            cls(
                description=BROWSER_SEQUENCE_DESCRIPTION,
                action_type=BrowserSequenceAction,
                observation_type=BrowserObservation,
                annotations=ToolAnnotations(
                    title="browser_sequence",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


class BrowserToolSet(ToolDefinition[BrowserAction, BrowserObservation]):
    """A set of all browser tools.

    This tool set includes all available browser-related tools
      for interacting with web pages.

    The toolset automatically checks for Chromium availability
    when created and automatically installs it if missing.
    """

    # Shared executor: reuse a single Chromium/CDP instance across parent
    # and subagents to avoid CDP port conflicts in sandbox containers.
    _shared_executor: ClassVar["BrowserToolExecutor | None"] = None
    _shared_executor_lock: ClassVar[threading.Lock] = threading.Lock()
    _shared_executor_creation_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def is_usable(cls) -> bool:
        from openhands.tools.browser_use.impl import BrowserToolExecutor

        return BrowserToolExecutor.check_chromium_available() is not None

    @classmethod
    def _warn_config_ignored(cls, executor_config: dict[str, object]) -> None:
        if not executor_config:
            return
        _logger.warning(
            "BrowserToolSet.create() called with executor_config but a "
            "shared executor already exists. The config %s will be "
            "ignored. This typically happens when a subagent requests "
            "browser tools — it reuses the parent's browser session.",
            list(executor_config.keys()),
        )

    @classmethod
    def _get_or_create_shared_executor(
        cls,
        conv_state: "ConversationState",
        **executor_config,
    ) -> "BrowserToolExecutor":
        return cls.get_or_create_shared_executor(
            full_output_save_dir=conv_state.env_observation_persistence_dir,
            **executor_config,
        )

    @classmethod
    def get_or_create_shared_executor(
        cls,
        full_output_save_dir: str | None = None,
        **executor_config,
    ) -> "BrowserToolExecutor":
        """Return the process-wide browser executor used by agents and desktop APIs."""
        with cls._shared_executor_creation_lock:
            with cls._shared_executor_lock:
                executor = cls._shared_executor

            if executor is not None:
                # The screencast and desktop services create this executor with
                # no persistence directory, and in production they usually get
                # there first: the conversation's Browser tab opens as soon as
                # the sandbox reports RUNNING, minutes before the agent's first
                # browser call. Reusing that instance verbatim left
                # `full_output_save_dir` None for the whole conversation, so
                # `include_screenshot=True` put the frame in the LLM context but
                # never on disk -- and every screenshot the agent had to publish
                # as a file needed a hand-rolled capture path instead
                # (`pip install Pillow`, an X11 grab of the whole desktop, a raw
                # CDP keystroke to clear an overlay: about three minutes per
                # conversation). Adopting a directory only ever fills in a None,
                # so the first caller that knows one wins and no existing
                # persistence target is overwritten.
                if executor.full_output_save_dir is None and full_output_save_dir:
                    executor.full_output_save_dir = full_output_save_dir
                cls._warn_config_ignored(executor_config)
                return executor

            from openhands.tools.browser_use.impl import BrowserToolExecutor

            executor = BrowserToolExecutor(
                full_output_save_dir=full_output_save_dir,
                **executor_config,
            )
            with cls._shared_executor_lock:
                cls._shared_executor = executor
            return executor

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        **executor_config,
    ) -> list[ToolDefinition[BrowserAction, BrowserObservation]]:
        executor = cls._get_or_create_shared_executor(conv_state, **executor_config)

        # Each tool.create() returns a Sequence[Self], so we flatten the results
        tools: list[ToolDefinition[BrowserAction, BrowserObservation]] = []
        for tool_class in [
            BrowserNavigateTool,
            BrowserClickTool,
            BrowserGetStateTool,
            BrowserGetContentTool,
            BrowserGetSecretTool,
            BrowserTypeTool,
            BrowserFillFormTool,
            BrowserScrollTool,
            BrowserSetViewportTool,
            BrowserGoBackTool,
            BrowserListTabsTool,
            BrowserSwitchTabTool,
            BrowserCloseTabTool,
            BrowserGetStorageTool,
            BrowserSetStorageTool,
            BrowserStartRecordingTool,
            BrowserStopRecordingTool,
            BrowserStartVideoRecordingTool,
            BrowserStopVideoRecordingTool,
            BrowserSequenceTool,
        ]:
            tools.extend(tool_class.create(executor))
        return tools


register_tool(BrowserToolSet.name, BrowserToolSet)
