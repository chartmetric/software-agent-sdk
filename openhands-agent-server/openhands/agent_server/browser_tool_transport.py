"""Invoke native tools on the sandbox browser without starting an agent run."""

import threading
from urllib.parse import urlsplit

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from openhands.agent_server.event_service import EventService
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.tools.browser_use.definition import (
    BrowserAction,
    BrowserObservation,
    BrowserSequenceAction,
    BrowserToolSet,
)


_browser_call_lock = threading.Lock()

type BrowserJSONValue = (
    str
    | int
    | float
    | bool
    | None
    | list[BrowserJSONValue]
    | dict[str, BrowserJSONValue]
)


class BrowserToolCallRequest(BaseModel):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, BrowserJSONValue] = Field(default_factory=dict)
    settle_before_capture: bool = Field(
        default=False,
        description="Wait for stable compositor pixels before invoking the tool.",
    )
    preview_url: HttpUrl | None = Field(
        default=None, description="Workspace Preview origin for credential references."
    )


def _uses_credential_reference(value: object) -> bool:
    if isinstance(value, dict):
        return bool(value.get("secret_name")) or any(
            _uses_credential_reference(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_uses_credential_reference(child) for child in value)
    return False


def _require_credential_origin(
    request: BrowserToolCallRequest, action: BrowserAction
) -> None:
    if not _uses_credential_reference(request.arguments):
        return
    if isinstance(action, BrowserSequenceAction):
        raise HTTPException(
            403,
            "Use individual browser calls for registered credentials so the "
            "Preview origin is checked before each entry.",
        )
    preview = request.preview_url
    executor = BrowserToolSet.get_or_create_shared_executor()
    page = urlsplit(executor.browser_metadata()["url"])
    port = page.port or (443 if page.scheme == "https" else 80)
    if preview is None or (page.scheme, page.hostname, port) != (
        preview.scheme,
        preview.host,
        preview.port,
    ):
        raise HTTPException(
            403,
            "Open the bound workspace preview before typing registered credentials.",
        )


def list_browser_tools(
    event_service: EventService,
) -> list[dict[str, BrowserJSONValue]]:
    conversation = event_service.get_conversation()
    return [
        jsonable_encoder(tool.to_mcp_tool())
        for tool in BrowserToolSet.create(conversation.state)
    ]


def close_browser_transport() -> None:
    executor = BrowserToolSet._shared_executor
    if executor is not None:
        executor.close()


def call_browser_tool(
    event_service: EventService, request: BrowserToolCallRequest
) -> BrowserObservation:
    # Different tool names and conversations share the same sandbox browser.
    conversation = event_service.get_conversation()
    with _browser_call_lock, conversation._state:
        if conversation.state.execution_status == ConversationExecutionStatus.RUNNING:
            raise HTTPException(
                409,
                "The sandbox agent is running. Wait for it to stop "
                "before driving its browser.",
            )
        tools = BrowserToolSet.create(conversation.state)
        tool = next((tool for tool in tools if tool.name == request.tool_name), None)
        if tool is None:
            raise HTTPException(
                404, "Unknown browser tool. Read this sandbox's browser tool catalog."
            )
        try:
            action = tool.action_from_arguments(request.arguments)
        except ValidationError as exc:
            raise HTTPException(
                422, "Arguments do not match the native browser tool schema."
            ) from exc
        assert isinstance(action, BrowserAction)
        _require_credential_origin(request, action)
        if request.settle_before_capture:
            BrowserToolSet.get_or_create_shared_executor().wait_for_stable_frame()
        observation = tool(action, conversation=conversation)
        assert isinstance(observation, BrowserObservation)
        return observation
