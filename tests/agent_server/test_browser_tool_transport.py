"""An external controller uses the conversation's native sandbox browser."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

from openhands.agent_server.conversation_router import conversation_router
from openhands.agent_server.dependencies import get_event_service
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.tool import ToolExecutor
from openhands.tools.browser_use.definition import (
    BrowserGetStateTool,
    BrowserNavigateAction,
    BrowserNavigateTool,
    BrowserObservation,
    BrowserSequenceTool,
    BrowserTypeTool,
)
from openhands.tools.browser_use.impl import BrowserToolExecutor


def test_browser_call_uses_native_schema_and_preserves_original_frame():
    expected = BrowserObservation.from_text(
        text='{"url":"https://example.test","title":"Fixture"}',
        screenshot_data="b3JpZ2luYWwtZnJhbWU=",
    )
    state_lock = threading.Lock()

    def run_native(action, conversation):
        assert state_lock.locked(), "Browser RPC must hold the conversation state lock"
        return expected

    executor = Mock(spec=ToolExecutor, side_effect=run_native)
    (tool,) = BrowserNavigateTool.create(executor)
    conversation = SimpleNamespace(
        agent=SimpleNamespace(tools_map={tool.name: tool}),
        state=SimpleNamespace(execution_status=ConversationExecutionStatus.IDLE),
        _state=state_lock,
    )
    service = Mock()
    service.get_conversation.return_value = conversation
    app = FastAPI()
    app.include_router(conversation_router)
    app.dependency_overrides[get_event_service] = lambda: service
    path = f"/conversations/{uuid4()}/browser"

    with (
        patch(
            "openhands.tools.browser_use.definition.BrowserToolSet.create",
            return_value=[tool],
        ),
        TestClient(app) as client,
    ):
        catalog = client.get(f"{path}/tools")
        assert catalog.status_code == 200
        assert catalog.json() == jsonable_encoder([tool.to_mcp_tool()])
        result = client.post(
            f"{path}/call",
            json={"tool_name": tool.name, "arguments": {"url": "https://example.test"}},
        )
        assert result.status_code == 200
        assert result.json() == expected.model_dump(mode="json")
        for payload, status in (
            ({"tool_name": "terminal", "arguments": {}}, 404),
            ({"tool_name": tool.name, "arguments": {}}, 422),
        ):
            denied = client.post(f"{path}/call", json=payload)
            assert denied.status_code == status
        conversation.state.execution_status = ConversationExecutionStatus.RUNNING
        busy = client.post(
            f"{path}/call",
            json={"tool_name": tool.name, "arguments": {"url": "https://example.test"}},
        )
        assert busy.status_code == 409

    assert executor.call_count == 1
    action = executor.call_args.args[0]
    assert isinstance(action, BrowserNavigateAction)
    assert action.url == "https://example.test"
    assert executor.call_args.args[1] is conversation


def test_browser_call_settles_local_cdp_before_the_requested_capture():
    expected = BrowserObservation.from_text(
        text='{"url":"https://example.test"}', screenshot_data="cGl4ZWxz"
    )
    settled = False

    def wait_for_stable_frame():
        nonlocal settled
        settled = True
        return True

    def run_native(action, conversation):
        assert settled
        return expected

    executor = Mock(spec=BrowserToolExecutor, side_effect=run_native)
    executor.wait_for_stable_frame.side_effect = wait_for_stable_frame
    (tool,) = BrowserGetStateTool.create(executor)
    conversation = SimpleNamespace(
        state=SimpleNamespace(execution_status=ConversationExecutionStatus.IDLE),
        _state=threading.Lock(),
    )
    service = Mock()
    service.get_conversation.return_value = conversation
    app = FastAPI()
    app.include_router(conversation_router)
    app.dependency_overrides[get_event_service] = lambda: service

    with (
        patch(
            "openhands.tools.browser_use.definition.BrowserToolSet.create",
            return_value=[tool],
        ),
        patch(
            "openhands.tools.browser_use.definition.BrowserToolSet.get_or_create_shared_executor",
            return_value=executor,
        ),
        TestClient(app) as client,
    ):
        result = client.post(
            f"/conversations/{uuid4()}/browser/call",
            json={
                "tool_name": tool.name,
                "arguments": {"include_screenshot": True},
                "settle_before_capture": True,
            },
        )

    assert result.status_code == 200
    assert result.json() == expected.model_dump(mode="json")
    executor.wait_for_stable_frame.assert_called_once_with()


@pytest.mark.parametrize(
    "preview_url,page_url,status,sequence",
    [
        ("https://preview.example", "https://preview.example/profile", 200, False),
        ("https://preview.example", "https://other.example", 403, False),
        (None, "https://preview.example/profile", 403, False),
        ("https://preview.example", "https://preview.example/profile", 403, True),
    ],
)
def test_credential_reference_requires_current_preview_origin(
    preview_url, page_url, status, sequence
):
    from openhands.tools.browser_use.impl import BrowserToolExecutor

    executor = Mock(
        spec=BrowserToolExecutor,
        return_value=BrowserObservation.from_text(text="Typed"),
    )
    executor.browser_metadata.return_value = {"url": page_url, "title": "Page"}
    (tool,) = (BrowserSequenceTool if sequence else BrowserTypeTool).create(executor)
    arguments = {"index": 0, "secret_name": "account"}
    if sequence:
        arguments = {
            "steps": [
                {"action": "navigate", "arguments": {"url": "https://other.example"}},
                {"action": "type", "arguments": {"index": 0, "secret_name": "account"}},
            ]
        }
    service = Mock()
    service.get_conversation.return_value = SimpleNamespace(
        state=SimpleNamespace(execution_status=ConversationExecutionStatus.IDLE),
        _state=threading.Lock(),
    )
    app = FastAPI()
    app.include_router(conversation_router)
    app.dependency_overrides[get_event_service] = lambda: service
    with (
        patch(
            "openhands.tools.browser_use.definition.BrowserToolSet.create",
            return_value=[tool],
        ),
        patch(
            "openhands.tools.browser_use.definition.BrowserToolSet.get_or_create_shared_executor",
            return_value=executor,
        ),
        TestClient(app) as client,
    ):
        result = client.post(
            f"/conversations/{uuid4()}/browser/call",
            json={
                "tool_name": tool.name,
                "arguments": arguments,
                "preview_url": preview_url,
            },
        )
    assert result.status_code == status
    assert executor.call_count == (1 if status == 200 else 0)


def test_browser_rpc_captures_real_local_page(tmp_path, monkeypatch):
    import base64
    import io
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from PIL import Image

    from openhands.sdk.conversation.secret_registry import SecretRegistry
    from openhands.tools.browser_use.definition import BrowserToolSet
    from openhands.tools.browser_use.impl import BrowserToolExecutor

    if BrowserToolExecutor.check_chromium_available() is None:
        pytest.skip("Chromium is not installed")

    class Page(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object):
            pass

        def do_GET(self):
            body = (
                b"<html><head><title>Sandbox capture fixture</title></head>"
                b'<body style="margin:0;background:#123456;color:white">'
                b"Sandbox capture fixture</body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Page)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/"
    conversation = SimpleNamespace(
        state=SimpleNamespace(
            execution_status=ConversationExecutionStatus.IDLE,
            env_observation_persistence_dir=str(tmp_path),
            secret_registry=SecretRegistry(),
        ),
        _state=threading.RLock(),
    )
    service = Mock()
    service.get_conversation.return_value = conversation
    app = FastAPI()
    app.include_router(conversation_router)
    app.dependency_overrides[get_event_service] = lambda: service
    monkeypatch.setattr(BrowserToolSet, "_shared_executor", None)
    try:
        with TestClient(app) as client:
            path = f"/conversations/{uuid4()}/browser/call"
            result = client.post(
                path,
                json={"tool_name": "browser_navigate", "arguments": {"url": url}},
            )
            assert result.status_code == 200
            result = client.post(
                path,
                json={
                    "tool_name": "browser_get_state",
                    "arguments": {"include_screenshot": True},
                },
            )
        assert result.status_code == 200
        observation = BrowserObservation.model_validate(result.json())
        assert not observation.is_error, observation.text
        assert "Sandbox capture fixture" in observation.text
        assert observation.screenshot_data
        pixels = base64.b64decode(observation.screenshot_data, validate=True)
        with Image.open(io.BytesIO(pixels)) as image:
            rgb = image.convert("RGB").getpixel((100, 100))
            assert isinstance(rgb, tuple)
            assert all(
                abs(actual - expected) <= 5
                for actual, expected in zip(rgb, (18, 52, 86))
            )
    finally:
        executor = BrowserToolSet._shared_executor
        if executor is not None:
            executor.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
