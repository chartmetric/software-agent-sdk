import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.types.utils import ModelResponse
from pydantic import SecretStr

from openhands.sdk.agent import Agent
from openhands.sdk.conversation import Conversation, LocalConversation
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.sdk.llm import LLM, Message, MessageToolCall, TextContent, ThinkingBlock
from openhands.sdk.llm.utils.metrics import Metrics
from openhands.sdk.llm.utils.telemetry import Telemetry
from openhands.tools.browser_use.definition import (
    BrowserFillFormAction,
    BrowserFillFormTool,
    BrowserFormField,
    BrowserGetContentAction,
    BrowserGetSecretAction,
    BrowserObservation,
    BrowserTypeAction,
    BrowserTypeTool,
)
from openhands.tools.browser_use.server import CustomBrowserUseServer


SECRET_VALUE = "browser-secret-123"


def _conversation_with_secrets(secrets: dict[str, str]) -> LocalConversation:
    secret_registry = SecretRegistry()
    secret_registry.update_secrets(secrets)
    conversation = SimpleNamespace(
        state=SimpleNamespace(secret_registry=secret_registry)
    )
    return cast(LocalConversation, cast(Any, conversation))


def test_browser_get_secret_exposes_raw_value_only_to_live_llm_context(
    mock_browser_executor,
):
    conversation = _conversation_with_secrets(
        {"TEST_ACCOUNT": json.dumps({"password": SECRET_VALUE})}
    )

    observation = mock_browser_executor(
        BrowserGetSecretAction(secret_name="TEST_ACCOUNT", json_field="password"),
        conversation,
    )
    event = ObservationEvent(
        observation=observation,
        action_id="action-1",
        tool_name="browser_get_secret",
        tool_call_id="call-1",
    )

    observation_content = observation.to_llm_content[0]
    event_content = event.to_llm_message().content[0]
    assert isinstance(observation_content, TextContent)
    assert isinstance(event_content, TextContent)
    assert observation_content.text == SECRET_VALUE
    assert event_content.text == SECRET_VALUE
    assert observation.text == "<secret-hidden>"
    assert SECRET_VALUE not in observation.model_dump_json()
    assert SECRET_VALUE not in event.model_dump_json()
    assert SECRET_VALUE not in json.dumps(event.model_dump(mode="json"))
    assert SECRET_VALUE not in event.visualize.plain
    assert SECRET_VALUE not in str(event)


def test_browser_get_secret_fails_closed_when_secret_cannot_be_resolved(
    mock_browser_executor,
):
    result = mock_browser_executor(
        BrowserGetSecretAction(secret_name="MISSING"),
        _conversation_with_secrets({}),
    )

    assert result.is_error is True
    assert result.text == "Unable to resolve the registered browser secret"


def test_browser_type_can_resolve_json_field_without_exposing_value(
    mock_browser_executor,
):
    conversation = _conversation_with_secrets(
        {"TEST_ACCOUNT": json.dumps({"password": SECRET_VALUE})}
    )

    with patch.object(
        mock_browser_executor,
        "type_secret_text",
        new_callable=AsyncMock,
        return_value="Typed <secret> into element 2",
    ) as type_secret_text:
        result = mock_browser_executor(
            BrowserTypeAction(
                index=2,
                secret_name="TEST_ACCOUNT",
                json_field="password",
            ),
            conversation,
        )

    type_secret_text.assert_awaited_once_with(2, SECRET_VALUE)
    assert SECRET_VALUE not in result.model_dump_json()


def test_browser_fill_form_resolves_multiple_fields_without_exposing_values(
    mock_browser_executor,
):
    email = "private@example.com"
    conversation = _conversation_with_secrets(
        {"TEST_ACCOUNT": json.dumps({"email": email, "password": SECRET_VALUE})}
    )
    fields = [
        BrowserFormField(
            index=1,
            secret_name="TEST_ACCOUNT",
            json_field="email",
        ),
        BrowserFormField(
            index=2,
            secret_name="TEST_ACCOUNT",
            json_field="password",
        ),
    ]

    with patch.object(
        mock_browser_executor,
        "fill_form",
        new_callable=AsyncMock,
        return_value=BrowserObservation.from_text(text="Final state"),
    ) as fill_form:
        result = mock_browser_executor(
            BrowserFillFormAction(fields=fields, submit_index=3),
            conversation,
        )

    fill_form.assert_awaited_once_with(
        fields,
        3,
        False,
        {0: email, 1: SECRET_VALUE},
    )
    serialized = result.model_dump_json()
    assert email not in serialized
    assert SECRET_VALUE not in serialized


def test_browser_fill_form_rejects_screenshot_with_runtime_secrets(
    mock_browser_executor,
):
    conversation = _conversation_with_secrets({"LOGIN_PASSWORD": SECRET_VALUE})

    with patch.object(
        mock_browser_executor,
        "fill_form",
        new_callable=AsyncMock,
    ) as fill_form:
        result = mock_browser_executor(
            BrowserFillFormAction(
                fields=[BrowserFormField(index=2, secret_name="LOGIN_PASSWORD")],
                include_screenshot=True,
            ),
            conversation,
        )

    fill_form.assert_not_awaited()
    assert result.is_error is True
    assert "separate browser_get_state" in result.text


def test_browser_fill_form_masks_nested_literal_secret_before_persistence(
    mock_browser_executor,
):
    llm = LLM(
        usage_id="browser-fill-form-secret-test",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(llm=llm, tools=[], include_default_tools=[])
    tool = BrowserFillFormTool.create(mock_browser_executor)[0]
    agent._initialized = True
    agent._tools = {tool.name: tool}
    conversation = Conversation(agent=agent)
    conversation.state.secret_registry.update_secrets({"LOGIN_PASSWORD": SECRET_VALUE})
    tool_call = MessageToolCall(
        id="call-fill-form",
        name=tool.name,
        arguments=json.dumps(
            {
                "fields": [{"index": 2, "text": SECRET_VALUE}],
                "submit_index": 3,
            }
        ),
        origin="completion",
    )

    action_event = agent._get_action_event(
        tool_call,
        conversation,
        "response-fill-form",
        lambda event: None,
    )

    assert isinstance(action_event, ActionEvent)
    assert isinstance(action_event.action, BrowserFillFormAction)
    assert action_event.action.fields[0].text == "<secret-hidden>"
    assert SECRET_VALUE not in action_event.model_dump_json()

    with patch.object(
        mock_browser_executor,
        "fill_form",
        new_callable=AsyncMock,
        return_value=BrowserObservation.from_text(text="Final state"),
    ) as fill_form:
        agent._execute_action_event(conversation, action_event)

    fill_form.assert_awaited_once_with(
        [BrowserFormField(index=2, text=SECRET_VALUE)],
        3,
        False,
        {0: SECRET_VALUE},
    )


def test_browser_type_action_event_persists_masked_data_and_executes_raw_text(
    mock_browser_executor,
):
    llm = LLM(
        usage_id="browser-secret-test",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(llm=llm, tools=[], include_default_tools=[])
    tool = BrowserTypeTool.create(mock_browser_executor)[0]
    agent._initialized = True
    agent._tools = {tool.name: tool}
    conversation = Conversation(agent=agent)
    conversation.state.secret_registry.update_secrets({"LOGIN_PASSWORD": SECRET_VALUE})
    assert (
        conversation.state.secret_registry.get_secret_value("LOGIN_PASSWORD")
        == SECRET_VALUE
    )
    tool_call = MessageToolCall(
        id="call-1",
        name=tool.name,
        arguments=json.dumps(
            {
                "index": 2,
                "text": SECRET_VALUE,
                "summary": f"Type {SECRET_VALUE}",
            }
        ),
        origin="completion",
    )
    emitted_events = []

    action_event = agent._get_action_event(
        tool_call,
        conversation,
        "response-1",
        emitted_events.append,
        thought=[TextContent(text=f"Use {SECRET_VALUE}")],
        reasoning_content=f"Typing {SECRET_VALUE}",
        thinking_blocks=[
            ThinkingBlock(thinking=f"Secret is {SECRET_VALUE}", signature="sig")
        ],
    )

    assert isinstance(action_event, ActionEvent)
    assert isinstance(action_event.action, BrowserTypeAction)
    assert action_event.action.text == "<secret-hidden>"
    assert SECRET_VALUE not in action_event.tool_call.arguments
    assert SECRET_VALUE not in (action_event.summary or "")
    assert SECRET_VALUE not in action_event.thought[0].text
    assert SECRET_VALUE not in (action_event.reasoning_content or "")
    thinking_block = action_event.thinking_blocks[0]
    assert isinstance(thinking_block, ThinkingBlock)
    assert SECRET_VALUE not in thinking_block.thinking
    assert SECRET_VALUE not in action_event.model_dump_json()
    assert SECRET_VALUE not in json.dumps(action_event.model_dump(mode="json"))
    assert SECRET_VALUE not in action_event.visualize.plain
    assert SECRET_VALUE not in str(action_event)

    with patch.object(
        mock_browser_executor,
        "type_secret_text",
        new_callable=AsyncMock,
        return_value="Typed <secret> into element 2",
    ) as type_secret_text:
        observation_events = agent._execute_action_event(conversation, action_event)

    type_secret_text.assert_awaited_once_with(2, SECRET_VALUE)
    assert SECRET_VALUE not in observation_events[0].model_dump_json()


def test_browser_executor_masks_secret_echoes_from_later_observations(
    mock_browser_executor,
):
    conversation = _conversation_with_secrets({"LOGIN_PASSWORD": SECRET_VALUE})
    conversation.state.secret_registry.get_secret_value("LOGIN_PASSWORD")

    with patch.object(
        mock_browser_executor,
        "get_content",
        new_callable=AsyncMock,
        return_value=f"Page echoed {SECRET_VALUE}",
    ):
        result = mock_browser_executor(BrowserGetContentAction(), conversation)

    assert result.text == "Page echoed <secret-hidden>"


def test_agent_masks_secret_echoed_in_final_message():
    llm = LLM(
        usage_id="browser-secret-test",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(llm=llm, tools=[], include_default_tools=[])
    conversation = Conversation(agent=agent)
    conversation.state.secret_registry.update_secrets({"LOGIN_PASSWORD": SECRET_VALUE})
    conversation.state.secret_registry.get_secret_value("LOGIN_PASSWORD")
    emitted_events = []

    event = agent._emit_message_event(
        Message(role="assistant", content=[TextContent(text=SECRET_VALUE)]),
        cast(Any, SimpleNamespace(id="response-1")),
        conversation,
        emitted_events.append,
    )

    content = event.llm_message.content[0]
    assert isinstance(content, TextContent)
    assert content.text == "<secret-hidden>"
    assert SECRET_VALUE not in event.model_dump_json()


def test_completion_telemetry_callback_masks_live_secret_context():
    registry = SecretRegistry()
    registry.update_secrets({"LOGIN_PASSWORD": SECRET_VALUE})
    registry.get_secret_value("LOGIN_PASSWORD")
    telemetry = Telemetry(
        model_name="test-model",
        log_enabled=True,
        metrics=Metrics(),
    )
    callback_data: list[str] = []
    telemetry.set_log_completions_callback(
        lambda _filename, log_data: callback_data.append(log_data)
    )
    telemetry.on_request(
        {"messages": [{"role": "tool", "content": SECRET_VALUE}]},
        log_masker=registry.mask_secrets_in_output,
    )

    telemetry.log_llm_call(ModelResponse(id="response-1", choices=[]), 0.0)

    assert len(callback_data) == 1
    assert SECRET_VALUE not in callback_data[0]
    assert "<secret-hidden>" in callback_data[0]


async def test_browser_server_marks_secret_input_as_sensitive():
    node = MagicMock()
    type_event = MagicMock()
    completed = asyncio.get_running_loop().create_future()
    completed.set_result(None)
    browser_session = SimpleNamespace(
        get_dom_element_by_index=AsyncMock(return_value=node),
        event_bus=SimpleNamespace(dispatch=MagicMock(return_value=completed)),
    )
    server = object.__new__(CustomBrowserUseServer)
    server.browser_session = cast(Any, browser_session)

    with patch(
        "openhands.tools.browser_use.server.TypeTextEvent", return_value=type_event
    ) as type_text_event:
        result = await server._type_secret_text(2, SECRET_VALUE)

    dispatched = browser_session.event_bus.dispatch.call_args.args[0]
    type_text_event.assert_called_once_with(
        node=node,
        text=SECRET_VALUE,
        is_sensitive=True,
        sensitive_key_name="secret",
    )
    assert dispatched is type_event
    assert result == "Typed <secret> into element 2"
