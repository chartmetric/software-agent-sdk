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


class TestBrowserStateSaysWhereOnThePageItWasRead:
    """`browser_get_state` has to make an absence falsifiable.

    Upstream 0.11.9 returns the interactive elements and nothing about the
    document they came from, so a reader cannot tell a short page from the top
    of a long one. Measured across two production runs on 2026-08-27, nine
    observations carried no scroll position and both runs decided a component
    was missing from a page they had only seen the top of.
    """

    @staticmethod
    def _server_at(scroll_y: int, page_height: int, base: str):
        import types

        from openhands.tools.browser_use.server import CustomBrowserUseServer

        server = CustomBrowserUseServer.__new__(CustomBrowserUseServer)
        page_info = types.SimpleNamespace(
            viewport_width=390,
            viewport_height=844,
            page_width=390,
            page_height=page_height,
            scroll_x=0,
            scroll_y=scroll_y,
        )
        page = types.SimpleNamespace(
            evaluate=lambda script: _coro({"items": [], "total": 0, "truncated": False})
        )
        server.browser_session = cast(
            Any,
            types.SimpleNamespace(
                _cached_browser_state_summary=types.SimpleNamespace(
                    page_info=page_info
                ),
                get_current_page=lambda: _coro(page),
            ),
        )

        async def upstream(self, include_screenshot=False):
            return base

        CustomBrowserUseServer.__mro__[1]._get_browser_state = upstream
        return CustomBrowserUseServer, server

    @staticmethod
    def _page(elements=()):
        import json

        return json.dumps(
            {
                "url": "https://preview.example/artist/3648",
                "title": "Artist",
                "tabs": [],
                "interactive_elements": list(elements),
            }
        )

    def test_it_reports_how_many_screens_are_still_below(self):
        import asyncio
        import json

        cls, server = self._server_at(844, 4220, self._page([{"index": 1}]))

        state = json.loads(asyncio.run(cls._get_browser_state(server, False)))

        # The number the agent acts on. Pixels would need a viewport height it
        # has to find somewhere else before they mean anything.
        assert state["pages_below"] == 3.0
        assert state["pages_above"] == 1.0
        assert state["scroll"] == {"x": 0, "y": 844}
        # Upstream's own content is carried through untouched.
        assert state["interactive_elements"] == [{"index": 1}]

    def test_a_page_that_fits_reports_nothing_below(self):
        import asyncio
        import json

        cls, server = self._server_at(0, 800, self._page())

        state = json.loads(asyncio.run(cls._get_browser_state(server, False)))

        assert state["pages_below"] == 0.0

    def test_an_upstream_error_is_passed_through_unchanged(self):
        """It is not this method's job to rewrite "no session active"."""
        import asyncio

        cls, server = self._server_at(0, 800, "Error: No browser session active")

        assert (
            asyncio.run(cls._get_browser_state(server, False))
            == "Error: No browser session active"
        )


class TestScrollingToSomethingRatherThanTowardsIt:
    """A target has to cost one call, not one call per screen.

    `browser_scroll` moved 500 pixels and took only a direction. Measured on
    conversation 6aa229b4 (2026-08-27): the run scrolled to y=2500 on an
    authenticated page, watched `pages_below` rise from 4.0 to 9.7 while it did
    -- the page grows as it loads -- wrote "I haven't reached the panel yet",
    and opened a pull request against a component it never saw. The edit was a
    guess read off the source.
    """

    @staticmethod
    def _server(evaluate_result):
        import types

        from openhands.tools.browser_use.server import CustomBrowserUseServer

        seen: dict = {}

        class Page:
            async def evaluate(self, script, arg):
                seen["script"], seen["arg"] = script, arg
                return evaluate_result

        server = CustomBrowserUseServer.__new__(CustomBrowserUseServer)
        server.browser_session = cast(
            Any, types.SimpleNamespace(get_current_page=lambda: _coro(Page()))
        )
        return CustomBrowserUseServer, server, seen

    def test_the_target_is_reached_in_one_call_and_centred(self):
        import asyncio

        cls, server, seen = self._server("Noteworthy Insights")

        result = asyncio.run(cls._scroll_to_text(server, "Noteworthy"))

        assert "Noteworthy Insights" in result
        assert seen["arg"] == "Noteworthy"
        # Centred, not merely brought to an edge, where a sticky header covers
        # it and the capture shows the wrong thing.
        assert "block: 'center'" in seen["script"]
        # What the page shows, so a match cannot come from a hidden node.
        assert "innerText" in seen["script"]

    def test_it_ignores_text_the_page_never_draws(self):
        """A framework's data island holds the very words being looked for.

        Measured in production 2026-08-27 on conversation a735c192, the first
        run to have this tool: it asked for "Noteworthy Insights" and was
        scrolled to Next.js's `__NEXT_DATA__` blob, whose JSON mentions the
        heading, at the bottom of the document. `innerText` on a `<script>`
        falls back to `textContent`, so the blob matched and -- being last in
        the wrapper both share -- won the walk.
        """
        import asyncio

        cls, server, seen = self._server("Noteworthy Insights")

        asyncio.run(cls._scroll_to_text(server, "Noteworthy"))

        # Run the selector the page is actually given, against a document shaped
        # like the one that failed: a wrapper holding both the heading and, last,
        # the data blob. Asserting on the script's text would pass against a
        # rule that never fires.
        script = seen["script"]
        chosen = _pick_in_fake_dom(script, "Noteworthy")
        assert chosen == "H2#real", chosen

    def test_not_finding_it_names_what_to_do_instead(self):
        """The run gets one turn on this text, so it has to carry the remedy."""
        import asyncio

        cls, server, _ = self._server(None)

        result = asyncio.run(cls._scroll_to_text(server, "Nowhere"))

        for remedy in ("loaded", "tab", "browser_get_content"):
            assert remedy in result

    def test_a_target_wins_over_a_direction(self):
        """Naming what you are looking for already says which way to go."""
        from openhands.tools.browser_use.definition import BrowserScrollAction

        assert "to_text" in BrowserScrollAction.model_fields
        # `browser_sequence` reuses the same schema, so a sequenced scroll can
        # take a target too rather than only the standalone tool.
        from openhands.tools.browser_use.definition import _sequence_step_actions

        assert "to_text" in _sequence_step_actions()["scroll"].model_fields


async def _coro(value):
    return value


def _pick_in_fake_dom(selector_source: str, needle: str) -> str:
    """Which element the injected selector picks, on the DOM that broke it.

    A tiny stand-in for the page: the wrapper contains the heading and then the
    `__NEXT_DATA__` script, and `innerText` on a script falls back to its text
    because it is never rendered -- which is exactly why it matched. Only the
    rendered node reports client rects.
    """
    import re

    uses_client_rects = bool(
        re.search(r"getClientRects\(\)\.length\s*===\s*0", selector_source)
    )

    class Node:
        def __init__(self, name, text, rects, children=()):
            self.name, self.text, self.rects = name, text, rects
            self.children = list(children)

        def contains(self, other):
            return other is not self and any(
                child is other or child.contains(other) for child in self.children
            )

        def walk(self):
            for child in self.children:
                yield child
                yield from child.walk()

    heading = Node("H2#real", "Noteworthy Insights", 1)
    blob = Node("SCRIPT#__NEXT_DATA__", '{"panel":"Noteworthy Insights"}', 0)
    # The blob first, which is the order that loses: the walk keeps the last
    # element the current best still contains, so an unrendered node reached
    # while the best is still their shared wrapper takes it and the heading
    # after it never can. Verified against a real Chrome DOM on 2026-08-27 --
    # in this order the unguarded selector returns the script.
    wrapper = Node("DIV#__next", "Noteworthy Insights", 1, [blob, heading])
    body = Node("BODY", "Noteworthy Insights", 1, [wrapper])

    best = None
    for node in body.walk():
        if uses_client_rects and node.rects == 0:
            continue
        if needle.lower() not in node.text.lower():
            continue
        if best is None or best.contains(node):
            best = node
    return best.name if best else "none"
