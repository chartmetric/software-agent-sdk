"""A recorded action naming a tool this agent does not have must not raise.

`step()` begins by executing unmatched actions straight from the conversation
history, so an action recorded by an earlier agent -- with a different tool set
-- reaches `_execute_action_event` having never passed the check in
`_handle_tool_call`. Raising there told the model "this should not happen as it
was checked earlier", which names no tool it can call, and left the action
unmatched so the next turn picked it up again.

Measured on a production conversation (2026-08-27) whose execution run had died
mid-flight: fifteen `terminal`, `task` and `read_pull_request_reviews` actions
were left without observations, and the next control turn -- which has none of
those tools -- spent itself raising on them instead of answering. The answer came
from a fallback.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from litellm import ChatCompletionMessageToolCall
from litellm.types.utils import Function
from pydantic import SecretStr

from openhands.sdk.agent import Agent
from openhands.sdk.conversation import Conversation, LocalConversation
from openhands.sdk.event import ActionEvent
from openhands.sdk.event.error_classification import AGENT_OUTCOME
from openhands.sdk.event.llm_convertible import AgentErrorEvent
from openhands.sdk.llm import LLM, TextContent
from openhands.sdk.llm.message import MessageToolCall
from openhands.sdk.tool import Action, Observation, Tool, ToolExecutor, register_tool
from openhands.sdk.tool.tool import ToolDefinition


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class _KeptAction(Action):
    value: str = ""


class _KeptObservation(Observation):
    result: str = ""


class _KeptExecutor(ToolExecutor[_KeptAction, _KeptObservation]):
    def __call__(self, action: _KeptAction, conversation=None) -> _KeptObservation:
        return _KeptObservation(result=action.value)


class _KeptTool(ToolDefinition[_KeptAction, _KeptObservation]):
    name = "kept_tool"

    @classmethod
    def create(cls, conv_state: "ConversationState | None" = None) -> Sequence[Self]:
        return [
            cls(
                description="A tool this agent does have",
                action_type=_KeptAction,
                observation_type=_KeptObservation,
                executor=_KeptExecutor(),
            )
        ]


register_tool("KeptTool", _KeptTool)


def _agent_and_conversation() -> tuple[Agent, LocalConversation]:
    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(llm=llm, tools=[Tool(name="KeptTool")])
    conversation = Conversation(agent=agent, callbacks=[])
    # The conversation holds its own copy, and only builds its tools map when it
    # first runs -- do that here so `tools_map` is populated.
    conversation.agent._initialize(conversation.state)
    assert isinstance(conversation.agent, Agent)
    return conversation.agent, conversation


def _recorded_action(tool_name: str) -> ActionEvent:
    """An action for `tool_name`, in the shape the conversation log records one."""
    call = MessageToolCall.from_chat_tool_call(
        ChatCompletionMessageToolCall(
            id="call_recorded_earlier",
            type="function",
            function=Function(name=tool_name, arguments="{}"),
        )
    )
    return ActionEvent(
        source="agent",
        thought=[TextContent(text="recorded by an earlier agent")],
        action=_KeptAction(value="x"),
        tool_name=tool_name,
        tool_call_id=call.id,
        tool_call=call,
        llm_response_id="response_recorded_earlier",
    )


def test_a_tool_this_agent_does_not_have_is_an_error_event_not_a_raise() -> None:
    agent, conversation = _agent_and_conversation()

    events = agent._execute_action_event(conversation, _recorded_action("terminal"))

    assert len(events) == 1
    error = events[0]
    assert isinstance(error, AgentErrorEvent)
    # Matched to the action, so the next turn does not pick it up again.
    assert error.tool_call_id == "call_recorded_earlier"
    # The agent's outcome, not an internal fault it can do nothing about.
    assert error.classification == AGENT_OUTCOME


def test_the_error_names_the_tools_the_agent_does_have() -> None:
    agent, conversation = _agent_and_conversation()

    events = agent._execute_action_event(conversation, _recorded_action("terminal"))

    assert isinstance(events[0], AgentErrorEvent)
    message = events[0].error
    # A refusal that names no alternative spends the turn rediscovering one.
    assert "kept_tool" in message
    assert "should not happen" not in message
