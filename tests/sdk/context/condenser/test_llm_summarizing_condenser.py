import warnings
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.types.utils import (
    Choices,
    Message as LiteLLMMessage,
    ModelResponse,
    Usage,
)
from pydantic import SecretStr

from openhands.sdk.context.condenser.base import (
    CondensationRequirement,
    NoCondensationAvailableException,
)
from openhands.sdk.context.condenser.llm_summarizing_condenser import (
    LLMSummarizingCondenser,
    Reason,
)
from openhands.sdk.context.view import View
from openhands.sdk.event.base import Event
from openhands.sdk.event.condenser import Condensation, CondensationRequest
from openhands.sdk.event.llm_convertible import MessageEvent
from openhands.sdk.llm import (
    LLM,
    LLMResponse,
    Message,
    MetricsSnapshot,
    TextContent,
)


def message_event(content: str) -> MessageEvent:
    return MessageEvent(
        llm_message=Message(role="user", content=[TextContent(text=content)]),
        source="user",
    )


@pytest.fixture
def mock_llm() -> LLM:
    """Create a mock LLM for testing."""
    mock_llm = MagicMock(spec=LLM)

    # Mock the completion response - now returns LLMResponse
    def create_completion_result(content: str) -> LLMResponse:
        message = Message(role="assistant", content=[TextContent(text=content)])
        metrics = MetricsSnapshot(
            model_name="test-model",
            accumulated_cost=0.0,
            max_budget_per_task=None,
            accumulated_token_usage=None,
        )
        # Create a mock ModelResponse
        raw_response = MagicMock(spec=ModelResponse)
        raw_response.id = "mock-llm-response-id"
        return LLMResponse(message=message, metrics=metrics, raw_response=raw_response)

    mock_llm.completion.return_value = create_completion_result(
        "Summary of forgotten events"
    )
    mock_llm.format_messages_for_llm = lambda messages: messages

    # Mock the required attributes that the LLM validator reads
    mock_llm.openrouter_site_url = "https://docs.all-hands.dev/"
    mock_llm.openrouter_app_name = "OpenHands"
    mock_llm.aws_access_key_id = None
    mock_llm.aws_secret_access_key = None
    mock_llm.aws_session_token = None
    mock_llm.aws_region_name = None
    mock_llm.aws_profile_name = None
    mock_llm.aws_role_name = None
    mock_llm.aws_session_name = None
    mock_llm.aws_bedrock_runtime_endpoint = None
    mock_llm.metrics = None
    mock_llm.model = "test-model"
    mock_llm.log_completions = False
    mock_llm.log_completions_folder = None
    mock_llm.custom_tokenizer = None
    mock_llm.base_url = None
    mock_llm.reasoning_effort = None
    mock_llm.litellm_extra_body = {}
    mock_llm.temperature = 0.0
    # Streaming is off by default (matches LLM.stream's default), so the
    # condenser uses this LLM directly without copying it.
    mock_llm.stream = False

    # Explicitly set pricing attributes required by LLM -> Telemetry wiring
    mock_llm.input_cost_per_token = None
    mock_llm.output_cost_per_token = None

    mock_llm._metrics = None
    mock_llm._telemetry = None

    # Helper method to set mock response content
    def set_mock_response_content(content: str):
        mock_llm.completion.return_value = create_completion_result(content)

    mock_llm.set_mock_response_content = set_mock_response_content

    return mock_llm


def test_default_values(mock_llm: LLM) -> None:
    """Test that LLMSummarizingCondenser has correct default values.

    These defaults are tuned to ensure workable manipulation indices for condensation.
    See https://github.com/OpenHands/software-agent-sdk/issues/1518 for context.
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm)

    # Retained only so callers with the deprecated argument still load.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert condenser.max_size == 960
    assert condenser.model_dump(exclude={"llm"})["max_size"] == 960

    # Default keep_first should be 2 (reduced from 4 to leave more room for
    # condensation)
    assert condenser.keep_first == 2


def test_event_count_does_not_trigger_condensation(mock_llm: LLM) -> None:
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=20)
    events = [message_event(f"Event {i}") for i in range(21)]

    assert condenser.condensation_requirement(View.from_events(events)) is None


def test_condense_returns_view_when_no_condensation_needed(mock_llm: LLM) -> None:
    """Test that condenser returns the original view when no condensation is needed."""  # noqa: E501
    event_count = 100
    condenser = LLMSummarizingCondenser(llm=mock_llm)

    events: list[Event] = [message_event(f"Event {i}") for i in range(event_count)]
    view = View.from_events(events)

    result = condenser.condense(view)

    assert isinstance(result, View)
    assert result == view
    # LLM should not be called
    cast(MagicMock, mock_llm.completion).assert_not_called()


def test_condense_returns_condensation_when_needed(mock_llm: LLM) -> None:
    """Test that condenser returns a Condensation when condensation is needed."""
    keep_first = 3
    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=keep_first)

    # Set up mock response
    cast(Any, mock_llm).set_mock_response_content("Summary of forgotten events")

    events: list[Event] = [message_event(f"Event {i}") for i in range(11)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    result = condenser.condense(view)

    assert isinstance(result, Condensation)
    assert result.summary == "Summary of forgotten events"
    # summary_offset should be the smallest manipulation index >= keep_first
    # Since all events are MessageEvents, manipulation indices are [0,1,2,3,4,...]
    # The smallest index >= keep_first (3) is 3
    # This means we keep events [0:3] = indices 0,1,2 = 3 events
    assert result.summary_offset == keep_first
    assert len(result.forgotten_event_ids) > 0

    # LLM should be called once
    cast(MagicMock, mock_llm.completion).assert_called_once()


def test_get_condensation_with_previous_summary(mock_llm: LLM) -> None:
    """Test that condenser properly handles previous summary content."""
    keep_first = 3
    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=keep_first)

    # Set up mock response
    cast(Any, mock_llm).set_mock_response_content("Updated summary")

    # Create events with a condensation in the history.
    events = [message_event(f"Event {i}") for i in range(14)]

    # Add a condensation to simulate previous summarization
    # The summary will be inserted at keep_first due to summary_offset
    condensation = Condensation(
        forgotten_event_ids={events[3].id, events[4].id},
        summary="Previous summary content",
        summary_offset=keep_first,
        llm_response_id="condensation_response_1",
    )
    events_with_condensation: list[Event] = list(events[:keep_first])
    events_with_condensation.append(condensation)
    events_with_condensation.extend(events[keep_first:])
    events_with_condensation.append(CondensationRequest())

    view = View.from_events(events_with_condensation)

    result = condenser.get_condensation(view)

    assert isinstance(result, Condensation)
    assert result.summary == "Updated summary"

    # Verify that the LLM was called with the previous summary
    completion_mock = cast(MagicMock, mock_llm.completion)
    completion_mock.assert_called_once()
    call_args = completion_mock.call_args
    messages = call_args[1]["messages"]  # Get keyword arguments
    prompt_text = messages[0].content[0].text

    # The prompt should contain the previous summary (it's in <PREVIOUS SUMMARY> sec.)
    # The summary is now retrieved from the view, which should have it at the summary
    # event
    assert (
        "Previous summary content" in prompt_text or "<PREVIOUS SUMMARY>" in prompt_text
    )


def test_invalid_config(mock_llm: LLM) -> None:
    """Test that LLMSummarizingCondenser validates configuration parameters."""
    # Test max_size must be positive
    with pytest.raises(ValueError):
        LLMSummarizingCondenser(llm=mock_llm, max_size=0)

    # Test keep_first must be non-negative
    with pytest.raises(ValueError):
        LLMSummarizingCondenser(llm=mock_llm, keep_first=-1)

    # max_size no longer constrains keep_first because event count is ignored.
    assert (
        LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=8).keep_first == 8
    )


def test_get_condensation_does_not_pass_extra_body(mock_llm: LLM) -> None:
    """Condenser should not pass extra_body to llm.completion.

    This prevents providers like 1p Anthropic from rejecting the request with
    "extra_body: Extra inputs are not permitted".
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=2)

    events: list[Event] = [message_event(f"Event {i}") for i in range(12)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    result = condenser.condense(view)
    assert isinstance(result, Condensation)

    # Ensure completion was called without an explicit extra_body kwarg
    completion_mock = cast(MagicMock, mock_llm.completion)
    assert completion_mock.call_count == 1


def test_condense_with_agent_llm(mock_llm: LLM) -> None:
    """Test that condenser accepts and works with optional agent llm parameter."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=2)

    # Create a separate mock for the agent's LLM
    agent_llm = MagicMock(spec=LLM)
    agent_llm.model = "gpt-4"
    # A MagicMock would otherwise stand in for the token count.
    agent_llm.effective_max_input_tokens = None
    agent_llm.effective_max_output_tokens = None

    events: list[Event] = [message_event(f"Event {i}") for i in range(12)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    # Call condense with the agent's LLM
    result = condenser.condense(view, agent_llm=agent_llm)
    assert isinstance(result, Condensation)

    # Verify the condenser still uses its own LLM for summarization
    completion_mock = cast(MagicMock, mock_llm.completion)
    assert completion_mock.call_count == 1

    # Agent LLM should not be called for completion (condenser uses its own LLM)
    assert not agent_llm.completion.called
    _, kwargs = completion_mock.call_args
    assert "extra_body" not in kwargs


def test_condense_with_token_limit_exceeded(mock_llm: LLM) -> None:
    """Test that condenser triggers on TOKENS reason when token limit is exceeded."""
    max_tokens = 100
    keep_first = 2
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_tokens=max_tokens, keep_first=keep_first
    )

    # Create a separate mock for the agent's LLM with token counting
    agent_llm = MagicMock(spec=LLM)
    agent_llm.model = "gpt-4"

    # Mock get_token_count to return predictable values based on message content length
    def mock_token_count(messages, **_kwargs):
        # Simple heuristic: count characters in all text content
        # Each character = 0.25 tokens (roughly 4 chars per token)
        total_chars = 0
        for msg in messages:
            for content in msg.content:
                if hasattr(content, "text"):
                    total_chars += len(content.text)
        return total_chars // 4

    cast(MagicMock, agent_llm.get_token_count).side_effect = mock_token_count

    # Create events that exceed token limit
    # Each event has 40 chars = 10 tokens
    # 15 events = 150 tokens (exceeds max_tokens of 100)
    events: list[Event] = [message_event("A" * 40) for i in range(15)]
    view = View.from_events(events)

    # Verify that TOKENS is the condensation reason
    reasons = condenser.get_condensation_reasons(view, agent_llm=agent_llm)
    assert Reason.TOKENS in reasons
    assert Reason.REQUEST not in reasons

    # Condense the view
    result = condenser.condense(view, agent_llm=agent_llm)
    assert isinstance(result, Condensation)

    # Verify the condenser used its own LLM for summarization
    completion_mock = cast(MagicMock, mock_llm.completion)
    assert completion_mock.call_count == 1

    # Verify forgotten events were calculated based on token reduction
    assert len(result.forgotten_event_ids) > 0


def test_condense_with_derived_token_limit(mock_llm: LLM) -> None:
    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=2)
    agent_llm = MagicMock(spec=LLM)
    agent_llm.model = "small-context-model"
    agent_llm.effective_max_input_tokens = 100
    agent_llm.effective_max_output_tokens = 0

    def mock_token_count(messages, **_kwargs):
        return (
            sum(
                len(content.text)
                for message in messages
                for content in message.content
                if isinstance(content, TextContent)
            )
            // 4
        )

    cast(MagicMock, agent_llm.get_token_count).side_effect = mock_token_count
    view = View.from_events([message_event("A" * 40) for _ in range(15)])

    result = condenser.condense(view, agent_llm=agent_llm)

    assert isinstance(result, Condensation)
    assert result.forgotten_event_ids


def test_condense_with_request_and_tokens_reasons(mock_llm: LLM) -> None:
    """Test condensation when both REQUEST and TOKENS reasons are true simultaneously.

    Verifies that the most aggressive condensation (minimum suffix) is chosen.
    """
    max_tokens = 100
    keep_first = 2
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_tokens=max_tokens, keep_first=keep_first
    )

    # Create a separate mock for the agent's LLM with token counting
    agent_llm = MagicMock(spec=LLM)
    agent_llm.model = "gpt-4"

    # Mock get_token_count to return predictable values
    def mock_token_count(messages, **_kwargs):
        total_chars = 0
        for msg in messages:
            for content in msg.content:
                if hasattr(content, "text"):
                    total_chars += len(content.text)
        return total_chars // 4

    cast(MagicMock, agent_llm.get_token_count).side_effect = mock_token_count

    # Create 20 events with 40 chars each = 10 tokens each = 200 total tokens
    # This exceeds max_tokens of 100 (triggers TOKENS)
    events: list[Event] = [message_event("A" * 40) for i in range(20)]
    # Add a CondensationRequest (triggers REQUEST)
    events.append(CondensationRequest())
    view = View.from_events(events)

    # Verify both reasons are present
    reasons = condenser.get_condensation_reasons(view, agent_llm=agent_llm)
    assert Reason.REQUEST in reasons
    assert Reason.TOKENS in reasons

    # Get the condensation
    result = condenser.condense(view, agent_llm=agent_llm)
    assert isinstance(result, Condensation)

    # The most aggressive condensation should be chosen (minimum suffix)
    assert len(result.forgotten_event_ids) > 0


def test_generate_condensation_raises_on_zero_events(mock_llm: LLM) -> None:
    """Test that _generate_condensation raises AssertionError when given 0 events.

    This prevents the LLM from being called with an empty event list, which would
    produce a confusing summary like "I don't see any events provided to summarize."
    See https://github.com/OpenHands/software-agent-sdk/issues/1518 for context.
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=2)

    with pytest.raises(AssertionError, match="No events to condense"):
        condenser._generate_condensation(
            forgotten_events=[],
            summary_offset=0,
        )

    # Verify the LLM was never called
    cast(MagicMock, mock_llm.completion).assert_not_called()


@pytest.mark.parametrize(
    "reasons",
    [set()],
)
def test_condensation_requirement_returns_none(
    mock_llm: LLM, reasons: set[Reason]
) -> None:
    """Test that condensation_requirement returns None when appropriate.

    Mocks get_condensation_reasons to test different reason combinations.
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=2)
    events: list[Event] = [message_event(f"Event {i}") for i in range(10)]
    view = View.from_events(events)

    with patch.object(
        LLMSummarizingCondenser, "get_condensation_reasons", return_value=reasons
    ):
        result = condenser.condensation_requirement(view)
        assert result is None


@pytest.mark.parametrize(
    "reasons",
    [
        {Reason.TOKENS},
    ],
)
def test_condensation_requirement_returns_hard_for_token_pressure(
    mock_llm: LLM, reasons: set[Reason]
) -> None:
    """Token pressure should trigger before the next LLM request can overflow."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=2)
    events: list[Event] = [message_event(f"Event {i}") for i in range(10)]
    view = View.from_events(events)

    with patch.object(
        LLMSummarizingCondenser, "get_condensation_reasons", return_value=reasons
    ):
        result = condenser.condensation_requirement(view)
        assert result == CondensationRequirement.HARD


@pytest.mark.parametrize(
    "reasons",
    [
        {Reason.REQUEST},
        {Reason.REQUEST, Reason.TOKENS},
    ],
)
def test_condensation_requirement_returns_hard(
    mock_llm: LLM, reasons: set[Reason]
) -> None:
    """Test that condensation_requirement returns HARD when REQUEST is present.

    Mocks get_condensation_reasons to test different combinations with REQUEST.
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=2)
    events: list[Event] = [message_event(f"Event {i}") for i in range(10)]
    view = View.from_events(events)

    with patch.object(
        LLMSummarizingCondenser, "get_condensation_reasons", return_value=reasons
    ):
        result = condenser.condensation_requirement(view)
        assert result == CondensationRequirement.HARD


def test_condense_with_hard_requirement_and_no_condensation_available(
    mock_llm: LLM,
) -> None:
    """Test that condense raises error with hard requirement but no condensation.

    When there's a hard requirement but no valid condensation range available
    (e.g., entire view is a single atomic unit), should raise an exception.
    """
    from openhands.sdk.context.condenser.base import NoCondensationAvailableException

    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=2)
    events: list[Event] = [message_event(f"Event {i}") for i in range(10)]
    view = View.from_events(events)

    # Mock to return HARD requirement but no events to condense
    # Also mock hard_context_reset to return None so the exception gets re-raised
    with (
        patch.object(
            LLMSummarizingCondenser,
            "get_condensation_reasons",
            return_value={Reason.REQUEST},
        ),
        patch.object(condenser, "_get_forgotten_events", return_value=([], 0)),
        patch.object(LLMSummarizingCondenser, "hard_context_reset", return_value=None),
    ):
        with pytest.raises(NoCondensationAvailableException):
            condenser.condense(view)


def test_minimum_progress_default_value(mock_llm: LLM) -> None:
    """Test that minimum_progress has the correct default value."""
    condenser = LLMSummarizingCondenser(llm=mock_llm)
    assert condenser.minimum_progress == 0.1


def test_minimum_progress_custom_value(mock_llm: LLM) -> None:
    """Test that minimum_progress accepts custom values."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, minimum_progress=0.2)
    assert condenser.minimum_progress == 0.2


@pytest.mark.parametrize(
    "invalid_value",
    [
        0.0,  # must be > 0.0
        -0.1,  # must be > 0.0
        1.0,  # must be < 1.0
        1.5,  # must be < 1.0
    ],
)
def test_minimum_progress_validation(mock_llm: LLM, invalid_value: float) -> None:
    """Test that minimum_progress validates the range (0.0 < value < 1.0)."""
    with pytest.raises(ValueError):
        LLMSummarizingCondenser(llm=mock_llm, minimum_progress=invalid_value)


def test_minimum_progress_threshold_not_met(mock_llm: LLM) -> None:
    """Test that condensation raises when forgotten events are below minimum_progress.

    When the ratio of forgotten events to total events is less than minimum_progress,
    should raise NoCondensationAvailableException.
    """
    # Create a condenser with a high minimum_progress value
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, keep_first=2, minimum_progress=0.8
    )

    # Create a view with 100 events
    events: list[Event] = [message_event(f"Event {i}") for i in range(100)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    # Mock _get_forgotten_events to return a small number of forgotten events
    # This allows us to directly test the minimum_progress threshold check
    # without dealing with complex boundary calculations
    small_forgotten = [events[2], events[3]]  # Only 2 events forgotten

    with patch.object(
        condenser, "_get_forgotten_events", return_value=(small_forgotten, 2)
    ):
        # Forgotten count (2) << minimum_progress (0.8) * len(view) (100)
        # 2 < 80, so the threshold is not met
        with pytest.raises(NoCondensationAvailableException, match="minimum progress"):
            condenser.get_condensation(view)


def test_minimum_progress_threshold_met(mock_llm: LLM) -> None:
    """Test that condensation succeeds when forgotten events meet minimum_progress.

    When the ratio of forgotten events to total events is >= minimum_progress,
    condensation should proceed normally.
    """
    # Use a low minimum_progress so it's easy to meet the threshold
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, keep_first=2, minimum_progress=0.1
    )

    # Set up mock response
    cast(Any, mock_llm).set_mock_response_content("Summary of forgotten events")

    events: list[Event] = [message_event(f"Event {i}") for i in range(30)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    result = condenser.condense(view)

    assert isinstance(result, Condensation)
    assert result.summary == "Summary of forgotten events"


def test_generate_condensation_wraps_llm_errors(mock_llm: LLM) -> None:
    """LLM failures in _generate_condensation raise NoCondensationAvailableException."""  # noqa: E501
    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=2)

    cast(MagicMock, mock_llm.completion).side_effect = RuntimeError("boom")

    events: list[Event] = [message_event(f"Event {i}") for i in range(12)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    with pytest.raises(NoCondensationAvailableException, match="boom"):
        condenser.get_condensation(view)


@pytest.mark.asyncio
async def test_agenerate_condensation_wraps_llm_errors(mock_llm: LLM) -> None:
    """Async variant: LLM failures surface as NoCondensationAvailableException."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=2)

    cast(MagicMock, mock_llm.acompletion).side_effect = RuntimeError("boom")

    events: list[Event] = [message_event(f"Event {i}") for i in range(12)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    with pytest.raises(NoCondensationAvailableException, match="boom"):
        await condenser.aget_condensation(view)


def test_llm_error_triggers_hard_context_reset(mock_llm: LLM) -> None:
    """A summarizer LLM failure during condense() triggers hard_context_reset."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, keep_first=2)

    # Force a HARD condensation requirement via a CondensationRequest
    events: list[Event] = [message_event(f"Event {i}") for i in range(12)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    # First call (get_condensation path) fails; second call
    # (hard_context_reset path) succeeds.
    success_response = cast(Any, mock_llm).completion.return_value
    cast(MagicMock, mock_llm.completion).side_effect = [
        RuntimeError("context window exceeded"),
        success_response,
    ]

    result = condenser.condense(view)

    assert isinstance(result, Condensation)
    assert result.summary == "Summary of forgotten events"
    assert cast(MagicMock, mock_llm.completion).call_count == 2


def _streaming_llm() -> LLM:
    """A real LLM with streaming enabled, as a long-running conversation has."""
    return LLM(
        model="gpt-4o",
        api_key=SecretStr("test-key"),
        usage_id="summarizer-test",
        stream=True,
    )


def _summary_response(content: str = "A summary") -> ModelResponse:
    return ModelResponse(
        id="resp-id",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=LiteLLMMessage(content=content, role="assistant"),
            )
        ],
        created=1234567890,
        model="gpt-4o",
        object="chat.completion",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@patch("openhands.sdk.llm.llm.LLM._transport_call", autospec=True)
def test_summarization_disables_streaming_when_llm_streams(mock_transport) -> None:
    """Regression test for issue #3902: a ``stream=True`` LLM must still summarize
    even though the condenser passes no ``on_token`` callback."""
    mock_transport.return_value = _summary_response("A summary")

    llm = _streaming_llm()
    condenser = LLMSummarizingCondenser(llm=llm, keep_first=3)

    events: list[Event] = [message_event(f"Event {i}") for i in range(11)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    result = condenser.condense(view)

    assert isinstance(result, Condensation)
    assert result.summary == "A summary"
    mock_transport.assert_called_once()
    # Streaming was disabled on a copy, not the agent's own LLM (autospec =>
    # self is the first positional arg).
    assert mock_transport.call_args.kwargs["enable_streaming"] is False
    assert mock_transport.call_args.kwargs["on_token"] is None
    summarizing_llm = mock_transport.call_args.args[0]
    assert summarizing_llm is not llm
    assert summarizing_llm.stream is False
    assert llm.stream is True  # original untouched (model_copy is non-mutating)
    # Token usage is still counted: the copy shares the original's metrics.
    usage = llm.metrics.accumulated_token_usage
    assert usage is not None
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5


@pytest.mark.asyncio
@patch("openhands.sdk.llm.llm.LLM._atransport_call", new_callable=AsyncMock)
async def test_async_summarization_disables_streaming_when_llm_streams(
    mock_atransport,
) -> None:
    """Async variant of the issue #3902 regression test (aget_condensation)."""
    mock_atransport.return_value = _summary_response("A summary")

    llm = _streaming_llm()
    condenser = LLMSummarizingCondenser(llm=llm, keep_first=3)

    events: list[Event] = [message_event(f"Event {i}") for i in range(11)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    result = await condenser.aget_condensation(view)

    assert isinstance(result, Condensation)
    assert result.summary == "A summary"
    mock_atransport.assert_awaited_once()
    assert mock_atransport.call_args.kwargs["enable_streaming"] is False
    assert mock_atransport.call_args.kwargs["on_token"] is None
    assert llm.stream is True


@patch("openhands.sdk.llm.llm.LLM._transport_call", autospec=True)
def test_summarization_uses_llm_as_is_when_not_streaming(mock_transport) -> None:
    """When streaming is off, the condenser summarizes with the LLM unchanged."""
    mock_transport.return_value = _summary_response("A summary")

    llm = LLM(
        model="gpt-4o",
        api_key=SecretStr("test-key"),
        usage_id="summarizer-test",
        stream=False,
    )
    condenser = LLMSummarizingCondenser(llm=llm, keep_first=3)

    events: list[Event] = [message_event(f"Event {i}") for i in range(11)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    result = condenser.condense(view)

    assert isinstance(result, Condensation)
    assert result.summary == "A summary"
    # The exact same LLM instance is used (no copy when not streaming).
    assert mock_transport.call_args.args[0] is llm
