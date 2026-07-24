"""Tests for Agent._extract_summary method."""

from unittest.mock import Mock

import mcp.types
import pytest
from pydantic import SecretStr

from openhands.sdk.agent import Agent
from openhands.sdk.llm import LLM
from openhands.sdk.mcp.client import MCPClient
from openhands.sdk.mcp.tool import MCPToolDefinition


@pytest.fixture
def agent():
    """Create a test agent."""
    return Agent(
        llm=LLM(
            usage_id="test-llm",
            model="test-model",
            api_key=SecretStr("test-key"),
            base_url="http://test",
        )
    )


@pytest.mark.parametrize(
    "summary_value,expected_result",
    [
        # Valid summary provided - use it
        ("testing file system", "testing file system"),
        # No summary provided - generate readable default
        (None, "test_tool: value"),
        # Non-string summary - generate readable default
        (123, "test_tool: value"),
        # Empty or whitespace-only - generate readable default
        ("", "test_tool: value"),
        ("   ", "test_tool: value"),
    ],
)
def test_extract_summary(agent, summary_value, expected_result):
    """Test _extract_summary method with various scenarios."""
    arguments = {"some_param": "value"}
    if summary_value is not None:
        arguments["summary"] = summary_value

    result = agent._extract_summary("test_tool", arguments)
    assert result == expected_result
    assert "summary" not in arguments


def test_default_summary_prefers_descriptive_argument(agent):
    """The fallback prefers well-known descriptive keys over dict order."""
    arguments = {"timeout": 100, "other": "ignored", "command": "git status"}

    result = agent._extract_summary("terminal", arguments)

    assert result == "terminal: git status"


def test_default_summary_never_dumps_raw_argument_json(agent):
    """The fallback must not leak the raw argument JSON into the summary."""
    arguments = {"command": "for i in $(seq 1 18); do sleep 5; done", "timeout": 100}

    result = agent._extract_summary("terminal", arguments)

    assert result == "terminal: for i in $(seq 1 18); do sleep 5; done"
    assert "{" not in result


def test_default_summary_clips_long_values_and_collapses_whitespace(agent):
    """Long or multi-line argument values are collapsed and truncated."""
    arguments = {"command": "echo start\n" + "x" * 200}

    result = agent._extract_summary("terminal", arguments)

    assert result.startswith("terminal: echo start x")
    assert result.endswith("…")
    assert len(result) <= len("terminal: ") + 80


def test_default_summary_without_string_arguments_uses_tool_name(agent):
    """With no usable string argument, the fallback is just the tool name."""
    result = agent._extract_summary("browser_get_state", {"include_screenshot": True})

    assert result == "browser_get_state"


def _make_mcp_tool_with_summary():
    """Create an MCP tool whose inputSchema declares 'summary' as required."""
    mcp_tool = mcp.types.Tool(
        name="jira_create_issue",
        description="Create a Jira issue",
        inputSchema={
            "type": "object",
            "properties": {
                "project_key": {"type": "string"},
                "summary": {"type": "string", "description": "Ticket title"},
                "issue_type": {"type": "string"},
            },
            "required": ["project_key", "summary", "issue_type"],
        },
    )
    client = Mock(spec=MCPClient)
    return MCPToolDefinition.create(mcp_tool, client)[0]


def test_extract_summary_preserves_mcp_tool_summary_param(agent):
    """_extract_summary must NOT pop 'summary' when the tool declares it."""
    tool = _make_mcp_tool_with_summary()
    arguments = {
        "project_key": "PROJ",
        "summary": "My ticket title",
        "issue_type": "Task",
    }

    result = agent._extract_summary(tool.name, arguments, tool=tool)

    # The tool's real "summary" value must remain in the dict
    assert arguments["summary"] == "My ticket title"
    # The tool's own summary value is reused as the event-level summary
    # (e.g. a Jira ticket title is descriptive enough for visualization)
    assert result == "My ticket title"


def test_mcp_tool_with_summary_param_roundtrip(agent):
    """End-to-end: summary must survive extraction and action validation."""
    tool = _make_mcp_tool_with_summary()
    arguments = {
        "project_key": "PROJ",
        "summary": "My ticket title",
        "issue_type": "Task",
    }

    # This is the exact call sequence from _get_action_event
    _summary = agent._extract_summary(tool.name, arguments, tool=tool)
    action = tool.action_from_arguments(arguments)

    # action_from_arguments should succeed (not raise ValidationError)
    assert action.data["summary"] == "My ticket title"
    assert action.data["project_key"] == "PROJ"


def test_extract_summary_mcp_tool_summary_missing_falls_back(agent):
    """When tool declares 'summary' but it's empty, fall back to default."""
    tool = _make_mcp_tool_with_summary()
    arguments = {
        "project_key": "PROJ",
        "summary": "",
        "issue_type": "Task",
    }

    result = agent._extract_summary(tool.name, arguments, tool=tool)

    # Empty summary → falls back to a readable default that skips the
    # empty tool-declared "summary" value
    assert result == "jira_create_issue: PROJ"
    # The empty value must still remain in arguments
    assert arguments["summary"] == ""


def test_extract_summary_still_pops_for_tools_without_summary_param(agent):
    """For tools that don't declare 'summary', it's still popped as meta."""
    mcp_tool = mcp.types.Tool(
        name="some_tool",
        description="A tool without a summary param",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
    )
    client = Mock(spec=MCPClient)
    tool = MCPToolDefinition.create(mcp_tool, client)[0]

    arguments = {"url": "https://example.com", "summary": "Fetch example"}
    result = agent._extract_summary(tool.name, arguments, tool=tool)

    assert result == "Fetch example"
    assert "summary" not in arguments
