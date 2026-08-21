"""Regression coverage for durable Agent state ownership."""

import uuid

import pytest

from openhands.sdk import LLM, Agent
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.io import LocalFileStore
from openhands.sdk.workspace import LocalWorkspace


def _agent(model: str) -> Agent:
    return Agent(llm=LLM(model=model, usage_id="default"), tools=[])


def test_resume_without_agent_keeps_persisted_agent(tmp_path):
    file_store = LocalFileStore(str(tmp_path))
    workspace = LocalWorkspace(working_dir=str(tmp_path))
    conversation_id = uuid.uuid4()
    state = ConversationState.create(
        id=conversation_id,
        agent=_agent("model-a"),
        workspace=workspace,
        file_store=file_store,
    )
    state.agent = _agent("model-b")

    reloaded = ConversationState.create(
        id=conversation_id,
        agent=None,
        workspace=workspace,
        file_store=LocalFileStore(str(tmp_path)),
    )

    assert reloaded.agent.llm.model == "model-b"


def test_resume_with_explicit_agent_preserves_override_behavior(tmp_path):
    file_store = LocalFileStore(str(tmp_path))
    workspace = LocalWorkspace(working_dir=str(tmp_path))
    conversation_id = uuid.uuid4()
    ConversationState.create(
        id=conversation_id,
        agent=_agent("model-a"),
        workspace=workspace,
        file_store=file_store,
    )

    reloaded = ConversationState.create(
        id=conversation_id,
        agent=_agent("model-c"),
        workspace=workspace,
        file_store=LocalFileStore(str(tmp_path)),
    )

    assert reloaded.agent.llm.model == "model-c"


def test_new_conversation_still_requires_agent(tmp_path):
    with pytest.raises(ValueError, match="agent is required"):
        ConversationState.create(
            id=uuid.uuid4(),
            agent=None,
            workspace=LocalWorkspace(working_dir=str(tmp_path)),
            file_store=LocalFileStore(str(tmp_path)),
        )
