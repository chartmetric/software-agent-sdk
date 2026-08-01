from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "server.yml"
AGENT_SERVER_SPEC = (
    REPO_ROOT
    / "openhands-agent-server"
    / "openhands"
    / "agent_server"
    / "agent-server.spec"
)


def test_server_workflow_passes_git_metadata_build_args() -> None:
    """The published agent-server images should embed git metadata."""
    workflow_text = SERVER_WORKFLOW.read_text(encoding="utf-8")

    assert "OPENHANDS_BUILD_GIT_SHA=${{ env.SDK_SHA }}" in workflow_text
    assert "OPENHANDS_BUILD_GIT_REF=${{ env.SDK_REF }}" in workflow_text


def test_server_workflow_publishes_python_image_to_project_ecr_with_oidc() -> None:
    """Production images should not depend on a developer registry token."""
    workflow_text = SERVER_WORKFLOW.read_text(encoding="utf-8")

    assert "id-token: write" in workflow_text
    assert (
        "arn:aws:iam::897744604563:role/cm-pilot-sdk-image-publisher" in workflow_text
    )
    assert (
        "897744604563.dkr.ecr.us-east-1.amazonaws.com/cm-pilot-agent-server"
        in workflow_text
    )
    assert "aws-actions/configure-aws-credentials@" in workflow_text
    assert "aws-actions/amazon-ecr-login@" in workflow_text
    assert "skopeo copy --all" in workflow_text
    assert "ecr_source_sha:" in workflow_text
    assert "inputs.ecr_source_sha != ''" in workflow_text
    assert "${SDK_SHA:0:7}-python" in workflow_text
    assert "${SDK_SHA}-python" in workflow_text
    assert "aws ecr describe-images" in workflow_text
    assert "secrets.GHCR_PAT" not in workflow_text


def test_agent_server_binary_copies_openhands_distribution_metadata() -> None:
    """The frozen binary should preserve OpenHands package metadata."""
    spec_text = AGENT_SERVER_SPEC.read_text(encoding="utf-8")

    for distribution in (
        "openhands-agent-server",
        "openhands-sdk",
        "openhands-tools",
        "openhands-workspace",
    ):
        assert f'*copy_metadata("{distribution}")' in spec_text
