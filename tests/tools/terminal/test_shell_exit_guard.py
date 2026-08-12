"""Guard against a top-level ``exit`` ending the persistent terminal session.

Motivation: the terminal tool keeps one long-lived tmux pane per session. A
command ending in ``exit $code`` terminates that pane, so the executor never
sees a closing prompt and blocks until the action timeout before it can
rebuild the pool. A production conversation sent
``timeout 240 npx tsc ...; code=$?; rm -f cfg; exit $code`` twice: each call
stalled ~240s and lost its result, while the same check re-run without ``exit``
finished in four seconds. This guard catches the command pre-execution and
replies with an actionable hint instead.
"""

import pytest

from openhands.tools.terminal.definition import (
    TerminalAction,
    TerminalObservation,
    looks_like_shell_exiting_command,
)
from openhands.tools.terminal.impl import TerminalExecutor


# --------------------------------------------------------------------------
# Unit tests for the heuristic.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command,expected",
    [
        # The exact shapes observed in production.
        (
            "timeout 240 npx tsc -p tsconfig.agent.json --noEmit; code=$?; "
            "rm -f tsconfig.agent.json; exit $code",
            "exit $code",
        ),
        (
            "git push origin HEAD; push_code=$?; rm -f /tmp/askpass; exit $push_code",
            "exit $push_code",
        ),
        # Bare and literal-status forms.
        ("exit", "exit"),
        ("make test; exit 1", "exit 1"),
        ("logout", "logout"),
        # `&&` / `||` chains still put `exit` in command position.
        ("test -f x && exit 0", "exit 0"),
        ("test -f x || exit 1", "exit 1"),
        # Conditionals run in the current shell, so their `exit` kills it.
        ('if [ "$c" -eq 0 ]; then echo ok; else exit 1; fi', "exit 1; fi"),
        ("for f in a b; do exit 2; done", "exit 2; done"),
        # A brace group is NOT a subshell.
        ("{ echo hi; exit 3; }", "exit 3; }"),
        # A newline separates commands just like `;`.
        ("echo one\nexit 4", "exit 4"),
    ],
)
def test_shell_exiting_commands_are_detected(command: str, expected: str) -> None:
    assert looks_like_shell_exiting_command(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        # A subshell exit does not end the caller's shell.
        "( make test; exit 1 )",
        "( exit 1 ) || echo failed",
        # Command substitution is a subshell too.
        "status=$(bash -c 'exit 3'; echo $?)",
        # Quoted text is data, not a command.
        "echo 'exit 1'",
        'echo "exit $code"',
        "grep -rn 'exit' src/",
        "awk '/x/ { exit }' file",
        # A heredoc body is data.
        "python - <<'PY'\nimport sys\nexit(1)\nPY",
        "cat <<EOF > /tmp/f\nexit 1\nEOF",
        # Words that merely contain or end with "exit".
        "exitcode=1",
        "./exit_handler.sh",
        "echo $?; exit_code=$?",
        # The recommended replacements must stay usable.
        'make test; code=$?; rm -f tmp; test "$code" -eq 0',
        'npx tsc --noEmit; code=$?; if [ "$code" -eq 0 ]; then echo ok; fi',
        # Comments are not commands.
        "# exit 1",
        # Ordinary commands and empty input.
        "ls -la",
        "",
        " ",
    ],
)
def test_safe_commands_are_not_flagged(command: str) -> None:
    assert looks_like_shell_exiting_command(command) is None


# --------------------------------------------------------------------------
# Executor-level integration: exiting command => structured error, no shell.
# --------------------------------------------------------------------------


_SHELL_SENTINEL = "Executor should not reach the shell when the command is rejected"


@pytest.fixture
def executor_without_shell() -> TerminalExecutor:
    """Build a TerminalExecutor without touching the real shell.

    ``__init__`` spins up a real tmux/subprocess session, so it is bypassed and
    both shell-execution paths raise. The guard runs *before* those paths, so a
    safe command must escape the guard and trigger the sentinel — that is how
    we prove the guard did not fire.
    """
    exe = TerminalExecutor.__new__(TerminalExecutor)
    exe._pool = None

    def _reach_shell(*_args: object, **_kwargs: object) -> TerminalObservation:
        raise AssertionError(_SHELL_SENTINEL)

    exe._execute_pooled = _reach_shell  # type: ignore[method-assign]
    exe._execute_single_session = _reach_shell  # type: ignore[method-assign]
    return exe


def test_exiting_command_returns_structured_error_without_shell(
    executor_without_shell: TerminalExecutor,
) -> None:
    action = TerminalAction(
        command="npx tsc --noEmit; code=$?; rm -f cfg.json; exit $code"
    )
    obs = executor_without_shell(action)

    assert isinstance(obs, TerminalObservation)
    assert obs.is_error is True
    assert obs.exit_code is None
    assert obs.command == action.command
    text = obs.text
    assert "exit $code" in text
    # The hint must teach a concrete recovery path.
    assert 'test "$code" -eq 0' in text
    assert "subshell" in text


def test_subshell_exit_reaches_the_shell(
    executor_without_shell: TerminalExecutor,
) -> None:
    """A subshell `exit` is harmless and must not be intercepted."""
    action = TerminalAction(command="( make test; exit 1 )")
    with pytest.raises(AssertionError, match=_SHELL_SENTINEL):
        executor_without_shell(action)


def test_keystroke_input_bypasses_the_guard(
    executor_without_shell: TerminalExecutor,
) -> None:
    """`is_input=True` forwards raw bytes to a running process, not a command."""
    action = TerminalAction(command="exit", is_input=True)
    with pytest.raises(AssertionError, match=_SHELL_SENTINEL):
        executor_without_shell(action)
