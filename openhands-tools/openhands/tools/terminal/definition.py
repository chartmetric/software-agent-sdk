"""Execute shell commands in a persistent terminal session."""

import os
import platform
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import Field


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState
from rich.text import Text

from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.sdk.utils import maybe_truncate
from openhands.tools.terminal.constants import (
    MAX_CMD_OUTPUT_SIZE,
    NO_CHANGE_TIMEOUT_SECONDS,
)
from openhands.tools.terminal.descriptions import (
    UNIX_TOOL_DESCRIPTION,
    WINDOWS_TOOL_DESCRIPTION,
)
from openhands.tools.terminal.metadata import CmdOutputMetadata


_LITERAL_ARG_HINT_TEMPLATE = (
    "[Tool-argument error] The `command` argument looks like a Python/JSON "
    "{literal_kind}, not a shell command. It starts with: {head!r}\n\n"
    "The `terminal` tool runs exactly ONE shell command at a time. To pass "
    "structured data or multi-line code:\n"
    "  - Write a script first with `file_editor` "
    '(command="create", path="/tmp/run.py", ...), then invoke it: '
    "`python /tmp/run.py`.\n"
    "  - Or use a heredoc inline, e.g.:\n"
    "        python - <<'EOF'\n"
    "        DATABASES = {{'default': {{...}}}}\n"
    "        # your code here\n"
    "        EOF\n\n"
    "Do not put a Python list/dict literal into the `command` field; the shell "
    "cannot interpret it."
)


def looks_like_python_literal_argument(command: str) -> str | None:
    """Detect when a tool call has packed structured data into `command`.

    Returns a short reason string (``"list literal"``, ``"nested list literal"``
    or ``"dict literal"``) if ``command`` appears to be a Python/JSON literal
    rather than a shell command, otherwise ``None``.

    Carefully distinguishes legitimate bash uses of ``[`` and ``[[`` (which
    are always followed by whitespace) from Python-style literals. See the
    accompanying tests for the full matrix.
    """
    stripped = command.lstrip()
    if len(stripped) < 2:
        return None
    a, b = stripped[0], stripped[1]
    # Top-level list literals: [{...}], ["..."], ['...']
    if a == "[" and b in ("{", '"', "'"):
        return "list literal"
    # Nested list literals: [[...]] — but bash `[[ -f x ]]` is followed by a
    # whitespace char, so we only flag `[[` followed by non-whitespace.
    if a == "[" and b == "[":
        if len(stripped) >= 3 and stripped[2] not in (" ", "\t"):
            return "nested list literal"
        return None
    # Top-level dict literals: {"key": ...} or {'key': ...}.
    # Bash group commands `{ ls; }` always have a space after `{`.
    if a == "{" and b in ('"', "'"):
        return "dict literal"
    return None


_SHELL_EXIT_HINT_TEMPLATE = (
    "[Tool-argument error] The `command` argument ends the persistent shell: "
    "it runs `{offender}` at the top level.\n\n"
    "The terminal is a long-lived tmux session shared across your tool calls. "
    "A top-level `exit` (or `logout`) terminates that shell, so the pane and "
    "its tmux server disappear, THIS command's output is lost, and the tool "
    "blocks until the timeout before it can rebuild the pool. Nothing was "
    "executed -- rewrite the command and send it again.\n\n"
    "To propagate or inspect a status without exiting, use a check that stays "
    "in the shell:\n"
    '  - `...; code=$?; rm -f tmp; test "$code" -eq 0`\n'
    '  - `...; code=$?; rm -f tmp; if [ "$code" -eq 0 ]; then echo ok; '
    'else echo "failed with $code"; fi`\n'
    "  - Or run the exiting part in a subshell: `( ...; exit 1 )`.\n\n"
    "Note that the tool already reports the exit code of the last command, so "
    "re-raising it with `exit` is rarely needed at all."
)

_SHELL_EXITING_BUILTINS = frozenset({"exit", "logout"})

# Words that introduce a command rather than being one, so an `exit` right
# after them is still in command position: `...; else exit 1; fi`.
_COMMAND_INTRODUCING_WORDS = frozenset(
    {"then", "else", "elif", "do", "!", "time", "command", "builtin", "eval"}
)

_HEREDOC_OPENER_RE = re.compile(
    r"<<-?\s*(?P<quote>['\"]?)(?P<delim>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)"
)

_WORD_RE = re.compile(r"[^\s;|&()<>{}'\"`]+")


def looks_like_shell_exiting_command(command: str) -> str | None:
    """Detect a top-level ``exit``/``logout`` that would kill the shell.

    The terminal tool keeps one persistent tmux pane per session, so a
    top-level ``exit`` destroys the pane mid-call: the command's own output is
    lost and the executor only discovers the dead session after the action
    timeout elapses. Catching the command before it runs turns a multi-minute
    stall into an immediate, actionable hint.

    Returns the offending trailing command text, or ``None`` when the command
    is safe. Deliberately conservative -- an ``exit`` inside quotes, a
    heredoc body, a subshell ``(...)``, or a command substitution is left
    alone, because those do not end the caller's shell. Missing a real case
    only restores the previous behaviour; a false positive would block a
    legitimate command, so ambiguity resolves to "safe".
    """
    depth = 0  # subshell / command-substitution nesting; > 0 cannot kill us
    at_command_start = True
    quote: str | None = None
    heredoc_delimiter: str | None = None

    for line in command.split("\n"):
        if heredoc_delimiter is not None:
            if line.strip() == heredoc_delimiter:
                heredoc_delimiter = None
            continue

        index = 0
        length = len(line)
        while index < length:
            char = line[index]

            if quote == "'":
                if char == "'":
                    quote = None
                index += 1
                continue
            if quote is not None:
                # Double quotes and backticks honour backslash escapes.
                if char == "\\":
                    index += 2
                    continue
                if char == quote:
                    quote = None
                index += 1
                continue

            if char == "\\":
                index += 2
                continue
            if char in ("'", '"', "`"):
                quote = char
                at_command_start = False
                index += 1
                continue
            if char == "#" and at_command_start:
                break  # comment runs to end of line
            if char == "$" and index + 1 < length and line[index + 1] in ("(", "{"):
                depth += 1
                index += 2
                continue
            if char == "(":
                depth += 1
                at_command_start = True
                index += 1
                continue
            if char in (")", "}"):
                # A closing brace may end `${...}` (tracked) or a brace group
                # (untracked, because it runs in the current shell).
                depth = max(depth - 1, 0)
                at_command_start = False
                index += 1
                continue
            if char in (";", "|", "&", "{"):
                # `{ ...; }` groups run in the current shell, so an `exit`
                # inside one still ends it: treat `{` as a plain separator.
                at_command_start = True
                index += 1
                continue
            if char in (" ", "\t"):
                index += 1
                continue
            if char == "<" and line.startswith("<<", index):
                opener = _HEREDOC_OPENER_RE.match(line, index)
                if opener is not None:
                    heredoc_delimiter = opener.group("delim")
                    index = opener.end()
                else:
                    index += 2
                at_command_start = False
                continue
            if char == ">":
                index += 1
                continue

            word_match = _WORD_RE.match(line, index)
            if word_match is None:
                index += 1
                continue
            word = word_match.group(0)
            if at_command_start and depth == 0 and word in _SHELL_EXITING_BUILTINS:
                return line[index:].strip()
            if word not in _COMMAND_INTRODUCING_WORDS:
                at_command_start = False
            index = word_match.end()

        # A newline separates commands just like `;` does.
        at_command_start = True

    return None


class TerminalAction(Action):
    """Schema for terminal command execution."""

    command: str = Field(
        description=(
            "The shell command to execute. Can be empty string to view"
            " additional logs when the previous exit code is `-1`. Can be a"
            " special key name when `is_input` is True: `C-c` (Ctrl+C),"
            " `C-d` (Ctrl+D/EOF), `C-z` (Ctrl+Z), or any `C-<letter>`"
            " for Ctrl sequences; navigation keys `UP`, `DOWN`, `LEFT`,"
            " `RIGHT`, `HOME`, `END`, `PGUP`, `PGDN`; and `TAB`, `ESC`,"
            " `BS` (Backspace), `ENTER`. You can only execute one command"
            " at a time. Use the platform-appropriate shell syntax described"
            " in the tool description when chaining commands."
        )
    )
    is_input: bool = Field(
        default=False,
        description="If True, the command is an input to the running process. If False, the command is executed in the terminal session. Default is False.",  # noqa
    )
    timeout: float | None = Field(
        default=None,
        ge=0,
        description=f"Optional. Sets a maximum time limit (in seconds) for running the command. If the command takes longer than this limit, you’ll be asked whether to continue or stop it. If you don’t set a value, the command will instead pause and ask for confirmation when it produces no new output for {NO_CHANGE_TIMEOUT_SECONDS} seconds. Use a higher value if the command is expected to take a long time (like installation or testing), or if it has a known fixed duration (like sleep).",  # noqa
    )
    reset: bool = Field(
        default=False,
        description="If True, reset the terminal by creating a new session. Use this only when the terminal becomes unresponsive. Note that all previously set environment variables and session state will be lost after reset. Cannot be used with is_input=True.",  # noqa
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation with a shell-style prompt."""
        content = Text()

        # Create PS1-style prompt
        content.append("$ ", style="bold green")

        # Add command with syntax highlighting
        if self.command:
            content.append(self.command, style="white")
        else:
            content.append("[empty command]", style="italic")

        # Add metadata if present
        if self.is_input:
            content.append(" ", style="white")
            content.append("(input to running process)", style="yellow")

        if self.timeout is not None:
            content.append(" ", style="white")
            content.append(f"[timeout: {self.timeout}s]", style="cyan")

        if self.reset:
            content.append(" ", style="white")
            content.append("[reset terminal]", style="red bold")

        return content


class TerminalObservation(Observation):
    """A ToolResult that can be rendered as a CLI output."""

    command: str | None = Field(
        description="The shell command that was executed. Can be empty string if the observation is from a previous command that hit soft timeout and is not yet finished.",  # noqa
    )
    exit_code: int | None = Field(
        default=None,
        description="The exit code of the command. -1 indicates the process hit the soft timeout and is not yet finished.",  # noqa
    )
    timeout: bool = Field(
        default=False, description="Whether the command execution timed out."
    )
    metadata: CmdOutputMetadata = Field(
        default_factory=CmdOutputMetadata,
        description="Additional metadata captured from PS1 after command execution.",
    )
    full_output_save_dir: str | None = Field(
        default=None,
        description="Directory where full output files are saved",
    )

    @property
    def command_id(self) -> int | None:
        """Get the command ID from metadata."""
        return self.metadata.pid

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        llm_content: list[TextContent | ImageContent] = []

        # If is_error is true, prepend error message
        if self.is_error:
            llm_content.append(TextContent(text=self.ERROR_MESSAGE_HEADER))

        # TerminalObservation always has content as a single TextContent
        content_text = self.text

        ret = f"{self.metadata.prefix}{content_text}{self.metadata.suffix}"
        if self.metadata.working_dir:
            ret += f"\n[Current working directory: {self.metadata.working_dir}]"
        if self.metadata.py_interpreter_path:
            ret += f"\n[Python interpreter: {self.metadata.py_interpreter_path}]"
        if self.metadata.exit_code != -1:
            ret += f"\n[Command finished with exit code {self.metadata.exit_code}]"

        # Use enhanced truncation with file saving if working directory is available
        truncated_text = maybe_truncate(
            content=ret,
            truncate_after=MAX_CMD_OUTPUT_SIZE,
            save_dir=self.full_output_save_dir,
            tool_prefix="terminal",
        )
        llm_content.append(TextContent(text=truncated_text))

        return llm_content

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation with terminal-style output formatting."""
        text = Text()

        if self.is_error:
            text.append("❌ ", style="red bold")
            text.append(self.ERROR_MESSAGE_HEADER, style="bold red")

        # TerminalObservation always has content as a single TextContent
        content_text = self.text

        if content_text:
            # Style the output based on content
            output_lines = content_text.split("\n")
            for line in output_lines:
                if line.strip():
                    # Color error-like lines differently
                    if any(
                        keyword in line.lower()
                        for keyword in ["error", "failed", "exception", "traceback"]
                    ):
                        text.append(line, style="red")
                    elif any(
                        keyword in line.lower() for keyword in ["warning", "warn"]
                    ):
                        text.append(line, style="yellow")
                    elif line.startswith("+ "):  # bash -x output
                        text.append(line, style="cyan")
                    else:
                        text.append(line, style="white")
                text.append("\n")

        # Add metadata with styling
        if hasattr(self, "metadata") and self.metadata:
            if self.metadata.working_dir:
                text.append("\n📁 ", style="blue")
                text.append(
                    f"Working directory: {self.metadata.working_dir}", style="blue"
                )

            if self.metadata.py_interpreter_path:
                text.append("\n🐍 ", style="green")
                text.append(
                    f"Python interpreter: {self.metadata.py_interpreter_path}",
                    style="green",
                )

            if (
                hasattr(self.metadata, "exit_code")
                and self.metadata.exit_code is not None
            ):
                if self.metadata.exit_code == 0:
                    text.append("\n✅ ", style="green")
                    text.append(f"Exit code: {self.metadata.exit_code}", style="green")
                elif self.metadata.exit_code == -1:
                    text.append("\n⏳ ", style="yellow")
                    text.append("Process still running (soft timeout)", style="yellow")
                else:
                    text.append("\n❌ ", style="red")
                    text.append(f"Exit code: {self.metadata.exit_code}", style="red")

        return text


class TerminalTool(ToolDefinition[TerminalAction, TerminalObservation]):
    """A ToolDefinition subclass that automatically initializes a TerminalExecutor with auto-detection."""  # noqa: E501

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        # When using the tmux backend, TmuxPanePool handles concurrency
        # internally via pane-level isolation — opt out of framework
        # serialization so parallel calls are allowed.
        # When using the subprocess backend there is only a single
        # session, so we declare a resource key to serialize terminal
        # calls against each other without blocking unrelated tools.
        if getattr(self.executor, "is_pooled", False):
            return DeclaredResources(keys=(), declared=True)
        return DeclaredResources(keys=("terminal:session",), declared=True)

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        username: str | None = None,
        no_change_timeout_seconds: int | None = None,
        terminal_type: Literal["tmux", "subprocess", "powershell"] | None = None,
        shell_path: str | None = None,
        executor: ToolExecutor | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> Sequence["TerminalTool"]:
        """Initialize TerminalTool with executor parameters.

        Args:
            conv_state: Conversation state to get working directory from.
                         If provided, working_dir will be taken from
                         conv_state.workspace
            username: Optional username for the shell session
            no_change_timeout_seconds: Timeout for no output change
            terminal_type: Force a specific session type:
                         ('tmux', 'subprocess', or 'powershell').
                         If None, auto-detect based on system capabilities:
                         - On Windows: PowerShell-backed backend
                         - On Unix-like systems: tmux if available, otherwise subprocess
            shell_path: Path to the shell binary. On Unix this applies to the
                       subprocess backend; on Windows it can point to a
                       PowerShell executable.
            env: Extra environment variables to add to the terminal session.
                 These are client-controlled session settings and are not part
                 of the LLM-facing TerminalAction schema.
        """
        # Import here to avoid circular imports
        from openhands.tools.terminal.impl import TerminalExecutor

        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        # Initialize the executor
        if executor is None:
            executor = TerminalExecutor(
                working_dir=working_dir,
                username=username,
                no_change_timeout_seconds=no_change_timeout_seconds,
                terminal_type=terminal_type,
                shell_path=shell_path,
                env=env,
                full_output_save_dir=conv_state.env_observation_persistence_dir,
            )

        tool_description = (
            WINDOWS_TOOL_DESCRIPTION
            if platform.system() == "Windows"
            else UNIX_TOOL_DESCRIPTION
        )

        # Initialize the parent ToolDefinition with the executor
        return [
            cls(
                action_type=TerminalAction,
                observation_type=TerminalObservation,
                description=tool_description,
                annotations=ToolAnnotations(
                    title="terminal",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# Automatically register the tool when this module is imported
register_tool(TerminalTool.name, TerminalTool)
