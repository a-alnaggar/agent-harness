from __future__ import annotations

from ..base import Tool, ToolError, ToolUnavailableError
from ..permissions import PermissionTier
from .command_filter import find_destructive_match
from .ssh_client import SshSandboxClient


class SandboxExecTool(Tool):
    """Runs a shell command on a remote sandbox machine over SSH - not the
    local workspace. Generic command execution is a different risk class
    than every other builtin tool (it isn't confined to create/read/modify
    inside a workspace root), so this stays STRICT: confirm every call, no
    "always" shortcut. Destructive-looking commands are also rejected
    client-side (see command_filter.py) to stay inside the spirit of the
    project's no-delete constraint even though this is a generic tool."""

    name = "sandbox_exec"
    permission_tier = PermissionTier.STRICT
    description = (
        "Run a shell command on an isolated remote sandbox machine over SSH "
        "(never the agent's own workspace). Use for anything that needs a "
        "real shell or OS - installing packages, running a script, checking "
        "system state - without touching local files. Destructive commands "
        "(rm, dd, force-push, etc.) are refused."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run on the sandbox host."},
        },
        "required": ["command"],
    }

    def __init__(self, client: SshSandboxClient):
        self._client = client

    def run(self, *, command: str) -> str:
        if not command or not command.strip():
            raise ToolError("'command' must not be empty")

        matched = find_destructive_match(command)
        if matched:
            raise ToolError(
                f"refused to run '{command}': looks destructive (matched pattern '{matched}'). "
                "sandbox_exec cannot run destructive commands."
            )

        try:
            result = self._client.run_command(command)
        except Exception as exc:
            # Unreachable host, rejected key, or a command that didn't finish
            # in time: none of these gets better by retrying right away.
            raise ToolUnavailableError(f"sandbox command failed: {exc}") from exc

        output = f"exit_code={result.exit_code}\nstdout:\n{result.stdout}"
        if result.stderr:
            output += f"\nstderr:\n{result.stderr}"
        return output
