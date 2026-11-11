"""SSH-backed sandbox: lets the agent run shell commands on a remote,
disposable machine instead of the local workspace."""

from __future__ import annotations

from .exec_tool import SandboxExecTool
from .ssh_client import SshCommandResult, SshSandboxClient

__all__ = ["SandboxExecTool", "SshCommandResult", "SshSandboxClient"]
