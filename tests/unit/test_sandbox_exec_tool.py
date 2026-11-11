import pytest

from agent.tools.base import ToolError
from agent.tools.permissions import PermissionTier
from agent.tools.sandbox.exec_tool import SandboxExecTool
from agent.tools.sandbox.ssh_client import SshCommandResult


class FakeSshClient:
    def __init__(self, result=None, error=None):
        self.result = result or SshCommandResult(exit_code=0, stdout="ok\n", stderr="")
        self.error = error
        self.commands_run = []

    def run_command(self, command: str) -> SshCommandResult:
        self.commands_run.append(command)
        if self.error is not None:
            raise self.error
        return self.result


def test_sandbox_exec_is_strict_tier():
    assert SandboxExecTool(FakeSshClient()).permission_tier == PermissionTier.STRICT


def test_runs_command_and_reports_exit_code_and_stdout():
    fake = FakeSshClient(SshCommandResult(exit_code=0, stdout="hello\n", stderr=""))
    tool = SandboxExecTool(fake)

    output = tool.run(command="echo hello")

    assert fake.commands_run == ["echo hello"]
    assert "exit_code=0" in output
    assert "hello" in output


def test_includes_stderr_when_present():
    fake = FakeSshClient(SshCommandResult(exit_code=1, stdout="", stderr="not found\n"))
    tool = SandboxExecTool(fake)

    output = tool.run(command="badcmd")

    assert "exit_code=1" in output
    assert "not found" in output


def test_rejects_empty_command():
    tool = SandboxExecTool(FakeSshClient())

    with pytest.raises(ToolError, match="must not be empty"):
        tool.run(command="   ")


def test_refuses_destructive_command_without_calling_the_client():
    fake = FakeSshClient()
    tool = SandboxExecTool(fake)

    with pytest.raises(ToolError, match="refused to run"):
        tool.run(command="rm -rf /")

    assert fake.commands_run == []


def test_wraps_connection_failures_as_tool_error():
    tool = SandboxExecTool(FakeSshClient(error=ConnectionError("no route to host")))

    with pytest.raises(ToolError, match="sandbox command failed"):
        tool.run(command="ls")


def test_connection_failures_mark_the_tool_unavailable_not_just_failed():
    """So the loop stops offering it for the request instead of asking the
    user to approve every retry."""
    from agent.tools.base import ToolUnavailableError

    tool = SandboxExecTool(FakeSshClient(error=TimeoutError("timed out")))

    with pytest.raises(ToolUnavailableError, match="timed out"):
        tool.run(command="pwd")


def test_a_blocked_command_is_an_ordinary_error_not_an_unavailable_tool():
    from agent.tools.base import ToolUnavailableError

    with pytest.raises(ToolError) as caught:
        SandboxExecTool(FakeSshClient()).run(command="rm -rf /tmp/x")

    assert not isinstance(caught.value, ToolUnavailableError)
