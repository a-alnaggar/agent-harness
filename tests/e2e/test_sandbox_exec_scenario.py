"""Remote-execution scenario (A7): sandbox_exec from
config file and SANDBOX_SSH_KEY to command output, through the real startup
path (`agent.__main__.main` -> load_config -> SshSandboxClient ->
ReactLoop's STRICT gate). The only fake is paramiko's SSHClient: no
connection is ever opened and no command runs anywhere. Every command a test
"runs" is one of the harmless ones from the manual real-VM scenario (M6)."""

from __future__ import annotations

import base64
import io
import threading
from pathlib import Path
from typing import Any

import paramiko
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

import agent.__main__ as main_module

NON_ZERO = "sh -c 'printf \"expected-error\\n\" >&2; exit 3'"
# What the fake host answers, per command: (exit status, stdout, stderr).
REMOTE = {
    "printf 'sandbox-ok\\n'": (0, "sandbox-ok\n", ""),
    "uname -a": (0, "Linux sandbox 6.12.0 #1 SMP x86_64 GNU/Linux\n", ""),
    "pwd": (0, "/home/sandbox\n", ""),
    NON_ZERO: (3, "", "expected-error\n"),
}


class FakeChannel:
    def __init__(self, status: int):
        self._status = status
        self.status_event = threading.Event()
        self.status_event.set()

    def recv_exit_status(self) -> int:
        return self._status


class FakeStdout(io.BytesIO):
    def __init__(self, data: bytes, status: int):
        super().__init__(data)
        self.channel = FakeChannel(status)


class FakeTransport:
    def __init__(self):
        self.active = True
        self.keepalive = None

    def is_active(self):
        return self.active

    def set_keepalive(self, seconds):
        self.keepalive = seconds


class FakeSSHClient:
    """Stands in for paramiko.SSHClient at the network boundary."""

    instances: list["FakeSSHClient"] = []
    connect_error: Exception | None = None

    def __init__(self) -> None:
        self.connected_with: dict[str, Any] | None = None
        self.commands: list[str] = []
        self.closed = False
        FakeSSHClient.instances.append(self)

    def set_missing_host_key_policy(self, policy: Any) -> None:
        pass

    def connect(self, **kwargs: Any) -> None:
        if FakeSSHClient.connect_error is not None:
            raise FakeSSHClient.connect_error
        self.connected_with = kwargs
        self.transport = FakeTransport()

    def get_transport(self) -> Any:
        return getattr(self, "transport", None)

    def exec_command(self, command: str, timeout: int) -> tuple[None, FakeStdout, io.BytesIO]:
        self.commands.append(command)
        status, stdout, stderr = REMOTE[command]
        return None, FakeStdout(stdout.encode(), status), io.BytesIO(stderr.encode())

    def close(self) -> None:
        self.closed = True


class ScriptedLLM:
    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)

    def chat(self, messages, tools=None, tool_choice="auto"):
        return self._responses.pop(0)


class ScriptedUser:
    _DECISIONS = {"y": "once", "a": "always", "n": "deny"}

    def __init__(self, *keys: str):
        self._keys = list(keys)
        self.asked: list[str] = []

    def __call__(self, tool_name: str, arguments: dict, description: str) -> str:
        self.asked.append(f"{tool_name}: {arguments['command']}")
        return self._DECISIONS[self._keys.pop(0)]


SANDBOX_BLOCK = """
sandbox:
  host: sandbox.example.invalid
  user: sandbox
  port: 2222
  timeout: 5
"""


def generated_key_base64() -> str:
    """A throwaway key in the format `ssh-keygen -t ed25519` writes, base64
    encoded exactly like the setup guide's one-liner."""
    pem = ed25519.Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, serialization.NoEncryption(),
    )
    return base64.b64encode(pem).decode()


@pytest.fixture
def start_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Returns start(config_yaml, llm, turns) -> (registry, tool observations
    per turn). Config comes from a real file via AGENT_CONFIG_PATH."""
    FakeSSHClient.instances = []
    FakeSSHClient.connect_error = None
    monkeypatch.setattr(paramiko, "SSHClient", FakeSSHClient)
    monkeypatch.delenv("SANDBOX_SSH_KEY", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    def start(config_yaml: str, llm: ScriptedLLM | None = None, turns: list | None = None):
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("model: m\n" + config_yaml, encoding="utf-8")
        monkeypatch.setenv("AGENT_CONFIG_PATH", str(config_path))
        monkeypatch.setattr(main_module, "LLMClient", lambda **_: llm or ScriptedLLM([]))
        seen: dict[str, Any] = {"observations": []}

        def scripted_cli(*, config, engine, registry, context_manager) -> None:
            seen["registry"] = registry
            for prompt, confirm in turns or []:
                before = len(engine._loop.messages)
                engine.run(prompt, confirm_callback=confirm)
                seen["observations"].append(
                    [m["content"] for m in engine._loop.messages[before:] if m.get("role") == "tool"]
                )

        monkeypatch.setattr(main_module, "start_cli", scripted_cli)
        main_module.main()
        return seen["registry"], seen["observations"]

    return start


def sandbox_call(command: str) -> dict[str, Any]:
    return {"tool_call": {"id": "call_sandbox", "name": "sandbox_exec", "arguments": {"command": command}}}


def answer(text: str) -> dict[str, Any]:
    return {"final_answer": text}


# --- availability: all of block + key, or nothing ----------------------------

def test_without_a_sandbox_block_the_tool_is_absent_and_nothing_is_said(start_agent, monkeypatch, capsys):
    monkeypatch.setenv("SANDBOX_SSH_KEY", generated_key_base64())

    registry, _ = start_agent("")

    assert not registry.has_tool("sandbox_exec")
    assert "sandbox" not in capsys.readouterr().err.lower()
    assert FakeSSHClient.instances == []


def test_with_a_block_but_no_key_the_tool_is_absent_and_a_note_says_why(start_agent, capsys):
    registry, _ = start_agent(SANDBOX_BLOCK)

    assert not registry.has_tool("sandbox_exec")
    assert "'sandbox' is configured but SANDBOX_SSH_KEY is not set; sandbox_exec is disabled." in capsys.readouterr().err


@pytest.mark.parametrize("bad_key, reason", [
    ("~/.ssh/sandbox_key", "must be a private key (PEM/OpenSSH text) or its base64 encoding"),
    (base64.b64encode(b"not a key").decode(), "not a usable private key"),
])
def test_a_malformed_key_stops_startup_with_a_clear_error_and_no_connection(start_agent, monkeypatch, capsys, bad_key, reason):
    monkeypatch.setenv("SANDBOX_SSH_KEY", bad_key)

    with pytest.raises(SystemExit) as exit_info:
        start_agent(SANDBOX_BLOCK)

    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "Sandbox key error: SANDBOX_SSH_KEY" in err and reason in err
    assert bad_key not in err
    assert FakeSSHClient.instances == []


@pytest.mark.parametrize("secret", ["private_key: abc", "password: hunter2"])
def test_a_secret_in_the_config_file_is_refused_at_startup(start_agent, capsys, secret):
    with pytest.raises(SystemExit):
        start_agent(SANDBOX_BLOCK + f"  {secret}\n")

    assert "must not be committed" in capsys.readouterr().err


# --- invocation: STRICT approval, output, exit status -------------------------

def test_approved_harmless_commands_run_and_report_stdout_and_exit_status(start_agent, monkeypatch):
    monkeypatch.setenv("SANDBOX_SSH_KEY", generated_key_base64())
    # "a" (always) is offered by the prompt but means nothing for a STRICT
    # tool: the next two calls are asked again.
    user = ScriptedUser("a", "y", "y")
    llm = ScriptedLLM([
        sandbox_call("printf 'sandbox-ok\\n'"),
        sandbox_call("uname -a"),
        sandbox_call("pwd"),
        answer("The sandbox answered."),
    ])

    registry, observations = start_agent(SANDBOX_BLOCK, llm, [("Check the sandbox.", user)])

    assert registry.get("sandbox_exec").permission_tier.value == "strict"
    assert user.asked == ["sandbox_exec: printf 'sandbox-ok\\n'", "sandbox_exec: uname -a", "sandbox_exec: pwd"]
    assert observations[0] == [
        "exit_code=0\nstdout:\nsandbox-ok\n",
        "exit_code=0\nstdout:\nLinux sandbox 6.12.0 #1 SMP x86_64 GNU/Linux\n",
        "exit_code=0\nstdout:\n/home/sandbox\n",
    ]
    (ssh,) = FakeSSHClient.instances
    assert ssh.commands == ["printf 'sandbox-ok\\n'", "uname -a", "pwd"]
    assert {k: ssh.connected_with[k] for k in ("hostname", "port", "username", "timeout")} == {
        "hostname": "sandbox.example.invalid", "port": 2222, "username": "sandbox", "timeout": 5,
    }
    assert isinstance(ssh.connected_with["pkey"], paramiko.Ed25519Key)
    assert ssh.connected_with["look_for_keys"] is False and ssh.connected_with["allow_agent"] is False
    assert ssh.closed, "the SSH connection is closed when the session ends"


def test_a_non_zero_exit_is_reported_with_its_status_and_stderr(start_agent, monkeypatch):
    monkeypatch.setenv("SANDBOX_SSH_KEY", generated_key_base64())
    llm = ScriptedLLM([sandbox_call(NON_ZERO), answer("It exited with status 3.")])

    _, observations = start_agent(SANDBOX_BLOCK, llm, [("Run the failing check.", ScriptedUser("y"))])

    # A failing remote command is a normal observation, not a tool error: the
    # model sees the status and stderr and decides what to do next.
    assert observations[0] == ["exit_code=3\nstdout:\n\nstderr:\nexpected-error\n"]


def test_a_denied_or_unanswerable_call_never_reaches_the_host(start_agent, monkeypatch):
    monkeypatch.setenv("SANDBOX_SSH_KEY", generated_key_base64())
    llm = ScriptedLLM([
        sandbox_call("uname -a"), answer("Not approved."),
        sandbox_call("uname -a"), answer("Could not ask."),
    ])

    _, observations = start_agent(SANDBOX_BLOCK, llm, [("Check the sandbox.", ScriptedUser("n")), ("Again.", None)])

    assert observations[0][0].startswith("Error: user denied permission to run 'sandbox_exec'")
    assert observations[1][0].startswith("Error: 'sandbox_exec' needs the user's confirmation, but there is no way to ask")
    assert FakeSSHClient.instances == []


def test_a_blocked_command_is_refused_even_when_approved_and_never_sent(start_agent, monkeypatch):
    monkeypatch.setenv("SANDBOX_SSH_KEY", generated_key_base64())
    llm = ScriptedLLM([sandbox_call("rm -rf /tmp/scenario"), answer("Refused.")])

    _, observations = start_agent(SANDBOX_BLOCK, llm, [("Clean up.", ScriptedUser("y"))])

    assert observations[0] == [
        "Error: refused to run 'rm -rf /tmp/scenario': looks destructive (matched pattern '\\brm\\b'). "
        "sandbox_exec cannot run destructive commands."
    ]
    assert FakeSSHClient.instances == [], "the filter runs before any connection is opened"


def test_a_host_that_rejects_the_key_gives_a_clear_error_observation(start_agent, monkeypatch):
    monkeypatch.setenv("SANDBOX_SSH_KEY", generated_key_base64())
    FakeSSHClient.connect_error = paramiko.AuthenticationException("Authentication failed.")
    llm = ScriptedLLM([sandbox_call("uname -a"), answer("The host rejected the key.")])

    _, observations = start_agent(SANDBOX_BLOCK, llm, [("Check the sandbox.", ScriptedUser("y"))])

    assert observations[0] == ["Error: sandbox command failed: Authentication failed. 'sandbox_exec' is unavailable for the rest of this request; do not call it again. Tell the user what failed."]
