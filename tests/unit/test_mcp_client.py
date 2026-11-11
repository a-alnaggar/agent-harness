"""mcp requires Python 3.10+; skip cleanly on older local interpreters.
Uses the real Gmail MCP server (src/agent/mcp_servers/gmail.py) as the
test fixture — discovery works without credentials, only tool execution
needs them, so these stay fully offline."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from agent.tools.base import ToolError  # noqa: E402
from agent.tools.mcp.client import McpClient, McpConnectionError  # noqa: E402

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def gmail_client(monkeypatch, tmp_path):
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(tmp_path / "missing-token.json"))
    monkeypatch.setenv("GMAIL_CREDENTIALS_PATH", str(tmp_path / "missing-client.json"))
    client = McpClient("gmail", sys.executable, ["-m", "agent.mcp_servers.gmail"])
    yield client
    client.close()


def test_connect_discovers_tools(gmail_client):
    tools = gmail_client.connect()
    names = {t["name"] for t in tools}
    assert names == {"search_emails", "read_email", "create_draft", "send_email"}


def test_discovered_tool_has_usable_schema(gmail_client):
    tools = gmail_client.connect()
    search = next(t for t in tools if t["name"] == "search_emails")
    assert search["description"]
    assert search["input_schema"]["required"] == ["query"]


def test_call_tool_without_credentials_raises_controlled_error(gmail_client):
    gmail_client.connect()
    with pytest.raises(ToolError, match="Gmail authorization unavailable"):
        gmail_client.call_tool("search_emails", {"query": "is:unread"})


def test_call_tool_before_connect_raises_controlled_error():
    client = McpClient("gmail", sys.executable, ["-m", "agent.mcp_servers.gmail"])
    with pytest.raises(McpConnectionError):
        client.call_tool("search_emails", {"query": "x"})


def test_connect_to_missing_command_raises_controlled_error():
    client = McpClient("bogus", "this-command-does-not-exist", [])
    with pytest.raises(McpConnectionError):
        client.connect(timeout=5.0)
    client.close()


def test_call_tool_truncates_an_oversized_result():
    client = McpClient("big-output", sys.executable, [str(_FIXTURES_DIR / "big_output_mcp_server.py")])
    client.connect()
    try:
        result = client.call_tool("dump_big_output", {})
    finally:
        client.close()

    assert len(result) < 10_000
    assert "truncated" in result
    assert result.startswith("x" * 100)


def test_close_before_connect_is_a_no_op():
    client = McpClient("gmail", sys.executable, ["-m", "agent.mcp_servers.gmail"])
    client.close()  # must not raise


# --- what an MCP server may inherit ------------------------------------------

def test_a_server_gets_the_base_environment_and_only_the_variables_its_config_names() -> None:
    from agent.tools.mcp.client import build_server_env

    source = {
        "PATH": "/bin", "HOME": "/home/x", "LANG": "C.UTF-8",
        "ACCESS_TOKEN": "model-secret", "SANDBOX_SSH_KEY": "key-secret", "NOTION_TOKEN": "notion",
        "GMAIL_TOKEN_PATH": "/gmail/token.json",
    }

    env = build_server_env(["NOTION_TOKEN"], source)

    assert env == {"PATH": "/bin", "HOME": "/home/x", "LANG": "C.UTF-8", "NOTION_TOKEN": "notion"}
    assert "ACCESS_TOKEN" not in env and "SANDBOX_SSH_KEY" not in env
    assert "GMAIL_TOKEN_PATH" not in env  # present in the agent's env, but this server did not ask for it


def test_listed_variables_that_are_not_set_are_simply_absent() -> None:
    from agent.tools.mcp.client import build_server_env

    assert build_server_env(["NOT_SET_ANYWHERE"], {"PATH": "/bin"}) == {"PATH": "/bin"}


def test_the_real_server_process_never_sees_the_agents_secrets(monkeypatch, tmp_path) -> None:
    """End to end: a child process spawned the way McpClient spawns servers."""
    import subprocess
    from agent.tools.mcp.client import build_server_env

    monkeypatch.setenv("ACCESS_TOKEN", "model-secret")
    monkeypatch.setenv("SANDBOX_SSH_KEY", "key-secret")
    monkeypatch.setenv("KEEP_ME", "yes")

    out = subprocess.run(
        [sys.executable, "-c", "import os; print(sorted(k for k in os.environ if k in ('ACCESS_TOKEN','SANDBOX_SSH_KEY','KEEP_ME')))"],
        env=build_server_env(["KEEP_ME"]), capture_output=True, text=True, check=True,
    ).stdout

    assert out.strip() == "['KEEP_ME']"
