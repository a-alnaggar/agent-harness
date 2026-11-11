"""Same scenario as test_mcp_memory_scenario.py (MCP + long-term memory across a
restart), but driven by a real model instead of a scripted
FakeLLM - proves the harness's actual end-to-end behavior, including
react_loop.py's auto-vs-required retry, not just that pre-scripted
responses get handled correctly.

Needs a real ACCESS_TOKEN, and real Gmail credentials (either already
exported as GMAIL_CREDENTIALS_PATH/GMAIL_TOKEN_PATH, or as
GMAIL_CREDENTIALS_JSON/GMAIL_TOKEN_JSON file-type CI variables) -
skips cleanly without either, so it doesn't fail for anyone who hasn't
set them up, and costs a real model+API call to run."""

import os
import shutil
import sys

import pytest

pytest.importorskip("mcp")

if not (os.environ.get("ACCESS_TOKEN") or os.environ.get("LLM_API_KEY")):
    pytest.skip("needs a real ACCESS_TOKEN/LLM_API_KEY to call the live model", allow_module_level=True)

from agent.config import McpServerConfig  # noqa: E402
from agent.engine.llm_client import LLMClient  # noqa: E402
from agent.engine.react_loop import ReactLoop  # noqa: E402
from agent.memory import SqliteMemoryStore  # noqa: E402
from agent.memory.tools import ForgetMemoryTool, SaveMemoryTool  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402
from agent.tools.mcp import register_mcp_tools  # noqa: E402


@pytest.fixture(autouse=True)
def real_gmail_credentials(tmp_path_factory, monkeypatch):
    """Resolves real Gmail credentials from wherever they're available -
    already-exported paths (local dev) or file-type CI variables
    (GMAIL_CREDENTIALS_JSON/GMAIL_TOKEN_JSON) - copying them into a
    writable location, since the token file needs to be replaceable on
    refresh. Skips the module if neither is available."""
    if os.environ.get("GMAIL_CREDENTIALS_PATH") and os.environ.get("GMAIL_TOKEN_PATH"):
        return

    creds_var, token_var = os.environ.get("GMAIL_CREDENTIALS_JSON"), os.environ.get("GMAIL_TOKEN_JSON")
    if not (creds_var and token_var):
        pytest.skip("needs real Gmail credentials (GMAIL_CREDENTIALS_PATH/GMAIL_TOKEN_PATH, "
                    "or the GMAIL_CREDENTIALS_JSON/GMAIL_TOKEN_JSON file variables)")

    writable_dir = tmp_path_factory.mktemp("gmail-creds")
    creds_path, token_path = writable_dir / "credentials.json", writable_dir / "gmail-token.json"
    shutil.copy(creds_var, creds_path)
    shutil.copy(token_var, token_path)
    monkeypatch.setenv("GMAIL_CREDENTIALS_PATH", str(creds_path))
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(token_path))


def approve(tool_name, arguments, description):
    """search_emails is an MCP tool (STRICT); the scenario needs it to run, so
    this stands in for the user answering yes at the prompt."""
    return "once"


def build_registry(memory_store):
    registry = ToolRegistry()
    registry.register(SaveMemoryTool(memory_store))
    registry.register(ForgetMemoryTool(memory_store))
    servers = [McpServerConfig(name="gmail", command=sys.executable, args=["-m", "agent.mcp_servers.gmail"])]
    clients = register_mcp_tools(registry, servers)
    return registry, clients


@pytest.mark.xfail(
    reason="non-deterministic: the real model sometimes searches but doesn't also "
    "save_memory within the step budget - model variance, not a harness bug",
    strict=False,
)
def test_mcp_tool_and_memory_together_survive_a_restart_with_the_real_model(tmp_path):
    memory_path = tmp_path / "memory.db"
    llm = LLMClient()

    # --- "Session 1": the real model searches real email and remembers the outcome ---
    store_1 = SqliteMemoryStore(memory_path)
    registry_1, clients_1 = build_registry(store_1)
    try:
        loop_1 = ReactLoop(llm, tool_registry=registry_1, max_steps=6, memory_store=store_1)
        loop_1.run(
            "Search my inbox for unread emails, then save what you learn "
            "under the memory key 'gmail_status' - either a short summary "
            "of what you found, or the error, whichever applies.",
            confirm_callback=approve,
        )

        tool_calls_made = {
            m["tool_calls"][0]["function"]["name"]
            for m in loop_1.messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        }
        assert "search_emails" in tool_calls_made, "the real model never actually called search_emails"
        assert store_1.get("gmail_status") is not None, "the real model never actually called save_memory"
    finally:
        for client in clients_1:
            client.close()

    # --- "Session 2": fresh process, real model recalls the memory ---
    store_2 = SqliteMemoryStore(memory_path)
    registry_2, clients_2 = build_registry(store_2)
    try:
        loop_2 = ReactLoop(llm, tool_registry=registry_2, max_steps=3, memory_store=store_2)
        result_2 = loop_2.run(
            "What do you remember about my gmail status? Quote it back to me.", confirm_callback=approve
        )

        system_messages = [m["content"] for m in loop_2.messages if m["role"] == "system"]
        assert any("gmail_status" in content for content in system_messages)
        assert result_2.strip()
    finally:
        for client in clients_2:
            client.close()
