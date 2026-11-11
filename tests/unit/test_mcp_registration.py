import sys

import pytest

pytest.importorskip("mcp")

from agent.config import McpServerConfig  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402
from agent.tools.mcp import register_mcp_tools  # noqa: E402
from agent.tools.permissions import PermissionTier  # noqa: E402


def test_registers_all_tools_from_a_working_server():
    registry = ToolRegistry()
    servers = [McpServerConfig(name="gmail", command=sys.executable, args=["-m", "agent.mcp_servers.gmail"])]

    clients = register_mcp_tools(registry, servers)

    try:
        names = {t.name for t in registry.list_tools()}
        assert {"search_emails", "read_email", "create_draft", "send_email"} <= names
        assert len(clients) == 1
    finally:
        for client in clients:
            client.close()


def test_tools_allowlist_registers_only_the_listed_tools():
    registry = ToolRegistry()
    servers = [McpServerConfig(
        name="gmail", command=sys.executable, args=["-m", "agent.mcp_servers.gmail"],
        tools=["search_emails", "read_email"],
    )]

    clients = register_mcp_tools(registry, servers)

    try:
        names = {t.name for t in registry.list_tools()}
        assert names == {"search_emails", "read_email"}
    finally:
        for client in clients:
            client.close()


def test_no_allowlist_registers_every_discovered_tool():
    registry = ToolRegistry()
    servers = [McpServerConfig(name="gmail", command=sys.executable, args=["-m", "agent.mcp_servers.gmail"], tools=None)]

    clients = register_mcp_tools(registry, servers)

    try:
        names = {t.name for t in registry.list_tools()}
        assert names == {"search_emails", "read_email", "create_draft", "send_email"}
    finally:
        for client in clients:
            client.close()


def test_a_broken_server_is_skipped_not_fatal(capsys):
    registry = ToolRegistry()
    servers = [
        McpServerConfig(name="bogus", command="this-command-does-not-exist", args=[]),
        McpServerConfig(name="gmail", command=sys.executable, args=["-m", "agent.mcp_servers.gmail"]),
    ]

    clients = register_mcp_tools(registry, servers)

    try:
        names = {t.name for t in registry.list_tools()}
        assert "search_emails" in names
        assert len(clients) == 1
        assert "bogus" in capsys.readouterr().err
    finally:
        for client in clients:
            client.close()


def test_tools_default_to_strict_permission_tier():
    """No config override at all - third-party tools never earn a lighter
    tier just by being present."""
    registry = ToolRegistry()
    servers = [McpServerConfig(name="gmail", command=sys.executable, args=["-m", "agent.mcp_servers.gmail"])]

    clients = register_mcp_tools(registry, servers)

    try:
        assert all(t.permission_tier == PermissionTier.STRICT for t in registry.list_tools())
    finally:
        for client in clients:
            client.close()


def test_auto_tools_and_confirm_tools_set_the_right_tier_per_tool():
    registry = ToolRegistry()
    servers = [McpServerConfig(
        name="gmail", command=sys.executable, args=["-m", "agent.mcp_servers.gmail"],
        auto_tools=["search_emails", "read_email"],
        confirm_tools=["create_draft"],
    )]

    clients = register_mcp_tools(registry, servers)

    try:
        assert registry.get("search_emails").permission_tier == PermissionTier.REGULAR
        assert registry.get("read_email").permission_tier == PermissionTier.REGULAR
        assert registry.get("create_draft").permission_tier == PermissionTier.CONFIRM
        # not listed in either - defaults to STRICT, e.g. send_email
        assert registry.get("send_email").permission_tier == PermissionTier.STRICT
    finally:
        for client in clients:
            client.close()


def test_no_configured_servers_registers_nothing():
    registry = ToolRegistry()

    clients = register_mcp_tools(registry, [])

    assert clients == []
    assert registry.list_tools() == []
