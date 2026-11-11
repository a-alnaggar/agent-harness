"""Standalone MCP test fixture: a single tool that returns an oversized
string, used to verify McpClient truncates large results before they'd
otherwise bloat every subsequent LLM call's context."""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("big-output-test-fixture")


@mcp.tool()
def dump_big_output() -> str:
    return "x" * 10_000


if __name__ == "__main__":
    mcp.run(transport="stdio")
