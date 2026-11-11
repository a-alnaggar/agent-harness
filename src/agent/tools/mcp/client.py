"""MCP client: connects to one configured MCP server over stdio and lets
the harness discover and call its tools.

Runs its own background thread with a dedicated asyncio event loop. The
rest of the harness (ToolRegistry.call) is synchronous, but the MCP
session has to stay open for the whole CLI session, not just one async
block, so a plain `asyncio.run()` per call would restart the server
subprocess every time — this keeps one persistent connection instead.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..base import ToolError

# src/agent/tools/mcp/client.py -> src/. A server spawned via `python -m
# agent.mcp_servers.X` needs this on PYTHONPATH whenever `agent` isn't
# installed as a real package (e.g. under pytest, where pythonpath in
# pyproject.toml only affects this process's sys.path, not a child's).
_SRC_DIR = Path(__file__).resolve().parents[3]

# Some MCP servers (e.g. Notion's) return large, deeply-nested JSON for
# routine calls - a single database schema easily runs several KB. Fed back
# into context on every subsequent turn, that both wastes tokens and, in
# practice, degrades the model's ability to continue the task. Unlike
# Gmail's own tools (which cap their own output deliberately), most
# third-party servers don't self-limit, so this is a generic safety net.
_MAX_RESULT_CHARS = 4000


# An MCP server is separate (often third-party, sometimes fetched by `npx -y`)
# code. It must not inherit the agent's environment, which holds the model
# token and the sandbox key. It gets only what a process needs to run, plus
# the variables its config lists by name under `env`.
_BASE_ENV = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ", "TMPDIR",
    "SYSTEMROOT", "COMSPEC", "PATHEXT",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
})


def build_server_env(passthrough: list[str], source: Optional[dict[str, str]] = None) -> dict[str, str]:
    """The environment handed to one MCP server: the base allowlist plus the
    named variables the server's config asks for (only those actually set)."""
    source = os.environ if source is None else source
    wanted = _BASE_ENV | set(passthrough)
    return {name: value for name, value in source.items() if name in wanted}


class McpConnectionError(ToolError):
    """Raised when connecting to or discovering an MCP server fails."""


class McpClient:
    def __init__(self, name: str, command: str, args: list[str], env: Optional[list[str]] = None):
        self.name = name
        self._env_names = list(env or [])
        self._command = command
        self._args = args
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._stdio_cm = None

    def connect(self, timeout: float = 15.0) -> list[dict[str, Any]]:
        """Start the server and discover its tools. Never raises a bare
        exception — connection failures surface as McpConnectionError so
        one broken server doesn't take down the whole agent."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        try:
            return self._run(self._connect_async(), timeout)
        except Exception as exc:
            self.close()
            raise McpConnectionError(f"could not connect to MCP server '{self.name}': {exc}") from exc

    async def _connect_async(self) -> list[dict[str, Any]]:
        env = build_server_env(self._env_names)
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_SRC_DIR), existing_pythonpath]))
        command = shutil.which(self._command) or self._command
        params = StdioServerParameters(command=command, args=self._args, env=env)
        self._stdio_cm = stdio_client(params)
        read, write = await self._stdio_cm.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        result = await self._session.list_tools()
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in result.tools
        ]

    def call_tool(self, tool_name: str, arguments: dict[str, Any], timeout: float = 30.0) -> str:
        if self._session is None:
            raise McpConnectionError(f"MCP server '{self.name}' is not connected")
        try:
            result = self._run(self._session.call_tool(tool_name, arguments), timeout)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"MCP tool '{tool_name}' on '{self.name}' failed: {exc}") from exc

        text = "\n".join(part.text for part in result.content if hasattr(part, "text"))
        if result.is_error:
            raise ToolError(text or f"MCP tool '{tool_name}' on '{self.name}' failed")
        if len(text) > _MAX_RESULT_CHARS:
            omitted = len(text) - _MAX_RESULT_CHARS
            text = f"{text[:_MAX_RESULT_CHARS]}\n... [truncated, {omitted} more characters omitted]"
        return text

    def _run(self, coro: Any, timeout: float) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def close(self) -> None:
        if self._loop is None:
            return

        async def _shutdown() -> None:
            if self._session is not None:
                await self._session.__aexit__(None, None, None)
            if self._stdio_cm is not None:
                await self._stdio_cm.__aexit__(None, None, None)

        try:
            self._run(_shutdown(), timeout=5.0)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        self._loop = None
        self._session = None
