"""Setup check: `python -m agent.doctor` reports, feature by feature, whether
this machine and environment can run the agent, and what to do if not.

It reads the same config file and environment the agent does, plus a .env
in the current directory (run it from the repo root, whose .env the agent
also reads when started from a source checkout), but never
prints a secret value - only whether it is set and, for the sandbox key,
whether the agent would accept it. Nothing is contacted over the network:
a set token or key is not proof the service will accept it.

Deliberately importable on an old Python (no 3.10+ syntax at runtime, and
agent.config imported lazily), so running it with the wrong interpreter
still says so instead of failing with a traceback.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

MIN_PYTHON = (3, 12)
TOKEN_VARS = ("ACCESS_TOKEN", "LLM_API_KEY", "OPENAI_API_KEY")
CORE_MODULES = ("openai", "dotenv", "yaml", "rich", "mcp", "prometheus_client", "fitz")
GMAIL_MODULES = ("mcp", "googleapiclient", "google_auth_oauthlib", "google_auth_httplib2", "email_validator")


@dataclass
class Check:
    feature: str
    required: bool
    ok: bool
    message: str = "OK"
    fix: str = ""


def _is_set(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def _missing_modules(names: tuple[str, ...]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def check_python() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] < MIN_PYTHON:
        return Check("Python", True, False, f"{version} is too old, 3.12 is required",
                     "Use Python 3.12 (or run everything in the container: ./scripts/run.sh).")
    missing = _missing_modules(CORE_MODULES)
    if missing:
        return Check("Python", True, False, f"missing packages: {', '.join(missing)}",
                     'pip install -e ".[gmail,sandbox]"')
    return Check("Python", True, True, f"OK ({version})")


def load_agent_config() -> tuple[Any, Optional[Check]]:
    """The parsed config, or a failed required check saying why it can't be loaded."""
    try:
        from agent.config import ConfigError, load_config
    except Exception as exc:
        return None, Check("Core configuration", True, False, f"cannot import the agent ({exc})",
                           'Install it first: pip install -e ".[gmail,sandbox]"')
    try:
        return load_config(), None
    except ConfigError as exc:
        return None, Check("Core configuration", True, False, f"config error: {exc}",
                           "Fix the file, or point AGENT_CONFIG_PATH at a valid one.")


def check_core(config: Any) -> Check:
    config_path = os.environ.get("AGENT_CONFIG_PATH", "config/default.yaml")
    token_var = next((name for name in TOKEN_VARS if _is_set(name)), None)
    if token_var is None:
        return Check("Core configuration", True, False, "missing ACCESS_TOKEN",
                     "cp .env.example .env, then set ACCESS_TOKEN to your LLM API key.")
    return Check("Core configuration", True, True,
                 f"OK (config {config_path}, model {config.model}, token from {token_var})")


def check_gmail(config: Any) -> Check:
    if not any(server.name == "gmail" for server in config.mcp_servers):
        return Check("Gmail MCP", False, False, "disabled (no 'gmail' entry under mcp_servers)")
    missing = _missing_modules(GMAIL_MODULES)
    if missing:
        return Check("Gmail MCP", False, False, "missing the gmail extra",
                     'pip install -e ".[gmail]"  (already in the Docker image)')
    from mcp.server.mcpserver.exceptions import ToolError

    from agent.mcp_servers.gmail_auth import credential_paths

    try:
        credentials, token = credential_paths()
    except ToolError as exc:
        return Check("Gmail MCP", False, False, f"invalid path setting: {exc}",
                     "Set GMAIL_CREDENTIALS_PATH and GMAIL_TOKEN_PATH to absolute file paths, "
                     "or remove them (and any empty 'NAME=' lines) from .env to use /gmail.")
    if not credentials.is_file():
        return Check("Gmail MCP", False, False, f"missing GMAIL_CREDENTIALS_PATH (no file at {credentials})",
                     "Put the OAuth desktop-client JSON at .secrets/gmail/credentials.json "
                     "(./scripts/run.sh mounts it at /gmail) or set GMAIL_CREDENTIALS_PATH. See the README (Gmail).")
    if not token.is_file():
        return Check("Gmail MCP", False, False, f"missing GMAIL_TOKEN_PATH (no token at {token})",
                     "Authorize the test account once: python -m agent.mcp_servers.gmail_auth")
    allowlist = "recipient allowlist set" if _is_set("GMAIL_ALLOWED_RECIPIENTS") else "no recipient allowlist"
    return Check("Gmail MCP", False, True, f"OK (credentials and token present, {allowlist}; token validity not checked)")


def check_notion(config: Any) -> Check:
    if not any(server.name == "notion" for server in config.mcp_servers):
        return Check("Notion MCP", False, False, "disabled (no 'notion' entry under mcp_servers)")
    if shutil.which("npx") is None:
        return Check("Notion MCP", False, False, "missing npx (Node.js)",
                     "Install Node.js 18+ with npm, or use the Docker image, which has it.")
    if not _is_set("NOTION_TOKEN"):
        return Check("Notion MCP", False, False, "missing NOTION_TOKEN",
                     "Create an internal integration at https://www.notion.so/my-integrations, "
                     "share the target pages with it, and set NOTION_TOKEN in .env.")
    return Check("Notion MCP", False, True, "OK (npx found, NOTION_TOKEN set; package fetched by npx on first start)")


def check_sandbox(config: Any) -> Check:
    sandbox = config.sandbox
    if sandbox is None:
        return Check("Remote execution", False, False, "disabled (no 'sandbox' block in config)")
    target = f"{sandbox.user}@{sandbox.host}:{sandbox.port}"
    if sandbox.private_key is None:
        return Check("Remote execution", False, False, f"missing SANDBOX_SSH_KEY (sandbox block targets {target})",
                     "Set SANDBOX_SSH_KEY to the base64 of a dedicated passphrase-less key. See the README (Remote execution).")
    if importlib.util.find_spec("paramiko") is None:
        return Check("Remote execution", False, False, "missing the sandbox extra",
                     'pip install -e ".[sandbox]"  (already in the Docker image)')
    from agent.tools.sandbox.ssh_client import load_private_key

    try:
        key = load_private_key(sandbox.private_key)
    except ValueError as exc:
        # load_private_key's messages never quote the key itself.
        return Check("Remote execution", False, False, f"invalid SANDBOX_SSH_KEY: {exc}",
                     "Re-encode it: base64 < sandbox_key | tr -d '\\n'  (the key content, not its path).")
    return Check("Remote execution", False, True,
                 f"OK ({key.get_name()} key accepted, target {target}; connection not tested)")


def check_web_search() -> Check:
    if not _is_set("TAVILY_API_KEY"):
        return Check("Web search", False, False, "missing TAVILY_API_KEY",
                     "Get a free key at https://tavily.com and set TAVILY_API_KEY in .env.")
    return Check("Web search", False, True, "OK (TAVILY_API_KEY set)")


def check_monitoring(config: Any) -> Check:
    if not config.monitoring.enabled:
        return Check("Monitoring", False, True,
                     "off in this config (./scripts/run.sh turns it on for its own runs)")
    metrics = config.monitoring.metrics
    return Check("Monitoring", False, True, f"OK (enabled, metrics on {metrics.host}:{metrics.port})")


def check_container_runtime() -> Check:
    """The same choice scripts/run.sh makes: CONTAINER_ENGINE if set, else
    the first of docker/podman that is installed and answering."""
    forced = os.environ.get("CONTAINER_ENGINE", "").strip()
    candidates = [forced] if forced else ["docker", "podman"]
    for engine in candidates:
        if shutil.which(engine) is None:
            continue
        try:
            running = subprocess.run([engine, "info"], capture_output=True, timeout=20).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            running = False
        if running:
            return Check("Container runtime", False, True, f"OK ({engine})")
        if forced:
            return Check("Container runtime", False, False, f"{engine} is installed but not running",
                         f"Start {engine} (for rootless Podman: podman machine start, or log in to a user session).")
    wanted = forced or "docker or podman"
    return Check("Container runtime", False, False, f"missing {wanted}",
                 "Needed only on the host, for ./scripts/run.sh and the docker/podman commands in the README.")


def run_checks() -> list[Check]:
    checks = [check_python()]
    config, failed = load_agent_config()
    if failed is not None:
        checks.append(failed)
    else:
        checks += [
            check_core(config),
            check_gmail(config),
            check_notion(config),
            check_sandbox(config),
            check_web_search(),
            check_monitoring(config),
        ]
    checks.append(check_container_runtime())
    return checks


def _load_dotenv() -> Optional[Path]:
    """Loads ./.env without overriding real environment variables, the same
    precedence the agent uses."""
    env_file = Path.cwd() / ".env"
    if not env_file.is_file():
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None
    load_dotenv(env_file, override=False)
    return env_file


def report(checks: list[Check], env_file: Optional[Path]) -> str:
    lines = [f"Environment: {'.env loaded from ' + str(env_file) if env_file else 'no .env here, process environment only'}"]
    for title, required in (("Required", True), ("Optional", False)):
        lines.append(title)
        for check in (c for c in checks if c.required == required):
            lines.append(f"  {check.feature}: {check.message}")
            if not check.ok and check.fix:
                lines.append(f"    fix: {check.fix}")
    failed = [c.feature for c in checks if c.required and not c.ok]
    lines.append(f"Result: NOT READY ({', '.join(failed)})" if failed else "Result: ready to run the agent")
    return "\n".join(lines)


def main() -> int:
    env_file = _load_dotenv()
    checks = run_checks()
    print(report(checks, env_file))
    return 1 if any(c.required and not c.ok for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
