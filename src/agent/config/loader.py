from __future__ import annotations

import json
import os
import re
import math
import ipaddress
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Optional, Union

import yaml

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
DEFAULT_MODEL = "qwen-agentworld-35b-a3b"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 60
# Context window of the Qwen model this was developed against, found
# empirically (the deployment's model-info endpoints rejected our scoped
# API key, so it isn't queryable at runtime). Override it in the config
# to match your model.
DEFAULT_CONTEXT_WINDOW_TOKENS = 262144
DEFAULT_COMPACTION_THRESHOLD_PERCENT = 80.0
DEFAULT_COMPACTION_KEEP_LAST_USER_TURNS = 2

_TOKEN_ENV_VARS = ("LLM_API_KEY", "OPENAI_API_KEY", "ACCESS_TOKEN")
_SANDBOX_SSH_KEY_ENV_VAR = "SANDBOX_SSH_KEY"
DEFAULT_SANDBOX_SSH_PORT = 22
DEFAULT_SANDBOX_SSH_TIMEOUT = 15


class ConfigError(Exception):
    pass


@dataclass
class McpServerConfig:
    name: str
    command: str
    args: list[str]
    tools: Optional[list[str]] = None
    # Permission tiers for this server's tools - see agent.tools.permissions.
    # Anything not listed in either defaults to STRICT (asks every call, no
    # "always" shortcut): third-party MCP tools never earn a lighter tier
    # just by being present, only via explicit config here.
    auto_tools: Optional[list[str]] = None
    confirm_tools: Optional[list[str]] = None
    # Names of environment variables passed through to this server. The server
    # gets nothing else from the agent's environment (see tools/mcp/client.py).
    env: Optional[list[str]] = None


@dataclass
class MetricsConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    # Keep the /metrics endpoint up this long after the agent finishes, so
    # Prometheus can take one last scrape. Without it, whatever happened after
    # the final scrape (up to one interval) is lost the moment the process
    # exits. Set slightly above the scrape interval when a scraper is in use.
    linger_seconds: float = 0.0


@dataclass
class JsonlConfig:
    enabled: bool = True
    path: str = ".agent_monitoring/events.jsonl"
    queue_capacity: int = 1024
    max_record_bytes: int = 8192
    max_file_bytes: int = 10485760
    backup_count: int = 3


@dataclass
class MonitoringConfig:
    enabled: bool = False
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    jsonl: JsonlConfig = field(default_factory=JsonlConfig)
    shutdown_timeout_seconds: float = 2.0


@dataclass
class PermissionsConfig:
    # Tool names (builtin or MCP) that are blocked outright: never run,
    # never asked about. Wins over any tier the tool or its MCP server
    # config would otherwise give it.
    deny_tools: list[str] = field(default_factory=list)


@dataclass
class CourseMaterialConfig:
    id: str
    title: str
    path: str


@dataclass
class CourseConfig:
    title: str
    materials: list[CourseMaterialConfig] = field(default_factory=list)


@dataclass
class SandboxConfig:
    host: str
    user: str
    # None means no key was supplied: the sandbox_exec tool stays off rather
    # than failing startup for everyone who simply doesn't hold the key.
    private_key: Optional[str] = None
    port: int = DEFAULT_SANDBOX_SSH_PORT
    timeout: int = DEFAULT_SANDBOX_SSH_TIMEOUT


@dataclass
class SubAgentConfigEntry:
    role: str
    allowed_tools: list[str] = field(default_factory=list)
    max_steps: int = 6
    description: str = ""
    require_successful_tool: bool = False


@dataclass
class AgentConfig:
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    base_url: Optional[str] = None
    access_token: Optional[str] = None
    mcp_servers: list[McpServerConfig] = field(default_factory=list)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    courses: dict[str, CourseConfig] = field(default_factory=dict)
    subagents: dict[str, SubAgentConfigEntry] = field(default_factory=dict)
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    compaction_threshold_percent: float = DEFAULT_COMPACTION_THRESHOLD_PERCENT
    compaction_keep_last_user_turns: int = DEFAULT_COMPACTION_KEEP_LAST_USER_TURNS
    sandbox: Optional[SandboxConfig] = None


def _read_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text) or {}
    elif path.suffix == ".json":
        data = json.loads(text) if text.strip() else {}
    else:
        raise ConfigError(
            f"unsupported config format '{path.suffix}', expected .yaml, .yml, or .json"
        )

    if not isinstance(data, dict):
        raise ConfigError(f"'{path}' must contain a mapping at the top level")
    return data


def _resolve_access_token() -> Optional[str]:
    for name in _TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def load_config(path: Union[Path, str, None] = None) -> AgentConfig:
    config_path = Path(path) if path is not None else Path(
        os.environ.get("AGENT_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    )
    data = _read_file(config_path)

    # Guards against committing the token, per the lab's requirement.
    if "access_token" in data:
        raise ConfigError(
            f"'{config_path}' must not contain an access_token — "
            "set it via an environment variable instead (see .env.example)"
        )
    if isinstance(data.get("sandbox"), dict):
        for secret_field in ("password", "private_key"):
            if secret_field in data["sandbox"]:
                raise ConfigError(
                    f"'{config_path}' sandbox.{secret_field} must not be committed — "
                    f"set the key via the {_SANDBOX_SSH_KEY_ENV_VAR} environment variable instead"
                )

    model = data.get("model", DEFAULT_MODEL)
    temperature = data.get("temperature", DEFAULT_TEMPERATURE)
    max_tokens = data.get("max_tokens", DEFAULT_MAX_TOKENS)
    timeout = data.get("timeout", DEFAULT_TIMEOUT)
    base_url = data.get("base_url")

    if not isinstance(model, str) or not model:
        raise ConfigError("'model' must be a non-empty string")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ConfigError("'temperature' must be a number")
    if not (0 <= temperature <= 2):
        raise ConfigError("'temperature' must be between 0 and 2")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ConfigError("'max_tokens' must be a positive integer")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ConfigError("'timeout' must be a positive integer")
    if base_url is not None and not isinstance(base_url, str):
        raise ConfigError("'base_url' must be a string")

    context_window_tokens = data.get("context_window_tokens", DEFAULT_CONTEXT_WINDOW_TOKENS)
    compaction_threshold_percent = data.get("compaction_threshold_percent", DEFAULT_COMPACTION_THRESHOLD_PERCENT)
    compaction_keep_last_user_turns = data.get(
        "compaction_keep_last_user_turns", DEFAULT_COMPACTION_KEEP_LAST_USER_TURNS
    )

    if isinstance(context_window_tokens, bool) or not isinstance(context_window_tokens, int) or context_window_tokens <= 0:
        raise ConfigError("'context_window_tokens' must be a positive integer")
    if isinstance(compaction_threshold_percent, bool) or not isinstance(compaction_threshold_percent, (int, float)):
        raise ConfigError("'compaction_threshold_percent' must be a number")
    if not (0 < compaction_threshold_percent <= 100):
        raise ConfigError("'compaction_threshold_percent' must be between 0 and 100")
    if (
        isinstance(compaction_keep_last_user_turns, bool)
        or not isinstance(compaction_keep_last_user_turns, int)
        or compaction_keep_last_user_turns < 0
    ):
        raise ConfigError("'compaction_keep_last_user_turns' must be a non-negative integer")

    mcp_servers = _parse_mcp_servers(data.get("mcp_servers", []))
    courses = _parse_courses(data.get("courses", {}))
    subagents = _parse_subagents(data.get("subagents", {}))
    sandbox = _parse_sandbox(data.get("sandbox"))

    return AgentConfig(
        model=model,
        temperature=float(temperature),
        max_tokens=max_tokens,
        timeout=timeout,
        base_url=base_url,
        access_token=_resolve_access_token(),
        mcp_servers=mcp_servers,
        monitoring=_parse_monitoring(data.get("monitoring", {})),
        permissions=_parse_permissions(data.get("permissions", {})),
        courses=courses,
        subagents=subagents,
        context_window_tokens=context_window_tokens,
        compaction_threshold_percent=float(compaction_threshold_percent),
        compaction_keep_last_user_turns=compaction_keep_last_user_turns,
        sandbox=sandbox,
    )


def _parse_mcp_servers(raw: Any) -> list[McpServerConfig]:
    if not isinstance(raw, list):
        raise ConfigError("'mcp_servers' must be a list")

    servers = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"'mcp_servers[{i}]' must be a mapping")
        name = entry.get("name")
        command = entry.get("command")
        args = entry.get("args", [])
        tools = entry.get("tools")
        auto_tools = entry.get("auto_tools")
        confirm_tools = entry.get("confirm_tools")
        env = entry.get("env")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"'mcp_servers[{i}].name' must be a non-empty string")
        if not isinstance(command, str) or not command:
            raise ConfigError(f"'mcp_servers[{i}].command' must be a non-empty string")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ConfigError(f"'mcp_servers[{i}].args' must be a list of strings")
        if tools is not None and (not isinstance(tools, list) or not all(isinstance(t, str) for t in tools)):
            raise ConfigError(f"'mcp_servers[{i}].tools' must be a list of strings")
        for field_name, value in (("auto_tools", auto_tools), ("confirm_tools", confirm_tools)):
            if value is not None and (not isinstance(value, list) or not all(isinstance(t, str) for t in value)):
                raise ConfigError(f"'mcp_servers[{i}].{field_name}' must be a list of strings")
        if env is not None and (
            not isinstance(env, list)
            or not all(isinstance(v, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v) for v in env)
        ):
            raise ConfigError(f"'mcp_servers[{i}].env' must be a list of environment variable names")
        if auto_tools and confirm_tools and (set(auto_tools) & set(confirm_tools)):
            raise ConfigError(f"'mcp_servers[{i}]': a tool can't be in both auto_tools and confirm_tools")
        servers.append(McpServerConfig(
            name=name, command=command, args=args, tools=tools,
            auto_tools=auto_tools, confirm_tools=confirm_tools, env=env,
        ))
    return servers


def _parse_sandbox(raw: Any) -> Optional[SandboxConfig]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("'sandbox' must be a mapping")

    host = raw.get("host")
    user = raw.get("user")
    port = raw.get("port", DEFAULT_SANDBOX_SSH_PORT)
    timeout = raw.get("timeout", DEFAULT_SANDBOX_SSH_TIMEOUT)

    if not isinstance(host, str) or not host:
        raise ConfigError("'sandbox.host' must be a non-empty string")
    if not isinstance(user, str) or not user:
        raise ConfigError("'sandbox.user' must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not (0 < port < 65536):
        raise ConfigError("'sandbox.port' must be a valid port number")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ConfigError("'sandbox.timeout' must be a positive integer")

    private_key = os.environ.get(_SANDBOX_SSH_KEY_ENV_VAR) or None
    return SandboxConfig(host=host, user=user, private_key=private_key, port=port, timeout=timeout)


def _parse_permissions(raw: Any) -> PermissionsConfig:
    if raw is None:
        return PermissionsConfig()
    if not isinstance(raw, dict):
        raise ConfigError("'permissions' must be a mapping")
    unknown = set(raw) - {"deny_tools"}
    if unknown:
        raise ConfigError(f"unknown 'permissions' setting(s): {', '.join(sorted(unknown))}")
    deny_tools = raw.get("deny_tools") or []
    if not isinstance(deny_tools, list) or not all(isinstance(name, str) and name for name in deny_tools):
        raise ConfigError("'permissions.deny_tools' must be a list of tool names")
    return PermissionsConfig(deny_tools=list(dict.fromkeys(deny_tools)))


def _parse_monitoring(raw: Any) -> MonitoringConfig:
    def mapping(value: Any, allowed: set[str]) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) - allowed:
            raise ConfigError("invalid monitoring configuration fields")
        return value

    def boolean(value: Any) -> bool:
        if type(value) is not bool:
            raise ConfigError("monitoring flags must be booleans")
        return value

    def integer(value: Any, minimum: int, maximum: int) -> int:
        if type(value) is not int or not minimum <= value <= maximum:
            raise ConfigError("monitoring integer is outside its permitted range")
        return value

    raw = mapping(raw, {"enabled", "metrics", "jsonl", "shutdown_timeout_seconds"})
    metrics = mapping(raw.get("metrics", {}), {"enabled", "host", "port", "linger_seconds"})
    logs = mapping(raw.get("jsonl", {}), set(JsonlConfig.__dataclass_fields__))
    host = metrics.get("host", "127.0.0.1")
    try:
        if type(host) is not str:
            raise ValueError
        ipaddress.ip_address(host)
    except ValueError:
        raise ConfigError("monitoring host must be a literal IP address") from None
    path = logs.get("path", JsonlConfig.path)
    if type(path) is not str or not path or "\x00" in path or Path(path).is_absolute() or (
        PureWindowsPath(path).drive or ".." in PureWindowsPath(path).parts or ".." in Path(path).parts
    ):
        raise ConfigError("monitoring log path must be a confined relative file path")
    timeout = raw.get("shutdown_timeout_seconds", 2.0)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 10:
        raise ConfigError("monitoring shutdown timeout must be between 0 and 10 seconds")
    linger = metrics.get("linger_seconds", 0.0)
    if type(linger) not in (int, float) or not math.isfinite(linger) or not 0 <= linger <= 30:
        raise ConfigError("monitoring metrics linger must be between 0 and 30 seconds")
    log_config = JsonlConfig(
        enabled=boolean(logs.get("enabled", True)), path=path,
        queue_capacity=integer(logs.get("queue_capacity", 1024), 1, 65536),
        max_record_bytes=integer(logs.get("max_record_bytes", 8192), 1024, 8192),
        max_file_bytes=integer(logs.get("max_file_bytes", 10485760), 8192, 1073741824),
        backup_count=integer(logs.get("backup_count", 3), 0, 10),
    )
    return MonitoringConfig(
        enabled=boolean(raw.get("enabled", False)),
        metrics=MetricsConfig(
            boolean(metrics.get("enabled", True)), host, integer(metrics.get("port", 8000), 1, 65535), float(linger),
        ),
        jsonl=log_config, shutdown_timeout_seconds=float(timeout),
    )



def _parse_courses(raw: Any) -> dict[str, CourseConfig]:
    if not isinstance(raw, dict):
        raise ConfigError("'courses' must be a mapping")

    courses: dict[str, CourseConfig] = {}
    for course_id, course_data in raw.items():
        if not isinstance(course_data, dict):
            raise ConfigError(f"'courses.{course_id}' must be a mapping")
        title = course_data.get("title", course_id)
        raw_materials = course_data.get("materials", [])
        if not isinstance(raw_materials, list):
            raise ConfigError(f"'courses.{course_id}.materials' must be a list")
        materials: list[CourseMaterialConfig] = []
        for j, mat in enumerate(raw_materials):
            if not isinstance(mat, dict):
                raise ConfigError(f"'courses.{course_id}.materials[{j}]' must be a mapping")
            mat_id = mat.get("id", "")
            mat_title = mat.get("title", "")
            mat_path = mat.get("path", "")
            if not mat_id or not mat_path:
                raise ConfigError(f"'courses.{course_id}.materials[{j}]' requires 'id' and 'path'")
            materials.append(CourseMaterialConfig(id=mat_id, title=mat_title, path=mat_path))
        courses[course_id] = CourseConfig(title=title, materials=materials)
    return courses


def _parse_subagents(raw: Any) -> dict[str, SubAgentConfigEntry]:
    if not isinstance(raw, dict):
        raise ConfigError("'subagents' must be a mapping")

    subagents: dict[str, SubAgentConfigEntry] = {}
    for name, sa_data in raw.items():
        if not isinstance(sa_data, dict):
            raise ConfigError(f"'subagents.{name}' must be a mapping")
        role = sa_data.get("role", "")
        if not role:
            raise ConfigError(f"'subagents.{name}.role' is required")
        allowed_tools = sa_data.get("allowed_tools", [])
        if not isinstance(allowed_tools, list):
            raise ConfigError(f"'subagents.{name}.allowed_tools' must be a list")
        max_steps = sa_data.get("max_steps", 6)
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ConfigError(f"'subagents.{name}.max_steps' must be a positive integer")
        description = sa_data.get("description", "")
        require_successful_tool = sa_data.get("require_successful_tool", False)
        if not isinstance(require_successful_tool, bool):
            raise ConfigError(f"'subagents.{name}.require_successful_tool' must be a boolean")
        subagents[name] = SubAgentConfigEntry(
            role=role,
            allowed_tools=allowed_tools,
            max_steps=max_steps,
            description=description,
            require_successful_tool=require_successful_tool,
        )
    return subagents
