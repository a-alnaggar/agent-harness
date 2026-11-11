import json

import pytest

from agent.config import AgentConfig, ConfigError, load_config

_TOKEN_ENV_VARS = ("LLM_API_KEY", "OPENAI_API_KEY", "ACCESS_TOKEN")


@pytest.fixture(autouse=True)
def clear_token_env(monkeypatch):
    for name in _TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_load_yaml_config(tmp_path):
    path = write(tmp_path, "config.yaml", "model: my-model\ntemperature: 0.5\n")

    config = load_config(path)

    assert config == AgentConfig(
        model="my-model", temperature=0.5, max_tokens=4096, timeout=60, base_url=None, access_token=None
    )


def test_load_json_config(tmp_path):
    path = write(tmp_path, "config.json", json.dumps({"model": "my-model", "temperature": 0.7}))

    config = load_config(path)

    assert config.model == "my-model"
    assert config.temperature == 0.7


def test_defaults_used_when_keys_missing(tmp_path):
    path = write(tmp_path, "config.yaml", "{}\n")

    config = load_config(path)

    assert config.model == "qwen-agentworld-35b-a3b"
    assert config.temperature == 0.2
    assert config.max_tokens == 4096
    assert config.timeout == 60


def test_base_url_is_optional(tmp_path):
    path = write(tmp_path, "config.yaml", "model: m\ntemperature: 0.1\nbase_url: https://example.com/v1\n")

    config = load_config(path)

    assert config.base_url == "https://example.com/v1"


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_unsupported_extension_raises(tmp_path):
    path = write(tmp_path, "config.txt", "model: m\n")

    with pytest.raises(ConfigError, match="unsupported config format"):
        load_config(path)


def test_non_mapping_top_level_raises(tmp_path):
    path = write(tmp_path, "config.yaml", "- one\n- two\n")

    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_invalid_model_type_raises(tmp_path):
    path = write(tmp_path, "config.yaml", "model: 123\n")

    with pytest.raises(ConfigError, match="'model'"):
        load_config(path)


def test_empty_model_raises(tmp_path):
    path = write(tmp_path, "config.yaml", "model: ''\n")

    with pytest.raises(ConfigError, match="'model'"):
        load_config(path)


def test_invalid_temperature_type_raises(tmp_path):
    path = write(tmp_path, "config.yaml", "temperature: 'hot'\n")

    with pytest.raises(ConfigError, match="'temperature'"):
        load_config(path)


def test_temperature_out_of_range_raises(tmp_path):
    path = write(tmp_path, "config.yaml", "temperature: 5\n")

    with pytest.raises(ConfigError, match="'temperature'"):
        load_config(path)


def test_boolean_temperature_rejected(tmp_path):
    path = write(tmp_path, "config.yaml", "temperature: true\n")

    with pytest.raises(ConfigError, match="'temperature'"):
        load_config(path)


def test_invalid_max_tokens_raises(tmp_path):
    path = write(tmp_path, "config.yaml", "max_tokens: -1\n")

    with pytest.raises(ConfigError, match="'max_tokens'"):
        load_config(path)


def test_non_integer_max_tokens_raises(tmp_path):
    path = write(tmp_path, "config.yaml", "max_tokens: 'lots'\n")

    with pytest.raises(ConfigError, match="'max_tokens'"):
        load_config(path)


def test_invalid_timeout_raises(tmp_path):
    path = write(tmp_path, "config.yaml", "timeout: 0\n")

    with pytest.raises(ConfigError, match="'timeout'"):
        load_config(path)


def test_access_token_in_file_is_rejected(tmp_path):
    path = write(tmp_path, "config.yaml", "model: m\naccess_token: sk-should-not-be-here\n")

    with pytest.raises(ConfigError, match="must not contain an access_token"):
        load_config(path)


def test_access_token_read_from_env(tmp_path, monkeypatch):
    path = write(tmp_path, "config.yaml", "model: m\ntemperature: 0.2\n")
    monkeypatch.setenv("ACCESS_TOKEN", "secret-value")

    config = load_config(path)

    assert config.access_token == "secret-value"


def test_access_token_env_precedence(tmp_path, monkeypatch):
    path = write(tmp_path, "config.yaml", "model: m\ntemperature: 0.2\n")
    monkeypatch.setenv("ACCESS_TOKEN", "low-priority")
    monkeypatch.setenv("OPENAI_API_KEY", "mid-priority")
    monkeypatch.setenv("LLM_API_KEY", "high-priority")

    config = load_config(path)

    assert config.access_token == "high-priority"


def test_access_token_none_when_unset(tmp_path):
    path = write(tmp_path, "config.yaml", "model: m\ntemperature: 0.2\n")

    config = load_config(path)

    assert config.access_token is None


def test_default_config_path_env_var(tmp_path, monkeypatch):
    path = write(tmp_path, "config.yaml", "model: from-env-path\ntemperature: 0.3\n")
    monkeypatch.setenv("AGENT_CONFIG_PATH", str(path))

    config = load_config()

    assert config.model == "from-env-path"


def test_repo_default_config_is_loadable():
    config = load_config("config/default.yaml")

    assert config.model
    assert 0 <= config.temperature <= 2


def test_mcp_server_with_no_tools_key_registers_everything(tmp_path):
    path = write(tmp_path, "config.yaml", """
model: m
temperature: 0.2
mcp_servers:
  - name: gmail
    command: python3
    args: ["-m", "agent.mcp_servers.gmail"]
""")

    config = load_config(path)

    assert config.mcp_servers[0].tools is None


def test_mcp_server_tools_allowlist_is_parsed(tmp_path):
    path = write(tmp_path, "config.yaml", """
model: m
temperature: 0.2
mcp_servers:
  - name: notion
    command: npx
    args: ["-y", "@notionhq/notion-mcp-server"]
    tools:
      - API-post-search
      - API-post-page
""")

    config = load_config(path)

    assert config.mcp_servers[0].tools == ["API-post-search", "API-post-page"]


def test_mcp_server_tools_must_be_a_list_of_strings(tmp_path):
    path = write(tmp_path, "config.yaml", """
model: m
temperature: 0.2
mcp_servers:
  - name: notion
    command: npx
    args: []
    tools: "API-post-search"
""")

    with pytest.raises(ConfigError, match="tools"):
        load_config(path)


def test_courses_and_subagents_loaded_correctly(tmp_path):
    path = write(
        tmp_path,
        "config.yaml",
        """
model: m
temperature: 0.2
courses:
  cs101:
    title: "Intro to CS"
    materials:
      - id: "lec1"
        title: "Lecture 1"
        path: "data/lec1.pdf"
subagents:
  course_knowledge:
    role: "You are course assistant."
    allowed_tools: ["search_course_material"]
    max_steps: 4
    description: "Course specialist"
""",
    )

    config = load_config(path)

    assert "cs101" in config.courses
    course = config.courses["cs101"]
    assert course.title == "Intro to CS"
    assert len(course.materials) == 1
    assert course.materials[0].id == "lec1"
    assert course.materials[0].title == "Lecture 1"
    assert course.materials[0].path == "data/lec1.pdf"

    assert "course_knowledge" in config.subagents
    sa = config.subagents["course_knowledge"]
    assert sa.role == "You are course assistant."
    assert sa.allowed_tools == ["search_course_material"]
    assert sa.max_steps == 4
    assert sa.description == "Course specialist"
    assert sa.require_successful_tool is False


def test_subagent_can_require_successful_tool(tmp_path):
    path = write(
        tmp_path,
        "config.yaml",
        """
model: m
subagents:
  memory_manager:
    role: "Manage memory"
    require_successful_tool: true
""",
    )

    config = load_config(path)

    assert config.subagents["memory_manager"].require_successful_tool is True


def test_courses_must_be_dict(tmp_path):
    path = write(tmp_path, "config.yaml", "model: m\ntemperature: 0.2\ncourses: [1, 2]\n")
    with pytest.raises(ConfigError, match="'courses' must be a mapping"):
        load_config(path)


def test_course_material_missing_id_or_path_raises(tmp_path):
    path = write(
        tmp_path,
        "config.yaml",
        """
model: m
temperature: 0.2
courses:
  cs101:
    title: "Intro"
    materials:
      - title: "No ID or path"
""",
    )
    with pytest.raises(ConfigError, match="requires 'id' and 'path'"):
        load_config(path)


def test_subagent_missing_role_raises(tmp_path):
    path = write(
        tmp_path,
        "config.yaml",
        """
model: m
temperature: 0.2
subagents:
  helper:
    allowed_tools: []
""",
    )
    with pytest.raises(ConfigError, match="role"):
        load_config(path)


def test_subagent_invalid_max_steps_raises(tmp_path):
    path = write(
        tmp_path,
        "config.yaml",
        """
model: m
temperature: 0.2
subagents:
  helper:
    role: "Help"
    max_steps: -2
""",
    )
    with pytest.raises(ConfigError, match="max_steps"):
        load_config(path)


def test_compaction_settings_default_when_not_specified(tmp_path):
    path = write(tmp_path, "config.yaml", "model: m\ntemperature: 0.2\n")

    config = load_config(path)

    assert config.context_window_tokens == 262144
    assert config.compaction_threshold_percent == 80.0
    assert config.compaction_keep_last_user_turns == 2


def test_compaction_settings_are_read_from_config(tmp_path):
    path = write(
        tmp_path,
        "config.yaml",
        """
model: m
temperature: 0.2
context_window_tokens: 32000
compaction_threshold_percent: 70
compaction_keep_last_user_turns: 5
""",
    )

    config = load_config(path)

    assert config.context_window_tokens == 32000
    assert config.compaction_threshold_percent == 70.0
    assert config.compaction_keep_last_user_turns == 5


def test_context_window_tokens_must_be_a_positive_integer(tmp_path):
    path = write(tmp_path, "config.yaml", "model: m\ntemperature: 0.2\ncontext_window_tokens: -5\n")

    with pytest.raises(ConfigError, match="context_window_tokens"):
        load_config(path)


def test_compaction_threshold_percent_must_be_between_0_and_100(tmp_path):
    path = write(tmp_path, "config.yaml", "model: m\ntemperature: 0.2\ncompaction_threshold_percent: 150\n")

    with pytest.raises(ConfigError, match="compaction_threshold_percent"):
        load_config(path)


def test_compaction_keep_last_user_turns_must_be_non_negative(tmp_path):
    path = write(tmp_path, "config.yaml", "model: m\ntemperature: 0.2\ncompaction_keep_last_user_turns: -1\n")

    with pytest.raises(ConfigError, match="compaction_keep_last_user_turns"):
        load_config(path)


def test_mcp_server_auto_and_confirm_tools_default_to_none(tmp_path):
    path = write(tmp_path, "config.yaml", """
model: m
temperature: 0.2
mcp_servers:
  - name: gmail
    command: python3
    args: ["-m", "agent.mcp_servers.gmail"]
""")

    config = load_config(path)

    assert config.mcp_servers[0].auto_tools is None
    assert config.mcp_servers[0].confirm_tools is None


def test_mcp_server_auto_and_confirm_tools_are_parsed(tmp_path):
    path = write(tmp_path, "config.yaml", """
model: m
temperature: 0.2
mcp_servers:
  - name: gmail
    command: python3
    args: []
    auto_tools:
      - search_emails
      - read_email
    confirm_tools:
      - create_draft
""")

    config = load_config(path)

    assert config.mcp_servers[0].auto_tools == ["search_emails", "read_email"]
    assert config.mcp_servers[0].confirm_tools == ["create_draft"]


def test_mcp_server_auto_tools_must_be_a_list_of_strings(tmp_path):
    path = write(tmp_path, "config.yaml", """
model: m
temperature: 0.2
mcp_servers:
  - name: gmail
    command: python3
    args: []
    auto_tools: "search_emails"
""")

    with pytest.raises(ConfigError, match="auto_tools"):
        load_config(path)


def test_mcp_server_confirm_tools_must_be_a_list_of_strings(tmp_path):
    path = write(tmp_path, "config.yaml", """
model: m
temperature: 0.2
mcp_servers:
  - name: gmail
    command: python3
    args: []
    confirm_tools: "create_draft"
""")

    with pytest.raises(ConfigError, match="confirm_tools"):
        load_config(path)


def test_mcp_server_tool_cannot_be_in_both_auto_and_confirm_tools(tmp_path):
    path = write(tmp_path, "config.yaml", """
model: m
temperature: 0.2
mcp_servers:
  - name: gmail
    command: python3
    args: []
    auto_tools:
      - search_emails
    confirm_tools:
      - search_emails
""")

    with pytest.raises(ConfigError, match="both auto_tools and confirm_tools"):
        load_config(path)


def test_repo_default_config_classifies_gmail_and_notion_tools(tmp_path):
    """Regression guard for the actual shipped config: gmail's send_email
    and notion's write tools must not silently end up in auto_tools."""
    config = load_config("config/default.yaml")

    gmail = next(s for s in config.mcp_servers if s.name == "gmail")
    notion = next(s for s in config.mcp_servers if s.name == "notion")

    assert "send_email" not in (gmail.auto_tools or [])
    assert "send_email" not in (gmail.confirm_tools or [])
    assert "search_emails" in gmail.auto_tools
    assert "API-post-page" in notion.confirm_tools
    assert "API-post-page" not in (notion.auto_tools or [])


def test_sandbox_is_none_when_not_configured(tmp_path):
    path = write(tmp_path, "config.yaml", "model: my-model\n")

    config = load_config(path)

    assert config.sandbox is None


def test_sandbox_is_parsed_from_config_and_env_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_SSH_KEY", "key-material")
    path = write(tmp_path, "config.yaml", """
model: my-model
sandbox:
  host: sandbox.example.com
  user: agent
  port: 2222
  timeout: 30
""")

    config = load_config(path)

    assert config.sandbox.host == "sandbox.example.com"
    assert config.sandbox.user == "agent"
    assert config.sandbox.port == 2222
    assert config.sandbox.timeout == 30
    assert config.sandbox.private_key == "key-material"


def test_sandbox_defaults_port_and_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_SSH_KEY", "key-material")
    path = write(tmp_path, "config.yaml", """
model: my-model
sandbox:
  host: sandbox.example.com
  user: agent
""")

    config = load_config(path)

    assert config.sandbox.port == 22
    assert config.sandbox.timeout == 15


@pytest.mark.parametrize("secret_field", ["password", "private_key"])
def test_sandbox_secret_in_yaml_is_rejected(tmp_path, monkeypatch, secret_field):
    monkeypatch.setenv("SANDBOX_SSH_KEY", "key-material")
    path = write(tmp_path, "config.yaml", f"""
model: my-model
sandbox:
  host: sandbox.example.com
  user: agent
  {secret_field}: hardcoded-oops
""")

    with pytest.raises(ConfigError, match=f"sandbox.{secret_field} must not be committed"):
        load_config(path)


def test_sandbox_without_a_key_loads_with_no_key_instead_of_failing(tmp_path, monkeypatch):
    """Not holding the key must not stop anyone from starting the agent -
    the tool just stays off (see __main__)."""
    monkeypatch.delenv("SANDBOX_SSH_KEY", raising=False)
    path = write(tmp_path, "config.yaml", """
model: my-model
sandbox:
  host: sandbox.example.com
  user: agent
""")

    config = load_config(path)

    assert config.sandbox.host == "sandbox.example.com"
    assert config.sandbox.private_key is None


def test_an_empty_key_env_var_counts_as_no_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_SSH_KEY", "")
    path = write(tmp_path, "config.yaml", """
model: my-model
sandbox:
  host: sandbox.example.com
  user: agent
""")

    assert load_config(path).sandbox.private_key is None


def test_sandbox_requires_host_and_user(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_SSH_KEY", "key-material")
    path = write(tmp_path, "config.yaml", """
model: my-model
sandbox:
  user: agent
""")

    with pytest.raises(ConfigError, match="sandbox.host"):
        load_config(path)


def test_permissions_default_to_denying_nothing(tmp_path):
    config = load_config(write(tmp_path, "config.yaml", "model: m\n"))

    assert config.permissions.deny_tools == []


def test_permissions_deny_tools_are_parsed_and_deduplicated_in_order(tmp_path):
    path = write(tmp_path, "config.yaml", """
model: m
permissions:
  deny_tools:
    - send_email
    - API-patch-page
    - send_email
""")

    assert load_config(path).permissions.deny_tools == ["send_email", "API-patch-page"]


def test_an_empty_permissions_block_is_fine(tmp_path):
    assert load_config(write(tmp_path, "config.yaml", "model: m\npermissions:\n")).permissions.deny_tools == []


@pytest.mark.parametrize("permissions_yaml, message", [
    ("permissions: just-a-string", "'permissions' must be a mapping"),
    ("permissions:\n  deny_tools: send_email", "must be a list of tool names"),
    ("permissions:\n  deny_tools: [1, 2]", "must be a list of tool names"),
    ("permissions:\n  deny_tools: ['']", "must be a list of tool names"),
    ("permissions:\n  deny_tool: [send_email]", "unknown 'permissions' setting"),
])
def test_invalid_permissions_are_rejected(tmp_path, permissions_yaml, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, "config.yaml", f"model: m\n{permissions_yaml}"))


def test_the_shipped_default_config_denies_nothing_by_default():
    assert load_config("config/default.yaml").permissions.deny_tools == []


@pytest.mark.parametrize("sandbox_yaml, expected_message", [
    ("sandbox: just-a-string", "'sandbox' must be a mapping"),
    ("sandbox:\n  host: h\n", "sandbox.user"),
    ("sandbox:\n  host: h\n  user: u\n  port: 70000\n", "sandbox.port"),
    ("sandbox:\n  host: h\n  user: u\n  port: true\n", "sandbox.port"),
    ("sandbox:\n  host: h\n  user: u\n  timeout: 0\n", "sandbox.timeout"),
])
def test_invalid_sandbox_settings_are_rejected(tmp_path, sandbox_yaml, expected_message):
    path = write(tmp_path, "config.yaml", f"model: m\n{sandbox_yaml}")

    with pytest.raises(ConfigError, match=expected_message):
        load_config(path)


def test_mcp_server_env_is_optional_and_parsed_as_names(tmp_path):
    path = write(tmp_path, "config.yaml", """
model: m
mcp_servers:
  - name: a
    command: c
  - name: b
    command: c
    env: [NOTION_TOKEN, GMAIL_TOKEN_PATH]
""")

    servers = load_config(path).mcp_servers

    assert servers[0].env is None
    assert servers[1].env == ["NOTION_TOKEN", "GMAIL_TOKEN_PATH"]


@pytest.mark.parametrize("env_yaml", ["env: NOTION_TOKEN", "env: [1]", "env: ['A=b']", "env: ['has space']", "env: ['']"])
def test_mcp_server_env_rejects_anything_but_variable_names(tmp_path, env_yaml):
    path = write(tmp_path, "config.yaml", f"model: m\nmcp_servers:\n  - name: a\n    command: c\n    {env_yaml}\n")

    with pytest.raises(ConfigError, match="list of environment variable names"):
        load_config(path)


def test_the_shipped_config_passes_each_mcp_server_only_what_it_needs():
    servers = {s.name: s for s in load_config("config/default.yaml").mcp_servers}

    assert set(servers["gmail"].env) == {"GMAIL_CREDENTIALS_PATH", "GMAIL_TOKEN_PATH", "GMAIL_ALLOWED_RECIPIENTS"}
    assert servers["notion"].env == ["NOTION_TOKEN"]
    assert not {"ACCESS_TOKEN", "SANDBOX_SSH_KEY"} & {v for s in servers.values() for v in s.env}
