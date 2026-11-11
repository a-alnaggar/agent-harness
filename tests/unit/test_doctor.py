import base64
import os
import subprocess
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

import agent.doctor as doctor
from agent.config import AgentConfig, McpServerConfig, MonitoringConfig, SandboxConfig

SECRET = "super-secret-value-that-must-never-be-printed"
ENV_VARS = (
    "ACCESS_TOKEN", "LLM_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY", "NOTION_TOKEN", "SANDBOX_SSH_KEY",
    "GMAIL_CREDENTIALS_PATH", "GMAIL_TOKEN_PATH", "GMAIL_ALLOWED_RECIPIENTS", "CONTAINER_ENGINE", "AGENT_CONFIG_PATH",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # main() loads .env into os.environ; a private copy keeps that from
    # leaking into other tests.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def make_config(**overrides):
    defaults = dict(model="m", temperature=0.2, max_tokens=100, timeout=10)
    defaults.update(overrides)
    return AgentConfig(**defaults)


def server(name):
    return McpServerConfig(name=name, command="x", args=[])


def ed25519_base64():
    pem = ed25519.Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, serialization.NoEncryption(),
    )
    return base64.b64encode(pem).decode()


# --- Python ---

def test_python_too_old_is_a_required_failure(monkeypatch):
    monkeypatch.setattr(doctor.sys, "version_info", (3, 9, 6))

    check = doctor.check_python()

    assert (check.required, check.ok, check.message) == (True, False, "3.9.6 is too old, 3.12 is required")


def test_python_reports_missing_packages(monkeypatch):
    monkeypatch.setattr(doctor, "CORE_MODULES", ("yaml", "no_such_package_for_doctor"))

    check = doctor.check_python()

    assert not check.ok
    assert check.message == "missing packages: no_such_package_for_doctor"
    assert "pip install" in check.fix


def test_python_ok_in_the_test_environment():
    assert doctor.check_python().ok


# --- Core configuration ---

def test_core_without_any_token_is_missing_access_token():
    check = doctor.check_core(make_config())

    assert (check.required, check.ok, check.message) == (True, False, "missing ACCESS_TOKEN")


def test_core_names_the_variable_the_token_came_from_but_never_its_value(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", SECRET)

    check = doctor.check_core(make_config())

    assert check.ok
    assert "token from LLM_API_KEY" in check.message
    assert SECRET not in check.message


def test_an_invalid_config_file_is_a_required_failure(tmp_path, monkeypatch):
    bad = tmp_path / "bad.yaml"
    bad.write_text("access_token: nope\n")
    monkeypatch.setenv("AGENT_CONFIG_PATH", str(bad))

    config, check = doctor.load_agent_config()

    assert config is None
    assert check.required and not check.ok
    assert "must not contain an access_token" in check.message


# --- Gmail MCP ---

def test_gmail_disabled_when_not_configured():
    assert doctor.check_gmail(make_config()).message.startswith("disabled")


def test_gmail_missing_credentials_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GMAIL_CREDENTIALS_PATH", str(tmp_path / "credentials.json"))
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(tmp_path / "gmail-token.json"))

    check = doctor.check_gmail(make_config(mcp_servers=[server("gmail")]))

    assert not check.ok
    assert check.message.startswith("missing GMAIL_CREDENTIALS_PATH")


def test_gmail_missing_token_points_at_the_auth_command(tmp_path, monkeypatch):
    (tmp_path / "credentials.json").write_text("{}")
    monkeypatch.setenv("GMAIL_CREDENTIALS_PATH", str(tmp_path / "credentials.json"))
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(tmp_path / "gmail-token.json"))

    check = doctor.check_gmail(make_config(mcp_servers=[server("gmail")]))

    assert check.message.startswith("missing GMAIL_TOKEN_PATH")
    assert "python -m agent.mcp_servers.gmail_auth" in check.fix


def test_gmail_relative_path_is_reported(monkeypatch):
    monkeypatch.setenv("GMAIL_CREDENTIALS_PATH", "credentials.json")

    check = doctor.check_gmail(make_config(mcp_servers=[server("gmail")]))

    assert check.message.startswith("invalid path setting")


def test_gmail_ok_when_both_files_exist(tmp_path, monkeypatch):
    for name in ("credentials.json", "gmail-token.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setenv("GMAIL_CREDENTIALS_PATH", str(tmp_path / "credentials.json"))
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(tmp_path / "gmail-token.json"))

    check = doctor.check_gmail(make_config(mcp_servers=[server("gmail")]))

    assert check.ok and "no recipient allowlist" in check.message


# --- Notion MCP ---

def test_notion_disabled_when_not_configured():
    assert doctor.check_notion(make_config()).message == "disabled (no 'notion' entry under mcp_servers)"


def test_notion_needs_npx(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)

    assert doctor.check_notion(make_config(mcp_servers=[server("notion")])).message == "missing npx (Node.js)"


def test_notion_needs_its_token(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/npx")

    assert doctor.check_notion(make_config(mcp_servers=[server("notion")])).message == "missing NOTION_TOKEN"


def test_notion_ok_without_printing_the_token(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/npx")
    monkeypatch.setenv("NOTION_TOKEN", SECRET)

    check = doctor.check_notion(make_config(mcp_servers=[server("notion")]))

    assert check.ok and SECRET not in check.message


# --- Remote execution ---

def sandbox(private_key=None):
    return SandboxConfig(host="sandbox.example.invalid", user="sandbox", private_key=private_key)


def test_sandbox_disabled_without_a_block():
    assert doctor.check_sandbox(make_config()).message == "disabled (no 'sandbox' block in config)"


def test_sandbox_block_without_key_is_missing_the_key():
    check = doctor.check_sandbox(make_config(sandbox=sandbox()))

    assert check.message == "missing SANDBOX_SSH_KEY (sandbox block targets sandbox@sandbox.example.invalid:22)"


def test_sandbox_invalid_key_is_reported_without_echoing_it():
    check = doctor.check_sandbox(make_config(sandbox=sandbox(private_key=SECRET)))

    assert not check.ok
    assert check.message.startswith("invalid SANDBOX_SSH_KEY:")
    assert SECRET not in check.message and SECRET not in check.fix


def test_sandbox_valid_key_is_accepted_without_echoing_it():
    key = ed25519_base64()

    check = doctor.check_sandbox(make_config(sandbox=sandbox(private_key=key)))

    assert check.ok
    assert check.message.startswith("OK (ssh-ed25519 key accepted")
    assert key not in check.message


# --- Web search, monitoring ---

def test_web_search_missing_then_set(monkeypatch):
    assert doctor.check_web_search().message == "missing TAVILY_API_KEY"
    monkeypatch.setenv("TAVILY_API_KEY", SECRET)
    check = doctor.check_web_search()
    assert check.ok and SECRET not in check.message


def test_monitoring_off_or_on():
    assert doctor.check_monitoring(make_config()).message.startswith("off in this config")
    assert "enabled" in doctor.check_monitoring(make_config(monitoring=MonitoringConfig(enabled=True))).message


# --- Container runtime ---

def fake_engines(monkeypatch, installed, running):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/usr/bin/{name}" if name in installed else None)
    monkeypatch.setattr(
        doctor.subprocess, "run",
        lambda args, **kwargs: SimpleNamespace(returncode=0 if args[0] in running else 1),
    )


def test_container_runtime_prefers_a_running_docker(monkeypatch):
    fake_engines(monkeypatch, installed={"docker", "podman"}, running={"docker", "podman"})

    assert doctor.check_container_runtime().message == "OK (docker)"


def test_container_runtime_falls_back_to_podman_when_docker_is_not_running(monkeypatch):
    fake_engines(monkeypatch, installed={"docker", "podman"}, running={"podman"})

    assert doctor.check_container_runtime().message == "OK (podman)"


def test_container_runtime_honours_container_engine(monkeypatch):
    fake_engines(monkeypatch, installed={"docker", "podman"}, running={"docker"})
    monkeypatch.setenv("CONTAINER_ENGINE", "podman")

    assert doctor.check_container_runtime().message == "podman is installed but not running"


def test_container_runtime_missing(monkeypatch):
    fake_engines(monkeypatch, installed=set(), running=set())

    check = doctor.check_container_runtime()

    assert not check.ok and not check.required
    assert check.message == "missing docker or podman"


def test_container_runtime_timeout_counts_as_not_running(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)

    def hang(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 20)

    monkeypatch.setattr(doctor.subprocess, "run", hang)

    assert doctor.check_container_runtime().message == "missing docker or podman"


# --- the whole command ---

def write_config(tmp_path, body):
    path = tmp_path / "agent.yaml"
    path.write_text("model: m\n" + body)
    return path


def test_main_is_ready_with_only_the_token_and_never_prints_secrets(tmp_path, monkeypatch, capsys):
    fake_engines(monkeypatch, installed=set(), running=set())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_CONFIG_PATH", str(write_config(tmp_path, "sandbox:\n  host: h\n  user: u\n")))
    monkeypatch.setenv("ACCESS_TOKEN", SECRET)
    monkeypatch.setenv("SANDBOX_SSH_KEY", SECRET)

    code = doctor.main()

    out = capsys.readouterr().out
    assert code == 0
    assert SECRET not in out
    assert "Core configuration: OK" in out
    assert "Remote execution: invalid SANDBOX_SSH_KEY" in out
    assert "Notion MCP: disabled" in out
    assert "Container runtime: missing docker or podman" in out
    assert out.rstrip().endswith("Result: ready to run the agent")


def test_main_fails_without_a_token_and_says_what_to_do(tmp_path, monkeypatch, capsys):
    fake_engines(monkeypatch, installed=set(), running=set())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_CONFIG_PATH", str(write_config(tmp_path, "")))

    code = doctor.main()

    out = capsys.readouterr().out
    assert code == 1
    assert "Core configuration: missing ACCESS_TOKEN" in out
    assert "fix: cp .env.example .env" in out
    assert "Result: NOT READY (Core configuration)" in out


def test_main_reads_dot_env_without_overriding_the_environment(tmp_path, monkeypatch, capsys):
    fake_engines(monkeypatch, installed=set(), running=set())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_CONFIG_PATH", str(write_config(tmp_path, "")))
    monkeypatch.setenv("TAVILY_API_KEY", "")
    (tmp_path / ".env").write_text(f"ACCESS_TOKEN={SECRET}\nTAVILY_API_KEY=from-dot-env\n")

    code = doctor.main()

    out = capsys.readouterr().out
    assert code == 0
    assert f".env loaded from {tmp_path / '.env'}" in out
    assert "Core configuration: OK" in out
    # an empty TAVILY_API_KEY already in the environment wins over .env
    assert "Web search: missing TAVILY_API_KEY" in out
    assert SECRET not in out
