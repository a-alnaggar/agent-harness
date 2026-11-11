"""scripts/run.sh, exercised for real (bash, no mocks of the script itself)
against a fake container engine that records every call it receives. What is
asserted is the contract with Docker *and* Podman: which commands are issued,
in which order, with which flags."""

from __future__ import annotations

import os
import shutil
import socket
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SEPARATOR = "\x1f"

FAKE_ENGINE = """#!/usr/bin/env bash
printf '%s\\x1f' "$@" >> "$FAKE_LOG"
printf '\\n' >> "$FAKE_LOG"
case "$1 $2" in
  "info "*) exit "${FAKE_INFO_EXIT:-0}" ;;
  "network inspect") exit 1 ;;
esac
if [ "$1" = run ] && [[ "$*" == *"AGENT_CONFIG_PATH=/app/config/run.yaml"* ]]; then
  exit "${FAKE_AGENT_EXIT:-0}"
fi
exit 0
"""


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str
    calls: list[list[str]]

    def with_verb(self, *verb: str) -> list[list[str]]:
        return [call for call in self.calls if call[: len(verb)] == list(verb)]

    def index_of(self, predicate) -> int:
        return next(i for i, call in enumerate(self.calls) if predicate(call))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def project(tmp_path):
    """A throwaway copy of the launcher's surroundings, so a run never touches
    the real repo (workspace/, .run/, .env)."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/run.sh", tmp_path / "scripts/run.sh")
    for name in ("config/default.yaml",):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("fake-engine", "curl"):  # curl stub: every readiness probe succeeds at once
        path = bin_dir / name
        path.write_text(FAKE_ENGINE if name == "fake-engine" else "#!/usr/bin/env bash\nexit 0\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return tmp_path


def run_launcher(project: Path, *args: str, env: dict[str, str] | None = None, engine: str | None = "fake-engine") -> Result:
    log = project / "calls.log"
    log.write_text("")
    environment = {
        "PATH": f"{project / 'bin'}:/usr/bin:/bin",
        "HOME": str(project),
        "FAKE_LOG": str(log),
        "ACCESS_TOKEN": "test-token",
        "PROMETHEUS_PORT": str(free_port()),
        "GRAFANA_PORT": str(free_port()),
    }
    if engine is not None:
        environment["CONTAINER_ENGINE"] = str(project / "bin" / engine)
    environment.update(env or {})
    done = subprocess.run(
        ["bash", str(project / "scripts/run.sh"), *args], env=environment, cwd=project,
        capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
    )
    calls = [line.split(SEPARATOR)[:-1] for line in log.read_text().splitlines() if line]
    return Result(done.returncode, done.stdout, done.stderr, calls)


def agent_call(result: Result) -> list[str]:
    return next(call for call in result.calls if call[0] == "run" and "AGENT_CONFIG_PATH=/app/config/run.yaml" in call)


def test_default_run_starts_monitoring_then_runs_the_agent_last_in_the_foreground(project):
    result = run_launcher(project)

    assert result.returncode == 0, result.stderr
    prometheus = result.index_of(lambda c: c[:1] == ["run"] and "agent-harness-prometheus" in c)
    grafana = result.index_of(lambda c: c[:1] == ["run"] and "agent-harness-grafana" in c)
    agent = result.index_of(lambda c: c[:1] == ["run"] and "AGENT_CONFIG_PATH=/app/config/run.yaml" in c)
    assert prometheus < agent and grafana < agent
    assert agent == len(result.calls) - 1  # the agent is the last thing launched: no attach step, no wait
    assert "-d" not in agent_call(result)  # foreground, so the user is at its prompt immediately
    assert "-d" in result.calls[prometheus] and "-d" in result.calls[grafana]  # only the monitoring is background


def test_the_agent_is_reachable_by_prometheus_under_the_name_it_scrapes(project):
    call = agent_call(run_launcher(project))

    assert call[call.index("--network-alias") + 1] == "agent"  # config/prometheus/prometheus.yml targets agent:8000
    assert call[call.index("--network") + 1] == "agent-harness-monitoring"


def test_default_run_builds_generates_a_monitoring_config_and_creates_the_network(project):
    result = run_launcher(project)

    assert result.with_verb("build")
    generator = next(c for c in result.calls if "--entrypoint" in c)
    assert "/generate.py" in generator and "/out/agent.yaml" in generator
    assert result.with_verb("network", "create") == [["network", "create", "agent-harness-monitoring"]]
    mounted = agent_call(result)[agent_call(result).index("-v") + 1]
    assert mounted.endswith("/.run/agent.yaml:/app/config/run.yaml:ro,z")


def test_monitoring_ports_are_published_on_localhost_only(project):
    result = run_launcher(project, env={"PROMETHEUS_PORT": "19191", "GRAFANA_PORT": "13131"})

    published = [c[c.index("-p") + 1] for c in result.calls if "-p" in c]
    assert sorted(published) == ["127.0.0.1:13131:3000", "127.0.0.1:19191:9090"]


def test_grafana_opens_on_the_dashboard_without_a_login(project):
    grafana = next(c for c in run_launcher(project).calls if "agent-harness-grafana" in c and c[0] == "run")

    assert "GF_AUTH_ANONYMOUS_ENABLED=true" in grafana
    assert "GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer" in grafana
    assert any(a.startswith("GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=") and a.endswith("agent-monitoring.json") for a in grafana)
    assert not any(a.startswith("GF_SECURITY_ADMIN_PASSWORD") for a in grafana)


def test_a_configured_grafana_admin_password_is_passed_through(project):
    grafana = next(c for c in run_launcher(project, env={"GRAFANA_ADMIN_PASSWORD": "s3cret"}).calls
                   if "agent-harness-grafana" in c and c[0] == "run")

    assert "GF_SECURITY_ADMIN_PASSWORD=s3cret" in grafana


def test_every_host_path_mount_carries_the_selinux_label_podman_needs(project):
    (project / ".secrets/gmail").mkdir(parents=True)
    for name in ("credentials.json", "gmail-token.json"):
        (project / ".secrets/gmail" / name).write_text("{}")
    result = run_launcher(project)

    mounts = [c[i + 1] for c in result.calls for i, a in enumerate(c) if a == "-v"]
    host_mounts = [m for m in mounts if m.startswith("/")]
    assert host_mounts and all(m.endswith(":z") or m.endswith(",z") for m in host_mounts), host_mounts


def test_no_monitoring_runs_only_the_agent_with_the_config_mounted_directly(project):
    result = run_launcher(project, "--no-monitoring")

    assert result.returncode == 0, result.stderr
    assert [c[0] for c in result.calls] == ["build", "run"]
    call = agent_call(result)
    assert "--network" not in call and "--name" not in call
    assert any(a.endswith("/config/default.yaml:/app/config/run.yaml:ro,z") for a in call)


def test_stop_removes_the_monitoring_containers_and_keeps_the_data(project):
    result = run_launcher(project, "--stop")

    assert result.returncode == 0
    assert result.with_verb("rm", "-f", "agent-harness-prometheus", "agent-harness-grafana")
    assert not result.with_verb("volume") and not result.with_verb("build")


def test_clean_also_deletes_the_stored_data(project):
    result = run_launcher(project, "--clean")

    assert result.with_verb("volume", "rm")
    assert {"agent-harness-prometheus-data", "agent-harness-grafana-data"} <= set(result.with_verb("volume", "rm")[0])


def test_the_agents_exit_status_is_the_launchers_exit_status(project):
    assert run_launcher(project, env={"FAKE_AGENT_EXIT": "7"}).returncode == 7


def test_it_says_where_monitoring_lives_and_how_to_stop_it(project):
    result = run_launcher(project, env={"GRAFANA_PORT": "13131"})

    assert "http://localhost:13131" in result.stdout
    assert "./scripts/run.sh --stop" in result.stdout


def test_a_missing_llm_token_fails_early_with_the_fix(project):
    result = run_launcher(project, env={"ACCESS_TOKEN": ""})

    assert result.returncode == 1
    assert ".env" in result.stderr and "ACCESS_TOKEN" in result.stderr
    assert not result.with_verb("build")


def test_the_token_can_come_from_dot_env_instead_of_the_environment(project):
    (project / ".env").write_text('ACCESS_TOKEN="from-dotenv"\n')

    result = run_launcher(project, env={"ACCESS_TOKEN": ""})

    assert result.returncode == 0, result.stderr
    assert "--env-file" in agent_call(result)


def test_a_busy_port_fails_readably_before_anything_is_started(project):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        result = run_launcher(project, env={"PROMETHEUS_PORT": str(port)})

    assert result.returncode == 1
    assert f"Port {port} is already in use" in result.stderr and "PROMETHEUS_PORT=<port>" in result.stderr
    assert not [c for c in result.calls if c[:2] == ["run", "-d"]]


def test_gmail_files_are_mounted_as_their_own_directory_never_the_repo(project):
    (project / ".secrets/gmail").mkdir(parents=True)
    for name in ("credentials.json", "gmail-token.json"):
        (project / ".secrets/gmail" / name).write_text("{}")

    mounts = [a for a in agent_call(run_launcher(project))]
    gmail = [m for m in mounts if m.endswith(":/gmail:z")]

    assert len(gmail) == 1 and gmail[0].endswith("/.secrets/gmail:/gmail:z")
    assert not any(m.endswith(":/repo") or ":/repo:" in m for m in mounts)


def test_gmail_files_left_in_the_repo_root_are_copied_into_the_mounted_folder(project):
    for name in ("credentials.json", "gmail-token.json"):
        (project / name).write_text(f"{name} content")

    result = run_launcher(project)

    for name in ("credentials.json", "gmail-token.json"):
        assert (project / ".secrets/gmail" / name).read_text() == f"{name} content"
    assert oct((project / ".secrets/gmail").stat().st_mode & 0o777) == "0o700"
    assert any(a.endswith("/.secrets/gmail:/gmail:z") for a in agent_call(result))
    assert "Copied credentials.json" in result.stderr


def test_a_token_refreshed_inside_the_folder_is_not_overwritten_by_an_older_root_copy(project):
    """The agent rewrites .secrets/gmail/gmail-token.json on refresh; a stale
    copy left in the repo root must not undo that."""
    folder = project / ".secrets/gmail"
    folder.mkdir(parents=True)
    (project / "gmail-token.json").write_text("old")
    (folder / "gmail-token.json").write_text("refreshed")
    (folder / "credentials.json").write_text("{}")
    os.utime(project / "gmail-token.json", (1, 1))

    result = run_launcher(project)

    assert (folder / "gmail-token.json").read_text() == "refreshed"
    assert "Copied" not in result.stderr


def test_only_half_of_the_gmail_files_keeps_gmail_off_and_says_why(project):
    (project / "credentials.json").write_text("{}")

    result = run_launcher(project)

    assert "needs both credentials.json and gmail-token.json" in result.stderr
    assert not any(a.endswith(":/gmail:z") for a in agent_call(result))


def test_a_custom_gmail_dir_is_used_as_is_without_copying_root_files_into_it(project, tmp_path_factory):
    custom = tmp_path_factory.mktemp("gmail")
    (project / "credentials.json").write_text("{}")

    run_launcher(project, env={"GMAIL_DIR": str(custom)})

    assert not (custom / "credentials.json").exists()


def test_the_course_material_is_copied_where_the_indexer_looks(project):
    (project / "resources/courses/c1").mkdir(parents=True)
    (project / "resources/courses/c1/syllabus.pdf").write_text("pdf")

    run_launcher(project)

    assert (project / "workspace/resources/courses/c1/syllabus.pdf").read_text() == "pdf"


def test_without_a_tty_the_agent_still_gets_stdin_so_tasks_can_be_piped(project):
    call = agent_call(run_launcher(project))

    assert "-i" in call and "-it" not in call


def test_it_uses_the_engine_named_in_container_engine(project):
    result = run_launcher(project)

    assert result.calls  # the fake, chosen via CONTAINER_ENGINE, received every call


def test_it_falls_back_to_podman_when_docker_is_installed_but_not_running(project):
    """Auto-detection: docker's `info` fails (daemon down), podman's works."""
    for name, info_exit in (("docker", 1), ("podman", 0)):
        path = project / "bin" / name
        path.write_text(FAKE_ENGINE.replace('"${FAKE_INFO_EXIT:-0}"', str(info_exit)))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        (project / f"{name}.log").write_text("")
    result = run_launcher(project, engine=None)

    assert result.returncode == 0, result.stderr
    # the fake logs into one shared file, so use the presence of a build as the signal
    # that the launcher settled on an engine at all, and check it was not docker:
    assert result.with_verb("build")
    assert "Building the agent image (podman)" in result.stdout


def test_it_reports_when_neither_engine_is_available(project):
    minimal = project / "nobin"
    minimal.mkdir()
    for tool in ("bash", "dirname", "sed", "grep", "cut", "tail", "mkdir", "cp", "cat", "seq", "sleep"):
        found = shutil.which(tool, path="/usr/bin:/bin")
        if found:
            (minimal / tool).symlink_to(found)
    if any(shutil.which(engine, path="/usr/bin:/bin") for engine in ("docker", "podman")):
        pytest.skip("a real engine is installed in a system directory")
    result = run_launcher(project, engine=None, env={"PATH": str(minimal)})

    assert result.returncode == 1
    assert "Neither a running docker nor podman" in result.stderr


def test_quotes_around_dot_env_values_are_removed_before_the_container_sees_them(project):
    """--env-file keeps quotes as part of the value: TAVILY_API_KEY="tvly-x"
    reached the agent with the quotes and Tavily answered 401."""
    original = 'ACCESS_TOKEN=plain\nTAVILY_API_KEY="tvly-x"\nNOTION_TOKEN=\'ntn-y\'\n# comment="kept"\nODD="a"b\n'
    (project / ".env").write_text(original)

    call = agent_call(run_launcher(project))

    passed = call[call.index("--env-file") + 1]
    assert passed.endswith(".run/agent.env")
    assert (project / ".run/agent.env").read_text().splitlines() == [
        "ACCESS_TOKEN=plain", "TAVILY_API_KEY=tvly-x", "NOTION_TOKEN=ntn-y", '# comment="kept"', 'ODD="a"b',
    ]
    assert (project / ".env").read_text() == original  # the user's file is left alone
    assert oct((project / ".run/agent.env").stat().st_mode & 0o777) == "0o600"
