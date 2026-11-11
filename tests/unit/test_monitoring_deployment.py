import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from agent.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def read_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def test_generate_monitoring_config_carries_over_all_capabilities(tmp_path) -> None:
    """The generator (run by run_monitoring.sh when a real ACCESS_TOKEN is
    set) must produce a config with default.yaml's real mcp_servers,
    courses, and subagents intact - not a stripped-down copy - with only
    monitoring turned on and rebound to 0.0.0.0."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_monitoring_config import generate

    dest = tmp_path / "generated.yaml"
    generate(ROOT / "config/default.yaml", dest)
    config = load_config(dest)

    assert config.base_url is None
    assert config.monitoring.enabled and config.monitoring.metrics.host == "0.0.0.0"
    default_config = load_config(ROOT / "config/default.yaml")
    assert {s.name for s in config.mcp_servers} == {s.name for s in default_config.mcp_servers}
    assert set(config.courses) == set(default_config.courses)
    assert set(config.subagents) == set(default_config.subagents)


def seconds(value: str) -> float:
    assert value.endswith("s")
    return float(value[:-1])


def test_generated_config_keeps_metrics_up_past_the_last_scrape(tmp_path) -> None:
    """Otherwise everything after the final scrape is lost when the agent exits
    (seen live: 6 runs started, 5 completed, the last delegation missing)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_monitoring_config import generate

    dest = tmp_path / "generated.yaml"
    generate(ROOT / "config/default.yaml", dest)
    scrape_interval = seconds(read_yaml("config/prometheus/prometheus.yml")["global"]["scrape_interval"])

    assert load_config(dest).monitoring.metrics.linger_seconds > scrape_interval


def test_default_config_keeps_monitoring_off_bound_to_localhost_with_no_linger() -> None:
    monitoring = load_config(ROOT / "config/default.yaml").monitoring

    assert not monitoring.enabled
    assert monitoring.metrics.host == "127.0.0.1"
    assert monitoring.metrics.linger_seconds == 0


def test_the_delay_from_event_to_dashboard_is_a_few_seconds_not_tens() -> None:
    """Worst case is roughly scrape interval + dashboard refresh (Grafana's
    minimum refresh is 5s). It used to be 15s + 15s."""
    scrape = read_yaml("config/prometheus/prometheus.yml")["global"]
    source = read_yaml("config/grafana/provisioning/datasources/prometheus.yaml")["datasources"][0]
    dashboard = json.loads((ROOT / "config/grafana/dashboards/agent-monitoring.json").read_text(encoding="utf-8"))

    scrape_interval = seconds(scrape["scrape_interval"])
    assert scrape_interval <= 5
    assert seconds(scrape["scrape_timeout"]) < scrape_interval
    # Grafana sizes $__rate_interval from this, so it has to match the real scrape interval
    assert seconds(source["jsonData"]["timeInterval"]) == scrape_interval
    assert seconds(dashboard["refresh"]) <= 5


def test_the_dockerfile_uses_the_lab_base_image_pinned_by_digest() -> None:
    first_line = (ROOT / "Dockerfile").read_text().splitlines()[0]

    # fully qualified so Podman doesn't have to resolve a short name; it is
    # still python:3.12-slim-bookworm, the image the lab requires
    assert first_line.startswith("FROM docker.io/library/python:3.12-slim-bookworm@sha256:")


def test_the_launcher_pins_monitoring_images_and_publishes_only_on_localhost() -> None:
    script = (ROOT / "scripts/run.sh").read_text()

    assert 'PROMETHEUS_IMAGE="docker.io/prom/prometheus:v3.5.5@sha256:' in script
    assert 'GRAFANA_IMAGE="docker.io/grafana/grafana:12.3.3@sha256:' in script
    publishes = re.findall(r'-p "([^"]+)"', script)
    assert len(publishes) == 2 and all(p.startswith("127.0.0.1:") for p in publishes)


def test_the_launcher_never_mounts_the_repository_into_the_container() -> None:
    script = (ROOT / "scripts/run.sh").read_text()

    assert "/repo" not in script
    assert '"$PWD:' not in script and ":/repo" not in script
    commands = [line for line in script.splitlines() if not line.lstrip().startswith("#")]
    assert not any("compose" in line for line in commands)  # plain engine commands only: works the same on podman


def test_the_launcher_is_valid_bash_and_executable() -> None:
    script = ROOT / "scripts/run.sh"

    assert os.access(script, os.X_OK)
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_provisioning_matches_scrape_and_dashboard() -> None:
    scrape = read_yaml("config/prometheus/prometheus.yml")
    assert scrape["global"]["scrape_interval"] == "2s"
    assert scrape["scrape_configs"][0]["static_configs"][0]["targets"] == ["agent:8000"]
    source = read_yaml("config/grafana/provisioning/datasources/prometheus.yaml")["datasources"][0]
    assert source["uid"] == "agent-prometheus" and source["url"] == "http://prometheus:9090"
    provider = read_yaml("config/grafana/provisioning/dashboards/agent.yaml")["providers"][0]
    assert not provider["allowUiUpdates"]
    dashboard = json.loads((ROOT / "config/grafana/dashboards/agent-monitoring.json").read_text(encoding="utf-8"))
    assert dashboard["uid"] == "agent-monitoring" and dashboard["refresh"] == "5s"
    assert len({panel["id"] for panel in dashboard["panels"]}) == 18
    expressions = []
    for panel in dashboard["panels"]:
        for target in panel["targets"]:
            assert target["datasource"]["uid"] == source["uid"]
            assert 'job="agent"' in target["expr"]
            expressions.append(target["expr"])
    joined = " ".join(expressions)
    for metric in ("agent_runs_completed_total", "agent_runs_active", "agent_llm_tokens_total",
                   "agent_tool_calls_total", "agent_memory_operations_total", "agent_permission_requests_total",
                   "agent_monitoring_log_dropped_total"):
        assert metric in joined
    for component in ("run", "llm", "tool", "memory", "permission"):
        assert f"agent_{component}_duration_seconds_bucket" in joined
    assert not any(panel["type"] == "text" for panel in dashboard["panels"])
    subagent_panels = [p for p in dashboard["panels"] if "sub-agent" in p["title"].lower()]
    assert len(subagent_panels) == 2
    subagent_exprs = " ".join(t["expr"] for p in subagent_panels for t in p["targets"])
    assert 'agent_id!="parent"' in subagent_exprs


def test_every_dashboard_query_has_deterministic_promtool_parse_coverage() -> None:
    dashboard = json.loads((ROOT / "config/grafana/dashboards/agent-monitoring.json").read_text(encoding="utf-8"))
    expected = {target["expr"].replace("$__rate_interval", "2m").replace("$__range", "4m")
                for panel in dashboard["panels"] for target in panel["targets"]}
    fixture = read_yaml("tests/fixtures/monitoring_rules.test.yaml")
    actual = {test["expr"] for test in fixture["tests"][0]["promql_expr_test"]}
    assert expected == actual


def test_aggregate_tool_latency_excludes_delegation_but_per_tool_panels_keep_it() -> None:
    """delegate_task spans a whole nested sub-agent run (seconds), while real
    tools take milliseconds: averaged together, "tool latency" was ~2.8s
    when every actual tool call took ~0.01s."""
    dashboard = json.loads((ROOT / "config/grafana/dashboards/agent-monitoring.json").read_text(encoding="utf-8"))
    by_title = {panel["title"]: [t["expr"] for t in panel["targets"]] for panel in dashboard["panels"]}

    for title in ("p95 component latency", "Mean component latency"):
        tool_exprs = [e for e in by_title[title] if "agent_tool_duration_seconds" in e]
        assert tool_exprs and all('origin!="subagent"' in e for e in tool_exprs), title
    per_tool = " ".join(by_title["p95 tool latency"])
    assert "by(le,tool,origin)" in per_tool and 'origin!="subagent"' not in per_tool


def test_every_host_the_monitoring_config_names_resolves_on_the_launcher_network() -> None:
    """Seen live: Grafana showed nothing because its data source asked for
    "prometheus", a name no container on the network answered to."""
    script = (ROOT / "scripts/run.sh").read_text()
    source = read_yaml("config/grafana/provisioning/datasources/prometheus.yaml")["datasources"][0]
    target = read_yaml("config/prometheus/prometheus.yml")["scrape_configs"][0]["static_configs"][0]["targets"][0]

    aliases = set(re.findall(r"--network-alias (\S+)", script))
    assert re.match(r"https?://([^:/]+)", source["url"]).group(1) in aliases
    assert target.split(":")[0] in aliases
