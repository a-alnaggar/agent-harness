# Monitoring

Prometheus metrics, a Grafana dashboard, and structured JSONL traces for
agent runs, LLM calls, tool calls, memory operations, permission
decisions, and sub-agent runs. Off by default.

- [Quick start](#quick-start)
- [How it's wired](#how-its-wired)
- [What gets recorded](#what-gets-recorded)
- [Prometheus metrics](#prometheus-metrics)
- [Grafana dashboard](#grafana-dashboard)
- [Structured JSONL logs](#structured-jsonl-logs)
- [Windows setup](#windows-setup)
- [Offline validation](#offline-validation)

## Quick start

Runs the agent, Prometheus, and Grafana together, with Docker or Podman.

```bash
./scripts/run.sh
```

You land straight at the `agent >` prompt. Prometheus and Grafana run in the
background; the agent runs in the foreground of your terminal, exactly like a
plain `docker run -it` (there is no separate attach step). It needs
`ACCESS_TOKEN` in `.env` (see `.env.example`).

- Grafana: http://localhost:3000 — opens on **Agent Harness Monitoring**, no
  login (viewer access, published on `127.0.0.1` only). To edit, log in as
  `admin` with `GRAFANA_ADMIN_PASSWORD` (default `admin`).
- Prometheus: http://localhost:9090
- Raw structured logs: `workspace/.agent_monitoring/events.jsonl`

`/exit` leaves Prometheus and Grafana running so you can keep browsing the
dashboard. `./scripts/run.sh --stop` stops them (stored data is kept),
`--clean` also deletes it, `--no-monitoring` runs the agent alone.

The agent runs with everything in `config/default.yaml` (real model, Gmail,
courses, sub-agents). For each run the script generates a monitoring-enabled
copy of that file (`scripts/generate_monitoring_config.py`, inside the image —
metrics bound to `0.0.0.0` so Prometheus can reach them) and mounts it; the
repo's own config files are never modified. `resources/courses/` is copied
into the workspace so `course_knowledge` has something to search.

**Docker or Podman.** Whichever is installed and running is used (docker
first); force one with `CONTAINER_ENGINE=podman`. The script sticks to
commands the two share — no compose — and fully qualifies image names so
Podman never has to guess a registry. Host folders are mounted with `:z` (the
SELinux label Podman needs on Fedora/RHEL; harmless elsewhere). Podman must be
version 4 or newer: containers find each other by name, which older CNI-based
Podman only does with an extra plugin.

**Ports** default to 3000 and 9090; if one is taken the script says so and
tells you which variable to set (`GRAFANA_PORT`, `PROMETHEUS_PORT`).

## How fresh is the dashboard?

An event reaches the dashboard after at most one Prometheus scrape plus one
Grafana refresh — a few seconds, not tens:

| Step | Setting | Value |
|---|---|---|
| Prometheus scrapes the agent | `scrape_interval` in `config/prometheus/prometheus.yml` | 2s |
| Grafana sizes its query windows from | `timeInterval` in the datasource | 2s (must match the scrape interval) |
| Dashboard re-queries | `refresh` in the dashboard JSON | 5s (Grafana's minimum) |

The agent's `/metrics` is an in-memory snapshot, so scraping it that often is
free. When the agent exits, the endpoint stays up for `metrics.linger_seconds`
(3s in the generated config, longer than the scrape interval) so Prometheus
takes one last scrape; without it, anything after the final scrape was lost
(runs completed, tool calls, tokens).

## How it's wired

`monitoring.enabled: true` makes `main()` build a `MonitoringRuntime`,
wrap its consumers (Prometheus, JSONL) in one `CompositeHook`, and inject
it into the SQLite store, core-profile manager, both tool registries,
`LLMClient`, and `ReactLoop`. Sub-agent loops get the same hook, with
`RunContext` threaded through `delegate_task` so their events nest under
the parent run.

Instrumentation is fail-open: if it breaks, it logs a diagnostic and
doesn't touch agent behavior.

## What gets recorded

`LifecycleEvent`/`RunContext` (`src/agent/observability.py`) carry a
timestamp, run/span/parent-span correlation, component, operation,
outcome, and duration. Attributes are narrow by design: token counts,
operation metadata, error class names — never prompts, messages, tool
arguments/results, memory values, file paths, or credentials.

| Family | Source | Notes |
|---|---|---|
| `agent.run.*` | `ReactLoop` | one per parent run and per sub-agent run, `agent_id` tells them apart |
| `llm.call.*` | `LLMClient` | token counts when the provider reports them |
| `tool.call.*` | `ToolRegistry` | tool identity + origin (`builtin`/`mcp`/`subagent`). `delegate_task` is origin `subagent`: its duration is a whole sub-agent run, so the aggregate tool-latency panels leave it out |
| `memory.operation.*` | SQLite store, core profile | read/save/forget, always inside the run that caused it (startup housekeeping is not recorded) |
| `permission.request.*` | `ReactLoop` confirmation gate | tier, decision, error class. Denied calls never reach `ToolRegistry`, so no `tool.call` event |

## Prometheus metrics

- `agent_runs_started`, `agent_runs_active`, `agent_runs_completed_total`,
  `agent_run_duration_seconds` — labeled by `agent_id`
- `agent_llm_calls_total`, `agent_llm_duration_seconds`, `agent_llm_tokens_total`
- `agent_tool_calls_total`, `agent_tool_duration_seconds` — labeled by tool + origin
- `agent_memory_operations_total`, `agent_memory_duration_seconds`
- `agent_permission_requests_total`, `agent_permission_duration_seconds`
- `agent_monitoring_log_dropped_total`

## Grafana dashboard

Auto-provisioned, no manual setup. Panels: throughput and active runs,
latency, success/failure rates, LLM tokens, tool usage, memory
operations, permission decisions, sub-agent runs and their latency.

## Structured JSONL logs

`JsonlHook`/`RotatingWriter` write one JSON event per line to
`workspace/.agent_monitoring/`, rotating at 10 MiB / 3 backups. This is
how you reconstruct one specific run, sub-agent delegations included, by
filtering on `run_id`. Grafana doesn't read these — no log datasource.

## Windows

`scripts/run.sh` is bash — use WSL (recommended) or Git Bash, with Docker
Desktop or Podman available inside it.

## Offline validation

```bash
pytest tests/unit/test_monitoring_deployment.py -q
docker run --rm --network none \
  --mount "type=bind,source=$(pwd)/config/prometheus,target=/checks,readonly" \
  --entrypoint promtool prom/prometheus:v3.5.5 check config /checks/prometheus.yml
docker run --rm --network none \
  --mount "type=bind,source=$(pwd)/tests/fixtures,target=/checks,readonly" \
  --entrypoint promtool prom/prometheus:v3.5.5 test rules /checks/monitoring_rules.test.yaml
```

`promtool test rules` checks every dashboard query against fixed sample
data with known expected values — catches a query that's syntactically
fine but numerically wrong, which pytest alone can't.
