#!/usr/bin/env bash
# Runs the agent with Prometheus + Grafana monitoring, using Docker or Podman.
# One command, and you land straight at the agent prompt:
#
#   ./scripts/run.sh
#
# Prometheus and Grafana start in the background; the agent runs in the
# foreground of this terminal, exactly like a plain `docker run -it`. Only
# commands both engines share are used (no compose), so it behaves the same
# with either.
#
#   ./scripts/run.sh --no-monitoring   agent only (same as the plain docker/podman run)
#   ./scripts/run.sh --stop            stop Prometheus + Grafana (data is kept)
#   ./scripts/run.sh --clean           stop them and delete their stored data
#
# Environment: CONTAINER_ENGINE=docker|podman, GRAFANA_PORT, PROMETHEUS_PORT,
# CONFIG_SOURCE (default config/default.yaml), GMAIL_DIR (default .secrets/gmail).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=${AGENT_IMAGE:-agent-harness}
NETWORK=agent-harness-monitoring
PROMETHEUS=agent-harness-prometheus
GRAFANA=agent-harness-grafana
AGENT=agent-harness-run
PROMETHEUS_VOLUME=agent-harness-prometheus-data
GRAFANA_VOLUME=agent-harness-grafana-data
# Fully qualified so Podman never has to guess a registry ("short-name" errors).
PROMETHEUS_IMAGE="docker.io/prom/prometheus:v3.5.5@sha256:332c2f43e7e389d74d3893b55bb02fbbd684208e681eeb604641d5d769c0fe2a"
GRAFANA_IMAGE="docker.io/grafana/grafana:12.3.3@sha256:9e1e77ade304069aee3196e9a4f210830e96e80ce9a2640891eccc324b152faf"
PROMETHEUS_PORT=${PROMETHEUS_PORT:-9090}
GRAFANA_PORT=${GRAFANA_PORT:-3000}
CONFIG_SOURCE=${CONFIG_SOURCE:-config/default.yaml}
GMAIL_DIR=${GMAIL_DIR:-$PWD/.secrets/gmail}

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }

mode=run
with_monitoring=1
for arg in "$@"; do
  case "$arg" in
    --no-monitoring) with_monitoring=0 ;;
    --stop) mode=stop ;;
    --clean) mode=clean ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

pick_engine() {
  if [ -n "${CONTAINER_ENGINE:-}" ]; then echo "$CONTAINER_ENGINE"; return 0; fi
  local candidate
  for candidate in docker podman; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" info >/dev/null 2>&1; then
      echo "$candidate"; return 0
    fi
  done
  return 1
}
ENGINE=$(pick_engine) || { echo "Neither a running docker nor podman was found." >&2; exit 1; }

remove_monitoring() { "$ENGINE" rm -f "$PROMETHEUS" "$GRAFANA" >/dev/null 2>&1 || true; }

if [ "$mode" = stop ] || [ "$mode" = clean ]; then
  remove_monitoring
  if [ "$mode" = clean ]; then
    "$ENGINE" volume rm -f "$PROMETHEUS_VOLUME" "$GRAFANA_VOLUME" >/dev/null 2>&1 || true
    echo "Monitoring stopped and its stored data deleted."
  else
    echo "Monitoring stopped (stored data kept; --clean deletes it)."
  fi
  "$ENGINE" network rm "$NETWORK" >/dev/null 2>&1 || true
  exit 0
fi

# Value of a variable from the environment, else from .env (no sourcing: .env
# is data, not a script).
env_value() {
  local key=$1 value=${!1:-}
  if [ -z "$value" ] && [ -f .env ]; then
    value=$(grep -E "^${key}=" .env | tail -n 1 | cut -d= -f2- | sed -e "s/^[\"']//" -e "s/[\"']\$//" || true)
  fi
  printf '%s' "$value"
}

if [ -z "$(env_value ACCESS_TOKEN)$(env_value LLM_API_KEY)$(env_value OPENAI_API_KEY)" ]; then
  echo "No LLM token found. Copy .env.example to .env and set ACCESS_TOKEN (your LLM API key)." >&2
  exit 1
fi

echo "Building the agent image ($ENGINE)..."
"$ENGINE" build -q -t "$IMAGE" . >/dev/null

mkdir -p workspace .run
# The course-material indexer only looks inside /workspace.
if [ -d resources/courses ]; then
  mkdir -p workspace/resources
  cp -R resources/courses workspace/resources/
fi

# Gmail credentials live in their own directory (never the whole repo): the
# token file is replaced on refresh, which needs a directory, not a single file.
# Files dropped in the repo root are copied there (when newer), so that still works.
if [ "$GMAIL_DIR" = "$PWD/.secrets/gmail" ]; then
  for name in credentials.json gmail-token.json; do
    if [ -f "$name" ] && { [ ! -f "$GMAIL_DIR/$name" ] || [ "$name" -nt "$GMAIL_DIR/$name" ]; }; then
      install -d -m 700 "$GMAIL_DIR"
      cp "$name" "$GMAIL_DIR/$name"
      echo "Copied $name from the repo root to .secrets/gmail/ (the folder the container sees)." >&2
    fi
  done
fi
gmail_args=()
if [ -f "$GMAIL_DIR/credentials.json" ] && [ -f "$GMAIL_DIR/gmail-token.json" ]; then
  gmail_args=(-v "$GMAIL_DIR:/gmail:z")
elif [ -f "$GMAIL_DIR/credentials.json" ] || [ -f "$GMAIL_DIR/gmail-token.json" ]; then
  echo "Note: Gmail stays off - $GMAIL_DIR needs both credentials.json and gmail-token.json" >&2
  echo "      (see the README, Gmail section, for creating the token)." >&2
fi

if [ "$with_monitoring" = 1 ]; then
  # A monitoring-enabled copy of the chosen config (metrics bound to 0.0.0.0 so
  # Prometheus can reach it). Generated inside the image: no host Python needed.
  "$ENGINE" run --rm \
    -v "$PWD/$CONFIG_SOURCE:/in/source.yaml:ro,z" \
    -v "$PWD/scripts/generate_monitoring_config.py:/generate.py:ro,z" \
    -v "$PWD/.run:/out:z" \
    --entrypoint python "$IMAGE" /generate.py --source /in/source.yaml --dest /out/agent.yaml >/dev/null
  config_file="$PWD/.run/agent.yaml"
else
  config_file="$PWD/$CONFIG_SOURCE"
fi

wait_for() {
  local url=$1 label=$2 i
  for i in $(seq 1 90); do
    if command -v curl >/dev/null 2>&1; then
      curl -fsS -o /dev/null "$url" 2>/dev/null && return 0
    elif command -v wget >/dev/null 2>&1; then
      wget -q -O /dev/null "$url" 2>/dev/null && return 0
    else
      sleep 10; return 0
    fi
    sleep 1
  done
  echo "Warning: $label did not answer at $url yet." >&2
}

# Fail early, and readably, if something else already owns a port we publish.
require_free_port() {
  if (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; then
    echo "Port $1 is already in use (an old monitoring stack, or another program)." >&2
    echo "Free it, or pick another one: $2=<port> ./scripts/run.sh" >&2
    exit 1
  fi
}

agent_network_args=()
if [ "$with_monitoring" = 1 ]; then
  remove_monitoring
  require_free_port "$PROMETHEUS_PORT" PROMETHEUS_PORT
  require_free_port "$GRAFANA_PORT" GRAFANA_PORT
  "$ENGINE" network inspect "$NETWORK" >/dev/null 2>&1 || "$ENGINE" network create "$NETWORK" >/dev/null

  # Grafana's data source points at http://prometheus:9090, so that name has
  # to resolve on the network (container names alone don't match it).
  "$ENGINE" run -d --name "$PROMETHEUS" --network "$NETWORK" --network-alias prometheus \
    -p "127.0.0.1:${PROMETHEUS_PORT}:9090" \
    -v "$PROMETHEUS_VOLUME:/prometheus" \
    -v "$PWD/config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro,z" \
    "$PROMETHEUS_IMAGE" \
    --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus \
    --storage.tsdb.retention.time=7d --storage.tsdb.retention.size=1GB >/dev/null

  grafana_env=(
    -e GF_AUTH_ANONYMOUS_ENABLED=true -e GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
    -e GF_USERS_ALLOW_SIGN_UP=false
    -e GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/agent-monitoring.json
    -e GF_ANALYTICS_REPORTING_ENABLED=false -e GF_ANALYTICS_CHECK_FOR_UPDATES=false
    -e GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES=false
  )
  grafana_password=$(env_value GRAFANA_ADMIN_PASSWORD)
  [ -n "$grafana_password" ] && grafana_env+=(-e "GF_SECURITY_ADMIN_PASSWORD=$grafana_password")
  "$ENGINE" run -d --name "$GRAFANA" --network "$NETWORK" \
    -p "127.0.0.1:${GRAFANA_PORT}:3000" \
    "${grafana_env[@]}" \
    -v "$GRAFANA_VOLUME:/var/lib/grafana" \
    -v "$PWD/config/grafana/provisioning:/etc/grafana/provisioning:ro,z" \
    -v "$PWD/config/grafana/dashboards:/var/lib/grafana/dashboards:ro,z" \
    "$GRAFANA_IMAGE" >/dev/null

  wait_for "http://127.0.0.1:${PROMETHEUS_PORT}/-/ready" Prometheus
  wait_for "http://127.0.0.1:${GRAFANA_PORT}/api/health" Grafana

  agent_network_args=(--network "$NETWORK" --network-alias agent --name "$AGENT")
  "$ENGINE" rm -f "$AGENT" >/dev/null 2>&1 || true
  echo
  echo "Grafana:    http://localhost:${GRAFANA_PORT}   (opens on the dashboard, no login needed)"
  echo "Prometheus: http://localhost:${PROMETHEUS_PORT}"
  echo "Logs:       workspace/.agent_monitoring/events.jsonl"
  echo
fi

if [ -t 0 ] && [ -t 1 ]; then tty_flags=(-it); else tty_flags=(-i); fi
env_args=()
if [ -f .env ]; then
  # docker/podman --env-file keeps quotes as part of the value (a shell would
  # strip them), so KEY="value" reached the agent as "value" with the quotes -
  # seen live as a 401 from Tavily. Pass a copy with surrounding quotes removed.
  sed -E -e "s/^([A-Za-z_][A-Za-z0-9_]*)=\"(.*)\"[[:space:]]*$/\1=\2/" \
         -e "s/^([A-Za-z_][A-Za-z0-9_]*)='(.*)'[[:space:]]*$/\1=\2/" .env > .run/agent.env
  chmod 600 .run/agent.env
  env_args=(--env-file .run/agent.env)
fi

status=0
"$ENGINE" run --rm "${tty_flags[@]}" \
  ${agent_network_args[@]+"${agent_network_args[@]}"} \
  ${env_args[@]+"${env_args[@]}"} \
  -e AGENT_CONFIG_PATH=/app/config/run.yaml \
  -v "$config_file:/app/config/run.yaml:ro,z" \
  -v "$PWD/workspace:/workspace:z" \
  ${gmail_args[@]+"${gmail_args[@]}"} \
  "$IMAGE" || status=$?

if [ "$with_monitoring" = 1 ]; then
  echo
  echo "Monitoring is still running: http://localhost:${GRAFANA_PORT}"
  echo "Stop it with: ./scripts/run.sh --stop   (--clean also deletes its data)"
fi
exit "$status"
