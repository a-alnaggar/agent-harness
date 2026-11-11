#!/usr/bin/env bash
set -euo pipefail

# Docker or Podman: whichever is installed (override with CONTAINER_ENGINE).
ENGINE=${CONTAINER_ENGINE:-$(command -v docker >/dev/null 2>&1 && echo docker || echo podman)}

IMAGE_TAG=localhost/agent:e2e-local
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

"$ENGINE" build -t "$IMAGE_TAG" .

echo "--- boot smoke test (no token needed) ---"
echo "/exit" | "$ENGINE" run --rm -i -e ACCESS_TOKEN=dummy "$IMAGE_TAG"

echo "--- gmail MCP server: connection + tool discovery (no credentials needed) ---"
"$ENGINE" run --rm --entrypoint python3 "$IMAGE_TAG" -c "
from agent.tools.mcp.client import McpClient

client = McpClient('gmail', 'python3', ['-m', 'agent.mcp_servers.gmail'])
tools = client.connect(timeout=15)
names = [t['name'] for t in tools]
assert 'search_emails' in names, f'search_emails not discovered: {names}'
print('PASS: gmail MCP server connected, discovered tools:', names)
client.close()
"

if [ -z "${ACCESS_TOKEN:-}" ]; then
  echo "ACCESS_TOKEN not set in this shell; skipping the real-LLM tool-call check."
  exit 0
fi

echo "--- real task through a mounted workspace ---"
printf 'create a file called e2e_check.txt with the content docker e2e ok\n/exit\n' \
  | "$ENGINE" run --rm -i -e ACCESS_TOKEN="$ACCESS_TOKEN" -v "$WORKDIR:/workspace:z" "$IMAGE_TAG"

if [ "$(cat "$WORKDIR/e2e_check.txt")" != "docker e2e ok" ]; then
  echo "FAIL: file content did not match" >&2
  exit 1
fi

echo "PASS: container correctly wrote to the mounted workspace"
