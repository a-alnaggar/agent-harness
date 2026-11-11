FROM docker.io/library/python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

# Node.js runtime for stdio MCP servers executed via npx
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY config ./config

RUN pip install --no-cache-dir ".[gmail,sandbox]"

ENV AGENT_CONFIG_PATH=/app/config/default.yaml
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

ENTRYPOINT ["python", "-m", "agent"]
