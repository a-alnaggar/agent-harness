from __future__ import annotations

import threading
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from agent.observability import LifecycleEvent

BUCKETS = (0.01, 0.05, 0.1, 0.5, 1, 5, 15, 30, 60, 120)


class PrometheusHook:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        self._lock = threading.Lock()
        self._active: dict[tuple[str, str], str] = {}
        self._active_counts: dict[str, int] = {}
        self._tools: set[str] = set()
        self._agents: set[str] = set()
        self.started = Counter("agent_runs_started", "Observed parent and sub-agent runs started", ("agent_id",), registry=self.registry)
        self.active = Gauge("agent_runs_active", "Observed active parent and sub-agent runs", ("agent_id",), registry=self.registry)
        self.calls: dict[str, Counter] = {}
        self.durations: dict[str, Histogram] = {}
        for component, name, labels in (
            # agent_id distinguishes the parent run from each sub-agent's own
            # runs (agent_id is the sub-agent's configured name, e.g.
            # "course_knowledge") - this is what backs the "Sub-agents
            # activity" dashboard panel.
            ("agent", "run", ("outcome", "agent_id")), ("llm", "llm", ("outcome",)),
            ("tool", "tool", ("tool", "origin", "outcome")),
            ("memory", "memory", ("operation", "outcome")),
            ("permission", "permission", ("tool", "origin", "tier", "decision", "outcome")),
        ):
            counter_name = {"agent": "agent_runs_completed", "llm": "agent_llm_calls",
                            "tool": "agent_tool_calls", "memory": "agent_memory_operations",
                            "permission": "agent_permission_requests"}[component]
            self.calls[component] = Counter(counter_name, "Observed terminal operations", labels, registry=self.registry)
            self.durations[component] = Histogram(f"agent_{name}_duration_seconds", "Operation duration", labels,
                                                  buckets=BUCKETS, registry=self.registry)
        self.tokens = Counter("agent_llm_tokens", "Provider-reported tokens", ("direction",), registry=self.registry)
        self.dropped = Counter("agent_monitoring_log_dropped", "Dropped structured records", ("reason",), registry=self.registry)
        self.server: Any = None
        self.thread: threading.Thread | None = None

    def start_endpoint(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.server, self.thread = start_http_server(port, addr=host, registry=self.registry)

    def _agent_label(self, agent_id: str) -> str:
        if agent_id not in self._agents and len(self._agents) >= 128:
            return "other"
        self._agents.add(agent_id)
        return agent_id

    def emit(self, event: LifecycleEvent) -> None:
        with self._lock:
            key = (event.run_id, event.span_id)
            if event.event_name == "agent.run.start":
                agent_id = self._agent_label(event.agent_id)
                self.started.labels(agent_id=agent_id).inc()
                if len(self._active) < 1024:
                    self._active[key] = agent_id
                    self._active_counts[agent_id] = self._active_counts.get(agent_id, 0) + 1
                    self.active.labels(agent_id=agent_id).set(self._active_counts[agent_id])
                return
            if event.event_name.endswith(".start"):
                return
            component = event.component
            if component not in self.calls:
                return
            labels = {"outcome": event.outcome}
            if component == "agent":
                agent_id = self._active.pop(key, None) or self._agent_label(event.agent_id)
                self._active_counts[agent_id] = max(0, self._active_counts.get(agent_id, 1) - 1)
                self.active.labels(agent_id=agent_id).set(self._active_counts[agent_id])
                labels["agent_id"] = agent_id
            if component == "tool":
                name = event.attributes.get("tool_name", "unknown_tool")
                if name not in self._tools and len(self._tools) >= 128:
                    name = "other"
                else:
                    self._tools.add(name)
                labels.update(tool=name, origin=event.attributes.get("origin", "unknown"))
            elif component == "permission":
                name = event.attributes.get("tool_name", "unknown_tool")
                if name not in self._tools and len(self._tools) >= 128:
                    name = "other"
                else:
                    self._tools.add(name)
                labels.update(
                    tool=name,
                    origin=event.attributes.get("origin", "unknown"),
                    tier=event.attributes.get("tier", "unknown"),
                    decision=event.attributes.get("decision", "error"),
                )
            elif component == "memory":
                labels["operation"] = event.operation
            self.calls[component].labels(**labels).inc()
            if event.duration_seconds is not None:
                self.durations[component].labels(**labels).observe(event.duration_seconds)
            if event.event_name == "llm.call.finish":
                for field, direction in (("input_tokens", "input"), ("output_tokens", "output")):
                    if field in event.attributes:
                        self.tokens.labels(direction).inc(event.attributes[field])

    def record_drop(self, reason: str) -> None:
        if reason in {"full", "oversize", "invalid", "closed", "writer", "contention"}:
            self.dropped.labels(reason).inc()

    def close(self, timeout: float = 2.0) -> None:
        if self.server is not None:
            server, thread = self.server, self.thread
            self.server = None
            def stop() -> None:
                server.shutdown()
                server.server_close()
                if thread is not None:
                    thread.join()
            closer = threading.Thread(target=stop, daemon=True)
            closer.start()
            closer.join(timeout)
