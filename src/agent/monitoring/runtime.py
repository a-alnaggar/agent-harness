from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

from agent.config.loader import MonitoringConfig
from agent.observability import notify_monitoring_failure


class MonitoringRuntime:
    def __init__(
        self, config: MonitoringConfig, root: Path,
        diagnostic: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.hooks: list[Any] = []
        self.resources: list[Any] = []
        self.timeout = config.shutdown_timeout_seconds
        self.diagnostic = diagnostic
        self.clock = clock
        self.sleep = sleep
        self.linger = 0.0
        if not config.enabled:
            return
        metrics = None
        if config.metrics.enabled:
            try:
                from .prometheus import PrometheusHook
                metrics = PrometheusHook()
                metrics.start_endpoint(config.metrics.host, config.metrics.port)
                self.hooks.append(metrics)
                self.resources.append(metrics)
                self.linger = config.metrics.linger_seconds
            except BaseException:
                notify_monitoring_failure(diagnostic)
        if config.jsonl.enabled:
            try:
                from .jsonl import JsonlHook, RotatingWriter
                logs = config.jsonl
                writer = RotatingWriter(root, logs.path, logs.max_file_bytes, logs.backup_count)
                hook = JsonlHook(writer, logs.queue_capacity, logs.max_record_bytes, diagnostic,
                                 metrics.record_drop if metrics is not None else None)
                self.hooks.append(hook)
                self.resources.insert(0, hook)
            except BaseException:
                notify_monitoring_failure(diagnostic)

    def close(self) -> None:
        resources, self.resources = self.resources, []
        if not resources:
            return
        if self.linger > 0:
            print("Flushing final metrics to Prometheus...", file=sys.stderr)
            try:
                self.sleep(self.linger)
            except BaseException:
                notify_monitoring_failure(self.diagnostic)
        deadline = self.clock() + self.timeout
        for resource in resources:
            try:
                resource.close(max(0.0, deadline - self.clock()))
            except BaseException:
                notify_monitoring_failure(self.diagnostic)
