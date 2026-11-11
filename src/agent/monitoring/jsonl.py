from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Callable

from agent.observability import LifecycleEvent, notify_monitoring_failure


class RotatingWriter:
    def __init__(self, root: Path, path: str, max_file_bytes: int, backup_count: int) -> None:
        self.root = root.resolve()
        self.path = self.root / path
        self.maximum = max_file_bytes
        self.backups = backup_count
        self._check(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._check(self.path)

    def _check(self, path: Path) -> None:
        path.resolve().relative_to(self.root)
        current = path
        while current != self.root:
            if current.is_symlink():
                raise ValueError("monitoring destinations must not be symlinks")
            current = current.parent

    def __call__(self, line: str) -> None:
        encoded = (line + "\n").encode("utf-8")
        self._check(self.path)
        if self.path.exists() and self.path.stat().st_size + len(encoded) > self.maximum:
            if self.backups:
                for number in range(self.backups, 0, -1):
                    source = self.path if number == 1 else Path(f"{self.path}.{number - 1}")
                    destination = Path(f"{self.path}.{number}")
                    self._check(source)
                    self._check(destination)
                    if source.exists():
                        os.replace(source, destination)
            else:
                self._write(b"", truncate=True)
        self._write(encoded)

    def _write(self, encoded: bytes, truncate: bool = False) -> None:
        flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if truncate else os.O_APPEND)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            if os.name == "posix":
                os.fchmod(stream.fileno(), 0o600)
            stream.write(encoded)


class JsonlHook:
    def __init__(
        self, writer: Callable[[str], None], queue_capacity: int = 1024,
        max_record_bytes: int = 8192, diagnostic: Callable[[str], None] | None = None,
        on_drop: Callable[[str], None] | None = None,
    ) -> None:
        self.queue: queue.Queue[str] = queue.Queue(maxsize=queue_capacity)
        self.maximum = max_record_bytes
        self.writer = writer
        self.diagnostic = diagnostic
        self.on_drop = on_drop
        self._accept_lock = threading.Lock()
        self._closing = threading.Event()
        self._diagnostic_pending = threading.Event()
        self._diagnostic_reported = False
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._run, name="agent-jsonl", daemon=True)
        self.thread.start()

    def _drop(self, reason: str) -> None:
        if self.on_drop is not None:
            try:
                self.on_drop(reason)
            except BaseException:
                self._diagnostic_pending.set()
        self._diagnostic_pending.set()

    def _report_failure(self) -> None:
        if self._diagnostic_pending.is_set() and not self._diagnostic_reported:
            self._diagnostic_reported = True
            notify_monitoring_failure(self.diagnostic)

    def emit(self, event: LifecycleEvent) -> None:
        try:
            values = {field: getattr(event, field) for field in (
                "timestamp", "event_name", "run_id", "span_id", "parent_span_id",
                "agent_id", "component", "operation", "outcome", "duration_seconds",
            )}
            values["attributes"] = dict(event.attributes)
            validated = LifecycleEvent(**values)
            values["timestamp"] = validated.timestamp.isoformat()
            line = json.dumps(values, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
            if len(line.encode("utf-8")) + 1 > self.maximum:
                self._drop("oversize")
                return
        except BaseException:
            self._drop("invalid")
            return
        if not self._accept_lock.acquire(blocking=False):
            self._drop("contention")
            return
        reason = None
        try:
            if self._closing.is_set():
                reason = "closed"
            else:
                try:
                    self.queue.put_nowait(line)
                except queue.Full:
                    reason = "full"
        finally:
            self._accept_lock.release()
        if reason is not None:
            self._drop(reason)

    def _run(self) -> None:
        try:
            while not self._closing.is_set() or not self.queue.empty():
                try:
                    line = self.queue.get(timeout=0.1)
                except queue.Empty:
                    self._report_failure()
                    continue
                try:
                    self.writer(line)
                except BaseException:
                    self._drop("writer")
                finally:
                    self.queue.task_done()
                    self._report_failure()
        finally:
            self._report_failure()
            self.finished.set()

    def close(self, timeout: float = 2.0) -> None:
        with self._accept_lock:
            self._closing.set()
        self.thread.join(timeout)
