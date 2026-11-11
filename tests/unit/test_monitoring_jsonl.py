import json
import os
import threading
from pathlib import Path

import pytest

from agent.monitoring.jsonl import JsonlHook, RotatingWriter
from test_monitoring_prometheus import event


def test_jsonl_serializes_only_safe_fields_and_drains() -> None:
    records: list[str] = []
    hook = JsonlHook(records.append)
    hook.emit(event("llm.call.finish", "llm", attributes={"input_tokens": 7}, parent_span_id="parent-span"))
    hook.close()
    hook.close()
    value = json.loads(records[0])
    assert value["attributes"] == {"input_tokens": 7}
    assert value["parent_span_id"] == "parent-span"
    assert value["timestamp"].endswith("+00:00")
    assert len(records) == 1 and hook.finished.is_set()


def test_blocked_writer_does_not_block_emit_and_queue_drops_newest() -> None:
    entered, release = threading.Event(), threading.Event()
    records: list[str] = []
    drops: list[str] = []
    def writer(line: str) -> None:
        entered.set()
        assert release.wait(2)
        records.append(line)
    hook = JsonlHook(writer, queue_capacity=1, on_drop=drops.append)
    try:
        hook.emit(event("llm.call.finish", "llm"))
        assert entered.wait(2)
        hook.emit(event("llm.call.finish", "llm"))
        hook.emit(event("llm.call.finish", "llm"))
        assert drops == ["full"] and hook.queue.qsize() == 1
        hook.close(timeout=0)
        assert hook.thread.is_alive()
    finally:
        release.set()
        assert hook.finished.wait(2)
    assert len(records) == 2
    hook.emit(event("llm.call.finish", "llm"))
    assert drops[-1] == "closed"


def test_invalid_oversize_and_writer_failure_are_content_free() -> None:
    drops: list[str] = []
    diagnostics: list[str] = []
    attempted = threading.Event()
    def writer(line: str) -> None:
        attempted.set()
        raise OSError("private-path-value")
    hook = JsonlHook(writer, on_drop=drops.append, diagnostic=diagnostics.append)
    forged = event("llm.call.finish", "llm")
    object.__setattr__(forged, "attributes", {"response": "private-email-data"})
    hook.emit(forged)
    hook.emit(event("llm.call.finish", "llm"))
    assert attempted.wait(2)
    hook.close()
    assert drops == ["invalid", "writer"]
    assert set(diagnostics) == {"Monitoring failure; agent execution continues."}
    small = JsonlHook(lambda _: None, max_record_bytes=1, on_drop=drops.append)
    small.emit(event("llm.call.finish", "llm"))
    small.close()
    assert drops[-1] == "oversize"


def test_rotation_is_bounded_and_owner_restricted(tmp_path: Path) -> None:
    writer = RotatingWriter(tmp_path, "logs/events.jsonl", 12, 2)
    for index in range(5):
        writer(str(index) * 8)
    files = list((tmp_path / "logs").iterdir())
    assert len(files) == 3
    assert all(path.stat().st_size <= 12 for path in files)
    assert (tmp_path / "logs/events.jsonl").read_text().strip() == "44444444"
    if os.name == "posix":
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in files)


def test_destinations_cannot_escape_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RotatingWriter(tmp_path, "../outside.jsonl", 1024, 1)


def test_backup_symlinks_are_rejected(tmp_path: Path) -> None:
    from unittest.mock import patch
    writer = RotatingWriter(tmp_path, "events.jsonl", 10, 1)
    writer("12345678")
    original = Path.is_symlink
    with patch.object(Path, "is_symlink", lambda path: path.name.endswith(".1") or original(path)):
        with pytest.raises(ValueError):
            writer("abcdefgh")


def test_drop_diagnostic_runs_once_on_worker_not_producer() -> None:
    threads: list[int] = []
    hook = JsonlHook(lambda _: None, max_record_bytes=1,
                     diagnostic=lambda _: threads.append(threading.get_ident()))
    producer = threading.get_ident()
    for _ in range(3):
        hook.emit(event("llm.call.finish", "llm"))
    hook.close()
    assert len(threads) == 1 and threads[0] != producer
