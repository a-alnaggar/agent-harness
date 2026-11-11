from pathlib import Path
from unittest.mock import Mock
import pytest

from agent.config.loader import MonitoringConfig
from agent.monitoring.runtime import MonitoringRuntime


def test_disabled_runtime_creates_nothing(tmp_path: Path) -> None:
    runtime = MonitoringRuntime(MonitoringConfig(), tmp_path)
    assert runtime.hooks == [] and runtime.resources == []
    assert list(tmp_path.iterdir()) == []
    runtime.close()


def test_consumers_initialize_independently_and_fail_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.monitoring.prometheus as metrics
    diagnostics: list[str] = []
    monkeypatch.setattr(metrics.PrometheusHook, "start_endpoint", Mock(side_effect=OSError("private-error")))
    runtime = MonitoringRuntime(MonitoringConfig(enabled=True), tmp_path, diagnostics.append)
    assert len(runtime.hooks) == 1
    runtime.close()
    assert diagnostics == ["Monitoring failure; agent execution continues."]


def test_shutdown_shares_one_deadline_and_preserves_cleanup(tmp_path: Path) -> None:
    ticks = iter([10.0, 10.5, 11.0])
    runtime = MonitoringRuntime(MonitoringConfig(), tmp_path, clock=lambda: next(ticks))
    first, second = Mock(), Mock()
    first.close.side_effect = RuntimeError("private-error")
    runtime.resources = [first, second]
    runtime.close()
    first.close.assert_called_once_with(1.5)
    second.close.assert_called_once_with(1.0)
    runtime.close()


# --- lingering after the agent finishes, so Prometheus gets a last scrape ------

def linger_runtime(tmp_path: Path, linger: float, diagnostics=None, sleep=None, metrics_enabled=True):
    import agent.monitoring.prometheus as metrics
    from agent.config.loader import JsonlConfig, MetricsConfig

    config = MonitoringConfig(
        enabled=True, metrics=MetricsConfig(enabled=metrics_enabled, port=1, linger_seconds=linger),
        jsonl=JsonlConfig(enabled=False),
    )
    original = metrics.PrometheusHook.start_endpoint
    metrics.PrometheusHook.start_endpoint = lambda self, host, port: original(self, "127.0.0.1", 0)
    try:
        return MonitoringRuntime(config, tmp_path, diagnostics, sleep=sleep or (lambda seconds: None))
    finally:
        metrics.PrometheusHook.start_endpoint = original


def test_close_waits_for_the_configured_linger_before_shutting_the_endpoint_down(tmp_path: Path, capsys) -> None:
    order = []
    runtime = linger_runtime(tmp_path, 3.0, sleep=lambda seconds: order.append(("sleep", seconds)))
    runtime.resources[0].close = lambda timeout=2.0: order.append(("closed", None))

    runtime.close()

    assert order == [("sleep", 3.0), ("closed", None)]
    assert "Flushing final metrics" in capsys.readouterr().err


def test_no_linger_means_no_wait_and_no_message(tmp_path: Path, capsys) -> None:
    slept = []
    runtime = linger_runtime(tmp_path, 0.0, sleep=slept.append)

    runtime.close()

    assert slept == [] and capsys.readouterr().err == ""


def test_without_a_metrics_endpoint_there_is_nothing_to_wait_for(tmp_path: Path) -> None:
    slept = []
    runtime = linger_runtime(tmp_path, 3.0, sleep=slept.append, metrics_enabled=False)

    runtime.close()

    assert slept == []


def test_a_failing_sleep_does_not_stop_shutdown(tmp_path: Path) -> None:
    diagnostics: list[str] = []

    def broken(seconds: float) -> None:
        raise KeyboardInterrupt

    runtime = linger_runtime(tmp_path, 3.0, diagnostics.append, sleep=broken)
    closed = Mock()
    runtime.resources = [closed]

    runtime.close()

    closed.close.assert_called_once()
    assert diagnostics == ["Monitoring failure; agent execution continues."]
