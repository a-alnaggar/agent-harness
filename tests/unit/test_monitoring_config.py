import pytest

from agent.config.loader import ConfigError, MonitoringConfig, _parse_monitoring


def test_monitoring_defaults_preserve_disabled_behavior() -> None:
    assert _parse_monitoring({}) == MonitoringConfig()


@pytest.mark.parametrize("raw", [
    None, {"enabled": 1}, {"unknown": True}, {"metrics": {"port": 0}},
    {"metrics": {"host": "localhost"}}, {"jsonl": {"queue_capacity": 0}},
    {"jsonl": {"path": "../events.jsonl"}}, {"jsonl": {"path": "C:\\secret.jsonl"}},
    {"jsonl": {"max_record_bytes": 9000}}, {"shutdown_timeout_seconds": float("nan")},
])
def test_invalid_monitoring_configuration_is_rejected(raw: object) -> None:
    with pytest.raises(ConfigError):
        _parse_monitoring(raw)


def test_independent_consumers_can_be_disabled() -> None:
    config = _parse_monitoring({"enabled": True, "metrics": {"enabled": False},
                                "jsonl": {"enabled": False}})
    assert config.enabled and not config.metrics.enabled and not config.jsonl.enabled


def test_metrics_linger_defaults_to_zero_and_is_configurable() -> None:
    assert _parse_monitoring({}).metrics.linger_seconds == 0
    assert _parse_monitoring({"metrics": {"linger_seconds": 3}}).metrics.linger_seconds == 3.0
    assert _parse_monitoring({"metrics": {"linger_seconds": 2.5}}).metrics.linger_seconds == 2.5


@pytest.mark.parametrize("linger", [-1, 31, float("nan"), float("inf"), "3", True, None])
def test_metrics_linger_must_be_a_bounded_number(linger: object) -> None:
    with pytest.raises(ConfigError):
        _parse_monitoring({"metrics": {"linger_seconds": linger}})
