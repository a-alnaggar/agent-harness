"""Unit tests for the `agent.__main__` composition root: error paths exit
cleanly with a message, and the happy path wires ReactLoop/CLI correctly
and always closes MCP clients, even if the CLI itself raises."""

from unittest.mock import MagicMock
from pathlib import Path

import pytest

import agent.__main__ as main_module
from agent.config import AgentConfig, ConfigError
from agent.memory import CoreProfileManager, SqliteMemoryStore


def make_config(**overrides):
    defaults = dict(model="m", temperature=0.2, max_tokens=100, timeout=10, base_url=None, access_token="tok")
    defaults.update(overrides)
    return AgentConfig(**defaults)


def test_config_error_exits_with_message(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "load_config", MagicMock(side_effect=ConfigError("bad config")))

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 1
    assert "Config error: bad config" in capsys.readouterr().err


def test_llm_client_error_exits_with_message(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main_module, "load_config", MagicMock(return_value=make_config()))
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(main_module, "register_builtin_tools", MagicMock())
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[]))
    monkeypatch.setattr(main_module, "LLMClient", MagicMock(side_effect=RuntimeError("no key")))

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 1
    assert "LLM client error: no key" in capsys.readouterr().err


def test_happy_path_wires_loop_and_closes_mcp_clients(monkeypatch, tmp_path):
    config = make_config()
    monkeypatch.setattr(main_module, "load_config", MagicMock(return_value=config))
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(main_module, "register_builtin_tools", MagicMock())

    fake_mcp_client = MagicMock()
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[fake_mcp_client]))
    monkeypatch.setattr(main_module, "LLMClient", MagicMock())
    start_cli_mock = MagicMock()
    monkeypatch.setattr(main_module, "start_cli", start_cli_mock)

    main_module.main()

    assert start_cli_mock.call_count == 1
    _, kwargs = start_cli_mock.call_args
    assert kwargs["engine"].__class__.__name__ == "ReactEngineAdapter"
    fake_mcp_client.close.assert_called_once()


def test_mcp_clients_are_closed_even_if_start_cli_raises(monkeypatch, tmp_path):
    config = make_config()
    monkeypatch.setattr(main_module, "load_config", MagicMock(return_value=config))
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(main_module, "register_builtin_tools", MagicMock())

    fake_mcp_client = MagicMock()
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[fake_mcp_client]))
    monkeypatch.setattr(main_module, "LLMClient", MagicMock())
    monkeypatch.setattr(main_module, "start_cli", MagicMock(side_effect=RuntimeError("cli crashed")))

    with pytest.raises(RuntimeError, match="cli crashed"):
        main_module.main()

    fake_mcp_client.close.assert_called_once()


def test_memory_store_points_at_the_workspace_root(monkeypatch, tmp_path):
    config = make_config()
    monkeypatch.setattr(main_module, "load_config", MagicMock(return_value=config))
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(main_module, "register_builtin_tools", MagicMock())
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[]))
    monkeypatch.setattr(main_module, "LLMClient", MagicMock())
    monkeypatch.setattr(main_module, "start_cli", MagicMock())
    memory_store_mock = MagicMock()
    monkeypatch.setattr(main_module, "SqliteMemoryStore", memory_store_mock)

    main_module.main()

    # once for the unobserved startup housekeeping, once for the store the agent uses
    assert memory_store_mock.call_args_list == [
        ((tmp_path / ".agent_memory" / "memory.db",), {}),
        ((tmp_path / ".agent_memory" / "memory.db",), {}),
    ]


def test_monitoring_injects_one_shared_hook_and_cleans_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from agent.config import MonitoringConfig
    import agent.monitoring.runtime as runtime_module

    runtime = MagicMock(hooks=[])
    runtime_constructor = MagicMock(return_value=runtime)
    monkeypatch.setattr(runtime_module, "MonitoringRuntime", runtime_constructor)
    monkeypatch.setattr(
        main_module,
        "load_config",
        MagicMock(return_value=make_config(monitoring=MonitoringConfig(enabled=True))),
    )
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(main_module, "register_builtin_tools", MagicMock())
    client = MagicMock()
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[client]))

    sqlite_memory = MagicMock()
    sqlite_memory.get_general_value.return_value = None
    core_profile = MagicMock()
    core_profile.format_for_prompt.return_value = ""
    sqlite_constructor = MagicMock(return_value=sqlite_memory)
    core_constructor = MagicMock(return_value=core_profile)
    parent_registry = MagicMock()
    parent_registry.list_tools.return_value = []
    subagent_registry = MagicMock()
    subagent_registry.list_tools.return_value = []
    registry_constructor = MagicMock(side_effect=[parent_registry, subagent_registry])
    llm_constructor = MagicMock()
    loop_constructor = MagicMock()
    monkeypatch.setattr(main_module, "SqliteMemoryStore", sqlite_constructor)
    monkeypatch.setattr(main_module, "CoreProfileManager", core_constructor)
    monkeypatch.setattr(main_module, "ToolRegistry", registry_constructor)
    monkeypatch.setattr(main_module, "LLMClient", llm_constructor)
    monkeypatch.setattr(main_module, "ReactLoop", loop_constructor)
    monkeypatch.setattr(main_module, "start_cli", MagicMock(side_effect=RuntimeError("cli failure")))

    with pytest.raises(RuntimeError, match="cli failure"):
        main_module.main()

    hooks = [
        sqlite_constructor.call_args.kwargs["hook"],
        core_constructor.call_args.kwargs["hook"],
        *(call.kwargs["hook"] for call in registry_constructor.call_args_list),
        llm_constructor.call_args.kwargs["hook"],
        loop_constructor.call_args.kwargs["hook"],
    ]
    assert len(hooks) == 6
    assert all(hook is hooks[0] for hook in hooks)
    client.close.assert_called_once()
    runtime.close.assert_called_once()


def test_disabled_monitoring_does_not_construct_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import agent.monitoring.runtime as runtime_module

    constructor = MagicMock(side_effect=AssertionError("must remain disabled"))
    monkeypatch.setattr(runtime_module, "MonitoringRuntime", constructor)
    monkeypatch.setattr(main_module, "load_config", MagicMock(return_value=make_config()))
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    for name in ("register_builtin_tools", "LLMClient", "start_cli"):
        monkeypatch.setattr(main_module, name, MagicMock())
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[]))

    main_module.main()

    constructor.assert_not_called()
    assert not (tmp_path / ".agent_monitoring").exists()


def test_memory_tools_are_isolated_to_memory_subagent(monkeypatch, tmp_path):
    config = make_config()
    monkeypatch.setattr(main_module, "load_config", MagicMock(return_value=config))
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[]))
    monkeypatch.setattr(main_module, "LLMClient", MagicMock())
    start_cli_mock = MagicMock()
    monkeypatch.setattr(main_module, "start_cli", start_cli_mock)

    main_module.main()

    registry = start_cli_mock.call_args.kwargs["registry"]
    names = [tool.name for tool in registry.list_tools()]
    assert "save_memory" not in names
    assert "forget" not in names

    delegate = registry.get("delegate_task")
    master_names = [tool.name for tool in delegate.runner.master_registry.list_tools()]
    assert master_names.count("save_memory") == 1
    assert master_names.count("forget") == 1
    assert master_names.count("update_core_preference") == 1


def test_startup_removes_duplicate_core_keys_from_sqlite(monkeypatch, tmp_path):
    memory_dir = tmp_path / ".agent_memory"
    core = CoreProfileManager(memory_dir / "core.md")
    core.update_preference("user_name", "Ahmed")
    store = SqliteMemoryStore(memory_dir / "memory.db")
    store.save_general(key="user_name", content="Old duplicate")

    monkeypatch.setattr(main_module, "load_config", MagicMock(return_value=make_config()))
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(main_module, "register_builtin_tools", MagicMock())
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[]))
    monkeypatch.setattr(main_module, "LLMClient", MagicMock())
    monkeypatch.setattr(main_module, "start_cli", MagicMock())

    main_module.main()

    assert CoreProfileManager(memory_dir / "core.md").get_preference("user_name") == "Ahmed"
    assert SqliteMemoryStore(memory_dir / "memory.db").get_general_value("user_name") is None


def test_deny_tools_reach_the_parent_loop_and_the_sub_agent_runner_as_one_shared_policy(monkeypatch, tmp_path):
    from agent.config import PermissionsConfig

    config = make_config(permissions=PermissionsConfig(deny_tools=["web_search"]))
    monkeypatch.setattr(main_module, "load_config", MagicMock(return_value=config))
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[]))
    monkeypatch.setattr(main_module, "LLMClient", MagicMock())
    start_cli_mock = MagicMock()
    monkeypatch.setattr(main_module, "start_cli", start_cli_mock)
    captured = {}

    import agent.subagents as subagents_pkg

    class CapturingRunner(subagents_pkg.SubAgentRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["runner"] = self

    monkeypatch.setattr(subagents_pkg, "SubAgentRunner", CapturingRunner)

    main_module.main()

    loop = start_cli_mock.call_args.kwargs["engine"]._loop
    assert loop.permission_policy.deny_tools == frozenset({"web_search"})
    assert captured["runner"].permission_policy is loop.permission_policy


def test_a_deny_entry_naming_no_registered_tool_warns_at_startup(monkeypatch, tmp_path, capsys):
    from agent.config import PermissionsConfig

    config = make_config(permissions=PermissionsConfig(deny_tools=["web_serch", "web_search"]))
    monkeypatch.setattr(main_module, "load_config", MagicMock(return_value=config))
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[]))
    monkeypatch.setattr(main_module, "LLMClient", MagicMock())
    monkeypatch.setattr(main_module, "start_cli", MagicMock())

    main_module.main()

    err = capsys.readouterr().err
    assert "'web_serch', which is not a registered tool" in err
    assert "'web_search', which is not a registered tool" not in err


def run_main_with_sandbox(monkeypatch, tmp_path, sandbox):
    monkeypatch.setattr(main_module, "load_config", MagicMock(return_value=make_config(sandbox=sandbox)))
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[]))
    monkeypatch.setattr(main_module, "LLMClient", MagicMock())
    start_cli_mock = MagicMock()
    monkeypatch.setattr(main_module, "start_cli", start_cli_mock)
    main_module.main()
    return start_cli_mock.call_args.kwargs["registry"]


def test_sandbox_configured_without_a_key_stays_off_and_says_so(monkeypatch, tmp_path, capsys):
    from agent.config import SandboxConfig

    registry = run_main_with_sandbox(monkeypatch, tmp_path, SandboxConfig(host="h", user="u", private_key=None))

    assert not registry.has_tool("sandbox_exec")
    assert "SANDBOX_SSH_KEY is not set; sandbox_exec is disabled" in capsys.readouterr().err


def test_sandbox_with_a_key_registers_the_tool_and_closes_the_client(monkeypatch, tmp_path):
    import agent.tools.sandbox as sandbox_pkg
    from agent.config import SandboxConfig

    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(sandbox_pkg, "SshSandboxClient", FakeClient)

    registry = run_main_with_sandbox(
        monkeypatch, tmp_path, SandboxConfig(host="h", user="u", private_key="the-key", port=2222, timeout=9)
    )

    assert registry.has_tool("sandbox_exec")
    assert created[0].kwargs == {"host": "h", "port": 2222, "user": "u", "private_key": "the-key", "timeout": 9}
    assert created[0].closed


def test_sandbox_with_a_malformed_key_exits_with_a_clear_error(monkeypatch, tmp_path, capsys):
    from agent.config import SandboxConfig

    with pytest.raises(SystemExit) as exc_info:
        run_main_with_sandbox(monkeypatch, tmp_path, SandboxConfig(host="h", user="u", private_key="garbage!!"))

    assert exc_info.value.code == 1
    assert "Sandbox key error" in capsys.readouterr().err


def test_the_parent_prompt_provider_receives_the_run_context(monkeypatch, tmp_path):
    """So the per-step core-profile read is recorded inside the run's trace."""
    config = make_config()
    monkeypatch.setattr(main_module, "load_config", MagicMock(return_value=config))
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[]))
    monkeypatch.setattr(main_module, "LLMClient", MagicMock())
    start_cli_mock = MagicMock()
    monkeypatch.setattr(main_module, "start_cli", start_cli_mock)

    main_module.main()

    loop = start_cli_mock.call_args.kwargs["engine"]._loop
    assert loop._provider_takes_context is True


def test_startup_housekeeping_emits_no_memory_events_because_it_belongs_to_no_run(monkeypatch, tmp_path):
    """Seen live: the legacy-key migration read every core key through the
    observed store, so each start recorded 13 memory reads with no run - 13
    rootless one-event "runs" in the trace, before the first real run."""
    from agent.config import MonitoringConfig
    import agent.monitoring.runtime as runtime_module

    class Recorder:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(event)

    recorder = Recorder()
    monkeypatch.setattr(runtime_module, "MonitoringRuntime", MagicMock(return_value=MagicMock(hooks=[recorder])))
    memory_dir = tmp_path / ".agent_memory"
    CoreProfileManager(memory_dir / "core.md").update_preference("user_name", "Ahmed")
    SqliteMemoryStore(memory_dir / "memory.db").save_general(key="user_name", content="legacy duplicate")
    monkeypatch.setattr(
        main_module, "load_config", MagicMock(return_value=make_config(monitoring=MonitoringConfig(enabled=True)))
    )
    monkeypatch.setattr(main_module.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(main_module, "register_mcp_tools", MagicMock(return_value=[]))
    monkeypatch.setattr(main_module, "LLMClient", MagicMock())
    monkeypatch.setattr(main_module, "start_cli", MagicMock())

    main_module.main()

    assert [e for e in recorder.events if e.component == "memory"] == []
    # ...and the migration itself still happened (the point of the housekeeping)
    assert SqliteMemoryStore(memory_dir / "memory.db").get_general_value("user_name") is None
