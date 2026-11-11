from __future__ import annotations

import sys
from pathlib import Path

from agent.cli_interface import start_cli
from agent.config import ConfigError, load_config
from agent.engine.llm_client import LLMClient
from agent.engine.react_loop import UNTRUSTED_CONTENT_NOTICE, ReactLoop
from agent.memory import (
    ALLOWED_CORE_KEYS,
    CoreProfileManager,
    ForgetMemoryTool,
    RecallMemoryTool,
    SaveFaqTool,
    SaveMemoryTool,
    SearchFaqTool,
    SqliteMemoryStore,
    UpdateCorePreferenceTool,
    route_memory_mutation,
)
from agent.orchestrator import CLIConfigView, ReactEngineAdapter
from agent.tools import ToolRegistry
from agent.tools.permissions import PermissionPolicy
from agent.tools.builtin import register_builtin_tools
from agent.tools.mcp import register_mcp_tools
from agent.observability import CompositeHook, notify_monitoring_failure

MAX_STEPS = 10


def _monitoring_diagnostic(message: str) -> None:
    print(message, file=sys.stderr)


def main() -> None:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    workspace_root = Path.cwd()

    runtime = None
    injection = {}
    mcp_clients = []
    sandbox_client = None
    try:
        if config.monitoring.enabled:
            try:
                from agent.monitoring.runtime import MonitoringRuntime
                runtime = MonitoringRuntime(config.monitoring, workspace_root, _monitoring_diagnostic)
                hook = CompositeHook(runtime.hooks, diagnostic=_monitoring_diagnostic)
                injection = {"hook": hook, "diagnostic": _monitoring_diagnostic}
            except BaseException:
                notify_monitoring_failure(_monitoring_diagnostic)

        memory_dir = workspace_root / ".agent_memory"
        # Startup housekeeping (run counter, pruning, migrating legacy core keys)
        # happens before any agent run exists. Observed, each of its reads would
        # be recorded as a memory operation belonging to no run - one rootless
        # one-event "run" per core key - so it uses unobserved handles on the
        # same files, and the observed stores are created after it.
        startup_store = SqliteMemoryStore(memory_dir / "memory.db")
        startup_store.advance_run_count()
        startup_store.prune_memory()
        startup_core = CoreProfileManager(memory_dir / "core.md")
        for key in ALLOWED_CORE_KEYS:
            old_value = startup_store.get_general_value(key)
            if not old_value or not isinstance(old_value, str) or not old_value.strip():
                continue
            if startup_core.get_preference(key) is None:
                startup_core.update_preference(key, old_value.strip())
            startup_store.delete_general(key)

        sqlite_memory = SqliteMemoryStore(memory_dir / "memory.db", **injection)
        core_mgr = CoreProfileManager(memory_dir / "core.md", **injection)

        registry = ToolRegistry(**injection)
        register_builtin_tools(registry, workspace_root)
        mcp_clients = register_mcp_tools(registry, config.mcp_servers)

        if config.sandbox is not None and config.sandbox.private_key is None:
            print(
                "Note: 'sandbox' is configured but SANDBOX_SSH_KEY is not set; sandbox_exec is disabled.",
                file=sys.stderr,
            )
        elif config.sandbox is not None:
            from agent.tools.sandbox import SandboxExecTool, SshSandboxClient

            try:
                sandbox_client = SshSandboxClient(
                    host=config.sandbox.host,
                    port=config.sandbox.port,
                    user=config.sandbox.user,
                    private_key=config.sandbox.private_key,
                    timeout=config.sandbox.timeout,
                )
            except ValueError as exc:
                print(f"Sandbox key error: {exc}", file=sys.stderr)
                sys.exit(1)
            registry.register(SandboxExecTool(sandbox_client))

        try:
            llm = LLMClient(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                base_url=config.base_url,
                api_key=config.access_token,
                context_window_tokens=config.context_window_tokens,
                **injection,
            )
        except RuntimeError as exc:
            print(f"LLM client error: {exc}", file=sys.stderr)
            sys.exit(1)

        from agent.subagents import DelegateTaskTool, SubAgentConfig, SubAgentRunner
        from agent.tools.pdf import (
            PdfKnowledgeIndexer,
            ReadCoursePageTool,
            ReadPdfTool,
            SearchCourseMaterialTool,
            SearchPdfTool,
        )

        indexer = PdfKnowledgeIndexer(base_courses_dir=workspace_root / "resources" / "courses")
        for course_id, course_cfg in config.courses.items():
            for mat in course_cfg.materials:
                pdf_path = workspace_root / mat.path
                if pdf_path.is_file():
                    indexer.index_pdf(
                        course_id=course_id,
                        document_id=mat.id,
                        document_title=mat.title,
                        file_path=pdf_path,
                    )

        subagent_master_registry = ToolRegistry(**injection)
        for tool in registry.list_tools():
            subagent_master_registry.register(tool)
        subagent_master_registry.register(SaveMemoryTool(sqlite_memory))
        subagent_master_registry.register(ForgetMemoryTool(sqlite_memory, core_mgr))
        subagent_master_registry.register(ReadPdfTool(workspace_root))
        subagent_master_registry.register(SearchPdfTool(workspace_root))
        subagent_master_registry.register(SearchCourseMaterialTool(indexer))
        subagent_master_registry.register(ReadCoursePageTool(indexer))
        subagent_master_registry.register(RecallMemoryTool(sqlite_memory))
        subagent_master_registry.register(SaveFaqTool(sqlite_memory))
        subagent_master_registry.register(SearchFaqTool(sqlite_memory))
        subagent_master_registry.register(UpdateCorePreferenceTool(core_mgr, sqlite_memory))

        # One policy for the whole session: the parent loop and every
        # sub-agent loop consult (and share approvals through) this object.
        permission_policy = PermissionPolicy(config.permissions.deny_tools)
        subagent_runner = SubAgentRunner(
            llm=llm, master_registry=subagent_master_registry,
            hook=injection.get("hook"), diagnostic=injection.get("diagnostic"),
            permission_policy=permission_policy,
        )
        for sa_name, sa_cfg in config.subagents.items():
            subagent_runner.register_subagent(SubAgentConfig(
                name=sa_name,
                role=sa_cfg.role,
                allowed_tools=sa_cfg.allowed_tools,
                max_steps=sa_cfg.max_steps,
                description=sa_cfg.description,
                require_successful_tool=sa_cfg.require_successful_tool,
            ))

        registry.register(DelegateTaskTool(runner=subagent_runner))
        # A typo here would leave the tool you meant to block fully usable,
        # so an unknown name is worth a loud line at startup.
        known_tools = {t.name for t in registry.list_tools()} | {t.name for t in subagent_master_registry.list_tools()}
        for name in sorted(permission_policy.deny_tools - known_tools):
            print(
                f"Warning: permissions.deny_tools lists '{name}', which is not a registered tool - "
                "nothing is blocked by that entry.",
                file=sys.stderr,
            )
        course_list_str = ", ".join(f"'{cid}' ({c.title})" for cid, c in config.courses.items()) or "none"
        parent_system_prompt_base = (
            "You are an autonomous AI engineering agent operating in a ReAct (Reasoning + Acting) loop.\n"
            f"Configured official courses: {course_list_str}.\n"
            "ROUTING INSTRUCTIONS:\n"
            "1. Course queries: If the user asks ANY question about course syllabus, grading, deadlines, exams, lectures, or course policies, "
            "do NOT inspect filesystem directories with list_directory or search the web. "
            "IMMEDIATELY delegate the task to the 'course_knowledge' subagent using delegate_task.\n"
            "2. Memory & User preferences: If the user asks to remember a fact, recall past notes/preferences, "
            "or manage course FAQs, delegate the task to the 'memory_manager' subagent using delegate_task.\n"
            "3. General PDF analysis: If the user asks to inspect general PDF files in the workspace, delegate to 'pdf_analyst'.\n"
            "4. Coding & workspace tasks: Use workspace tools (create_file, read_file, etc.) directly.\n"
            "Always provide a concise thought and use structured tool calls.\n\n"
            + UNTRUSTED_CONTENT_NOTICE
        )

        def build_parent_system_prompt(context=None) -> str:
            core_profile_text = core_mgr.format_for_prompt_with_context(context=context)
            if not core_profile_text:
                return parent_system_prompt_base
            return f"{parent_system_prompt_base}\n\n{core_profile_text}"

        loop = ReactLoop(
            llm,
            tool_registry=registry,
            max_steps=MAX_STEPS,
            system_prompt=parent_system_prompt_base,
            system_prompt_provider=build_parent_system_prompt,
            tool_router=route_memory_mutation,
            compaction_threshold_percent=config.compaction_threshold_percent,
            compaction_keep_last_user_turns=config.compaction_keep_last_user_turns,
            permission_policy=permission_policy,
            **injection,
        )
        engine = ReactEngineAdapter(loop)
        cli_config = CLIConfigView(config, workspace_root, max_iterations=MAX_STEPS)
        start_cli(config=cli_config, engine=engine, registry=registry, context_manager=engine)
    finally:
        for client in mcp_clients:
            try:
                client.close()
            except BaseException:
                notify_monitoring_failure(_monitoring_diagnostic)
        if sandbox_client is not None:
            try:
                sandbox_client.close()
            except Exception:
                pass
        if runtime is not None:
            try:
                runtime.close()
            except BaseException:
                notify_monitoring_failure(_monitoring_diagnostic)


if __name__ == "__main__":
    main()
