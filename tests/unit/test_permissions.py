"""PermissionPolicy: what a tool's effective tier is, and what "always"
means - the rules the loop enforces, tested without a loop."""

from agent.tools.base import Tool
from agent.tools.permissions import PermissionPolicy, PermissionTier


def make_tool(tier=None, name="some_tool"):
    class _Tool(Tool):
        description = "d"

        def run(self, **kwargs) -> str:
            return ""

    _Tool.name = name
    if tier is not None:
        _Tool.permission_tier = tier
    return _Tool()


def test_the_tools_declared_tier_is_used_when_nothing_is_denied():
    policy = PermissionPolicy()

    for tier in (PermissionTier.REGULAR, PermissionTier.CONFIRM, PermissionTier.STRICT):
        assert policy.tier_for("some_tool", make_tool(tier)) == tier


def test_a_tool_that_declares_no_tier_is_never_treated_as_regular():
    assert PermissionPolicy().tier_for("mystery", object()) == PermissionTier.CONFIRM


def test_a_denied_name_overrides_the_tools_own_tier():
    policy = PermissionPolicy(deny_tools=["send_email"])

    assert policy.tier_for("send_email", make_tool(PermissionTier.REGULAR)) == PermissionTier.DENY
    assert policy.tier_for("read_file", make_tool(PermissionTier.REGULAR)) == PermissionTier.REGULAR


def test_deny_list_is_immutable_after_construction():
    policy = PermissionPolicy(deny_tools=["a"])

    assert policy.deny_tools == frozenset({"a"})
    assert not hasattr(policy.deny_tools, "add")


def test_confirm_tools_remember_an_always_approval_per_tool_name():
    policy = PermissionPolicy()
    assert not policy.is_approved("create_draft", PermissionTier.CONFIRM)

    policy.remember("create_draft", PermissionTier.CONFIRM)

    assert policy.is_approved("create_draft", PermissionTier.CONFIRM)
    assert not policy.is_approved("other_tool", PermissionTier.CONFIRM)


def test_strict_tools_never_accept_a_remembered_approval():
    policy = PermissionPolicy()

    policy.remember("send_email", PermissionTier.STRICT)
    assert not policy.is_approved("send_email", PermissionTier.STRICT)

    # even if the same name had been approved while it was a CONFIRM tool
    policy.remember("send_email", PermissionTier.CONFIRM)
    assert not policy.is_approved("send_email", PermissionTier.STRICT)


def test_two_policies_do_not_share_approvals():
    first, second = PermissionPolicy(), PermissionPolicy()

    first.remember("create_draft", PermissionTier.CONFIRM)

    assert not second.is_approved("create_draft", PermissionTier.CONFIRM)
