import pytest

from agent.tools.sandbox.command_filter import find_destructive_match


@pytest.mark.parametrize("command", [
    "rm -rf /tmp/whatever",
    "sudo rm important.txt",
    "dd if=/dev/zero of=/dev/sda",
    "mkfs.ext4 /dev/sdb1",
    "shred -u secret.txt",
    "git push --force origin main",
    "git push -f origin main",
    "git reset --hard HEAD~1",
    "git clean -xdf",
    ":(){ :|:& };:",
])
def test_flags_destructive_commands(command):
    assert find_destructive_match(command) is not None


@pytest.mark.parametrize("command", [
    "ls -la",
    "echo hello",
    "python3 script.py",
    "git status",
    "git push origin main",
    "cat file.txt",
    "mkdir new_dir",
    "pip install requests",
])
def test_allows_ordinary_commands(command):
    assert find_destructive_match(command) is None
