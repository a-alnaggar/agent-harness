import base64
import io
import threading

import paramiko
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from agent.tools.sandbox.ssh_client import SshSandboxClient, load_private_key


def ed25519_openssh_pem(passphrase: bytes | None = None) -> str:
    """Same format `ssh-keygen -t ed25519` writes."""
    encryption = (
        serialization.BestAvailableEncryption(passphrase) if passphrase else serialization.NoEncryption()
    )
    return ed25519.Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, encryption
    ).decode()


def ecdsa_pem() -> str:
    buffer = io.StringIO()
    paramiko.ECDSAKey.generate().write_private_key(buffer)
    return buffer.getvalue()


def rsa_pem() -> str:
    buffer = io.StringIO()
    paramiko.RSAKey.generate(2048).write_private_key(buffer)
    return buffer.getvalue()


KEY = ed25519_openssh_pem()


class FakeChannel:
    def __init__(self, exit_code, finished=True):
        self.exit_code = exit_code
        self.status_event = threading.Event()
        if finished:
            self.status_event.set()

    def recv_exit_status(self):
        return self.exit_code


class FakeStdout(io.BytesIO):
    def __init__(self, data: bytes, exit_code: int):
        super().__init__(data)
        self.channel = FakeChannel(exit_code)


class FakeTransport:
    def __init__(self):
        self.active = True
        self.keepalive = None

    def is_active(self):
        return self.active

    def set_keepalive(self, seconds):
        self.keepalive = seconds


class FakeParamikoClient:
    instances = []

    def __init__(self):
        self.connected_with = None
        self.closed = False
        self.commands = []
        FakeParamikoClient.instances.append(self)

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def connect(self, hostname, port, username, pkey, look_for_keys, allow_agent, timeout,
                banner_timeout, auth_timeout):
        self.connected_with = dict(
            hostname=hostname, port=port, username=username, pkey=pkey,
            look_for_keys=look_for_keys, allow_agent=allow_agent, timeout=timeout,
            banner_timeout=banner_timeout, auth_timeout=auth_timeout,
        )
        self.transport = FakeTransport()

    def get_transport(self):
        return getattr(self, "transport", None)

    def exec_command(self, command, timeout):
        self.commands.append(command)
        stdout = FakeStdout(b"hello\n", exit_code=0)
        stderr = io.BytesIO(b"")
        return None, stdout, stderr

    def close(self):
        self.closed = True


def test_connects_lazily_on_first_command(monkeypatch):
    FakeParamikoClient.instances.clear()
    monkeypatch.setattr(paramiko, "SSHClient", FakeParamikoClient)

    client = SshSandboxClient(host="sandbox.example.com", port=22, user="agent", private_key=KEY)
    assert FakeParamikoClient.instances == []

    result = client.run_command("echo hello")

    fake = FakeParamikoClient.instances[0]
    connected = fake.connected_with
    assert (connected["hostname"], connected["port"], connected["username"], connected["timeout"]) == (
        "sandbox.example.com", 22, "agent", 15,
    )
    assert isinstance(connected["pkey"], paramiko.Ed25519Key)
    assert result.exit_code == 0
    assert result.stdout == "hello\n"


def test_only_the_dedicated_key_is_offered_never_ssh_agent_or_default_keys(monkeypatch):
    FakeParamikoClient.instances.clear()
    monkeypatch.setattr(paramiko, "SSHClient", FakeParamikoClient)

    SshSandboxClient(host="h", port=22, user="u", private_key=KEY).run_command("ls")

    connected = FakeParamikoClient.instances[0].connected_with
    assert connected["look_for_keys"] is False
    assert connected["allow_agent"] is False


def test_reuses_connection_across_commands(monkeypatch):
    FakeParamikoClient.instances.clear()
    monkeypatch.setattr(paramiko, "SSHClient", FakeParamikoClient)

    client = SshSandboxClient(host="h", port=22, user="u", private_key=KEY)
    client.run_command("one")
    client.run_command("two")

    assert len(FakeParamikoClient.instances) == 1
    assert FakeParamikoClient.instances[0].commands == ["one", "two"]


def test_close_is_idempotent_and_resets_connection(monkeypatch):
    FakeParamikoClient.instances.clear()
    monkeypatch.setattr(paramiko, "SSHClient", FakeParamikoClient)

    client = SshSandboxClient(host="h", port=22, user="u", private_key=KEY)
    client.run_command("one")
    fake = FakeParamikoClient.instances[0]

    client.close()
    client.close()

    assert fake.closed is True


def fingerprint(key: paramiko.PKey) -> bytes:
    return key.asbytes()


@pytest.mark.parametrize("make_pem, key_class", [
    (ed25519_openssh_pem, paramiko.Ed25519Key),
    (ecdsa_pem, paramiko.ECDSAKey),
    (rsa_pem, paramiko.RSAKey),
])
def test_load_private_key_detects_each_supported_key_type(make_pem, key_class):
    assert isinstance(load_private_key(make_pem()), key_class)


def test_load_private_key_accepts_base64_of_the_key_file():
    pem = ed25519_openssh_pem()
    encoded = base64.b64encode(pem.encode()).decode()

    assert fingerprint(load_private_key(encoded)) == fingerprint(load_private_key(pem))


def test_load_private_key_accepts_escaped_newlines_as_some_env_files_store_it():
    pem = ed25519_openssh_pem()

    assert fingerprint(load_private_key(pem.strip().replace("\n", "\\n"))) == fingerprint(load_private_key(pem))


def test_load_private_key_ignores_surrounding_whitespace():
    pem = ed25519_openssh_pem()

    assert fingerprint(load_private_key(f"\n  {pem}  \n")) == fingerprint(load_private_key(pem))


def test_load_private_key_rejects_text_that_is_neither_a_key_nor_base64():
    with pytest.raises(ValueError, match="PEM/OpenSSH text\\) or its base64"):
        load_private_key("this is not a key!")


def test_load_private_key_rejects_base64_that_does_not_decode_to_a_key():
    with pytest.raises(ValueError, match="not a usable private key"):
        load_private_key(base64.b64encode(b"just some bytes, no key here").decode())


def test_load_private_key_rejects_a_passphrase_protected_key():
    with pytest.raises(ValueError, match="without a passphrase"):
        load_private_key(ed25519_openssh_pem(passphrase=b"hunter2"))


def test_a_malformed_key_fails_at_construction_not_at_first_command():
    with pytest.raises(ValueError):
        SshSandboxClient(host="h", port=22, user="u", private_key="garbage!!")



class HangingParamikoClient(FakeParamikoClient):
    def exec_command(self, command, timeout):
        self.commands.append(command)
        stdout = FakeStdout(b"", exit_code=0)
        stdout.channel = FakeChannel(0, finished=False)
        return None, stdout, io.BytesIO(b"")


def test_a_command_that_never_finishes_times_out_instead_of_hanging(monkeypatch):
    FakeParamikoClient.instances.clear()
    monkeypatch.setattr(paramiko, "SSHClient", HangingParamikoClient)
    client = SshSandboxClient(host="h", port=22, user="u", private_key=KEY, timeout=0.05)

    with pytest.raises(TimeoutError, match="did not finish within"):
        client.run_command("sleep 1000")


def test_a_failed_connection_is_dropped_and_the_next_command_reconnects(monkeypatch):
    FakeParamikoClient.instances.clear()
    monkeypatch.setattr(paramiko, "SSHClient", HangingParamikoClient)
    client = SshSandboxClient(host="h", port=22, user="u", private_key=KEY, timeout=0.05)
    with pytest.raises(TimeoutError):
        client.run_command("sleep 1000")

    monkeypatch.setattr(paramiko, "SSHClient", FakeParamikoClient)
    result = client.run_command("echo hello")

    assert FakeParamikoClient.instances[0].closed
    assert len(FakeParamikoClient.instances) == 2 and result.stdout == "hello\n"



def test_a_connection_that_died_while_idle_is_replaced_before_the_next_command(monkeypatch):
    """Seen live: after a quiet spell every command "timed out" on the dead
    connection until the agent was restarted."""
    FakeParamikoClient.instances.clear()
    monkeypatch.setattr(paramiko, "SSHClient", FakeParamikoClient)
    client = SshSandboxClient(host="h", port=22, user="u", private_key=KEY)
    client.run_command("echo hello")
    FakeParamikoClient.instances[0].transport.active = False

    result = client.run_command("echo hello")

    assert FakeParamikoClient.instances[0].closed
    assert len(FakeParamikoClient.instances) == 2 and result.exit_code == 0


def test_every_phase_of_connecting_is_bounded_and_the_connection_kept_alive(monkeypatch):
    FakeParamikoClient.instances.clear()
    monkeypatch.setattr(paramiko, "SSHClient", FakeParamikoClient)

    SshSandboxClient(host="h", port=22, user="u", private_key=KEY, timeout=9).run_command("ls")

    fake = FakeParamikoClient.instances[0]
    assert fake.connected_with["banner_timeout"] == 9 and fake.connected_with["auth_timeout"] == 9
    assert fake.transport.keepalive == 30
