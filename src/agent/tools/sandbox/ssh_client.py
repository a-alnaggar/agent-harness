"""Thin wrapper around paramiko for the sandbox_exec tool. Kept separate
from SandboxExecTool so the tool itself can be tested against a fake
client without touching real SSH/paramiko internals."""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass

import paramiko

_KEY_TYPES = (paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey)


def load_private_key(text: str) -> paramiko.PKey:
    """Accepts the private key as raw PEM/OpenSSH text (real newlines, or
    literal \\n as some .env loaders store it) or as one base64 line - the
    single-line form survives `docker --env-file` and CI variables, which
    multi-line PEM does not."""
    text = text.strip().replace("\\n", "\n")
    if "-----BEGIN" not in text:
        try:
            text = base64.b64decode(text, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ValueError(
                "SANDBOX_SSH_KEY must be a private key (PEM/OpenSSH text) or its base64 encoding"
            ) from exc
    for key_type in _KEY_TYPES:
        try:
            return key_type.from_private_key(io.StringIO(text))
        except (paramiko.SSHException, ValueError):
            continue
    raise ValueError(
        "SANDBOX_SSH_KEY is not a usable private key - it must be an Ed25519, ECDSA, "
        "or RSA key generated without a passphrase"
    )


@dataclass
class SshCommandResult:
    exit_code: int
    stdout: str
    stderr: str


class SshSandboxClient:
    """One SSH connection to a single sandbox host. Connects lazily on the
    first command and stays open across calls within a session."""

    def __init__(self, host: str, port: int, user: str, private_key: str, timeout: int = 15):
        self.host = host
        self.port = port
        self.user = user
        self._pkey = load_private_key(private_key)
        self.timeout = timeout
        self._client: paramiko.SSHClient | None = None

    def _connect(self) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        # AutoAddPolicy skips host-key verification, which would be a real
        # problem for a long-lived production target - acceptable here only
        # because this is meant to point at a disposable lab sandbox host,
        # not a machine holding anything sensitive.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.host,
            port=self.port,
            username=self.user,
            pkey=self._pkey,
            # Only the dedicated sandbox key may authenticate - never fall
            # back to whatever keys happen to be in ~/.ssh or an ssh-agent.
            look_for_keys=False,
            allow_agent=False,
            timeout=self.timeout,
            banner_timeout=self.timeout,
            auth_timeout=self.timeout,
        )
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(30)
        return client

    def run_command(self, command: str) -> SshCommandResult:
        try:
            transport = self._client.get_transport() if self._client is not None else None
            if transport is None or not transport.is_active():
                # A connection that died while idle would otherwise time out
                # every command until the agent restarts.
                self.close()
                self._client = self._connect()
            _stdin, stdout, stderr = self._client.exec_command(command, timeout=self.timeout)
            # recv_exit_status() alone waits forever on a command that never ends.
            if not stdout.channel.status_event.wait(self.timeout):
                raise TimeoutError(f"command did not finish within {self.timeout} s")
            exit_code = stdout.channel.recv_exit_status()
            return SshCommandResult(
                exit_code=exit_code,
                stdout=stdout.read().decode("utf-8", errors="replace"),
                stderr=stderr.read().decode("utf-8", errors="replace"),
            )
        except Exception:
            # Don't reuse a connection that just failed; the next call reconnects.
            self.close()
            raise

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
