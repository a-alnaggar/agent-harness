from pathlib import Path
from unittest.mock import MagicMock

import pytest
from google.auth.exceptions import RefreshError
from mcp.server.mcpserver.exceptions import ToolError
from oauthlib.oauth2 import AccessDeniedError

from agent.mcp_servers import gmail_auth as auth


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    client, token = tmp_path / "client.json", tmp_path / "token.json"
    monkeypatch.setenv("GMAIL_CREDENTIALS_PATH", str(client))
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(token))
    return client, token


@pytest.mark.parametrize("value", ["", "relative.json"])
def test_invalid_paths(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMAIL_CREDENTIALS_PATH", value)
    with pytest.raises(ToolError):
        auth.credential_paths()


def test_credential_paths_default_to_the_mounted_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GMAIL_CREDENTIALS_PATH", raising=False)
    monkeypatch.delenv("GMAIL_TOKEN_PATH", raising=False)

    credential_path, token_path = auth.credential_paths()

    assert credential_path == Path("/gmail/credentials.json")
    assert token_path == Path("/gmail/gmail-token.json")


def test_same_paths(paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(paths[0]))
    with pytest.raises(ToolError, match="different"):
        auth.credential_paths()


def test_missing_token(paths: tuple[Path, Path]) -> None:
    with pytest.raises(ToolError, match="authorization unavailable"):
        auth.load_credentials()


@pytest.mark.parametrize("expired", [False, True])
def test_load_and_refresh(expired: bool, paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = MagicMock(valid=not expired, refresh_token="fake", to_json=lambda: '{"test": true}')
    monkeypatch.setattr(auth.Credentials, "from_authorized_user_file", lambda path: credentials)
    assert auth.load_credentials() is credentials
    credentials.has_scopes.assert_called_once_with(auth.SCOPES)
    if expired:
        credentials.refresh.assert_called_once()
        assert paths[1].read_text() == '{"test": true}'
    else:
        credentials.refresh.assert_not_called()
        assert not paths[1].exists()


def test_scope_failure(paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = MagicMock()
    credentials.has_scopes.return_value = False
    monkeypatch.setattr(auth.Credentials, "from_authorized_user_file", lambda path: credentials)
    with pytest.raises(ToolError, match="scopes"):
        auth.load_credentials()
    credentials.refresh.assert_not_called()


def test_refresh_failure(paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = MagicMock(valid=False, refresh_token="fake")
    credentials.refresh.side_effect = RefreshError("private")
    monkeypatch.setattr(auth.Credentials, "from_authorized_user_file", lambda path: credentials)
    with pytest.raises(ToolError, match="authorization unavailable") as caught:
        auth.load_credentials()
    assert "private" not in str(caught.value)


def test_authorize(paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    flow = MagicMock()
    flow.run_local_server.return_value.to_json.return_value = '{"test": true}'
    factory = MagicMock(return_value=flow)
    monkeypatch.setattr(auth.InstalledAppFlow, "from_client_secrets_file", factory)
    auth.authorize()
    factory.assert_called_once_with(str(paths[0]), auth.SCOPES)
    assert paths[1].read_text() == '{"test": true}'
    assert flow.run_local_server.call_args.kwargs["access_type"] == "offline"
    assert flow.run_local_server.call_args.kwargs["timeout_seconds"] == 600


def test_save_token_replaces_existing_file_atomically(paths: tuple[Path, Path]) -> None:
    token_path = paths[1]
    token_path.write_text('{"old": true}')
    credentials = MagicMock(to_json=lambda: '{"new": true}')

    auth.save_token(token_path, credentials)

    assert token_path.read_text() == '{"new": true}'
    assert not list(token_path.parent.glob(".gmail-token-*.json"))


def test_save_token_uses_a_sibling_temporary_file(paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    token_path = paths[1]
    credentials = MagicMock(to_json=lambda: '{"new": true}')
    original_replace = auth.os.replace
    replaced = {}

    def capture_replace(source: str, destination: Path) -> None:
        replaced["source"] = Path(source)
        replaced["destination"] = Path(destination)
        original_replace(source, destination)

    monkeypatch.setattr(auth.os, "replace", capture_replace)
    auth.save_token(token_path, credentials)

    assert replaced["source"].parent == token_path.parent
    assert replaced["destination"] == token_path


def test_save_token_keeps_existing_file_when_replacement_fails(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    token_path = paths[1]
    token_path.write_text('{"old": true}')
    credentials = MagicMock(to_json=lambda: '{"new": true}')
    monkeypatch.setattr(auth.os, "replace", lambda source, destination: (_ for _ in ()).throw(OSError("failed")))

    with pytest.raises(OSError, match="failed"):
        auth.save_token(token_path, credentials)

    assert token_path.read_text() == '{"old": true}'
    assert not list(token_path.parent.glob(".gmail-token-*.json"))


def test_service_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = MagicMock()
    monkeypatch.setattr(auth, "load_credentials", lambda: credentials)
    builder = MagicMock()
    monkeypatch.setattr(auth, "build", builder)
    auth.gmail_service()
    assert builder.call_args.kwargs["http"].http.timeout == 15
    assert builder.call_args.kwargs["static_discovery"] is True


def test_oauth_denied(paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    flow = MagicMock()
    flow.run_local_server.side_effect = AccessDeniedError(description="private")
    monkeypatch.setattr(auth.InstalledAppFlow, "from_client_secrets_file", lambda *args: flow)
    with pytest.raises(ToolError, match="OAuth setup failed"):
        auth.authorize()
    assert not paths[1].exists()
