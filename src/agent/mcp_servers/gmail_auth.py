"""Explicit OAuth setup keeps interactive authorization off MCP stdout."""

import os
import sys
import tempfile
from pathlib import Path

import httplib2
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from mcp.server.mcpserver.exceptions import ToolError
from oauthlib.oauth2 import OAuth2Error

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly",
          "https://www.googleapis.com/auth/gmail.compose"]


# Falls back to a dedicated directory mounted at /gmail (scripts/run.sh
# mounts .secrets/gmail there) - a directory, because the token file is
# replaced on refresh, and only that directory, never the whole repo. Either
# path is still overridable via its env var for a non-Docker setup.
_DEFAULT_PATHS = {
    "GMAIL_CREDENTIALS_PATH": "/gmail/credentials.json",
    "GMAIL_TOKEN_PATH": "/gmail/gmail-token.json",
}


def credential_paths() -> tuple[Path, Path]:
    paths = []
    for name, default in _DEFAULT_PATHS.items():
        raw = os.environ.get(name)
        value = (raw if raw is not None else default).strip()
        if not value:
            raise ToolError(f"Set {name} to an absolute private file path.")
        path = Path(value).expanduser()
        is_posix_root = sys.platform == "win32" and (value.startswith("/") or value.startswith("\\"))
        if not path.is_absolute() and not is_posix_root:
            raise ToolError(f"{name} must be an absolute path.")
        paths.append(path if is_posix_root else path.resolve())
    if paths[0] == paths[1]:
        raise ToolError("Credential and token paths must be different.")
    return paths[0], paths[1]


def save_token(path: Path, credentials: Credentials) -> None:
    serialized = credentials.to_json()
    # A sibling temporary file keeps replacement atomic on the same filesystem.
    descriptor, temporary = tempfile.mkstemp(prefix=".gmail-token-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_credentials() -> Credentials:
    _, token_path = credential_paths()
    try:
        credentials = Credentials.from_authorized_user_file(str(token_path))
        if not credentials.has_scopes(SCOPES):
            raise ToolError("Gmail token lacks required scopes. Run the OAuth setup command again.")
        if not credentials.valid:
            if not credentials.refresh_token:
                raise ToolError("Gmail authorization expired. Run the OAuth setup command again.")
            credentials.refresh(Request())
            save_token(token_path, credentials)
        return credentials
    except (OSError, ValueError, GoogleAuthError):
        raise ToolError("Gmail authorization unavailable. Run python -m agent.mcp_servers.gmail_auth.") from None


def gmail_service() -> object:
    credentials = load_credentials()
    return build("gmail", "v1", http=AuthorizedHttp(credentials, http=httplib2.Http(timeout=15)),
                 cache_discovery=False, static_discovery=True)


def authorize() -> None:
    credential_path, token_path = credential_paths()
    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(credential_path), SCOPES)
        credentials = flow.run_local_server(host="localhost", port=0, timeout_seconds=600,
                                            access_type="offline", prompt="consent")
        save_token(token_path, credentials)
    except (OSError, ValueError, GoogleAuthError, OAuth2Error, AttributeError):
        raise ToolError("OAuth setup failed. Check the desktop client file and test-user configuration.") from None


if __name__ == "__main__":
    try:
        authorize()
    except ToolError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
