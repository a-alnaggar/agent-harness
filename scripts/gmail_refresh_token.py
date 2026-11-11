"""Gmail OAuth refresh, run inside Docker. Binds to 0.0.0.0 so the
port-forward reaches it, but presents localhost:PORT as the redirect URI
so the browser can actually open it."""

import sys
sys.path.insert(0, "src")

from google_auth_oauthlib.flow import InstalledAppFlow
from agent.mcp_servers.gmail_auth import SCOPES, credential_paths, save_token

credential_path, token_path = credential_paths()
flow = InstalledAppFlow.from_client_secrets_file(str(credential_path), SCOPES)
credentials = flow.run_local_server(
    host="localhost", bind_addr="0.0.0.0", port=8765,
    open_browser=False, access_type="offline", prompt="consent",
)
save_token(token_path, credentials)
print("Saved token to", token_path)
