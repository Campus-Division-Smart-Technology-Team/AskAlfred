import os
import msal

AUTHORITY = f"https://login.microsoftonline.com/{os.environ['AZURE_TENANT_ID']}"
SCOPES = ["User.Read"]  # minimal, safe default


def build_msal_app(cache=None):
    return msal.ConfidentialClientApplication(
        client_id=os.environ["AZURE_CLIENT_ID"],
        authority=AUTHORITY,
        client_credential=os.environ["AZURE_CLIENT_SECRET"],
        token_cache=cache,
    )
