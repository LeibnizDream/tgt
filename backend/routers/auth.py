"""
Microsoft OneDrive / Microsoft Graph OAuth2 authentication router.

This module implements a three-step MSAL (Microsoft Authentication Library)
OAuth2 flow for the application:

1. ``GET /api/auth/start``    – Redirects the user to the Microsoft login page.
   If a valid token is already cached, the redirect is skipped and the user is
   sent straight to the success page.
2. ``GET /api/auth/redirect`` – Handles the OAuth2 callback, exchanges the
   authorization code for an access token, and caches it on disk.
3. ``GET /api/auth/me``       – Returns whether a cached account exists.
4. ``POST /api/auth/logout``  – Removes cached accounts from the token cache.

Credentials are read from environment variables ``TENANT_ID``, ``CLIENT_ID``,
and ``CLIENT_SECRET`` or from ``materials/secrets.env``.
"""
import os
import logging
from pathlib import Path

import msal
from dotenv import load_dotenv
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

router = APIRouter()
logger = logging.getLogger(__name__)

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
ACCOUNT = os.getenv("ACCOUNT")

if not (TENANT_ID and CLIENT_ID and CLIENT_SECRET):
    envf = Path(__file__).parent.parent / "materials" / "secrets.env"
    if envf.exists():
        load_dotenv(envf)
        TENANT_ID = os.getenv("TENANT_ID")
        CLIENT_ID = os.getenv("CLIENT_ID")
        CLIENT_SECRET = os.getenv("CLIENT_SECRET")
        ACCOUNT = os.getenv("ACCOUNT")
if not (TENANT_ID and CLIENT_ID and CLIENT_SECRET):
    raise ValueError("Missing OneDrive OAuth credentials")

# Keep your scopes as requested
SCOPES    = ["Files.ReadWrite.All", "User.Read"]
AUTH_URL  = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
GRAPH     = "https://graph.microsoft.com/v1.0"

CACHE_FILE = Path(__file__).parent.parent / "materials" / "token_cache.json"

def _save_cache(cache: msal.SerializableTokenCache):
    """Persist the MSAL token cache to ``materials/token_cache.json`` if it changed."""
    print("Saving token cache to", CACHE_FILE)
    if cache.has_state_changed:
        with open(CACHE_FILE, "w") as f:
            f.write(cache.serialize())

def _load_cache() -> msal.SerializableTokenCache:
    """Load and deserialize the MSAL token cache from disk, or return an empty cache."""
    print("Loading token cache from", CACHE_FILE)
    cache = msal.SerializableTokenCache()
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            cache.deserialize(f.read())
    return cache

def _build_msal_app(cache=None):
    """Create an MSAL ``ConfidentialClientApplication`` using the configured credentials."""
    return msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{TENANT_ID}",
        client_credential=CLIENT_SECRET,
        token_cache=cache
    )

def get_fresh_token() -> str:
    """Get a valid access token from the MSAL cache, refreshing silently if needed."""
    cache = _load_cache()
    app = _build_msal_app(cache)
    accounts = app.get_accounts()
    if not accounts:
        raise RuntimeError("No cached account. User must connect first.")
    result = app.acquire_token_silent(SCOPES, account=accounts[0])
    if not result or "access_token" not in result:
        raise RuntimeError("Token refresh failed. User must reconnect.")
    _save_cache(cache)
    return result["access_token"]


@router.get("/me")
async def auth_me():
    """Return whether a cached MSAL account exists."""
    cache = _load_cache()
    app = _build_msal_app(cache) 
    accounts = app.get_accounts()
    if accounts:
        return {"authenticated": True, "account": accounts[0].get("username")}
    return {"authenticated": False}


@router.post("/logout")
async def auth_logout():
    """Remove all cached accounts except the configured ACCOUNT."""
    if not CACHE_FILE.exists():
        return {"status": "logged_out"}

    cache = _load_cache()
    app = _build_msal_app(cache)
    accounts = app.get_accounts()

    for account in accounts:
        if not ACCOUNT or account.get("username") != ACCOUNT:
            app.remove_account(account)
            print(f"Removed account {account.get('username')} from cache.")

    _save_cache(cache)
    return {"status": "logged_out"}


@router.get("/start")
async def start_onedrive_auth(request: Request):
    """
    Begin the OneDrive OAuth2 login flow.

    Attempts a silent token acquisition first. If a valid cached token is found,
    the user is redirected directly to the frontend success page with the token
    in the URL fragment. Otherwise the user is redirected to the Microsoft
    consent/login page.
    """
    host   = request.headers.get("host")
    scheme = "http" if host.startswith(("localhost", "127.0.0.1")) else "https"
    redirect_uri = f"{scheme}://{host}/api/auth/redirect"

    cache = _load_cache()
    app = _build_msal_app(cache)

    # 1) Try silent first (no UI)
    if cache:
        accounts = app.get_accounts(username=ACCOUNT) if ACCOUNT else app.get_accounts()
        if accounts:
            print("Account(s) in cache:", ", ".join(a["username"] for a in accounts))
            result = app.acquire_token_silent(SCOPES, account=accounts[0])
            if result and "access_token" in result:
                print("Using cached token, no login required.")
                _save_cache(cache)
                token = result["access_token"]
                frontend_success = f"{scheme}://{host}/success-auth#token={token}"
                return RedirectResponse(url=frontend_success, status_code=302)
            else:
                print("Silent token acquisition failed, need to login.")

    print("No suitable token exists in cache. Redirecting to login.")
    eqp = {}
    if ACCOUNT:
        eqp["login_hint"]  = ACCOUNT
        # eqp["domain_hint"] = "leibniz-zas.de"

    auth_url = app.get_authorization_request_url(
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        response_mode="query",
        extra_query_parameters=eqp or None,
        prompt="select_account",
    )
    return RedirectResponse(url=auth_url, status_code=302)

@router.get("/redirect")
async def onedrive_auth_redirect(request: Request):
    """
    Handle the OAuth2 callback from Microsoft.

    Exchanges the authorization code present in the query string for an access
    token, persists the token cache, and redirects the user to the frontend
    success page with the token in the URL fragment.
    """
    code = request.query_params.get("code")
    if not code:
        return JSONResponse({"error": "No code in callback"}, status_code=400)

    host   = request.headers.get("host")
    scheme = "http" if host.startswith(("localhost", "127.0.0.1")) else "https"
    redirect_uri = f"{scheme}://{host}/api/auth/redirect"

    cache = _load_cache()
    app   = _build_msal_app(cache)
    result = app.acquire_token_by_authorization_code(code, scopes=SCOPES, redirect_uri=redirect_uri)

    if "access_token" not in result:
        return JSONResponse({"error": "Token error", "details": {k: result.get(k) for k in ("error","error_description","correlation_id")}}, status_code=400)

    _save_cache(cache)
    token = result["access_token"]
    frontend_success = f"{scheme}://{host}/success-auth#token={token}"
    return RedirectResponse(url=frontend_success, status_code=302)
