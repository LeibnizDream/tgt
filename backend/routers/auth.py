import os
import logging
from pathlib import Path

import msal
import jwt
import requests
from dotenv import load_dotenv
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

router = APIRouter()
logger = logging.getLogger(__name__)

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
if not (TENANT_ID and CLIENT_ID and CLIENT_SECRET):
    envf = Path(__file__).parent.parent / "materials" / "secrets.env"
    if envf.exists():
        load_dotenv(envf)
        TENANT_ID = os.getenv("TENANT_ID")
        CLIENT_ID = os.getenv("CLIENT_ID")
        CLIENT_SECRET = os.getenv("CLIENT_SECRET")
if not (TENANT_ID and CLIENT_ID and CLIENT_SECRET):
    raise ValueError("Missing OneDrive OAuth credentials")

# Keep your scopes as requested
SCOPES    = ["Files.ReadWrite.All", "User.Read"]
AUTH_URL  = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
GRAPH     = "https://graph.microsoft.com/v1.0"

def _build_msal_app(cache=None):
    return msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{TENANT_ID}",
        client_credential=CLIENT_SECRET,
        token_cache=cache
    )

def _decode_jwt_noverify(token: str) -> dict:
    """
    Decode a JWT without verifying the signature (debug only).
    """
    try:
        return jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
    except Exception as e:
        logger.warning("JWT decode failed: %s", e)
        return {}

def _log_permissions_debug(access_token: str, id_token: str | None = None) -> None:
    """
    Log account, tenant, audience, and granted scopes.
    """
    if id_token:
        it = _decode_jwt_noverify(id_token)
        logger.info("=== ID TOKEN ===")
        logger.info("preferred_username: %s", it.get("preferred_username"))
        logger.info("name              : %s", it.get("name"))
        logger.info("tid (tenant)      : %s", it.get("tid"))
        logger.info("aud               : %s", it.get("aud"))
        logger.info("=================")

    at = _decode_jwt_noverify(access_token)
    logger.info("=== ACCESS TOKEN ===")
    logger.info("scp (scopes)      : %s", at.get("scp"))   # this is the key bit for permissions
    logger.info("roles             : %s", at.get("roles"))
    logger.info("tid (tenant)      : %s", at.get("tid"))
    logger.info("aud               : %s", at.get("aud"))
    logger.info("====================")

def _graph_get(path: str, token: str) -> requests.Response:
    return requests.get(
        f"{GRAPH}{path}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15
    )

@router.get("/start")
async def start_onedrive_auth(request: Request):
    host   = request.headers.get("host")
    scheme = "http" if host.startswith(("localhost", "127.0.0.1")) else "https"
    redirect_uri = f"{scheme}://{host}/api/auth/redirect"

    msal_app = _build_msal_app()
    auth_url = msal_app.get_authorization_request_url(
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        response_mode="query",
        # Optional during testing: uncomment to force account picker
        # prompt="select_account",
    )
    return RedirectResponse(url=auth_url, status_code=302)

@router.get("/redirect")
async def onedrive_auth_redirect(request: Request):
    code = request.query_params.get("code")
    if not code:
        return JSONResponse({"error": "No code in callback"}, status_code=400)

    host   = request.headers.get("host")
    scheme = "http" if host.startswith(("localhost", "127.0.0.1")) else "https"
    redirect_uri = f"{scheme}://{host}/api/auth/redirect"

    msal_app = _build_msal_app()
    result = msal_app.acquire_token_by_authorization_code(
        code,
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )

    token = result.get("access_token")
    if not token:
        # Log MSAL error for diagnosis
        logger.warning("Token acquisition failed: %s", {k: result.get(k) for k in ("error","error_description","correlation_id")})
        return JSONResponse({"error": "Token error", "details": result}, status_code=400)

    # ---- PERMISSIONS DEBUG START ----
    _log_permissions_debug(access_token=token, id_token=result.get("id_token"))

    # Quick sanity checks to surface 403s early (permissions/resource issues)
    try:
        me = _graph_get("/me", token)
        logger.info("Graph /me -> %s", me.status_code)
        if me.status_code != 200:
            logger.warning("Graph /me body: %s", me.text[:500])

        drive = _graph_get("/me/drive/root", token)
        logger.info("Graph /me/drive/root -> %s", drive.status_code)
        if drive.status_code != 200:
            logger.warning("Graph /me/drive/root body: %s", drive.text[:500])
    except Exception as e:
        logger.warning("Graph sanity checks failed: %s", e)
    # ---- PERMISSIONS DEBUG END ----

    # Keep your existing frontend contract (token in hash)
    frontend_success = f"{scheme}://{host}/success-auth#token={token}"
    return RedirectResponse(url=frontend_success, status_code=302)
