from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_403_FORBIDDEN
from app.config.settings import settings, AuthMode
from app.core.security import decode_access_token
from typing import Optional

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
security_bearer = HTTPBearer(auto_error=False)

async def verify_api_key(
    api_key_header_val: Optional[str] = Security(api_key_header),
    bearer_val: Optional[HTTPAuthorizationCredentials] = Security(security_bearer)
) -> str:
    """
    Verifies API Key or JWT Token based on configured AUTH_MODE.
    Returns: 'master', 'anonymous', or 'sub' (subject) from JWT.
    """
    # 1. NONE Mode
    if settings.auth_mode == AuthMode.NONE:
        return "anonymous"

    # 2. KEY Mode
    if settings.auth_mode == AuthMode.KEY:
        # Check Header
        if api_key_header_val and api_key_header_val == settings.api_key:
            return "master"
        # Check Bearer (legacy support for key in bearer)
        if bearer_val and bearer_val.credentials == settings.api_key:
            return "master"
        
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )

    # 3. JWT Mode
    if settings.auth_mode == AuthMode.JWT:
        if not bearer_val:
             raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="Missing Bearer Token"
            )
        
        token_str = bearer_val.credentials
        payload = decode_access_token(token_str)
        
        if not payload:
             raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="Invalid or Expired Token"
            )
            
        client_id = payload.get("client_id")
        if not client_id or client_id not in settings.registered_client_ids:
             raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="Invalid Client ID"
            )
            
        return payload.get("sub", "unknown")

    # Fallback
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Authentication Configuration Error"
    )

async def verify_master_key(
    api_key_header_val: Optional[str] = Security(api_key_header),
    bearer_val: Optional[HTTPAuthorizationCredentials] = Security(security_bearer)
) -> str:
    """
    Strictly verifies ONLY the Master API Key.
    Used for administrative tasks and token generation.
    """
    token = None
    if api_key_header_val:
        token = api_key_header_val
    elif bearer_val:
        token = bearer_val.credentials
        
    if not token or token != settings.api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid Master API Key"
        )
    return "master"
