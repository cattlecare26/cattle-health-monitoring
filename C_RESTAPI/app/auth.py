"""
Authentication & authorization dependencies.
Supports dual auth: JWT Bearer token (for users) and X-API-Key (for ESP32 devices).
Provides role-based and farm-scoped access control.
"""

from typing import Optional

from fastapi import Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.user_services import decode_access_token, get_user_by_username

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(default=None),
) -> Optional[dict]:
    """
    Authenticate via JWT Bearer token or X-API-Key.
    Returns user dict for JWT auth, or None for valid API key auth (ESP32 devices).
    Raises 401 if neither is valid.
    """
    # Try JWT Bearer token first
    if credentials:
        payload = decode_access_token(credentials.credentials)
        if payload and "sub" in payload:
            user = await get_user_by_username(payload["sub"])
            if user and user.get("is_active", False):
                user.pop("hashed_password", None)
                return user
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Fall back to API key (ESP32 device auth)
    if x_api_key:
        if x_api_key == settings.API_SECRET_KEY:
            return None  # Valid API key, no user context
        raise HTTPException(status_code=401, detail="Invalid API key")

    raise HTTPException(status_code=401, detail="Authentication required: provide Bearer token or X-API-Key")


async def verify_api_key(x_api_key: str = Header(...)):
    """Legacy API key check — kept for backward compatibility on ESP32 routes."""
    if x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized request: Invalid API key")


async def require_authenticated_user(
    user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Require a JWT-authenticated user (not API key). Used for user-management endpoints."""
    if user is None:
        raise HTTPException(status_code=403, detail="This endpoint requires user authentication (JWT), not API key")
    return user


async def require_admin(
    user: dict = Depends(require_authenticated_user),
) -> dict:
    """Require an admin-role or super_admin-role user."""
    if user.get("role") not in ("admin", "super_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def require_role(allowed_roles: list[str]):
    """Factory: create a dependency that requires one of the given roles."""
    async def _check(user: Optional[dict] = Depends(get_current_user)) -> Optional[dict]:
        if user is None:
            return None  # API key auth — pass through (ESP32)
        if user.get("role") not in allowed_roles:
            raise HTTPException(status_code=403, detail=f"Access denied. Required role: {', '.join(allowed_roles)}")
        return user
    return _check


def require_farm_access(user: Optional[dict], farm_id: str) -> None:
    """
    Check that a JWT-authenticated user has access to the given farm.
    API key users (ESP32) bypass this check.
    Admins with empty farm_ids have access to all farms.
    """
    if user is None:
        return  # API key auth — no farm restriction
    if user.get("role") == "super_admin":
        return  # Super admin — access to all farms
    if user.get("role") == "admin" and not user.get("farm_ids"):
        return  # Admin with no farm restriction — access to all farms
    if farm_id not in user.get("farm_ids", []):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: you do not have access to farm '{farm_id}'",
        )
