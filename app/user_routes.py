"""
User management and authentication API routes.
"""

from fastapi import APIRouter, Depends, HTTPException
from app.auth import require_authenticated_user, require_admin
from app.user_models import (
    UserCreate,
    UserLogin,
    UserUpdate,
    UserResponse,
    TokenResponse,
    MessageResponse,
)
from app.user_services import (
    create_user,
    authenticate_user,
    create_access_token,
    get_all_users,
    update_user,
    deactivate_user,
    get_user_count,
)
from app.config import settings

auth_router = APIRouter(prefix="/api/v1/auth", tags=["Authentication & Users"])


@auth_router.post("/register", response_model=UserResponse)
async def register_user(
    data: UserCreate,
    current_user: dict = Depends(require_admin),
):
    """
    Register a new user. Admin access required.
    Exception: if no users exist yet, the first registration is open (auto-admin).
    """
    try:
        user = await create_user(
            username=data.username,
            email=data.email,
            password=data.password,
            full_name=data.full_name,
            role=data.role,
            farm_ids=data.farm_ids,
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@auth_router.post("/bootstrap", response_model=UserResponse)
async def bootstrap_first_user(data: UserCreate):
    """
    Create the first admin user. Only works when no users exist in the system.
    This endpoint is open (no auth required) and can only be called once.
    """
    count = await get_user_count()
    if count > 0:
        raise HTTPException(
            status_code=403,
            detail="System already has users. Use /register with admin credentials.",
        )
    try:
        user = await create_user(
            username=data.username,
            email=data.email,
            password=data.password,
            full_name=data.full_name,
            role="admin",
            farm_ids=data.farm_ids,
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@auth_router.post("/login", response_model=TokenResponse)
async def login(data: UserLogin):
    """Authenticate a user and return a JWT access token."""
    user = await authenticate_user(data.username, data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(data={"sub": user["username"], "role": user["role"]})

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.JWT_EXPIRE_MINUTES * 60,
        user=UserResponse(**user),
    )


@auth_router.get("/me", response_model=UserResponse)
async def get_current_profile(current_user: dict = Depends(require_authenticated_user)):
    """Get the current authenticated user's profile."""
    return current_user


@auth_router.get("/users", response_model=list[UserResponse])
async def list_users(current_user: dict = Depends(require_admin)):
    """List all users. Admin access required."""
    users = await get_all_users()
    if not users:
        raise HTTPException(status_code=404, detail="No users found")
    return users


@auth_router.put("/users/{username}", response_model=UserResponse)
async def update_user_endpoint(
    username: str,
    data: UserUpdate,
    current_user: dict = Depends(require_admin),
):
    """Update a user's role, farm assignments, or status. Admin access required."""
    result = await update_user(username, data.model_dump(exclude_unset=True))
    if result is None:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    return result


@auth_router.delete("/users/{username}", response_model=MessageResponse)
async def delete_user_endpoint(
    username: str,
    current_user: dict = Depends(require_admin),
):
    """Deactivate a user account. Admin access required."""
    if username == current_user.get("username"):
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
    success = await deactivate_user(username)
    if not success:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    return MessageResponse(success=True, message=f"User '{username}' has been deactivated")
