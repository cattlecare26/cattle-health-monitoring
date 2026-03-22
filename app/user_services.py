"""
User service layer: password hashing, JWT tokens, and user CRUD operations.
"""

from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import jwt, JWTError

from app.config import settings
from app.database import get_db
from app.logger import log_event


# ── Password Hashing (bcrypt) ──


def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


# ── JWT Token Management ──


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token with an expiration time."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.JWT_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token. Returns payload or None if invalid."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


# ── User CRUD Operations ──

_USER_PROJECTION = {"_id": 0, "hashed_password": 0}
_USER_FULL_PROJECTION = {"_id": 0}


async def get_user_by_username(username: str) -> Optional[dict]:
    """Fetch a user by username (includes hashed_password for auth)."""
    db = get_db()
    return await db.users.find_one({"username": username}, _USER_FULL_PROJECTION)


async def get_user_by_email(email: str) -> Optional[dict]:
    """Fetch a user by email."""
    db = get_db()
    return await db.users.find_one({"email": email}, _USER_FULL_PROJECTION)


async def get_user_count() -> int:
    """Get the total number of users in the system."""
    db = get_db()
    return await db.users.count_documents({})


async def create_user(
    username: str,
    email: str,
    password: str,
    full_name: str,
    role: str = "user",
    farm_ids: Optional[list[str]] = None,
) -> dict:
    """
    Create a new user with a bcrypt-hashed password.
    First user in the system is automatically assigned the admin role.
    Raises ValueError if username or email already exists.
    """
    db = get_db()

    # Check for existing username or email
    if await db.users.find_one({"username": username}):
        raise ValueError(f"Username '{username}' already exists")
    if await db.users.find_one({"email": email}):
        raise ValueError(f"Email '{email}' is already registered")

    # First user becomes admin automatically
    user_count = await get_user_count()
    if user_count == 0:
        role = "admin"

    now = datetime.utcnow()
    user_doc = {
        "username": username,
        "email": email,
        "hashed_password": hash_password(password),
        "full_name": full_name,
        "role": role,
        "farm_ids": farm_ids or [],
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }

    await db.users.insert_one(user_doc)

    await log_event(
        service="auth_api",
        level="INFO",
        action="create_user",
        collection="users",
        message=f"New user registered: {username} (role: {role})",
    )

    # Return without password
    user_doc.pop("hashed_password")
    user_doc.pop("_id", None)
    return user_doc


async def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Authenticate a user by username and password.
    Returns user dict (without password) if valid, None otherwise.
    """
    user = await get_user_by_username(username)
    if not user:
        return None
    if not user.get("is_active", False):
        return None
    if not verify_password(password, user["hashed_password"]):
        return None

    await log_event(
        service="auth_api",
        level="INFO",
        action="user_login",
        collection="users",
        message=f"User logged in: {username}",
    )

    # Return without password
    user.pop("hashed_password", None)
    return user


async def get_all_users() -> list[dict]:
    """Fetch all users (without passwords)."""
    db = get_db()
    cursor = db.users.find({}, _USER_PROJECTION).sort("created_at", 1)
    return await cursor.to_list(length=1000)


async def update_user(username: str, update_fields: dict) -> Optional[dict]:
    """
    Update user fields. Returns updated user or None if not found.
    Only updates provided non-None fields.
    """
    db = get_db()
    fields = {k: v for k, v in update_fields.items() if v is not None}
    if not fields:
        return await db.users.find_one({"username": username}, _USER_PROJECTION)

    fields["updated_at"] = datetime.utcnow()
    result = await db.users.update_one({"username": username}, {"$set": fields})
    if result.matched_count == 0:
        return None

    await log_event(
        service="auth_api",
        level="INFO",
        action="update_user",
        collection="users",
        message=f"User updated: {username}, fields: {list(fields.keys())}",
    )
    return await db.users.find_one({"username": username}, _USER_PROJECTION)


async def get_admins_by_farm_id(farm_id: str) -> list[dict]:
    """Fetch all active admin users assigned to a given farm."""
    db = get_db()
    cursor = db.users.find(
        {"role": "admin", "is_active": True, "farm_ids": farm_id},
        _USER_PROJECTION,
    )
    return await cursor.to_list(length=100)


async def deactivate_user(username: str) -> bool:
    """Deactivate a user account. Returns True if found and deactivated."""
    db = get_db()
    result = await db.users.update_one(
        {"username": username},
        {"$set": {"is_active": False, "updated_at": datetime.utcnow()}},
    )
    if result.matched_count == 0:
        return False

    await log_event(
        service="auth_api",
        level="WARNING",
        action="deactivate_user",
        collection="users",
        message=f"User deactivated: {username}",
    )
    return True
