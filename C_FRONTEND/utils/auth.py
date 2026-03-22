"""
Authentication utilities for Streamlit session state management.
Supports 3-tier role hierarchy: Super Admin → Admin → User.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_LANG = os.getenv("DEFAULT_LANG", "en")


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "authenticated": False,
        "token": None,
        "user": None,
        "role": None,
        "effective_role": None,  # super_admin | admin | user
        "lang": _DEFAULT_LANG,
        "theme": "light",
        "current_page": "dashboard",
        "selected_cattle_cid": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _detect_effective_role(user: dict) -> str:
    """
    Determine the effective role from user data.
    Super Admin = explicit super_admin role OR admin with empty farm_ids.
    Admin = admin with specific farm_ids (scoped).
    User = regular user role.
    """
    role = user.get("role", "user")
    if role == "super_admin":
        return "super_admin"
    if role == "admin":
        farm_ids = user.get("farm_ids", [])
        if not farm_ids:
            return "super_admin"
        return "admin"
    return "user"


def login_user(token: str, user: dict):
    """Store authentication data in session state."""
    st.session_state.authenticated = True
    st.session_state.token = token
    st.session_state.user = user
    st.session_state.role = user.get("role", "user")
    st.session_state.effective_role = _detect_effective_role(user)


def logout_user():
    """Clear authentication data from session state."""
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.effective_role = None
    st.session_state.current_page = "login"


def is_authenticated() -> bool:
    return st.session_state.get("authenticated", False)


def is_super_admin() -> bool:
    return st.session_state.get("effective_role") == "super_admin"


def is_admin() -> bool:
    return st.session_state.get("effective_role") in ("admin", "super_admin")


def is_user() -> bool:
    return st.session_state.get("effective_role") == "user"


def get_effective_role() -> str:
    return st.session_state.get("effective_role", "user")


def get_token() -> str:
    return st.session_state.get("token", "")


def get_user() -> dict:
    return st.session_state.get("user") or {}


def get_lang() -> str:
    return st.session_state.get("lang", "en")


def get_theme() -> str:
    return st.session_state.get("theme", "light")


def navigate_to(page: str, **kwargs):
    """Navigate to a different page, with optional extra state."""
    st.session_state.current_page = page
    for key, value in kwargs.items():
        st.session_state[key] = value
