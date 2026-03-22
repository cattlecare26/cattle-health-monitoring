"""
User management page.
Super Admin: can create admins and users, see all.
Admin: can only create users scoped to their farms.
"""

import streamlit as st
import re
import pandas as pd
from utils.translations import t
from utils.auth import get_lang, get_token, get_user, is_super_admin, get_effective_role
from utils.theme import get_palette
from services.api_client import api_get_users, api_register, api_update_user, api_deactivate_user
from components.navbar import render_navbar


def render():
    lang = get_lang()
    token = get_token()
    current_user = get_user()
    eff_role = get_effective_role()
    p = get_palette()

    render_navbar()

    title = t("user_management", lang) if is_super_admin() else t("create_user", lang)
    st.markdown(f"## 👥 {title}")
    st.markdown("---")

    if is_super_admin():
        tab_list, tab_add, tab_edit = st.tabs([
            f"All Users", f"Create User", f"Edit User"
        ])
        with tab_list:
            _render_user_list(lang, token, p)
        with tab_add:
            _render_add_user(lang, token, current_user, p, allow_admin=True)
        with tab_edit:
            _render_edit_user(lang, token, current_user, p)
    else:
        tab_list, tab_add = st.tabs([
            f"{t('total_users', lang)}", f"{t('create_user', lang)}"
        ])
        with tab_list:
            _render_user_list(lang, token, p, scope_farms=current_user.get("farm_ids", []))
        with tab_add:
            _render_add_user(lang, token, current_user, p, allow_admin=False)


def _render_user_list(lang: str, token: str, p: dict, scope_farms: list = None):
    users = api_get_users(token) or []

    if scope_farms:
        users = [u for u in users if u.get("role") == "user" and
                 any(f in scope_farms for f in u.get("farm_ids", []))]

    if not users:
        st.info(t("no_data", lang))
        return

    table = []
    for u in users:
        u_role = u.get("role", "user")
        if u_role == "super_admin" or (u_role == "admin" and not u.get("farm_ids")):
            role_label = t("super_admin", lang)
        elif u_role == "admin":
            role_label = t("admin", lang)
        else:
            role_label = t("user", lang)
        table.append({
            t("username", lang): u.get("username", ""),
            t("full_name", lang): u.get("full_name", ""),
            t("email", lang): u.get("email", ""),
            t("role", lang): role_label,
            t("assigned_farms", lang): ", ".join(u.get("farm_ids", [])) or "All",
            t("status", lang): "Active" if u.get("is_active") else "Inactive",
        })

    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    st.caption(f"Total: {len(users)}")


def _render_add_user(lang: str, token: str, current_user: dict, p: dict, allow_admin: bool):
    st.markdown(
        f"""<div style="background: {p['status_info_bg']}; border: 1px solid {p['border']};
            border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; color: {p['text_secondary']};">
            ℹ️ Fill in the details below to create a new {'admin or user' if allow_admin else 'user'} account.
            {'Leave Farm IDs empty for Super Admin access.' if allow_admin else ''}
        </div>""",
        unsafe_allow_html=True,
    )

    with st.form("add_user_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input(
                f"👤 {t('full_name', lang)}", placeholder="e.g. Dr. Ramesh Kumar",
                help="Full display name of the user"
            )
            username = st.text_input(
                f"🆔 {t('username', lang)}", placeholder="e.g. ramesh_k",
                help="Alphanumeric and underscores only, min 3 chars"
            )
            email = st.text_input(
                f"📧 {t('email', lang)}", placeholder="e.g. ramesh@farm.com",
                help="Must be a valid unique email address"
            )
        with col2:
            password = st.text_input(
                f"🔒 {t('password', lang)}", type="password", placeholder="Min 8 characters",
                help="Minimum 8 characters required"
            )
            confirm_password = st.text_input(
                f"🔒 Confirm Password", type="password", placeholder="Re-enter password",
                help="Must match the password above"
            )
            if allow_admin:
                role = st.selectbox(
                    f"🏷️ {t('role', lang)}", ["user", "admin"],
                    format_func=lambda x: f"⚕️ {t('admin', lang)} (Veterinary Doctor)" if x == "admin" else f"👤 {t('user', lang)} (Farmer)",
                    help="Admin = Vet Doctor, User = Farmer"
                )
            else:
                role = "user"
                st.text_input(f"🏷️ {t('role', lang)}", value=f"👤 {t('user', lang)} (Farmer)", disabled=True)

            if allow_admin:
                farm_ids_input = st.text_input(
                    f"🏠 {t('assigned_farms', lang)}",
                    placeholder="Farm-A, Farm-B (comma-separated)",
                    help="Leave empty for Super Admin (all-farm access)"
                )
            else:
                my_farms = current_user.get("farm_ids", [])
                farm_ids_input = st.multiselect(
                    f"🏠 {t('assigned_farms', lang)}", options=my_farms, default=my_farms,
                    help="Users can only be assigned to your farms"
                )

        submitted = st.form_submit_button(
            f"➕ {t('create_user', lang)}", use_container_width=True, type="primary"
        )

    if submitted:
        errors = []
        if not full_name or len(full_name.strip()) < 1:
            errors.append("👤 Full name is required")
        if not username or len(username.strip()) < 3:
            errors.append("🆔 Username must be at least 3 characters")
        if not re.match(r"^[a-zA-Z0-9_]+$", username.strip()) if username else True:
            errors.append("🆔 Username must contain only letters, numbers, and underscores")
        if not email or "@" not in email or "." not in email.split("@")[-1]:
            errors.append("📧 A valid email address is required")
        if not password or len(password) < 8:
            errors.append("🔒 Password must be at least 8 characters")
        if password != confirm_password:
            errors.append("🔒 Passwords do not match")

        if errors:
            for err in errors:
                st.error(err)
            return

        if allow_admin:
            farms = [f.strip() for f in farm_ids_input.split(",") if f.strip()] if isinstance(farm_ids_input, str) else farm_ids_input
        else:
            farms = farm_ids_input if isinstance(farm_ids_input, list) else []

        with st.spinner(t("loading", lang)):
            result, error_msg = api_register(
                token, username.strip().lower(), email.strip().lower(),
                password, full_name.strip(), role, farms,
            )

        if result:
            st.success(f"✅ User **{result.get('full_name', username)}** (`{result.get('username')}`) created successfully as **{result.get('role').upper()}**!")
            st.balloons()
            import time
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"❌ Failed to create user: {error_msg}")


def _render_edit_user(lang: str, token: str, current_user: dict, p: dict):
    users = api_get_users(token) or []

    if not users:
        st.info(t("no_data", lang))
        return

    selected_username = st.selectbox(
        "Select user to edit:",
        options=[u["username"] for u in users],
        format_func=lambda x: f"{x} — {next((u['full_name'] for u in users if u['username'] == x), '')}",
    )

    user = next((u for u in users if u["username"] == selected_username), None)
    if not user:
        return

    # Show current info
    st.markdown(
        f"""<div style="background: {p['card_bg']}; border: 1px solid {p['card_border']};
            border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem;
            box-shadow: 0 1px 3px {p['card_shadow']}; color: {p['text']};">
            <strong>{user.get('full_name', '')}</strong>
            <span style="color: {p['text_secondary']};"> &mdash; {user.get('email', '')}</span>
            <br><span style="color: {p['text_muted']}; font-size: 0.85rem;">
                Role: {user.get('role', '').upper()} &middot;
                Farms: {', '.join(user.get('farm_ids', [])) or 'All'} &middot;
                {'Active' if user.get('is_active') else 'Inactive'}
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.form("edit_user_form"):
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input(t("full_name", lang), value=user.get("full_name", ""))
            role_options = ["user", "admin", "super_admin"]
            current_role = user.get("role", "user")
            role_index = role_options.index(current_role) if current_role in role_options else 0
            role = st.selectbox(
                t("role", lang), role_options,
                index=role_index,
                format_func=lambda x: (
                    f"{t('super_admin', lang)}" if x == "super_admin"
                    else f"{t('admin', lang)} (Vet Doctor)" if x == "admin"
                    else f"{t('user', lang)} (Farmer)"
                ),
            )
        with col2:
            farm_ids = st.text_input(
                t("assigned_farms", lang),
                value=", ".join(user.get("farm_ids", [])),
                help="Comma-separated. Leave empty for Super Admin."
            )
            is_active = st.checkbox(t("active", lang), value=user.get("is_active", True))

        submitted = st.form_submit_button(f"{t('save', lang)}", use_container_width=True, type="primary")

    if submitted:
        farms = [f.strip() for f in farm_ids.split(",") if f.strip()] if farm_ids else []
        update_data = {"full_name": full_name.strip(), "role": role, "farm_ids": farms, "is_active": is_active}

        with st.spinner(t("loading", lang)):
            result = api_update_user(token, selected_username, update_data)

        if result:
            st.success(f"User **{selected_username}** updated successfully!")
            st.rerun()
        else:
            st.error("Failed to update user. Please check the inputs and try again.")

    # Deactivate
    if selected_username != current_user.get("username"):
        st.markdown("---")
        st.warning(t("confirm_deactivate", lang))
        if st.button(f"{t('deactivate_user', lang)}", type="secondary"):
            with st.spinner(t("loading", lang)):
                result = api_deactivate_user(token, selected_username)
            if result:
                st.success(f"User '{selected_username}' deactivated.")
                st.rerun()
            else:
                st.error("Failed to deactivate user.")
