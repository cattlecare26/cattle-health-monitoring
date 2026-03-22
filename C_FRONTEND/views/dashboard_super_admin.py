"""
Super Admin dashboard - full system overview with Admin > User > Cattle mapping.
"""

import streamlit as st
import pandas as pd
from utils.translations import t
from utils.auth import get_lang, get_token, get_user, navigate_to
from utils.theme import get_palette, health_color, health_bg, ACCENT
from services.api_client import (
    api_get_cattle_list, api_get_all_latest, api_get_users,
    api_get_recent_alerts,
)
from components.navbar import render_navbar


def render():
    lang = get_lang()
    token = get_token()
    user = get_user()
    p = get_palette()

    render_navbar()

    st.markdown(f"### 🛡️ {t('system_overview', lang)} — {t('welcome', lang)}, {user.get('full_name', 'Admin')}!")

    cattle_list = api_get_cattle_list(token) or []
    users = api_get_users(token) or []
    latest_data = api_get_all_latest(token) or []
    recent_alerts = api_get_recent_alerts(token, limit=20) or []

    admins = [u for u in users if u.get("role") in ("admin", "super_admin")]
    farmers = [u for u in users if u.get("role") == "user"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_card(p, "", t("total_admins", lang), str(len(admins)), ACCENT["yellow"]),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(_card(p, "👨‍🌾", t("total_users", lang), str(len(farmers)), ACCENT["purple"]),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(_card(p, "", t("total_cattle", lang), str(len(cattle_list)), ACCENT["green"]),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(_card(p, "", t("active_alerts", lang), str(len(recent_alerts)),
                          ACCENT["red"] if recent_alerts else ACCENT["green"]),
                    unsafe_allow_html=True)

    st.markdown("---")

    st.subheader(f"🗺️ {t('mapping_view', lang)}: Admin > User > Cattle")

    farm_to_admin = {}
    for a in admins:
        for farm in a.get("farm_ids", []):
            farm_to_admin.setdefault(farm, []).append(a)

    farm_to_users = {}
    for u in farmers:
        for farm in u.get("farm_ids", []):
            farm_to_users.setdefault(farm, []).append(u)

    farm_to_cattle = {}
    for c in cattle_list:
        farm = c.get("farm_id", "unassigned")
        farm_to_cattle.setdefault(farm, []).append(c)

    all_farms = sorted(set(list(farm_to_admin.keys()) + list(farm_to_users.keys()) + list(farm_to_cattle.keys())))

    if not all_farms:
        st.info(t("no_data", lang))
    else:
        for farm in all_farms:
            farm_admins = farm_to_admin.get(farm, [])
            farm_users = farm_to_users.get(farm, [])
            farm_cattle = farm_to_cattle.get(farm, [])

            with st.expander(f"**{farm}** — {len(farm_admins)} admin(s), {len(farm_users)} user(s), {len(farm_cattle)} cattle", expanded=True):
                col_a, col_u, col_c = st.columns(3)

                with col_a:
                    st.markdown("**⚕️ Admins**")
                    for a in farm_admins:
                        status_dot = "🟢" if a.get("is_active") else "🔴"
                        st.markdown(f"{status_dot} {a.get('full_name', a.get('username', ''))}")

                with col_u:
                    st.markdown("**👤 Users**")
                    for u in farm_users:
                        status_dot = "🟢" if u.get("is_active") else "🔴"
                        st.markdown(f"{status_dot} {u.get('full_name', u.get('username', ''))}")

                with col_c:
                    st.markdown("**🐄 Cattle**")
                    for c in farm_cattle:
                        st.markdown(f"CID {c['cid']}: {c.get('name', '')}")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader(f"👥 All Users")
        if users:
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
                    t("role", lang): role_label,
                    t("assigned_farms", lang): ", ".join(u.get("farm_ids", [])) or "All",
                    t("status", lang): "🟢" if u.get("is_active") else "🔴",
                })
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    with col_right:
        st.subheader(f"🔔 {t('recent_alerts', lang)}")
        if recent_alerts:
            for alert in recent_alerts[:8]:
                level = alert.get("status", "warning")
                emoji = "🔴" if level == "critical" else "🟡"
                st.markdown(
                    f"""<div style="padding: 0.5rem 0.75rem; margin: 0.2rem 0;
                        background: {health_bg(level, p)};
                        border-left: 3px solid {health_color(level, p)};
                        border-radius: 6px; font-size: 0.85rem; color: {p['text']};">
                        {emoji} <strong>CID {alert.get('cid', '?')}</strong> — {level.upper()}
                        <br><span style="color: {p['text_muted']}; font-size: 0.75rem;">{str(alert.get('timestamp', ''))[:19]}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ No recent alerts.")

    st.markdown("---")
    st.subheader(f"🐄 All Cattle")
    if cattle_list:
        latest_by_cid = {item["cid"]: item for item in (latest_data or [])}
        table = []
        for c in cattle_list:
            cid = c["cid"]
            sensor = latest_by_cid.get(cid, {})
            temp = sensor.get("temperature", 0) if sensor else 0
            bpm = sensor.get("heart", {}).get("bpm", 0) if sensor else 0

            farm = c.get("farm_id", "")
            owners = [u.get("full_name", u.get("username", "")) for u in farmers if farm in u.get("farm_ids", [])]

            table.append({
                "CID": cid,
                t("cattle_name", lang): c.get("name", ""),
                t("farm_id", lang): farm,
                t("owned_by", lang): ", ".join(owners) or "Unassigned",
                "🌡️": f"{temp:.1f}" if temp > 0 else "-",
                "❤️": f"{bpm:.0f}" if bpm > 0 else "-",
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        # Cattle detail drill-down
        cid_options = [c["cid"] for c in cattle_list]
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            selected_cid = st.selectbox(
                "Select cattle to view details:",
                options=cid_options,
                format_func=lambda x: f"CID {x} — {next((c['name'] for c in cattle_list if c['cid'] == x), '')}",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📊 View Details", type="primary", use_container_width=True):
                navigate_to("cattle_detail", selected_cattle_cid=selected_cid)
                st.rerun()


def _card(p: dict, icon: str, label: str, value: str, color: str) -> str:
    return f"""
    <div style="background: {p['card_bg']}; border: 1px solid {p['card_border']};
                border-radius: 10px; padding: 1rem; text-align: center;
                box-shadow: 0 1px 3px {p['card_shadow']};">
        <div style="font-size: 1.3rem;">{icon}</div>
        <div style="font-size: 1.6rem; font-weight: 700; color: {color};">{value}</div>
        <div style="color: {p['text_secondary']}; font-size: 0.8rem;">{label}</div>
    </div>
    """
