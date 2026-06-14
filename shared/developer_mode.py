"""
Hidden developer / pipeline diagnostics mode.

Researcher mode (default): show analysis tables only.
Developer mode: restore audit panels, file paths, and intermediate pipeline output.

Re-enable developer mode via:
- Sidebar · expander → Developer mode toggle
- Environment variable LAZY_EYE_DEVELOPER_MODE=1
"""

from __future__ import annotations

import os

import streamlit as st

DEVELOPER_MODE_SESSION_KEY = "lazy_eye_developer_mode"
DEVELOPER_MODE_ENV_VAR = "LAZY_EYE_DEVELOPER_MODE"


def _env_developer_mode_enabled() -> bool:
    value = os.environ.get(DEVELOPER_MODE_ENV_VAR, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def developer_mode_enabled() -> bool:
    if _env_developer_mode_enabled():
        return True
    return bool(st.session_state.get(DEVELOPER_MODE_SESSION_KEY, False))


def render_developer_mode_controls() -> None:
    """Obscure sidebar toggle; also honored when LAZY_EYE_DEVELOPER_MODE is set."""
    if _env_developer_mode_enabled():
        st.sidebar.caption("Developer mode (env)")
        return

    with st.sidebar.expander("·", expanded=False):
        st.caption("Pipeline diagnostics and audit panels")
        enabled = st.checkbox(
            "Developer mode",
            value=bool(st.session_state.get(DEVELOPER_MODE_SESSION_KEY, False)),
            key="developer_mode_toggle",
        )
        st.session_state[DEVELOPER_MODE_SESSION_KEY] = enabled
