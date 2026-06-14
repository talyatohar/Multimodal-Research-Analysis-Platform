"""Participants database — search, filter, hand-off to Analyse (UI shell)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.participant_service import delete_participant
from domain.cohorts import GROUP_FILTER_ALL, PARTICIPANT_GROUPS
from domain.participant_id import normalize_participant_id
from shared.cohort_state import reload_cohort_from_disk, summarize_tasks
from shared.session_keys import SELECTED_PARTICIPANT_ID
from shared.ui import apply_theme, configure_page, page_header, sidebar_brand

configure_page("Participants database")
apply_theme()
sidebar_brand()

MSG_HANDOFF = "Participant selected. Go to Analyse Participant from the sidebar."


def safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _select_for_analyse(participant_id: str) -> None:
    st.session_state[SELECTED_PARTICIPANT_ID] = normalize_participant_id(participant_id)
    st.info(MSG_HANDOFF, icon="ℹ️")


page_header(
    "Cohort registry",
    "Participants database",
    "Search by ID, filter by clinical group, inspect tasks. Choose a participant below, then open "
    "Analyse Participant from the sidebar — the ID will be pre-selected there.",
)

cohort = reload_cohort_from_disk()
q = st.text_input("Search by ID", placeholder="Exact or partial participant ID")
group = st.selectbox(
    "Filter by group",
    options=[GROUP_FILTER_ALL, *PARTICIPANT_GROUPS],
)

rows = []
for r in cohort:
    pid = r.get("participant_id") or ""
    if q.strip() and q.strip().lower() not in pid.lower():
        continue
    g = r.get("participant_group") or ""
    if group != GROUP_FILTER_ALL and g != group:
        continue
    rows.append(
        {
            "Participant ID": pid,
            "Name": r.get("participant_name") or "—",
            "Group": g or "—",
            "Age": str(r.get("participant_age")) if r.get("participant_age") is not None else "—",
            "Existing task": summarize_tasks(r),
        }
    )

df = pd.DataFrame(rows)
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Participant ID": st.column_config.TextColumn("Participant ID"),
        "Name": st.column_config.TextColumn("Name"),
        "Group": st.column_config.TextColumn("Group"),
        "Age": st.column_config.TextColumn("Age"),
        "Existing task": st.column_config.TextColumn("Existing task"),
    },
)

st.divider()
st.markdown("##### Select participant for Analyse")
ids = [r["Participant ID"] for r in rows if r["Participant ID"]]
if not ids:
    st.caption("No rows match the current filters — adjust search or register participants via Upload.")
    choice = None
else:
    st.caption("Click a participant ID, or pick from the list and use **Open Analyse participant**.")
    ncols = min(4, len(ids))
    cols = st.columns(ncols)
    for idx, rid in enumerate(ids):
        with cols[idx % ncols]:
            if st.button(rid, key=f"db_pick_pid_{rid}", use_container_width=True):
                _select_for_analyse(rid)

    choice = st.selectbox("Participant ID (dropdown)", options=ids, key="db_participant_dropdown")

    if st.button("Open Analyse participant", type="primary", disabled=choice is None):
        _select_for_analyse(choice)

st.caption(
    "Default view lists every participant when no search/filter is applied (loaded from participants_table.xlsx)."
)

st.divider()
st.markdown("##### Delete participant")
st.warning(
    "This permanently deletes the selected participant's registry row in `participants_table.xlsx` "
    "and the entire `database/participants/participant_<ID>/` folder. "
    "No other files are removed. This action cannot be undone.",
    icon="⚠️",
)

all_participant_ids = [r.get("participant_id") for r in cohort if r.get("participant_id")]
if not all_participant_ids:
    st.caption("No participants are registered — nothing to delete.")
else:
    delete_choice = st.selectbox(
        "Participant ID to delete",
        options=all_participant_ids,
        key="db_delete_participant",
    )
    delete_confirmed = st.checkbox(
        "I understand this will permanently delete the participant registry row and folder.",
        key="db_delete_confirm",
    )
    if st.button(
        "Delete participant",
        type="secondary",
        disabled=not delete_confirmed,
        key="db_delete_button",
    ):
        result = delete_participant(delete_choice)
        if result.ok:
            if st.session_state.get(SELECTED_PARTICIPANT_ID) == delete_choice:
                st.session_state.pop(SELECTED_PARTICIPANT_ID, None)
            reload_cohort_from_disk()
            st.success(result.message)
            safe_rerun()
        else:
            st.error(result.message)
