"""Temporary Streamlit UI for Corsano segmentation debugging."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.analysis.corsano_debug_report import (
    CORSANO_DEBUG_PARTICIPANT_ID,
    build_corsano_debug_report,
)
from backend.paths import participant_dir
from domain.participant_id import normalize_participant_id


def render_corsano_debug_section(participant_id: str) -> None:
    if normalize_participant_id(participant_id) != CORSANO_DEBUG_PARTICIPANT_ID:
        return

    with st.expander("Corsano Debug – Participant 11", expanded=True):
        st.caption(
            "Temporary debug output for Corsano segmentation only. "
            "This is not saved as analysis output."
        )
        report = build_corsano_debug_report(participant_dir(participant_id))

        resting_rows = report["resting_state_segments"]
        resting_summary = pd.DataFrame(
            [
                {
                    "resting_state_found": report["resting_state_found"],
                    "resting_task_start_utc": report["resting_state_task_start_utc"],
                    "resting_task_end_utc": report["resting_state_task_end_utc"],
                    "resting_activity_rows": resting_rows["activity_rows"],
                    "resting_hrv_rows": resting_rows["hrv_rows"],
                    "resting_acc_rows": resting_rows["acc_rows"],
                }
            ]
        )
        st.markdown("###### Resting state")
        st.dataframe(resting_summary, use_container_width=True, hide_index=True)

        for task_report in report["tasks"]:
            st.markdown(f"###### {task_report['task_name']}")
            task_window = pd.DataFrame(
                [
                    {
                        "task_name": task_report["task_name"],
                        "TASK_START_UTC": task_report["task_start_utc"],
                        "TASK_END_UTC": task_report["task_end_utc"],
                    }
                ]
            )
            st.dataframe(task_window, use_container_width=True, hide_index=True)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**activity.xlsx**")
                st.dataframe(task_report["activity"], use_container_width=True, hide_index=True)
            with col_b:
                st.markdown("**heart_rate_variability.xlsx**")
                st.dataframe(task_report["hrv"], use_container_width=True, hide_index=True)
            with col_c:
                st.markdown("**acc.xlsx**")
                st.dataframe(task_report["acc"], use_container_width=True, hide_index=True)

            table2_debug = task_report.get("table2_calculation") or {}
            if table2_debug:
                st.markdown("**Table 2 calculation debug (BPM / Respiration)**")
                calc_col_a, calc_col_b = st.columns(2)
                with calc_col_a:
                    st.markdown("*BPM*")
                    st.dataframe(table2_debug.get("bpm"), use_container_width=True, hide_index=True)
                with calc_col_b:
                    st.markdown("*Respiration*")
                    st.dataframe(table2_debug.get("respiration"), use_container_width=True, hide_index=True)
