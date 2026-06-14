"""Upload participant — persist to database/ per project specification."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from domain.feature_catalog import EEG_TASK_FEATURES
from backend.participant_service import ParticipantUpload, TaskUpload, register_participant
from domain.cohorts import PARTICIPANT_GROUPS
from domain.participant_id import normalize_participant_id
from domain.storage_layout import example_tree
from domain.tasks import TASK_NAME_PLACEHOLDER, TASK_NAMES
from shared.cohort_state import reload_cohort_from_disk
from shared.developer_mode import developer_mode_enabled
from shared.ui import apply_theme, configure_page, page_header, sidebar_brand


def _rerun() -> None:
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def _render_eeg_channel_list(
    channels: list[dict],
    channel_count: int | None = None,
    *,
    source: str = "eeg_metadata.json",
) -> None:
    """Show every channel in header order with total count."""
    if not channels:
        st.caption("Channel list not available.")
        return

    listed_count = len(channels)
    total = channel_count if channel_count is not None else listed_count
    st.markdown(f"**Complete EEG channel list** — {total} channels (order from {source})")
    if channel_count is not None and channel_count != listed_count:
        st.warning(
            f"Header channel_count is {channel_count} but the channel list contains {listed_count} entries."
        )

    rows = [
        {
            "order": channel.get("order", index),
            "name": channel.get("name"),
            "type": channel.get("type"),
            "unit": channel.get("unit"),
        }
        for index, channel in enumerate(channels, start=1)
    ]
    table_height = min(max(listed_count * 35 + 38, 120), 900)
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        height=table_height,
    )
    st.markdown("**Channel names (plain text)**")
    st.text(
        "\n".join(
            f"{row['order']}. {row['name']}"
            for row in rows
            if row.get("name")
        )
    )


configure_page("Upload participant")
apply_theme()
sidebar_brand()

page_header(
    "Registry & ingest",
    "Upload participant",
    "All fields optional except participant ID. Metadata appends to `participants_table.xlsx`; "
    "raw exports are copied once and never overwritten.",
)

st.session_state.setdefault("upload_task_blocks", [{"task_name": TASK_NAME_PLACEHOLDER}])


def add_task_block() -> None:
    st.session_state.upload_task_blocks.append({"task_name": TASK_NAME_PLACEHOLDER})


def remove_task_block(idx: int) -> None:
    blocks = st.session_state.upload_task_blocks
    if len(blocks) <= 1:
        return
    st.session_state.upload_task_blocks = [b for i, b in enumerate(blocks) if i != idx]


st.markdown("##### New participant")
pid = st.text_input("Participant ID")
pname = st.text_input("Participant name")
page = st.number_input("Participant age", min_value=0, max_value=120, value=0, step=1, help="0 = not provided")
pgroup = st.selectbox("Participant group", options=["— Select group —"] + PARTICIPANT_GROUPS, index=0)
notes = st.text_area("Notes")

st.divider()
st.markdown("##### Participant-level exports")
c_eye, c_acc = st.columns(2)
with c_eye:
    eye_files = st.file_uploader(
        "EyeTracking.xlsx (Tobii Pro Lab export — multiple files allowed)",
        type=["xlsx"],
        accept_multiple_files=True,
        help="Millisecond exports; single workbook per file. Tobii column bundle per project book.",
    )
    if developer_mode_enabled():
        with st.expander("Tobii export checklist (reference)"):
            st.markdown(
                "- Include timing columns (`Recording timestamp [ms]`, UTC start fields).\n"
                "- Gaze events, validity, pupil, stimulus metadata as documented.\n"
                "- Original uploads are never overwritten by preprocessing."
            )
with c_acc:
    acc_file = st.file_uploader("acc.xlsx", type=["xlsx"], accept_multiple_files=False)
    activity_file = st.file_uploader("activity.xlsx", type=["xlsx"], accept_multiple_files=False)
    hrv_file = st.file_uploader("heart_rate_variability.xlsx", type=["xlsx"], accept_multiple_files=False)

st.divider()
st.markdown("##### Task blocks")
task_uploads: list[TaskUpload] = []
for i, block in enumerate(st.session_state.upload_task_blocks):
    with st.expander(f"Task {i + 1}", expanded=(i == len(st.session_state.upload_task_blocks) - 1)):
        opts = [TASK_NAME_PLACEHOLDER] + list(TASK_NAMES)
        tix = opts.index(block["task_name"]) if block["task_name"] in opts else 0
        name = st.selectbox("Task", options=opts, index=tix, key=f"task_name_{i}")
        st.session_state.upload_task_blocks[i]["task_name"] = name
        f1, f2 = st.columns(2)
        with f1:
            ahdr_f = st.file_uploader("Task.ahdr (legacy BrainVision)", key=f"ahdr_{i}")
            eeg_f = st.file_uploader("Task.eeg (legacy BrainVision)", key=f"eeg_{i}")
        with f2:
            amrk_f = st.file_uploader("Task.amrk (legacy BrainVision)", key=f"amrk_{i}")
            eprime_f = st.file_uploader("Eprime.txt", type=["txt"], key=f"eprime_{i}")
        bvrf_files = st.file_uploader(
            "BrainVision Recorder BVRF files (.bvrh, .bvrd, .bvrm, .bvri)",
            type=["bvrh", "bvrd", "bvrm", "bvri"],
            accept_multiple_files=True,
            key=f"bvrf_{i}",
            help="Upload one or more files from a BVRF set. Files with the same basename are grouped as one recording. Saved unchanged under EEG_raw/.",
        )
        comp = st.number_input(
            "Reading comprehension assessment score (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            key=f"comp_{i}",
            help="Stored as reading_comprehension_score.txt alongside task exports.",
        )
        st.caption("Manual reading end times are based on user review of the Tobii screen recording.")
        manual_time_mode = st.selectbox(
            "Manual reading end time interpretation",
            options=["Local Israel Time (UTC+3 / UTC+2 depending on date)", "UTC directly"],
            key=f"manual_time_mode_{i}",
            help="Use Local Israel Time for times read from Tobii screen recordings; it is converted to UTC after E-Prime provides the experiment date.",
        )
        manual_end = st.text_input(
            "Manual reading end time (optional)",
            key=f"manual_end_{i}",
            placeholder="HH:MM:SS.sss or full datetime",
            help="Leave empty to use only the E-Prime task end. Time-only values are combined with the experiment date.",
        )
        eeg_offset = st.number_input(
            "EEG clock offset (seconds, optional)",
            value=0.0,
            step=0.001,
            format="%.3f",
            key=f"eeg_offset_{i}",
            help=(
                "EEG_PC_time − E-Prime/Tobii_PC_time. "
                "Example: EEG shows 15:21:50.422 and E-Prime shows 15:18:42.863 → enter +187.559. "
                "Leave at 0 if clocks match."
            ),
        )
        task_uploads.append(
            TaskUpload(
                task_name=name,
                ahdr=ahdr_f,
                eeg=eeg_f,
                amrk=amrk_f,
                bvrf_files=list(bvrf_files) if bvrf_files else [],
                eprime=eprime_f,
                comprehension_score=comp if comp > 0 else None,
                manual_reading_end_time=manual_end.strip() or None,
                manual_time_interpretation=manual_time_mode,
                eeg_clock_offset_seconds=float(eeg_offset),
            )
        )
        if st.button("Remove task block", key=f"rm_task_{i}"):
            remove_task_block(i)
            _rerun()

b1, b2 = st.columns(2)
with b1:
    if st.button("ADD TASK", use_container_width=True):
        add_task_block()
        _rerun()
with b2:
    if st.button("ADD PARTICIPANT", type="primary", use_container_width=True):
        upload = ParticipantUpload(
            participant_id=normalize_participant_id(pid),
            participant_name=pname.strip() or None,
            participant_age=None if page == 0 else int(page),
            participant_group=None if pgroup == "— Select group —" else pgroup,
            notes=notes.strip() or None,
            eye_tracking_files=list(eye_files) if eye_files else [],
            acc=acc_file,
            activity=activity_file,
            hrv=hrv_file,
            tasks=task_uploads,
        )
        result = register_participant(upload)
        if not result.ok:
            for err in result.errors:
                st.error(err)
        else:
            reload_cohort_from_disk()
            st.session_state.upload_task_blocks = [{"task_name": TASK_NAME_PLACEHOLDER}]
            st.success(result.message)
            if not developer_mode_enabled():
                for sync in result.sync_results:
                    task_label = sync.get("task_name", "Task")
                    if sync.get("success"):
                        st.success(f"**{task_label}** — registered and processed.")
                    else:
                        st.error(f"**{task_label}** — E-Prime parsing failed: {sync.get('message')}")
                for warning in result.warnings:
                    st.warning(warning)
            if developer_mode_enabled():
                from shared.upload_developer_diagnostics import render_upload_developer_diagnostics
                render_upload_developer_diagnostics(result, _render_eeg_channel_list)

if developer_mode_enabled():
    with st.expander("Target on-disk tree (reference)"):
        st.code(example_tree(normalize_participant_id(pid) or "123456789"), language="text")

st.caption(
    "On save: `database/participants_table.xlsx` is updated, `database/participants/participant_<id>/` "
    "is created, and E-Prime sync + eye segmentation run when the required files are present."
)
