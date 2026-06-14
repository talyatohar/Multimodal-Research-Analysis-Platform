"""Upload-page pipeline diagnostics (developer mode only)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from domain.feature_catalog import EEG_TASK_FEATURES


def render_upload_developer_diagnostics(result, render_eeg_channel_list) -> None:
    for sync in result.sync_results:
            task_label = sync.get("task_name", "Task")
            if sync.get("success"):
                st.success(f"**{task_label}** — {sync.get('message', 'E-Prime parsing succeeded.')}")
                st.markdown(
                    f"- **SessionStartDateTimeUtc (parsed):** `{sync.get('session_start_utc')}`\n"
                    f"- **TASK_START_UTC:** `{sync.get('task_start_utc')}`\n"
                    f"- **TASK_END_UTC:** `{sync.get('task_end_utc')}`\n"
                    f"- **Task duration:** `{sync.get('task_duration_ms')}` ms"
                )
                if sync.get("manual_task_end_utc"):
                    st.info(
                        "Manual reading end times are based on user review of the Tobii screen recording.\n\n"
                        f"- Entered/interpreted local time: `{sync.get('manual_task_end_local_time')}`\n"
                        f"- Converted UTC time: `{sync.get('manual_task_end_utc')}`"
                    )
                if sync.get("sync_json_path"):
                    st.caption(f"sync_window.json → `{sync['sync_json_path']}`")
                eye = sync.get("eye_segmentation") or {}
                if eye:
                    st.markdown("**Eye-tracking segmentation**")
                    st.markdown(
                        f"- **Files checked:** {len(eye.get('files_checked', []))}\n"
                        f"- **Files with overlap:** {', '.join(eye.get('files_with_overlap') or []) or '—'}\n"
                        f"- **Recordings checked:** {len(eye.get('recordings_checked', []))}\n"
                        f"- **Recordings with overlap:** {', '.join(eye.get('recordings_with_overlap') or []) or '—'}\n"
                        f"- **Rows before:** `{eye.get('rows_before', 0)}`\n"
                        f"- **Rows after:** `{eye.get('rows_after', 0)}`\n"
                        f"- **Task window:** `{eye.get('task_start_utc')}` → `{eye.get('task_end_utc')}`\n"
                        f"- **Buffers:** pre `{eye.get('pre_task_buffer_ms')}` ms, "
                        f"post `{eye.get('post_task_buffer_ms')}` ms\n"
                        f"- **Selected rows UTC:** `{eye.get('first_selected_row_utc')}` → "
                        f"`{eye.get('last_selected_row_utc')}`"
                    )
                    per_recording = eye.get("per_recording") or []
                    if per_recording:
                        st.markdown("**Per-recording time debug**")
                        for rec in per_recording:
                            title = (
                                f"{rec.get('file', '—')} / {rec.get('recording_name') or '—'} "
                                f"— overlap: {rec.get('overlap')}, rows: {rec.get('rows_after', 0)}"
                            )
                            with st.expander(title):
                                st.markdown(
                                    f"- **recording_name:** `{rec.get('recording_name')}`\n"
                                    f"- **recording_start_utc:** `{rec.get('recording_start_utc')}`\n"
                                    f"- **first_row_utc:** `{rec.get('first_row_utc')}`\n"
                                    f"- **last_row_utc:** `{rec.get('last_row_utc')}`\n"
                                    f"- **min Recording timestamp [ms]:** `{rec.get('min_recording_timestamp_ms')}`\n"
                                    f"- **max Recording timestamp [ms]:** `{rec.get('max_recording_timestamp_ms')}`\n"
                                    f"- **task_start_utc:** `{rec.get('task_start_utc')}`\n"
                                    f"- **task_end_utc:** `{rec.get('task_end_utc')}`\n"
                                    f"- **filter_start_utc:** `{rec.get('filter_start_utc')}`\n"
                                    f"- **filter_end_utc:** `{rec.get('filter_end_utc')}`\n"
                                    f"- **overlap:** `{rec.get('overlap')}`\n"
                                    f"- **rows_in_window:** `{rec.get('rows_in_window')}`\n"
                                    f"- **rows_after:** `{rec.get('rows_after')}`\n"
                                    f"- **first_selected_row_utc:** `{rec.get('first_selected_row_utc')}`\n"
                                    f"- **last_selected_row_utc:** `{rec.get('last_selected_row_utc')}`"
                                )
                                if rec.get("error"):
                                    st.caption(f"Error: {rec['error']}")
                    if eye.get("segment_path"):
                        st.caption(f"processed_eye_tracking_segment_eprime.xlsx → `{eye['segment_path']}`")
                    if eye.get("manual_segment_path"):
                        st.caption(f"processed_eye_tracking_segment_manual.xlsx → `{eye['manual_segment_path']}`")
                    if eye.get("report_path"):
                        st.caption(f"segmentation_report_eprime.json → `{eye['report_path']}`")
                    if eye.get("manual_report_path"):
                        st.caption(f"segmentation_report_manual.json → `{eye['manual_report_path']}`")
            else:
                st.error(f"**{task_label}** — E-Prime parsing failed: {sync.get('message')}")
                preview = sync.get("eprime_debug_preview")
                if preview:
                    st.markdown("**E-Prime file (first 30 lines):**")
                    st.code(preview, language=None)
    for inspection_entry in result.eeg_inspection_results:
        task_label = inspection_entry.get("task_name", "Task")
        metadata = inspection_entry.get("metadata") or {}
        st.markdown(f"**{task_label} — EEG inspection (Phase 1)**")
        if inspection_entry.get("metadata_json_path"):
            st.caption(f"eeg_metadata.json → `{inspection_entry['metadata_json_path']}`")
        for record in metadata.get("recordings", []):
            basename = record.get("basename", "—")
            files = record.get("files") or {}
            st.markdown(f"Recording set `{basename}`")
            st.markdown(
                f"- **Header file:** `{files.get('header', '—')}`\n"
                f"- **Data file:** `{files.get('data', '—')}` "
                f"({'present' if record.get('data_file_present') else 'missing'}, "
                f"{record.get('data_file_size_bytes', '—')} bytes)\n"
                f"- **Marker file:** `{files.get('marker', '—')}`\n"
                f"- **Impedance file:** `{files.get('impedance', '—')}`\n"
                f"- **Sampling rate (Hz):** `{record.get('sampling_rate_hz', '—')}`\n"
                f"- **Channel count:** `{record.get('channel_count', '—')}`\n"
                f"- **Markers count:** `{record.get('markers_count', '—')}`\n"
                f"- **Recording start:** `{record.get('recording_start_time', '—')}`\n"
                f"- **Recording end:** `{record.get('recording_end_time', '—')}`\n"
                f"- **Duration (s):** `{record.get('duration_seconds', '—')}`"
            )
            amplifier = record.get("amplifier")
            if amplifier:
                st.markdown(f"- **Amplifier:** `{amplifier}`")
            channels = record.get("channels") or []
            render_eeg_channel_list(
                channels,
                record.get("channel_count"),
                source="eeg_metadata.json / .bvrh",
            )
            impedance_qc = record.get("impedance_qc") or []
            if impedance_qc:
                st.markdown("**Impedance QC**")
                st.dataframe(pd.DataFrame(impedance_qc), use_container_width=True, hide_index=True)
            missing_required = record.get("missing_required_files") or []
            missing_optional = record.get("missing_optional_files") or []
            if missing_required:
                st.warning(
                    f"Recording set `{basename}` is missing required file(s): "
                    f"{', '.join(missing_required)}"
                )
            if missing_optional:
                st.info(
                    f"Recording set `{basename}` is missing optional file(s): "
                    f"{', '.join(missing_optional)}"
                )
    for segment_entry in result.eeg_segment_results:
        task_label = segment_entry.get("task_name", "Task")
        segment = segment_entry.get("metadata") or {}
        st.markdown(f"**{task_label} — EEG segmentation**")
        if segment_entry.get("segment_metadata_path"):
            st.caption(f"eeg_segment_metadata.json → `{segment_entry['segment_metadata_path']}`")
        if segment.get("ok"):
            st.markdown(
                f"- **Segment start (UTC):** `{segment.get('segment_start_utc', '—')}`\n"
                f"- **Segment end (UTC):** `{segment.get('segment_end_utc', '—')}`\n"
                f"- **Segment duration (s):** `{segment.get('segment_duration_seconds', '—')}`\n"
                f"- **Missing task start (s):** `{segment.get('missing_task_start_seconds', '—')}`\n"
                f"- **Missing task end (s):** `{segment.get('missing_task_end_seconds', '—')}`\n"
                f"- **EEG start sample index:** `{segment.get('eeg_start_sample_index', '—')}`\n"
                f"- **EEG end sample index:** `{segment.get('eeg_end_sample_index', '—')}`"
            )
        for warning in segment.get("warnings", []):
            if "Partial EEG-task overlap" in warning:
                st.warning(warning)
    for access_entry in result.eeg_raw_access_results:
        task_label = access_entry.get("task_name", "Task")
        info = access_entry.get("info") or {}
        st.markdown(f"**{task_label} — EEG raw signal access (Phase 2)**")
        if access_entry.get("segment_info_path"):
            st.caption(f"eeg_segment_info.json → `{access_entry['segment_info_path']}`")
        if info.get("read_successful"):
            st.success("Segment read successful.")
        else:
            st.error("Segment read failed or was not attempted.")
        st.markdown(
            f"- **Shape [samples, channels]:** `{info.get('shape', '—')}`\n"
            f"- **Channel count:** `{info.get('channel_count', '—')}`\n"
            f"- **Sample count:** `{info.get('sample_count', '—')}`\n"
            f"- **Sampling rate (Hz):** `{info.get('sampling_rate_hz', '—')}`\n"
            f"- **Segment duration (s):** `{info.get('segment_duration_seconds', '—')}`\n"
            f"- **Raw .bvrd size (bytes):** `{info.get('raw_data_file_size_bytes', '—')}`\n"
            f"- **Channel count matches .bvrh:** `{info.get('channel_count_matches_header', '—')}`\n"
            f"- **Sample count matches duration × rate:** `{info.get('sample_count_matches_duration', '—')}`"
        )
    for read_test_entry in result.eeg_raw_read_test_results:
        task_label = read_test_entry.get("task_name", "Task")
        test = read_test_entry.get("result") or {}
        st.markdown(f"**{task_label} — Raw EEG read test (Phase 5)**")
        if read_test_entry.get("raw_read_test_path"):
            st.caption(f"eeg_raw_read_test.json → `{read_test_entry['raw_read_test_path']}`")
        if test.get("read_success"):
            st.success(f"read_success: `{test.get('read_success')}`")
        else:
            st.error(f"read_success: `{test.get('read_success', False)}`")
        st.markdown(
            f"- **read_success:** `{test.get('read_success', '—')}`\n"
            f"- **expected_shape:** `{test.get('expected_shape', '—')}`\n"
            f"- **actual_shape:** `{test.get('actual_shape', '—')}`\n"
            f"- **expected_sample_count:** `{test.get('expected_sample_count', '—')}`\n"
            f"- **actual_sample_count:** `{test.get('actual_sample_count', '—')}`\n"
            f"- **channel_count:** `{test.get('channel_count', '—')}`\n"
            f"- **data_dtype:** `{test.get('data_dtype', '—')}`\n"
            f"- **raw_file_size_bytes:** `{test.get('raw_file_size_bytes', '—')}`"
        )
        if test.get("error_message"):
            st.error(f"error_message: {test['error_message']}")
        with st.expander("eeg_raw_read_test.json (full)"):
            st.json(test)
    for signal_qc_entry in result.eeg_raw_signal_qc_results:
        task_label = signal_qc_entry.get("task_name", "Task")
        signal_qc = signal_qc_entry.get("result") or {}
        st.markdown(f"**{task_label} — Descriptive raw signal QC (Phase 6)**")
        if signal_qc_entry.get("raw_signal_qc_path"):
            st.caption(f"eeg_raw_signal_qc.json → `{signal_qc_entry['raw_signal_qc_path']}`")
        st.info(signal_qc.get("qc_note", "Descriptive QC only. No exclusion decision is made."))
        if signal_qc.get("read_success"):
            st.markdown(
                f"- **Total NaN count:** `{signal_qc.get('total_nan_count', '—')}`\n"
                f"- **Total Inf count:** `{signal_qc.get('total_inf_count', '—')}`\n"
                f"- **Flat channels:** `{signal_qc.get('flat_channel_count', '—')}`"
            )
            flat_channels = signal_qc.get("flat_channels") or []
            if flat_channels:
                st.markdown(f"- **Flat channel list:** `{', '.join(flat_channels)}`")
            per_channel = signal_qc.get("per_channel") or []
            if per_channel:
                st.markdown("**Per-channel descriptive QC**")
                qc_table_height = min(max(len(per_channel) * 35 + 38, 120), 900)
                st.dataframe(
                    pd.DataFrame(per_channel),
                    use_container_width=True,
                    hide_index=True,
                    height=qc_table_height,
                )
        else:
            st.error("EEG segment could not be read for descriptive QC.")
            if signal_qc.get("error_message"):
                st.error(signal_qc["error_message"])
    for plan_entry in result.eeg_preprocessing_plan_results:
        task_label = plan_entry.get("task_name", "Task")
        plan = plan_entry.get("plan") or {}
        status = plan.get("status") or {}
        rois = plan.get("roi_definitions") or {}
        st.markdown(f"**{task_label} — EEG preprocessing plan**")
        if plan_entry.get("preprocessing_plan_path"):
            st.caption(f"eeg_preprocessing_plan.json → `{plan_entry['preprocessing_plan_path']}`")
        st.info(plan.get("note", "Plan and audit placeholders only. No EEG signals are modified."))
        st.markdown(
            f"- **Reference:** `{plan.get('reference', '—')}`\n"
            f"- **Band-pass (Hz):** `{plan.get('bandpass_hz', '—')}`\n"
            f"- **Notch (Hz):** `{plan.get('notch_hz', '—')}`\n"
            f"- **Bad channel policy:** `{plan.get('bad_channel_policy', '—')}`\n"
            f"- **ICA policy:** `{plan.get('ica_policy', '—')}`\n"
            f"- **Baseline source:** `{plan.get('baseline_source', '—')}`"
        )
        st.markdown("**ROI definitions**")
        for region, channels in rois.items():
            st.markdown(f"- **{region}:** `{', '.join(channels)}`")
        st.markdown("**Preprocessing audit (placeholders)**")
        st.markdown(
            f"- **Preprocessing completed:** `{status.get('preprocessing_completed', '—')}`\n"
            f"- **Bad channels flagged:** `{status.get('bad_channels_flagged', '—')}`\n"
            f"- **ICA completed:** `{status.get('ica_completed', '—')}`\n"
            f"- **Baseline available:** `{status.get('baseline_available', '—')}`"
        )
        with st.expander("eeg_preprocessing_plan.json (full)"):
            st.json(plan)
    for exec_entry in result.eeg_preprocessing_exec_results:
        task_label = exec_entry.get("task_name", "Task")
        audit = exec_entry.get("audit") or {}
        st.markdown(f"**{task_label} — Basic preprocessing (Phase 8)**")
        if exec_entry.get("preprocessed_segment_path"):
            st.caption(
                f"eeg_preprocessed_segment.npy → `{exec_entry['preprocessed_segment_path']}`"
            )
        if exec_entry.get("audit_path"):
            st.caption(f"eeg_preprocessing_audit.json → `{exec_entry['audit_path']}`")
        if audit.get("preprocessing_completed"):
            st.success("Basic preprocessing completed successfully.")
        else:
            st.error("Basic preprocessing failed.")
            if audit.get("error_message"):
                st.error(audit["error_message"])
        st.markdown(
            f"- **Original shape:** `{audit.get('original_shape', '—')}`\n"
            f"- **Preprocessed shape:** `{audit.get('preprocessed_shape', '—')}`\n"
            f"- **Sampling rate (Hz):** `{audit.get('sampling_rate_hz', '—')}`\n"
            f"- **Filters applied:** `{audit.get('filters_applied', '—')}`\n"
            f"- **Reference applied:** `{audit.get('reference_applied', '—')}`\n"
            f"- **Channels used:** `{len(audit.get('channels_used') or [])}`\n"
            f"- **Channels removed:** `{audit.get('channels_removed', '—')}`"
        )
        st.caption(audit.get("notes", ""))
        with st.expander("eeg_preprocessing_audit.json (full)"):
            st.json(audit)
    for pre_qc_entry in result.eeg_preprocessed_signal_qc_results:
        task_label = pre_qc_entry.get("task_name", "Task")
        pre_qc = pre_qc_entry.get("result") or {}
        st.markdown(f"**{task_label} — Preprocessed signal QC (Phase 9)**")
        if pre_qc_entry.get("preprocessed_qc_path"):
            st.caption(f"eeg_preprocessed_qc.json → `{pre_qc_entry['preprocessed_qc_path']}`")
        st.info(pre_qc.get("qc_note", "Preprocessed signal QC is descriptive only. No exclusion decision is made."))
        st.markdown(
            f"- **Preprocessing completed:** `{pre_qc.get('preprocessing_completed', '—')}`\n"
            f"- **Total NaN count:** `{pre_qc.get('total_nan_count', '—')}`\n"
            f"- **Total Inf count:** `{pre_qc.get('total_inf_count', '—')}`\n"
            f"- **Flat channels:** `{pre_qc.get('flat_channel_count', '—')}`"
        )
        if pre_qc.get("read_success"):
            per_channel = pre_qc.get("per_channel") or []
            if per_channel:
                st.markdown("**Per-channel QC**")
                qc_table_height = min(max(len(per_channel) * 35 + 38, 120), 900)
                st.dataframe(
                    pd.DataFrame(per_channel),
                    use_container_width=True,
                    hide_index=True,
                    height=qc_table_height,
                )
        else:
            st.error("Preprocessed segment could not be read for QC.")
            if pre_qc.get("error_message"):
                st.error(pre_qc["error_message"])
        with st.expander("eeg_preprocessed_qc.json (full)"):
            st.json(pre_qc)
    for feat_entry in result.eeg_task_level_features_results:
        task_label = feat_entry.get("task_name", "Task")
        features = feat_entry.get("features") or {}
        feature_export = {
            feature_name: features.get(feature_name, "—")
            for feature_name in EEG_TASK_FEATURES
        }
        st.markdown(f"**{task_label} — Task-level EEG features (Phase 10–22)**")
        if feat_entry.get("features_path"):
            st.caption(f"task_level_eeg_features.xlsx → `{feat_entry['features_path']}`")
        if feat_entry.get("features_json_path"):
            st.caption(f"task_level_eeg_features.json → `{feat_entry['features_json_path']}`")
        if feat_entry.get("table_3_path"):
            st.caption(f"table_3_eeg_data.xlsx → `{feat_entry['table_3_path']}`")
        for feature_name, feature_value in feature_export.items():
            if feature_value == "Not available":
                st.warning(f"**{feature_name}:** `{feature_value}`")
            else:
                st.success(f"**{feature_name}:** `{feature_value}`")
        if features.get("error_message"):
            st.caption(features["error_message"])
        st.markdown("**Table 3 · EEG Data (current row)**")
        table_3_path = feat_entry.get("table_3_path")
        if table_3_path:
            try:
                table_3_df = pd.read_excel(table_3_path, engine="openpyxl")
            except (OSError, ValueError):
                table_3_df = pd.DataFrame([feature_export])
        else:
            table_3_df = pd.DataFrame([feature_export])
        st.dataframe(table_3_df, use_container_width=True, hide_index=True)
        with st.expander("task_level_eeg_features.json (full)"):
            st.json(feature_export)
    for audit_entry in result.eeg_task_level_feature_audit_results:
        task_label = audit_entry.get("task_name", "Task")
        audit = audit_entry.get("audit") or {}
        st.markdown(f"**{task_label} — Task-level EEG feature completeness (Phase 22)**")
        if audit_entry.get("audit_path"):
            st.caption(f"eeg_task_level_feature_audit.json → `{audit_entry['audit_path']}`")
        st.info(audit.get("audit_note", "Completeness audit only."))
        missing_columns = audit.get("missing_feature_columns") or []
        st.markdown(
            f"- **Implemented EEG features:** `{audit.get('implemented_feature_count', '—')}` / "
            f"`{audit.get('expected_feature_count', '—')}`\n"
            f"- **Numeric features:** `{audit.get('numeric_feature_count', '—')}`\n"
            f"- **Not available features:** `{audit.get('not_available_feature_count', '—')}`"
        )
        if missing_columns:
            st.warning(f"**Missing columns:** `{', '.join(missing_columns)}`")
        else:
            st.success("All expected EEG feature columns are present in Table 3.")
        if audit.get("source_file"):
            st.caption(f"Source: `{audit['source_file']}`")
        with st.expander("eeg_task_level_feature_audit.json (full)"):
            st.json(audit)
    for baseline_entry in result.eeg_baseline_linkage_results:
        task_label = baseline_entry.get("task_name", "Task")
        status = baseline_entry.get("status") or {}
        st.markdown(f"**{task_label} — Resting state baseline linkage (Phase 15)**")
        if baseline_entry.get("baseline_status_path"):
            st.caption(f"eeg_baseline_status.json → `{baseline_entry['baseline_status_path']}`")
        baseline_available = status.get("baseline_available")
        availability_label = "Yes" if baseline_available else "No"
        if baseline_available:
            st.success(f"**Baseline available:** {availability_label}")
        else:
            st.warning(f"**Baseline available:** {availability_label}")
        st.markdown(f"- **Baseline task:** `{status.get('baseline_task_name', 'Resting state')}`")
        if status.get("baseline_task_path"):
            st.caption(f"Baseline path: `{status['baseline_task_path']}`")
        missing_features = status.get("missing_baseline_features") or []
        if missing_features:
            st.markdown(f"- **Missing baseline features:** `{', '.join(missing_features)}`")
        else:
            st.markdown("- **Missing baseline features:** none")
        if not baseline_available:
            st.caption(status.get("baseline_missing_message", ""))
        st.caption(status.get("note", ""))
        with st.expander("eeg_baseline_status.json (full)"):
            st.json(status)
    for qc_entry in result.eeg_qc_results:
        task_label = qc_entry.get("task_name", "Task")
        summary = qc_entry.get("summary") or {}
        st.markdown(f"**{task_label} — EEG QC summary (Phase 4)**")
        if qc_entry.get("qc_summary_path"):
            st.caption(f"eeg_qc_summary.json → `{qc_entry['qc_summary_path']}`")
        above_threshold = summary.get("channels_above_metadata_impedance_threshold")
        threshold_kohm = summary.get("high_impedance_kohm_threshold")
        above_threshold_display = (
            f"{len(above_threshold)} channel(s) above {threshold_kohm} kΩ (metadata inspection)"
            if isinstance(above_threshold, list)
            else above_threshold
        )
        st.markdown(
            f"- **Sampling rate (Hz):** `{summary.get('sampling_rate_hz', '—')}`\n"
            f"- **Channel count:** `{summary.get('channel_count', '—')}`\n"
            f"- **Segment duration (s):** `{summary.get('segment_duration_seconds', '—')}`\n"
            f"- **Partial overlap:** `{summary.get('partial_overlap', '—')}`\n"
            f"- **Missing task start (s):** `{summary.get('missing_task_start_seconds', '—')}`\n"
            f"- **Missing task end (s):** `{summary.get('missing_task_end_seconds', '—')}`\n"
            f"- **Markers count:** `{summary.get('markers_count', '—')}`\n"
            f"- **Impedance available:** `{summary.get('impedance_available', '—')}`\n"
            f"- **Channels above metadata impedance threshold:** `{above_threshold_display}`\n"
            f"- **EEGLAB .set detected:** `{summary.get('eeglab_set_detected', '—')}`"
        )
        st.caption(
            summary.get(
                "impedance_threshold_note",
                "Impedance threshold is used for metadata inspection only and is not an exclusion criterion.",
            )
        )
        if isinstance(above_threshold, list) and above_threshold:
            with st.expander("Channels above metadata impedance threshold"):
                st.dataframe(pd.DataFrame(above_threshold), use_container_width=True, hide_index=True)
    for compat_entry in result.eeg_eeglab_compat_results:
        task_label = compat_entry.get("task_name", "Task")
        status = compat_entry.get("status") or {}
        st.markdown(f"**{task_label} — EEGLAB compatibility (Phase 3)**")
        if status.get("set_file_detected"):
            st.success(status.get("message", "EEGLAB .set file detected"))
            for set_path in status.get("set_files", []):
                st.caption(f"`.set` → `{set_path}`")
        else:
            st.info(status.get("message", "EEGLAB .set export not implemented yet"))
    for sync_entry in result.eeg_sync_results:
        task_label = sync_entry.get("task_name", "Task")
        audit = sync_entry.get("audit") or {}
        st.markdown(f"**{task_label} — EEG clock sync**")
        if sync_entry.get("audit_path"):
            st.caption(f"eeg_time_audit.json → `{sync_entry['audit_path']}`")
        if sync_entry.get("settings_path"):
            st.caption(f"eeg_sync_settings.json → `{sync_entry['settings_path']}`")
        st.markdown(
            f"- **E-Prime task start (UTC):** `{audit.get('eprime_task_start_utc', '—')}`\n"
            f"- **E-Prime task end (UTC):** `{audit.get('eprime_task_end_utc', '—')}`\n"
            f"- **EEG start raw (UTC):** `{audit.get('eeg_recording_start_utc_raw', '—')}`\n"
            f"- **EEG end raw (UTC):** `{audit.get('eeg_recording_end_utc_raw', '—')}`\n"
            f"- **EEG clock offset (s):** `{audit.get('eeg_clock_offset_seconds', '—')}`\n"
            f"- **EEG start adjusted (UTC):** `{audit.get('eeg_recording_start_utc_adjusted', '—')}`\n"
            f"- **EEG end adjusted (UTC):** `{audit.get('eeg_recording_end_utc_adjusted', '—')}`\n"
            f"- **Raw overlap (s):** `{audit.get('raw_overlap_seconds', '—')}`\n"
            f"- **Adjusted overlap (s):** `{audit.get('adjusted_overlap_seconds', '—')}`"
        )
    for p in result.saved_paths:
        st.code(p, language=None)
    for w in result.warnings:
        st.warning(w)

