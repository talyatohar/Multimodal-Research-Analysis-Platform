"""Analyse participant — UI flow per project specification (no computations)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from backend.analysis.events import run_event_level_analysis, summarize_event_eeg_alignment_counts
from backend.analysis.feature_definitions import feature_tooltip
from backend.analysis.task_level_tables import TABLE_3_FILE, build_task_level_tables
from backend.eeg.task_level_feature_audit import detect_table_3_eeg_columns
from backend.paths import participant_dir, task_dir
from backend.sync.eprime import load_sync_window_json
from domain.error_catalog import (
    missing_corsano_timestamp,
    missing_eeg_bundle,
    missing_eprime,
    missing_eye_tracking,
    missing_resting_for_baseline,
    no_tobii_window_match,
)
from domain.feature_catalog import (
    EEG_TASK_FEATURES,
    EVENT_EEG_DISTRIBUTION_COLUMNS,
    EVENT_EEG_FEATURES,
    EVENT_TYPES,
    EYE_TASK_FEATURES,
    FEATURE_HELP,
    PHYSIO_TASK_FEATURES,
    QC_TASK_FEATURES,
)
from domain.sync_spec import EPRIME_FIELDS
from domain.participant_id import normalize_participant_id
from domain.resting_state import (
    RESTING_STATE_REGRESSION_SKIP_MESSAGE,
    is_resting_state_task,
)
from domain.tasks import TASK_NAMES
from shared.cohort_state import find_participant, participant_options, summarize_tasks
from shared.corsano_debug_ui import render_corsano_debug_section
from shared.developer_mode import developer_mode_enabled
from shared.session_keys import (
    ANALYSE_COMPARE_DF,
    ANALYSE_COMPARE_META,
    ANALYSE_LAST_PID,
    ANALYSE_TASK_LEVEL_DFS,
    ANALYSE_TASK_LEVEL_META,
    EVENT_LEVEL_RESULTS,
    SELECTED_PARTICIPANT_ID,
    UI_EVENT_LEVEL_UNLOCKED,
    UI_TASK_LEVEL_GENERATED,
    UI_TASK_LEVEL_UNLOCKED,
)
from shared.ui import (
    EVENT_EEG_DISTRIBUTION_TABLE_HEIGHT,
    EVENT_SUMMARY_TABLE_HEIGHT,
    apply_theme,
    configure_page,
    page_header,
    render_stable_dataframe,
    sidebar_brand,
)

configure_page("Analyse participant")
apply_theme()
sidebar_brand()

for k, default in (
    (UI_TASK_LEVEL_UNLOCKED, False),
    (UI_EVENT_LEVEL_UNLOCKED, False),
    (UI_TASK_LEVEL_GENERATED, False),
    (EVENT_LEVEL_RESULTS, {}),
):
    st.session_state.setdefault(k, default)


MISSING_VALUE = "Missing / not computed"


def _read_first_row(path) -> dict:
    if not path.is_file():
        return {}
    df = pd.read_excel(path, engine="openpyxl")
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _eye_table_path_for_window(folder, window_choice: str):
    if window_choice == "Manual Reading Window":
        path = folder / "table_1_eye_tracking_data_manual.xlsx"
        return path if path.is_file() else None
    eprime_path = folder / "table_1_eye_tracking_data_eprime.xlsx"
    if eprime_path.is_file():
        return eprime_path
    legacy_path = folder / "table_1_eye_tracking_data.xlsx"
    return legacy_path if legacy_path.is_file() else None


def _task_compare_window_options(folder) -> tuple[list[str], int]:
    manual_path = folder / "table_1_eye_tracking_data_manual.xlsx"
    eprime_path = _eye_table_path_for_window(folder, "E-Prime Window")
    options = []
    if eprime_path is not None:
        options.append("E-Prime Window")
    else:
        options.append("E-Prime Window")
    if manual_path.is_file():
        options.append("Manual Reading Window")
        return options, 1
    return options, 0


def _value_or_missing(row: dict, feature: str):
    if feature not in row:
        return MISSING_VALUE
    value = row.get(feature)
    if value is None:
        return MISSING_VALUE
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return MISSING_VALUE
        if stripped.startswith("Not available"):
            return stripped
        return stripped
    try:
        if pd.isna(value):
            return MISSING_VALUE
    except (TypeError, ValueError):
        pass
    return value


def _detect_eeg_columns_for_tasks(participant_id: str, tasks: list[str]) -> list[str]:
    detected: set[str] = set()
    for task_name in tasks:
        table_path = task_dir(participant_id, task_name) / TABLE_3_FILE
        detected.update(detect_table_3_eeg_columns(table_path))
    return [feature_name for feature_name in EEG_TASK_FEATURES if feature_name in detected]


def _build_compare_df(
    participant_id: str,
    tasks: list[str],
    selected_windows: dict[str, str],
    eye: list[str],
    physio: list[str],
    eeg: list[str],
    qc: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, str | None]]]:
    cols: list[str] = []
    for group in (eye, physio, eeg, qc):
        for name in group:
            if name not in cols:
                cols.append(name)
    rows = []
    warnings: list[str] = []
    source_paths: dict[str, dict[str, str | None]] = {}
    for task_name in tasks:
        folder = task_dir(participant_id, task_name)
        window_choice = selected_windows.get(task_name, "E-Prime Window")
        eye_path = _eye_table_path_for_window(folder, window_choice)
        physio_path = folder / "table_2_physiological_data.xlsx"
        eeg_path = folder / "table_3_eeg_data.xlsx"
        qc_path = folder / "table_4_quality_control.xlsx"

        eye_row = _read_first_row(eye_path) if eye_path is not None else {}
        physio_row = _read_first_row(physio_path)
        eeg_row = _read_first_row(eeg_path)
        qc_row = _read_first_row(qc_path)
        source_paths[task_name] = {
            "eye_tracking": str(eye_path.resolve()) if eye_path is not None else None,
            "physiological": str(physio_path.resolve()) if physio_path.is_file() else None,
            "eeg": str(eeg_path.resolve()) if eeg_path.is_file() else None,
            "quality_control": str(qc_path.resolve()) if qc_path.is_file() else None,
        }

        if eye and not eye_row:
            warnings.append(f"{task_name}: missing eye-tracking table for {window_choice}.")
        if physio and not physio_row:
            warnings.append(f"{task_name}: missing table_2_physiological_data.xlsx.")
        if eeg and not eeg_row:
            warnings.append(f"{task_name}: missing table_3_eeg_data.xlsx.")
        if qc and not qc_row:
            warnings.append(f"{task_name}: missing table_4_quality_control.xlsx.")

        out = {"task": task_name, "analysis_window_used": window_choice}
        for feature in eye:
            out[feature] = _value_or_missing(eye_row, feature)
        for feature in physio:
            out[feature] = _value_or_missing(physio_row, feature)
        for feature in eeg:
            out[feature] = _value_or_missing(eeg_row, feature)
        for feature in qc:
            out[feature] = _value_or_missing(qc_row, feature)
        rows.append(out)
    return pd.DataFrame(rows, columns=["task", "analysis_window_used"] + cols), warnings, source_paths


def _comparison_filename(raw_name: str) -> tuple[str | None, str | None]:
    name = raw_name.strip()
    if not name:
        return None, "Enter a comparison file name before saving."
    original = name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.rstrip(" .")
    if not name:
        return None, "Comparison file name contains no valid characters."
    stem = name[:-5] if name.lower().endswith(".xlsx") else name
    if stem.upper() in {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}:
        stem = f"_{stem}"
    filename = f"{stem}.xlsx"
    if filename != original and f"{original}.xlsx" != filename:
        return filename, f"Invalid Windows filename characters were replaced. Saving as `{filename}`."
    return filename, None


def _comparison_metadata_path(xlsx_path):
    stem = xlsx_path.stem
    return xlsx_path.with_name(f"{stem}_metadata.json")


def _display_single_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.iloc[[0]].reset_index(drop=True)


def _event_context(participant_id: str, task_name: str, event_type: str) -> tuple[str, str, str]:
    return (participant_id, task_name, event_type)


def _prepare_event_summary_display(df: pd.DataFrame) -> pd.DataFrame:
    return _display_single_row(df)


def _prepare_event_distribution_display(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(EVENT_EEG_DISTRIBUTION_COLUMNS)
    indexed: dict[str, pd.Series] = {}
    if not df.empty and "feature" in df.columns:
        for _, row in df.iterrows():
            key = str(row["feature"]).strip()
            if key in EVENT_EEG_FEATURES:
                indexed[key] = row

    rows: list[dict[str, Any]] = []
    for name in EVENT_EEG_FEATURES:
        if name in indexed:
            rows.append(indexed[name].to_dict())
        else:
            rows.append({column: (name if column == "feature" else None) for column in columns})
    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)


def _event_table_key(prefix: str, participant_id: str, task_name: str, event_type: str) -> str:
    safe_task = re.sub(r"[^\w-]", "_", task_name)
    safe_event = re.sub(r"[^\w-]", "_", event_type)
    return f"{prefix}_{participant_id}_{safe_task}_{safe_event}"


def _event_level_results() -> dict[str, Any]:
    results = st.session_state.get(EVENT_LEVEL_RESULTS)
    if not isinstance(results, dict):
        results = {}
        st.session_state[EVENT_LEVEL_RESULTS] = results
    return results


def _get_event_level_result(
    participant_id: str,
    task_name: str,
    event_type: str,
) -> dict[str, Any] | None:
    slot = (
        _event_level_results()
        .get(participant_id, {})
        .get(task_name, {})
        .get(event_type)
    )
    return slot if isinstance(slot, dict) else None


def _clear_event_level_results() -> None:
    st.session_state[EVENT_LEVEL_RESULTS] = {}


def _store_event_level_result(
    summary_df: pd.DataFrame,
    eeg_df: pd.DataFrame,
    *,
    participant_id: str,
    task_name: str,
    event_type: str,
    meta: dict[str, Any],
) -> bool:
    summary_display = _prepare_event_summary_display(summary_df)
    eeg_display = _prepare_event_distribution_display(eeg_df)
    if summary_display.empty or len(eeg_display) != len(EVENT_EEG_FEATURES):
        return False

    results = _event_level_results()
    results.setdefault(participant_id, {}).setdefault(task_name, {})[event_type] = {
        "summary_table": summary_display,
        "distribution_table": eeg_display,
        "summary_col_config": _feature_column_config(summary_display),
        "eeg_col_config": _feature_column_config(eeg_display),
        "meta": meta,
    }
    return True


def _parse_optional_positive_float(value: str, label: str) -> tuple[float | None, str | None]:
    text = value.strip()
    if not text:
        return None, None
    try:
        parsed = float(text)
    except ValueError:
        return None, f"{label} must be a number."
    if parsed <= 0:
        return None, f"{label} must be greater than 0 px."
    return parsed, None


def _feature_column_config(df: pd.DataFrame) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for col in df.columns:
        col_name = str(col)
        tooltip = feature_tooltip(col_name)
        label = f"{col_name} ⓘ" if tooltip else col_name
        series = df[col]
        if col_name == "feature" or not pd.api.types.is_numeric_dtype(series):
            config[col] = st.column_config.TextColumn(label=label, help=tooltip)
        else:
            config[col] = st.column_config.NumberColumn(label=label, help=tooltip)
    return config


def _clear_analyse_table_caches() -> None:
    st.session_state[UI_TASK_LEVEL_GENERATED] = False
    _clear_event_level_results()
    for key in (
        ANALYSE_TASK_LEVEL_DFS,
        ANALYSE_TASK_LEVEL_META,
        ANALYSE_COMPARE_DF,
        ANALYSE_COMPARE_META,
    ):
        st.session_state.pop(key, None)


page_header(
    "Multimodal analysis",
    "Analyse participant",
    "Progressive disclosure matches the spec: unlock task-level tools, generate cached tables once, "
    "compare tasks, then unlock event-level EEG summaries.",
)

opts = participant_options()
preferred = normalize_participant_id(st.session_state.get(SELECTED_PARTICIPANT_ID))
default_ix = 0
if preferred and preferred in opts:
    default_ix = opts.index(preferred)

pid = st.selectbox("Participants ID", options=opts, index=default_ix, key="analyse_pid_select")
if pid and not pid.startswith("—"):
    pid = normalize_participant_id(pid)

if st.session_state.get(ANALYSE_LAST_PID) != pid:
    st.session_state[ANALYSE_LAST_PID] = pid
    _clear_analyse_table_caches()

row = find_participant(pid) if pid and not pid.startswith("—") else None

if row:
    st.caption(
        f"**{row.get('participant_name') or '—'}** · {row.get('participant_group') or '—'} · "
        f"Age {row.get('participant_age') if row.get('participant_age') is not None else '—'}"
    )
    st.caption(f"Registered tasks: {summarize_tasks(row)}")
    render_corsano_debug_section(str(row["participant_id"]))
else:
    st.info("Register a participant on the Upload page to populate this selector.", icon="ℹ️")

st.divider()

st.markdown("##### Task Level Analysis")
if st.button("Task Level Analysis", type="secondary"):
    st.session_state[UI_TASK_LEVEL_UNLOCKED] = True

if st.session_state[UI_TASK_LEVEL_UNLOCKED]:
    available_task_names = [t.get("task_name") for t in (row or {}).get("tasks", []) if t.get("task_name")]
    task_options = available_task_names or list(TASK_NAMES)
    task_choice = st.selectbox(
        "Task Name",
        options=task_options,
        key="analyse_task_level_task_name",
    )
    selected_task_folder = task_dir(str(row["participant_id"]), task_choice) if row else None
    sync_window = load_sync_window_json(selected_task_folder) if selected_task_folder else None
    window_options = ["E-Prime Window"]
    if sync_window and sync_window.get("manual_task_end_utc"):
        window_options.append("Manual Reading Window")
    window_choice = st.selectbox(
        "Eye-tracking analysis window",
        options=window_options,
        key="analyse_eye_tracking_window_type",
        help="Manual reading end times are based on user review of the Tobii screen recording.",
    )
    if window_choice == "Manual Reading Window":
        st.info("Manual reading end times are based on user review of the Tobii screen recording.")
        if developer_mode_enabled() and sync_window:
            st.caption(
                f"Manual local time: `{sync_window.get('manual_task_end_local_time')}` · "
                f"Manual UTC: `{sync_window.get('manual_task_end_utc')}`"
            )
    force_eye_recompute = False
    if developer_mode_enabled():
        force_eye_recompute = st.checkbox(
            "Force recompute task-level tables",
            value=False,
            key="analyse_force_eye_tracking_recompute",
            help="When unchecked, existing task-level table Excel files are loaded from the task folder.",
        )
    with st.expander("Advanced settings · regression threshold override", expanded=False):
        st.caption(
            "Leave blank to estimate thresholds automatically from the processed eye-tracking segment. "
            "Manual values override the auto-estimated thresholds for this run."
        )
        tc1, tc2 = st.columns(2)
        with tc1:
            line_threshold_raw = st.text_input(
                "line_transition_threshold_px",
                key="analyse_line_transition_threshold_px",
                placeholder="Enter px",
                help="Same-line rule: abs(delta_y) < line_transition_threshold_px.",
            )
        with tc2:
            regression_threshold_raw = st.text_input(
                "regression_threshold_px",
                key="analyse_regression_threshold_px",
                placeholder="Enter px",
                help="Regression rule after same-line filtering: delta_x > regression_threshold_px.",
            )
    if st.button("GENERATE", type="primary", key="gen_task_level"):
        if not row:
            st.warning("Select a registered participant before generating task-level analysis.")
        else:
            line_threshold, line_threshold_error = _parse_optional_positive_float(
                line_threshold_raw,
                "line_transition_threshold_px",
            )
            regression_threshold, regression_threshold_error = _parse_optional_positive_float(
                regression_threshold_raw,
                "regression_threshold_px",
            )
            if line_threshold_error or regression_threshold_error:
                for error in (line_threshold_error, regression_threshold_error):
                    if error:
                        st.warning(error)
                st.stop()
            result = build_task_level_tables(
                task_dir(str(row["participant_id"]), task_choice),
                force_recompute=force_eye_recompute,
                line_transition_threshold_px=line_threshold,
                regression_threshold_px=regression_threshold,
                window_type="manual" if window_choice == "Manual Reading Window" else "eprime",
            )
            st.session_state[UI_TASK_LEVEL_GENERATED] = True
            st.session_state[ANALYSE_TASK_LEVEL_DFS] = result.tables
            st.session_state[ANALYSE_TASK_LEVEL_META] = {
                "task_name": task_choice,
                "window_choice": window_choice,
                "warnings": result.warnings,
                "loaded_existing": result.loaded_existing,
                "table_paths": {key: str(path.resolve()) for key, path in result.table_paths.items()},
                "quality_control": result.quality_control,
            }
            if all(result.loaded_existing.values()):
                st.success("Loaded existing task-level analysis tables.")
            else:
                st.success("Generated task-level analysis tables.")
            if developer_mode_enabled():
                eye_qc = result.quality_control.get("eye_tracking") if isinstance(result.quality_control, dict) else None
                regression_params = (eye_qc or {}).get("regression_parameters") if isinstance(eye_qc, dict) else None
                if regression_params:
                    st.info(
                        "Regression thresholds: "
                        f"line_transition_threshold_px={regression_params.get('line_transition_threshold_px')} "
                        f"({regression_params.get('line_transition_threshold_source')}), "
                        f"regression_threshold_px={regression_params.get('regression_threshold_px')} "
                        f"({regression_params.get('regression_threshold_source')})."
                    )

    if st.session_state[UI_TASK_LEVEL_GENERATED]:
        dfs_tl = st.session_state.get(ANALYSE_TASK_LEVEL_DFS)
        if not isinstance(dfs_tl, dict):
            dfs_tl = {}
        meta = st.session_state.get(ANALYSE_TASK_LEVEL_META) or {}
        task_label = meta.get("task_name", "—")
        st.caption(f"Task-level tables for task: **{task_label}** · {meta.get('window_choice', 'E-Prime Window')}.")
        table_specs = (
            ("t1", "Table 1 · Eye Tracking Data"),
            ("t2", "Table 2 · Physiological Data"),
            ("t3", "Table 3 · EEG Data"),
            ("t4", "Table 4 · Quality Control"),
        )
        table_paths = meta.get("table_paths") or {}
        loaded_existing = meta.get("loaded_existing") or {}
        warnings_by_table = meta.get("warnings") or {}
        for key, title in table_specs:
            df = dfs_tl.get(key)
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            df = _display_single_row(df)
            st.markdown(f"###### {title}")
            if developer_mode_enabled() and loaded_existing.get(key):
                st.caption("Loaded from saved table.")
            render_stable_dataframe(
                df,
                column_config=_feature_column_config(df),
                max_visible_rows=None,
            )
            if developer_mode_enabled():
                if table_paths.get(key):
                    st.caption(f"{table_paths[key]}")
            for warning in warnings_by_table.get(title, []):
                st.warning(warning)
        if is_resting_state_task(task_choice):
            eye_qc = (meta.get("quality_control") or {}).get("eye_tracking")
            info = RESTING_STATE_REGRESSION_SKIP_MESSAGE
            if isinstance(eye_qc, dict) and eye_qc.get("regression_detection_info"):
                info = eye_qc["regression_detection_info"]
            st.info(info, icon="ℹ️")
    if developer_mode_enabled() and st.session_state[UI_TASK_LEVEL_GENERATED]:
        qc_report = meta.get("quality_control")
        if isinstance(qc_report, dict):
            eye_qc = qc_report.get("eye_tracking")
            regression_params = (eye_qc or {}).get("regression_parameters") if isinstance(eye_qc, dict) else None
            if regression_params:
                st.caption(
                    "Regression thresholds: "
                    f"line `{regression_params.get('line_transition_threshold_px')}` px "
                    f"({regression_params.get('line_transition_threshold_source')}), "
                    f"regression `{regression_params.get('regression_threshold_px')}` px "
                    f"({regression_params.get('regression_threshold_source')})."
                )
            with st.expander("Eye-tracking quality control report"):
                st.json(qc_report)

    st.divider()
    st.markdown("##### Task comparison")
    compare_task_options = available_task_names or list(TASK_NAMES)
    compare_tasks = st.multiselect(
        "Task names",
        options=compare_task_options,
        key="analyse_compare_task_names",
    )
    cf1, cf2, cf3, cf4 = st.columns(4)
    with cf1:
        eye_ms = st.multiselect(
            "Eye tracking features",
            options=list(EYE_TASK_FEATURES),
            key="analyse_compare_eye_features",
        )
    with cf2:
        phys_ms = st.multiselect(
            "Physiological features",
            options=list(PHYSIO_TASK_FEATURES),
            key="analyse_compare_phys_features",
        )
    with cf3:
        if developer_mode_enabled() and row and compare_tasks:
            detected_eeg_columns = _detect_eeg_columns_for_tasks(str(row["participant_id"]), compare_tasks)
            st.caption(
                "EEG columns detected from table_3_eeg_data.xlsx: "
                f"`{len(detected_eeg_columns)}` of `{len(EEG_TASK_FEATURES)}` expected features."
            )
            if detected_eeg_columns:
                st.caption(f"Detected: `{', '.join(detected_eeg_columns)}`")
        eeg_ms = st.multiselect(
            "EEG features",
            options=list(EEG_TASK_FEATURES),
            key="analyse_compare_eeg_features",
            help="All task-level EEG features are listed. Values are read from each task's table_3_eeg_data.xlsx.",
        )
    with cf4:
        qc_ms = st.multiselect(
            "Quality control features",
            options=list(QC_TASK_FEATURES),
            key="analyse_compare_qc_features",
        )
    selected_compare_windows: dict[str, str] = {}
    if row and compare_tasks:
        st.caption("Select the eye-tracking analysis window for each task. Manual is the default when its table exists.")
        for task_name in compare_tasks:
            folder = task_dir(str(row["participant_id"]), task_name)
            window_opts, default_window_ix = _task_compare_window_options(folder)
            selected_compare_windows[task_name] = st.selectbox(
                f"{task_name} window",
                options=window_opts,
                index=default_window_ix,
                key=f"analyse_compare_window_{task_name}",
            )
    if st.button("COMPARE", key="compare_btn"):
        has_features = bool(eye_ms or phys_ms or eeg_ms or qc_ms)
        if not row:
            st.warning("Select a registered participant before running a comparison.")
        elif not compare_tasks:
            st.warning("Select at least one **task** before running a comparison.")
        elif not has_features:
            st.warning(
                "Select at least one **feature** in one or more categories (Eye / Physiological / EEG / QC) "
                "before running a comparison."
            )
        else:
            compare_df, compare_warnings, source_paths = _build_compare_df(
                str(row["participant_id"]),
                list(compare_tasks),
                selected_compare_windows,
                list(eye_ms),
                list(phys_ms),
                list(eeg_ms),
                list(qc_ms),
            )
            st.session_state[ANALYSE_COMPARE_DF] = compare_df
            st.session_state[ANALYSE_COMPARE_META] = {
                "tasks": list(compare_tasks),
                "windows": selected_compare_windows,
                "eye": list(eye_ms),
                "physio": list(phys_ms),
                "eeg": list(eeg_ms),
                "qc": list(qc_ms),
                "warnings": compare_warnings,
                "source_table_paths": source_paths,
            }
            st.success("Comparison table loaded from saved task-level tables.")

    compare_df = st.session_state.get(ANALYSE_COMPARE_DF)
    if isinstance(compare_df, pd.DataFrame) and not compare_df.empty:
        st.caption("Comparison table loaded from saved task-level tables; missing values are marked explicitly.")
        render_stable_dataframe(
            compare_df.reset_index(drop=True),
            column_config=_feature_column_config(compare_df),
        )
    compare_meta = st.session_state.get(ANALYSE_COMPARE_META) or {}
    for warning in compare_meta.get("warnings") or []:
        st.warning(warning)
    if isinstance(compare_df, pd.DataFrame) and not compare_df.empty:
        st.markdown("###### Save Comparison")
        comparison_name = st.text_input("Comparison file name", key="analyse_comparison_file_name")
        filename, filename_warning = _comparison_filename(comparison_name) if comparison_name else (None, None)
        if filename_warning:
            st.caption(filename_warning)
        comparisons_dir = participant_dir(str(row["participant_id"])) / "comparisons" if row else None
        target_path = comparisons_dir / filename if comparisons_dir and filename else None
        overwrite_existing = False
        if target_path and target_path.exists():
            overwrite_existing = st.checkbox(
                "Overwrite existing comparison file",
                value=False,
                key="analyse_comparison_overwrite",
                help="If unchecked, a timestamp suffix is added automatically.",
            )
        if st.button("Save comparison table", key="save_comparison_table"):
            if not row:
                st.warning("Select a registered participant before saving a comparison.")
            elif filename is None or comparisons_dir is None or target_path is None:
                st.warning(filename_warning or "Enter a comparison file name before saving.")
            else:
                comparisons_dir.mkdir(parents=True, exist_ok=True)
                save_path = target_path
                if save_path.exists() and not overwrite_existing:
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    save_path = save_path.with_name(f"{save_path.stem}_{stamp}{save_path.suffix}")
                compare_df.to_excel(save_path, index=False, engine="openpyxl")
                metadata = {
                    "participant_id": str(row["participant_id"]),
                    "selected_tasks": compare_meta.get("tasks", []),
                    "selected_features": {
                        "eye_tracking": compare_meta.get("eye", []),
                        "physiological": compare_meta.get("physio", []),
                        "eeg": compare_meta.get("eeg", []),
                        "quality_control": compare_meta.get("qc", []),
                    },
                    "analysis_window_used": compare_meta.get("windows", {}),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_table_paths": compare_meta.get("source_table_paths", {}),
                }
                metadata_path = _comparison_metadata_path(save_path)
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                st.success(f"Comparison table saved to `{save_path.resolve()}`")
                if developer_mode_enabled():
                    st.caption(f"Metadata saved to `{metadata_path.resolve()}`")

st.divider()
st.markdown("##### Event Level Analysis")
if st.button("Event Level Analysis", type="secondary"):
    st.session_state[UI_EVENT_LEVEL_UNLOCKED] = True

if st.session_state[UI_EVENT_LEVEL_UNLOCKED]:
    event_task_options: list[str] = []
    if row:
        event_task_options = [
            t.get("task_name") for t in row.get("tasks", []) if t.get("task_name")
        ]

    if not event_task_options:
        st.info("No task folders found for this participant.", icon="ℹ️")
    else:
        ev_task_sel = st.selectbox(
            "Task (event scope)",
            options=event_task_options,
            key="event_task_multiselect",
        )
        ev_type = st.selectbox(
            "Event type",
            options=list(EVENT_TYPES),
            key="event_type_selectbox",
        )

        current_event_context: tuple[str, str, str] | None = None
        if row:
            current_event_context = _event_context(str(row["participant_id"]), ev_task_sel, ev_type)

        cached_event_result = (
            _get_event_level_result(*current_event_context)
            if current_event_context is not None
            else None
        )

        gen_col, regen_col = st.columns([1, 1])
        with gen_col:
            generate_event = st.button("GENERATE", type="primary", key="event_generate_button")
        with regen_col:
            regenerate_event = st.button("Regenerate event-level analysis", key="regen_event")

        if generate_event or regenerate_event:
            if not row:
                st.warning("Select a registered participant before generating event-level tables.")
            elif current_event_context is None:
                st.warning("Select a registered participant before generating event-level tables.")
            else:
                participant_id, task_name, event_type = current_event_context
                task_folder = task_dir(participant_id, task_name)
                pipeline_result = run_event_level_analysis(
                    task_folder,
                    participant_id,
                    task_name,
                    event_type,
                    window_type="eprime",
                    force_recompute=regenerate_event,
                )
                event_result = pipeline_result.get("event_result") or {}
                alignment_result = pipeline_result.get("alignment_result") or {}
                epoch_result = pipeline_result.get("epoch_result") or {}
                feature_result = pipeline_result.get("feature_result") or {}
                aggregation_result = pipeline_result.get("aggregation_result") or {}
                loaded_from_cache = bool(pipeline_result.get("loaded_from_cache"))

                selected_summary = aggregation_result.get("selected_summary")
                if not isinstance(selected_summary, pd.DataFrame):
                    selected_summary = pd.DataFrame()
                eeg_df = aggregation_result.get("selected_distribution")
                if not isinstance(eeg_df, pd.DataFrame):
                    eeg_df = pd.DataFrame()

                event_meta = {
                    "event_type": event_type,
                    "task_name": task_name,
                    "database_path": event_result.get("database_path"),
                    "summary_path": aggregation_result.get("summary_path") or event_result.get("summary_path"),
                    "distribution_path": aggregation_result.get("distribution_path"),
                    "warnings": (
                        (event_result.get("warnings") or [])
                        + (alignment_result.get("warnings") or [])
                        + (epoch_result.get("warnings") or [])
                        + (feature_result.get("warnings") or [])
                        + (aggregation_result.get("warnings") or [])
                    ),
                    "loaded_from_cache": loaded_from_cache,
                    "event_count": int(
                        pd.to_numeric(selected_summary["number_of_events"], errors="coerce").fillna(0).iloc[0]
                    )
                    if not selected_summary.empty and "number_of_events" in selected_summary.columns
                    else 0,
                }
                stored = _store_event_level_result(
                    selected_summary,
                    eeg_df,
                    participant_id=participant_id,
                    task_name=task_name,
                    event_type=event_type,
                    meta=event_meta,
                )
                if stored:
                    cached_event_result = _get_event_level_result(participant_id, task_name, event_type)
                    if loaded_from_cache:
                        st.success("Loaded saved event-level analysis from disk.")
                    elif developer_mode_enabled():
                        st.success("Generated and saved event-level analysis outputs.")
                    else:
                        st.success("Event-level analysis complete.")
                else:
                    st.warning("Event-level tables could not be prepared from the saved outputs.")
                for warning in event_meta["warnings"]:
                    st.warning(warning)

        if cached_event_result:
            ev_meta = cached_event_result.get("meta") or {}
            ev_head = ev_meta.get("event_type", ev_type)
            participant_id, task_name, event_type = current_event_context  # type: ignore[misc]
            with st.container():
                st.markdown(f"###### Event summary · {ev_head}")
                if developer_mode_enabled():
                    task_label = ev_meta.get("task_name", ev_task_sel)
                    st.caption(f"Task: `{task_label}`")
                    if ev_meta.get("database_path"):
                        st.caption(f"event_database.xlsx → `{ev_meta['database_path']}`")
                    if ev_meta.get("summary_path"):
                        st.caption(f"event_summary.xlsx → `{ev_meta['summary_path']}`")
                    if ev_meta.get("distribution_path"):
                        st.caption(f"event_level_eeg_distribution.xlsx → `{ev_meta['distribution_path']}`")
                render_stable_dataframe(
                    cached_event_result["summary_table"],
                    height=EVENT_SUMMARY_TABLE_HEIGHT,
                    column_config=cached_event_result.get("summary_col_config"),
                    key=_event_table_key("event_summary_table", participant_id, task_name, event_type),
                )

                st.markdown("###### Event-level EEG distribution")
                alignment_counts = summarize_event_eeg_alignment_counts(
                    task_dir(participant_id, task_name),
                    event_type,
                )
                if alignment_counts["detected"] > 0 and alignment_counts["aligned"] == 0:
                    st.info(
                        "Events were detected, but no EEG features could be computed because all events "
                        "fall outside the available EEG segment.",
                        icon="ℹ️",
                    )
                    st.caption(
                        f"Detected events: {alignment_counts['detected']} · "
                        f"EEG-aligned events: {alignment_counts['aligned']} · "
                        f"Outside-EEG events: {alignment_counts['outside']}"
                    )
                render_stable_dataframe(
                    cached_event_result["distribution_table"],
                    height=EVENT_EEG_DISTRIBUTION_TABLE_HEIGHT,
                    column_config=cached_event_result.get("eeg_col_config"),
                    key=_event_table_key(
                        "event_eeg_distribution_table",
                        participant_id,
                        task_name,
                        event_type,
                    ),
                )

st.divider()
if developer_mode_enabled():
    with st.expander("Feature glossary (click-through placeholder)"):
        all_feats = sorted(set(FEATURE_HELP))
        pick = st.selectbox("Feature", options=all_feats, index=0, key="analyse_glossary_feature")
        st.write(FEATURE_HELP.get(pick, "No description available."))

    with st.expander("Synchronization inputs (reference)"):
        st.markdown(
            "**E-Prime parser targets** — extract UTC session anchor plus fixation/story timing fields "
            f"including: `{', '.join(EPRIME_FIELDS)}` (see project book for window math)."
        )
        st.markdown(
            "**BrainVision** — `.amrk` first marker → recording start (local → UTC offset rule); "
            "`.ahdr` `SamplingInterval` → Hz."
        )
        st.markdown(
            "**Tobii** — search all participant `EyeTracking.xlsx` files until a recording spans "
            "`TASK_START_UTC`…`TASK_END_UTC` in ms resolution."
        )
        st.markdown("**Corsano** — clip `timestamp` (Unix ms) to the E-Prime window; ignore `date` for alignment.")

    with st.expander("Modular processing & error handling (spec)"):
        st.markdown(
            "- Each modality runs only when required files exist; missing data blocks that module only.\n"
            "- Internal flags such as `eye_tracking_uploaded`, `eeg_uploaded`, `task_level_features_generated` "
            "gate generate-vs-load without surfacing as a primary dashboard tile.\n"
            "- User-facing errors must name the missing dependency."
        )
        st.code(
            "\n".join(
                [
                    missing_eye_tracking(),
                    missing_eeg_bundle(),
                    missing_eprime(),
                    missing_resting_for_baseline(),
                    missing_corsano_timestamp(),
                    no_tobii_window_match(),
                ]
            ),
            language="text",
        )
