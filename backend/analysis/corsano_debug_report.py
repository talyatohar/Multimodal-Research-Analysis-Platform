"""Temporary Corsano segmentation debug report (participant 11 only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.analysis.corsano_task_level import (
    ACC_FILE,
    ACTIVITY_FILE,
    BPM_QUALITY_MIN,
    HRV_FILE,
    INSUFFICIENT_DATA,
    OUTLIER_REMOVAL_MIN_SAMPLES,
    OUTLIER_REMOVAL_NOTE,
    RESP_QUALITY_MIN,
    RESTING_STATE_TASK,
    _activity_features,
    _expanded_sparse_boundary,
    _load_corsano_excel,
    _mean_and_std,
    _mean_only,
    _parse_task_window,
    _resting_state_folder,
    _trim_percentile_outliers,
)
from backend.sync.eprime import load_sync_window_json

CORSANO_DEBUG_PARTICIPANT_ID = "11"


def _iso_timestamp(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return pd.Timestamp(value).isoformat()


def _discover_task_folders(participant_folder: Path) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    if not participant_folder.is_dir():
        return tasks
    for child in sorted(participant_folder.iterdir()):
        if child.is_dir() and (child / "sync_window.json").is_file():
            tasks.append((child.name, child))
    return tasks


def _file_span(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "total_rows_in_file": 0,
            "first_timestamp_utc": "—",
            "last_timestamp_utc": "—",
        }
    return {
        "total_rows_in_file": int(len(df)),
        "first_timestamp_utc": _iso_timestamp(df["timestamp_utc"].min()),
        "last_timestamp_utc": _iso_timestamp(df["timestamp_utc"].max()),
    }


def _inspect_sparse_segment(
    df: pd.DataFrame | None,
    task_start: pd.Timestamp,
    task_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    span = _file_span(df)
    if df is None or df.empty:
        return pd.DataFrame(), {
            **span,
            "strict_window_rows": 0,
            "expanded_window_rows": 0,
            "expanded_start_utc": "—",
            "expanded_end_utc": "—",
        }

    strict = df.loc[
        (df["timestamp_utc"] >= task_start) & (df["timestamp_utc"] <= task_end)
    ].copy()
    strict_count = int(len(strict))
    if not strict.empty:
        return strict, {
            **span,
            "strict_window_rows": strict_count,
            "expanded_window_rows": strict_count,
            "expanded_start_utc": _iso_timestamp(task_start),
            "expanded_end_utc": _iso_timestamp(task_end),
        }

    expanded_start, expanded_end = _expanded_sparse_boundary(df, task_start, task_end)
    if expanded_start is None or expanded_end is None or expanded_start > expanded_end:
        return pd.DataFrame(), {
            **span,
            "strict_window_rows": 0,
            "expanded_window_rows": 0,
            "expanded_start_utc": _iso_timestamp(expanded_start),
            "expanded_end_utc": _iso_timestamp(expanded_end),
        }

    expanded = df.loc[
        (df["timestamp_utc"] >= expanded_start) & (df["timestamp_utc"] <= expanded_end)
    ].copy()
    return expanded, {
        **span,
        "strict_window_rows": 0,
        "expanded_window_rows": int(len(expanded)),
        "expanded_start_utc": _iso_timestamp(expanded_start),
        "expanded_end_utc": _iso_timestamp(expanded_end),
    }


def _inspect_strict_segment(
    df: pd.DataFrame | None,
    task_start: pd.Timestamp,
    task_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    span = _file_span(df)
    if df is None or df.empty:
        return pd.DataFrame(), {
            **span,
            "strict_window_rows": 0,
            "expanded_window_rows": 0,
            "expanded_start_utc": "—",
            "expanded_end_utc": "—",
        }

    strict = df.loc[
        (df["timestamp_utc"] >= task_start) & (df["timestamp_utc"] <= task_end)
    ].copy()
    return strict, {
        **span,
        "strict_window_rows": int(len(strict)),
        "expanded_window_rows": int(len(strict)),
        "expanded_start_utc": _iso_timestamp(task_start),
        "expanded_end_utc": _iso_timestamp(task_end),
    }


def _activity_quality_counts(segment: pd.DataFrame) -> dict[str, int]:
    if segment.empty:
        return {
            "bpm_non_null_rows": 0,
            "bpm_q_ge_3_rows": 0,
            "respiration_rate_non_null_rows": 0,
            "resp_q_ge_3_rows": 0,
        }
    bpm = pd.to_numeric(segment["bpm"], errors="coerce") if "bpm" in segment.columns else pd.Series(dtype=float)
    bpm_q = pd.to_numeric(segment["bpm_q"], errors="coerce") if "bpm_q" in segment.columns else pd.Series(dtype=float)
    resp = (
        pd.to_numeric(segment["respiration_rate"], errors="coerce")
        if "respiration_rate" in segment.columns
        else pd.Series(dtype=float)
    )
    resp_q = pd.to_numeric(segment["resp_q"], errors="coerce") if "resp_q" in segment.columns else pd.Series(dtype=float)
    return {
        "bpm_non_null_rows": int(bpm.notna().sum()),
        "bpm_q_ge_3_rows": int((bpm_q >= BPM_QUALITY_MIN).sum()),
        "respiration_rate_non_null_rows": int(resp.notna().sum()),
        "resp_q_ge_3_rows": int((resp_q >= RESP_QUALITY_MIN).sum()),
    }


def _hrv_quality_counts(segment: pd.DataFrame) -> dict[str, int]:
    if segment.empty:
        return {"rmssd_non_null_rows": 0, "si_non_null_rows": 0}
    rmssd = pd.to_numeric(segment["rmssd"], errors="coerce") if "rmssd" in segment.columns else pd.Series(dtype=float)
    si = pd.to_numeric(segment["si"], errors="coerce") if "si" in segment.columns else pd.Series(dtype=float)
    return {
        "rmssd_non_null_rows": int(rmssd.notna().sum()),
        "si_non_null_rows": int(si.notna().sum()),
    }


def _acc_complete_rows(segment: pd.DataFrame) -> int:
    if segment.empty:
        return 0
    required = ("accX", "accY", "accZ")
    if any(column not in segment.columns for column in required):
        return 0
    x = pd.to_numeric(segment["accX"], errors="coerce")
    y = pd.to_numeric(segment["accY"], errors="coerce")
    z = pd.to_numeric(segment["accZ"], errors="coerce")
    return int((x.notna() & y.notna() & z.notna()).sum())


def _metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()],
        columns=["metric", "value"],
    )


def _format_samples(series: pd.Series) -> str:
    if series.empty:
        return "[]"
    return str([round(float(value), 6) if float(value) == float(value) else value for value in series.tolist()])


def _trace_activity_feature_calculation(
    activity_segment: pd.DataFrame,
    *,
    value_col: str,
    q_col: str,
    q_min: int,
    feature_label: str,
) -> dict[str, Any]:
    rows_before_quality_filter = 0
    rows_after_quality_filter = 0
    rows_after_outlier_removal = 0
    final_mean_count = 0
    final_std_count = 0
    samples_after_quality_filter = pd.Series(dtype=float)
    samples_after_outlier_removal = pd.Series(dtype=float)
    insufficient_mean_reason = "—"
    insufficient_std_reason = "—"
    computed_mean: float | str = INSUFFICIENT_DATA
    computed_std: float | str = INSUFFICIENT_DATA

    if activity_segment.empty:
        insufficient_mean_reason = f"{feature_label}: activity segment is empty after window segmentation."
        insufficient_std_reason = insufficient_mean_reason
    elif value_col not in activity_segment.columns:
        insufficient_mean_reason = f"{feature_label}: missing column '{value_col}' in segmented activity rows."
        insufficient_std_reason = insufficient_mean_reason
    elif q_col not in activity_segment.columns:
        insufficient_mean_reason = f"{feature_label}: missing column '{q_col}' in segmented activity rows."
        insufficient_std_reason = insufficient_mean_reason
    else:
        values_raw = pd.to_numeric(activity_segment[value_col], errors="coerce")
        rows_before_quality_filter = int(values_raw.notna().sum())

        quality = pd.to_numeric(activity_segment[q_col], errors="coerce")
        quality_pass_mask = quality >= q_min
        samples_after_quality_filter = pd.to_numeric(
            activity_segment.loc[quality_pass_mask, value_col],
            errors="coerce",
        ).dropna()
        rows_after_quality_filter = int(len(samples_after_quality_filter))

        if rows_after_quality_filter == 0:
            insufficient_mean_reason = (
                f"{feature_label}: no rows with non-null {value_col} and {q_col} >= {q_min}."
            )
            insufficient_std_reason = insufficient_mean_reason
        else:
            outlier_removal_applied = rows_after_quality_filter >= OUTLIER_REMOVAL_MIN_SAMPLES
            samples_after_outlier_removal = _trim_percentile_outliers(samples_after_quality_filter)
            rows_after_outlier_removal = int(len(samples_after_outlier_removal))
            if rows_after_outlier_removal == 0:
                if outlier_removal_applied:
                    insufficient_mean_reason = (
                        f"{feature_label}: all {rows_after_quality_filter} quality-filtered sample(s) "
                        "were removed by 1st–99th percentile outlier removal."
                    )
                else:
                    insufficient_mean_reason = (
                        f"{feature_label}: no quality-filtered samples available for calculation."
                    )
                insufficient_std_reason = insufficient_mean_reason
            else:
                final_mean_count = rows_after_outlier_removal
                final_std_count = rows_after_outlier_removal
                computed_mean = _mean_only(samples_after_quality_filter)
                _, computed_std = _mean_and_std(samples_after_quality_filter)
                if computed_mean == INSUFFICIENT_DATA:
                    insufficient_mean_reason = (
                        f"{feature_label}: mean returned Insufficient Data despite "
                        f"{rows_after_outlier_removal} sample(s) after outlier removal."
                    )
                else:
                    insufficient_mean_reason = "—"
                if computed_std == INSUFFICIENT_DATA:
                    if final_std_count < 2:
                        insufficient_std_reason = (
                            f"{feature_label}: only {final_std_count} valid sample(s) after filtering; "
                            "variability requires at least 2 samples."
                        )
                    else:
                        insufficient_std_reason = (
                            f"{feature_label}: variability returned Insufficient Data despite "
                            f"{rows_after_outlier_removal} sample(s) after outlier removal."
                        )
                else:
                    insufficient_std_reason = "—"

    outlier_removal_applied = (
        rows_after_quality_filter >= OUTLIER_REMOVAL_MIN_SAMPLES
        if rows_after_quality_filter > 0
        else False
    )

    return {
        "rows_before_quality_filter": rows_before_quality_filter,
        "rows_after_quality_filter": rows_after_quality_filter,
        "rows_after_outlier_removal": rows_after_outlier_removal,
        "outlier_removal_applied": outlier_removal_applied,
        "outlier_removal_note": OUTLIER_REMOVAL_NOTE,
        "final_sample_count_used_for_mean": final_mean_count,
        "final_sample_count_used_for_std": final_std_count,
        "samples_after_quality_filter": _format_samples(samples_after_quality_filter),
        "samples_after_outlier_removal": _format_samples(samples_after_outlier_removal),
        "computed_mean": computed_mean,
        "computed_std": computed_std,
        "insufficient_data_reason_mean": insufficient_mean_reason,
        "insufficient_data_reason_std": insufficient_std_reason,
    }


def _table2_calculation_debug(
    participant_folder: Path,
    task_folder: Path,
    activity_segment: pd.DataFrame,
) -> dict[str, Any]:
    resting_folder = _resting_state_folder(participant_folder)
    resting_available = resting_folder is not None
    resting_segment = pd.DataFrame()
    if resting_folder is not None:
        resting_window = _parse_task_window(resting_folder)
        activity = _load_corsano_excel(participant_folder, ACTIVITY_FILE)
        if resting_window is not None and activity is not None:
            resting_segment, _ = _inspect_sparse_segment(activity, *resting_window)

    activity_features = (
        _activity_features(
            activity_segment,
            resting_segment,
            resting_available=resting_available,
        )
        if not activity_segment.empty
        else {
            "mean_bpm": INSUFFICIENT_DATA,
            "mean_respiration_rate": INSUFFICIENT_DATA,
            "respiration_variability": INSUFFICIENT_DATA,
        }
    )

    bpm_trace = _trace_activity_feature_calculation(
        activity_segment,
        value_col="bpm",
        q_col="bpm_q",
        q_min=BPM_QUALITY_MIN,
        feature_label="BPM",
    )
    resp_trace = _trace_activity_feature_calculation(
        activity_segment,
        value_col="respiration_rate",
        q_col="resp_q",
        q_min=RESP_QUALITY_MIN,
        feature_label="Respiration",
    )

    saved_table_2: dict[str, Any] = {}
    table_2_path = task_folder / "table_2_physiological_data.xlsx"
    if table_2_path.is_file():
        saved_df = pd.read_excel(table_2_path, engine="openpyxl")
        if not saved_df.empty:
            saved_table_2 = saved_df.iloc[0].to_dict()

    bpm_metrics = {
        **{k: v for k, v in bpm_trace.items() if not k.startswith("samples_")},
        "pipeline_mean_bpm": activity_features.get("mean_bpm"),
        "saved_table_2_mean_bpm": saved_table_2.get("mean_bpm", "—"),
        "samples_after_quality_filter": bpm_trace["samples_after_quality_filter"],
        "samples_after_outlier_removal": bpm_trace["samples_after_outlier_removal"],
    }
    resp_metrics = {
        **{k: v for k, v in resp_trace.items() if not k.startswith("samples_")},
        "pipeline_mean_respiration_rate": activity_features.get("mean_respiration_rate"),
        "pipeline_respiration_variability": activity_features.get("respiration_variability"),
        "saved_table_2_mean_respiration_rate": saved_table_2.get("mean_respiration_rate", "—"),
        "saved_table_2_respiration_variability": saved_table_2.get("respiration_variability", "—"),
        "samples_after_quality_filter": resp_trace["samples_after_quality_filter"],
        "samples_after_outlier_removal": resp_trace["samples_after_outlier_removal"],
    }

    return {
        "bpm": _metrics_table(bpm_metrics),
        "respiration": _metrics_table(resp_metrics),
    }


def build_corsano_debug_report(participant_folder: Path) -> dict[str, Any]:
    activity = _load_corsano_excel(participant_folder, ACTIVITY_FILE)
    hrv = _load_corsano_excel(participant_folder, HRV_FILE)
    acc = _load_corsano_excel(participant_folder, ACC_FILE)

    resting_folder = participant_folder / RESTING_STATE_TASK
    resting_found = resting_folder.is_dir() and load_sync_window_json(resting_folder) is not None
    resting_window = _parse_task_window(resting_folder) if resting_found else None
    resting_segments = {"activity_rows": 0, "hrv_rows": 0, "acc_rows": 0}

    if resting_window is not None:
        resting_start, resting_end = resting_window
        activity_rest, _ = _inspect_sparse_segment(activity, resting_start, resting_end)
        hrv_rest, _ = _inspect_sparse_segment(hrv, resting_start, resting_end)
        acc_rest, _ = _inspect_strict_segment(acc, resting_start, resting_end)
        resting_segments = {
            "activity_rows": int(len(activity_rest)),
            "hrv_rows": int(len(hrv_rest)),
            "acc_rows": int(len(acc_rest)),
        }

    task_reports: list[dict[str, Any]] = []
    for task_name, task_folder in _discover_task_folders(participant_folder):
        task_window = _parse_task_window(task_folder)
        if task_window is None:
            task_reports.append(
                {
                    "task_name": task_name,
                    "task_start_utc": "—",
                    "task_end_utc": "—",
                    "activity": _metrics_table({"error": "sync_window.json missing or invalid"}),
                    "hrv": _metrics_table({"error": "sync_window.json missing or invalid"}),
                    "acc": _metrics_table({"error": "sync_window.json missing or invalid"}),
                }
            )
            continue

        task_start, task_end = task_window
        activity_segment, activity_meta = _inspect_sparse_segment(activity, task_start, task_end)
        hrv_segment, hrv_meta = _inspect_sparse_segment(hrv, task_start, task_end)
        acc_segment, acc_meta = _inspect_strict_segment(acc, task_start, task_end)

        activity_metrics = {
            **activity_meta,
            **_activity_quality_counts(activity_segment),
        }
        hrv_metrics = {
            **hrv_meta,
            **_hrv_quality_counts(hrv_segment),
        }
        acc_metrics = {
            **acc_meta,
            "acc_complete_rows": _acc_complete_rows(acc_segment),
        }

        task_reports.append(
            {
                "task_name": task_name,
                "task_start_utc": _iso_timestamp(task_start),
                "task_end_utc": _iso_timestamp(task_end),
                "activity": _metrics_table(activity_metrics),
                "hrv": _metrics_table(hrv_metrics),
                "acc": _metrics_table(acc_metrics),
                "table2_calculation": _table2_calculation_debug(
                    participant_folder,
                    task_folder,
                    activity_segment,
                ),
            }
        )

    return {
        "resting_state_found": resting_found,
        "resting_state_task_start_utc": _iso_timestamp(resting_window[0]) if resting_window else "—",
        "resting_state_task_end_utc": _iso_timestamp(resting_window[1]) if resting_window else "—",
        "resting_state_segments": resting_segments,
        "tasks": task_reports,
    }
