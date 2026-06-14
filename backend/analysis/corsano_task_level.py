"""Corsano physiological (Table 2) and motion QC (Table 4) task-level features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.sync.eprime import load_sync_window_json
from domain.feature_catalog import PHYSIO_TASK_FEATURES, QC_TASK_FEATURES
from domain.storage_layout import TASK_COMPREHENSION_FILE
from domain.resting_state import (
    RESTING_STATE_TASK,
    is_resting_state_task,
    physio_baseline_change_zeros,
)
INSUFFICIENT_DATA = "Insufficient Data"
MISSING_RESTING_STATE = "Missing Resting State"
NOT_AVAILABLE = "Not Available"
SPARSE_BOUNDARY_SECONDS = 60.0
BPM_QUALITY_MIN = 3
RESP_QUALITY_MIN = 3
OUTLIER_REMOVAL_MIN_SAMPLES = 10
OUTLIER_REMOVAL_NOTE = (
    "Outlier removal is applied only when at least 10 valid samples are available."
)
SPARSE_TIMESTAMP_WARNING = "Corsano file has sparse timestamps; nearest available sample was used."
BASELINE_MISSING_WARNING = "Baseline comparison unavailable: Missing Resting State task."

ACTIVITY_FILE = "activity.xlsx"
HRV_FILE = "heart_rate_variability.xlsx"
ACC_FILE = "acc.xlsx"

VALID_LEFT_COL = "Validity left"
VALID_RIGHT_COL = "Validity right"
SEGMENT_FILE_BY_WINDOW = {
    "eprime": "processed_eye_tracking_segment_eprime.xlsx",
    "manual": "processed_eye_tracking_segment_manual.xlsx",
}


@dataclass
class CorsanoTaskLevelResult:
    physiology_row: dict[str, Any]
    quality_control_row: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _parse_task_window(task_folder: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    window = load_sync_window_json(task_folder)
    if not window:
        return None
    start = pd.to_datetime(window.get("task_start_utc"), errors="coerce", utc=True)
    end = pd.to_datetime(window.get("task_end_utc"), errors="coerce", utc=True)
    if pd.isna(start) or pd.isna(end):
        return None
    return start, end


def _load_corsano_excel(participant_folder: Path, filename: str) -> pd.DataFrame | None:
    path = participant_folder / filename
    if not path.is_file():
        return None
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except (OSError, ValueError):
        return None
    if "timestamp" not in df.columns:
        return None
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True, errors="coerce")
    out = out.loc[out["timestamp_utc"].notna()].copy()
    return out


def _append_sparse_timestamp_warning(warnings: list[str]) -> None:
    if SPARSE_TIMESTAMP_WARNING not in warnings:
        warnings.append(SPARSE_TIMESTAMP_WARNING)


def _segment_corsano_strict(
    df: pd.DataFrame | None,
    task_start: pd.Timestamp,
    task_end: pd.Timestamp,
    warnings: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        warnings.append(f"Physiological Data: missing or empty {label}.")
        return pd.DataFrame()

    segmented = df.loc[
        (df["timestamp_utc"] >= task_start) & (df["timestamp_utc"] <= task_end)
    ].copy()
    if segmented.empty:
        warnings.append(f"Physiological Data: no {label} samples inside the task window.")
    return segmented


def _expanded_sparse_boundary(
    df: pd.DataFrame,
    task_start: pd.Timestamp,
    task_end: pd.Timestamp,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    expanded_start: pd.Timestamp | None = None
    before_start = df.loc[df["timestamp_utc"] <= task_start]
    if not before_start.empty:
        last_before = before_start["timestamp_utc"].max()
        if (task_start - last_before).total_seconds() <= SPARSE_BOUNDARY_SECONDS:
            expanded_start = last_before

    expanded_end: pd.Timestamp | None = None
    after_end = df.loc[df["timestamp_utc"] >= task_end]
    if not after_end.empty:
        first_after = after_end["timestamp_utc"].min()
        if (first_after - task_end).total_seconds() <= SPARSE_BOUNDARY_SECONDS:
            expanded_end = first_after

    return expanded_start, expanded_end


def _segment_corsano_sparse(
    df: pd.DataFrame | None,
    task_start: pd.Timestamp,
    task_end: pd.Timestamp,
    warnings: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        warnings.append(f"Physiological Data: missing or empty {label}.")
        return pd.DataFrame()

    strict = df.loc[
        (df["timestamp_utc"] >= task_start) & (df["timestamp_utc"] <= task_end)
    ].copy()
    if not strict.empty:
        return strict

    expanded_start, expanded_end = _expanded_sparse_boundary(df, task_start, task_end)
    if expanded_start is None or expanded_end is None or expanded_start > expanded_end:
        warnings.append(
            f"Physiological Data: no {label} samples within the task window or expanded 60-second boundaries."
        )
        return pd.DataFrame()

    segmented = df.loc[
        (df["timestamp_utc"] >= expanded_start) & (df["timestamp_utc"] <= expanded_end)
    ].copy()
    if segmented.empty:
        warnings.append(
            f"Physiological Data: no {label} samples within the task window or expanded 60-second boundaries."
        )
        return pd.DataFrame()

    _append_sparse_timestamp_warning(warnings)
    return segmented


def _trim_percentile_outliers(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty or len(numeric) < OUTLIER_REMOVAL_MIN_SAMPLES:
        return numeric
    low, high = numeric.quantile([0.01, 0.99])
    return numeric[(numeric >= low) & (numeric <= high)]


def _mean_and_std(series: pd.Series) -> tuple[float | str, float | str]:
    trimmed = _trim_percentile_outliers(series)
    if trimmed.empty:
        return INSUFFICIENT_DATA, INSUFFICIENT_DATA
    mean_val = float(trimmed.mean())
    if len(trimmed) < 2:
        return mean_val, INSUFFICIENT_DATA
    return mean_val, float(trimmed.std(ddof=1))


def _mean_only(series: pd.Series) -> float | str:
    trimmed = _trim_percentile_outliers(series)
    if trimmed.empty:
        return INSUFFICIENT_DATA
    return float(trimmed.mean())


def _change_from_baseline(
    task_value: float | str,
    resting_value: float | str,
    *,
    resting_available: bool,
) -> float | str:
    if not resting_available:
        return MISSING_RESTING_STATE
    if task_value == INSUFFICIENT_DATA or resting_value == INSUFFICIENT_DATA:
        return INSUFFICIENT_DATA
    if isinstance(task_value, str) or isinstance(resting_value, str):
        return INSUFFICIENT_DATA
    return float(task_value) - float(resting_value)


def _motion_magnitude(df: pd.DataFrame) -> pd.Series:
    required = ("accX", "accY", "accZ")
    if df.empty or any(column not in df.columns for column in required):
        return pd.Series(dtype=float)
    x = pd.to_numeric(df["accX"], errors="coerce")
    y = pd.to_numeric(df["accY"], errors="coerce")
    z = pd.to_numeric(df["accZ"], errors="coerce")
    return np.sqrt(x**2 + y**2 + z**2).dropna()


def _is_numeric_like(series: pd.Series) -> bool:
    converted = pd.to_numeric(series.dropna(), errors="coerce")
    return not converted.empty and converted.notna().all()


def _validity_mask(df: pd.DataFrame, col: str, warnings: list[str]) -> pd.Series | None:
    if col not in df.columns:
        warnings.append(f"Quality Control: missing '{col}'; validity percentage unavailable.")
        return None
    series = df[col]
    if _is_numeric_like(series):
        return pd.to_numeric(series, errors="coerce") == 0
    text = series.astype("string").str.strip().str.casefold()
    valid_tokens = {"valid", "true", "yes", "ok", "0"}
    invalid_tokens = {"invalid", "false", "no", "1", "2", "3", "4"}
    recognised = text.isin(valid_tokens | invalid_tokens)
    if not bool(recognised.any()):
        warnings.append(f"Quality Control: could not recognise validity coding in '{col}'.")
        return None
    return text.isin(valid_tokens)


def _validity_percentage(mask: pd.Series | None, total_rows: int) -> float | str:
    if mask is None or total_rows == 0:
        return NOT_AVAILABLE
    return float(mask.sum() / total_rows * 100.0)


def _or_validity_mask(left: pd.Series | None, right: pd.Series | None) -> pd.Series | None:
    if left is not None and right is not None:
        return left | right
    if left is not None:
        return left
    if right is not None:
        return right
    return None


def _eye_tracking_segment_path(task_folder: Path, window_type: str) -> Path | None:
    preferred = task_folder / SEGMENT_FILE_BY_WINDOW.get(window_type, SEGMENT_FILE_BY_WINDOW["eprime"])
    if preferred.is_file():
        return preferred
    legacy = task_folder / "processed_eye_tracking_segment.xlsx"
    return legacy if legacy.is_file() else None


def _eye_validity_percentages(
    task_folder: Path,
    *,
    window_type: str,
    warnings: list[str],
) -> tuple[float | str, float | str, float | str]:
    segment_path = _eye_tracking_segment_path(task_folder, window_type)
    if segment_path is None:
        warnings.append(
            "Quality Control: processed eye-tracking segment missing; validity percentages unavailable."
        )
        return NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE
    try:
        df = pd.read_excel(segment_path, engine="openpyxl")
    except (OSError, ValueError) as exc:
        warnings.append(f"Quality Control: could not read eye-tracking segment ({exc}).")
        return NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE
    if df.empty:
        warnings.append("Quality Control: processed eye-tracking segment is empty.")
        return NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE

    total_rows = len(df)
    left_valid = _validity_mask(df, VALID_LEFT_COL, warnings)
    right_valid = _validity_mask(df, VALID_RIGHT_COL, warnings)
    combined_valid = _or_validity_mask(left_valid, right_valid)
    return (
        _validity_percentage(left_valid, total_rows),
        _validity_percentage(right_valid, total_rows),
        _validity_percentage(combined_valid, total_rows),
    )


def _reading_comprehension_score(task_folder: Path, warnings: list[str]) -> float | str:
    path = task_folder / TASK_COMPREHENSION_FILE
    if not path.is_file():
        warnings.append(
            f"Quality Control: missing {TASK_COMPREHENSION_FILE}; Reading_comprehension_assessment_score unavailable."
        )
        return NOT_AVAILABLE
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        warnings.append(f"Quality Control: {TASK_COMPREHENSION_FILE} is empty.")
        return NOT_AVAILABLE
    try:
        return float(text)
    except ValueError:
        warnings.append(f"Quality Control: could not parse {TASK_COMPREHENSION_FILE} as a number.")
        return NOT_AVAILABLE


def _resting_state_folder(participant_folder: Path) -> Path | None:
    resting_folder = participant_folder / RESTING_STATE_TASK
    if not resting_folder.is_dir():
        return None
    if load_sync_window_json(resting_folder) is None:
        return None
    return resting_folder


def _filtered_activity_column(
    segment: pd.DataFrame,
    q_col: str,
    value_col: str,
    q_min: int,
) -> pd.Series:
    if segment.empty or q_col not in segment.columns or value_col not in segment.columns:
        return pd.Series(dtype=float)
    quality = pd.to_numeric(segment[q_col], errors="coerce")
    return pd.to_numeric(segment.loc[quality >= q_min, value_col], errors="coerce").dropna()


def _activity_features(
    activity_segment: pd.DataFrame,
    resting_segment: pd.DataFrame | None,
    *,
    resting_available: bool,
) -> dict[str, float | str]:
    bpm = _filtered_activity_column(activity_segment, "bpm_q", "bpm", BPM_QUALITY_MIN)
    resp = _filtered_activity_column(activity_segment, "resp_q", "respiration_rate", RESP_QUALITY_MIN)

    mean_bpm = _mean_only(bpm)
    mean_resp, resp_var = _mean_and_std(resp)

    resting_mean_bpm = (
        _mean_only(_filtered_activity_column(resting_segment, "bpm_q", "bpm", BPM_QUALITY_MIN))
        if resting_segment is not None and not resting_segment.empty
        else INSUFFICIENT_DATA
    )
    resting_mean_resp, _ = (
        _mean_and_std(_filtered_activity_column(resting_segment, "resp_q", "respiration_rate", RESP_QUALITY_MIN))
        if resting_segment is not None and not resting_segment.empty
        else (INSUFFICIENT_DATA, INSUFFICIENT_DATA)
    )

    return {
        "mean_bpm": mean_bpm,
        "bpm_change_from_baseline": _change_from_baseline(
            mean_bpm,
            resting_mean_bpm,
            resting_available=resting_available,
        ),
        "mean_respiration_rate": mean_resp,
        "respiration_variability": resp_var,
        "respiration_change_from_baseline": _change_from_baseline(
            mean_resp,
            resting_mean_resp,
            resting_available=resting_available,
        ),
    }


def _hrv_features(
    hrv_segment: pd.DataFrame,
    resting_segment: pd.DataFrame | None,
    *,
    resting_available: bool,
) -> dict[str, float | str]:
    rmssd = pd.to_numeric(hrv_segment.get("rmssd"), errors="coerce").dropna()
    si = pd.to_numeric(hrv_segment.get("si"), errors="coerce").dropna()
    mean_rmssd, rmssd_var = _mean_and_std(rmssd)
    mean_si, si_var = _mean_and_std(si)

    resting_mean_rmssd, _ = (
        _mean_and_std(pd.to_numeric(resting_segment.get("rmssd"), errors="coerce").dropna())
        if resting_segment is not None and not resting_segment.empty
        else (INSUFFICIENT_DATA, INSUFFICIENT_DATA)
    )
    resting_mean_si, _ = (
        _mean_and_std(pd.to_numeric(resting_segment.get("si"), errors="coerce").dropna())
        if resting_segment is not None and not resting_segment.empty
        else (INSUFFICIENT_DATA, INSUFFICIENT_DATA)
    )

    return {
        "mean_rmssd": mean_rmssd,
        "rmssd_variability": rmssd_var,
        "rmssd_change_from_baseline": _change_from_baseline(
            mean_rmssd,
            resting_mean_rmssd,
            resting_available=resting_available,
        ),
        "mean_si": mean_si,
        "si_variability": si_var,
        "si_change_from_baseline": _change_from_baseline(
            mean_si,
            resting_mean_si,
            resting_available=resting_available,
        ),
    }


def _motion_features(
    acc_segment: pd.DataFrame,
    resting_acc_segment: pd.DataFrame | None,
    *,
    resting_available: bool,
    warnings: list[str],
) -> dict[str, float | str]:
    magnitude = _motion_magnitude(acc_segment)
    if magnitude.empty:
        return {
            "mean_motion_magnitude": INSUFFICIENT_DATA,
            "motion_variability": INSUFFICIENT_DATA,
            "high_motion_percentage": INSUFFICIENT_DATA,
        }

    mean_motion = float(magnitude.mean())
    motion_var = float(magnitude.std(ddof=1)) if len(magnitude) > 1 else 0.0

    if not resting_available or resting_acc_segment is None or resting_acc_segment.empty:
        warnings.append(
            "Quality Control: Resting state missing; high_motion_percentage uses Missing Resting State."
        )
        return {
            "mean_motion_magnitude": mean_motion,
            "motion_variability": motion_var,
            "high_motion_percentage": MISSING_RESTING_STATE,
        }

    resting_magnitude = _motion_magnitude(resting_acc_segment)
    if resting_magnitude.empty:
        return {
            "mean_motion_magnitude": mean_motion,
            "motion_variability": motion_var,
            "high_motion_percentage": MISSING_RESTING_STATE,
        }

    resting_mean = float(resting_magnitude.mean())
    resting_std = float(resting_magnitude.std(ddof=1)) if len(resting_magnitude) > 1 else 0.0
    threshold = resting_mean + 2.0 * resting_std
    high_motion_pct = float((magnitude > threshold).sum() / len(magnitude) * 100.0)
    return {
        "mean_motion_magnitude": mean_motion,
        "motion_variability": motion_var,
        "high_motion_percentage": high_motion_pct,
    }


def build_corsano_task_level_features(
    task_folder: Path,
    *,
    window_type: str = "eprime",
) -> CorsanoTaskLevelResult:
    warnings: list[str] = []
    participant_folder = task_folder.parent
    task_window = _parse_task_window(task_folder)
    if task_window is None:
        warnings.append("Physiological Data: sync_window.json missing or invalid task_start_utc/task_end_utc.")
        physiology_row = {feature: INSUFFICIENT_DATA for feature in PHYSIO_TASK_FEATURES}
        left_pct, right_pct, combined_pct = _eye_validity_percentages(
            task_folder,
            window_type=window_type,
            warnings=warnings,
        )
        qc_row = {
            "left_valid_percentage": left_pct,
            "right_valid_percentage": right_pct,
            "combined_valid_percentage": combined_pct,
            "mean_motion_magnitude": INSUFFICIENT_DATA,
            "motion_variability": INSUFFICIENT_DATA,
            "high_motion_percentage": MISSING_RESTING_STATE,
            "Reading_comprehension_assessment_score": _reading_comprehension_score(task_folder, warnings),
        }
        return CorsanoTaskLevelResult(
            physiology_row=physiology_row,
            quality_control_row=qc_row,
            warnings=warnings,
        )

    task_start, task_end = task_window
    resting_folder = _resting_state_folder(participant_folder)
    resting_available = resting_folder is not None
    if not resting_available:
        warnings.append(BASELINE_MISSING_WARNING)

    activity = _load_corsano_excel(participant_folder, ACTIVITY_FILE)
    hrv = _load_corsano_excel(participant_folder, HRV_FILE)
    acc = _load_corsano_excel(participant_folder, ACC_FILE)

    activity_segment = _segment_corsano_sparse(activity, task_start, task_end, warnings, label=ACTIVITY_FILE)
    hrv_segment = _segment_corsano_sparse(hrv, task_start, task_end, warnings, label=HRV_FILE)
    acc_segment = _segment_corsano_strict(acc, task_start, task_end, warnings, label=ACC_FILE)

    resting_activity_segment = pd.DataFrame()
    resting_hrv_segment = pd.DataFrame()
    resting_acc_segment = pd.DataFrame()
    if resting_folder is not None:
        resting_window = _parse_task_window(resting_folder)
        if resting_window is not None:
            resting_start, resting_end = resting_window
            resting_activity = _load_corsano_excel(participant_folder, ACTIVITY_FILE)
            resting_hrv = _load_corsano_excel(participant_folder, HRV_FILE)
            resting_acc = _load_corsano_excel(participant_folder, ACC_FILE)
            resting_activity_segment = _segment_corsano_sparse(
                resting_activity,
                resting_start,
                resting_end,
                warnings,
                label=f"Resting state {ACTIVITY_FILE}",
            )
            resting_hrv_segment = _segment_corsano_sparse(
                resting_hrv,
                resting_start,
                resting_end,
                warnings,
                label=f"Resting state {HRV_FILE}",
            )
            resting_acc_segment = _segment_corsano_strict(
                resting_acc,
                resting_start,
                resting_end,
                warnings,
                label=f"Resting state {ACC_FILE}",
            )

    physiology_row: dict[str, Any] = {feature: INSUFFICIENT_DATA for feature in PHYSIO_TASK_FEATURES}
    if not activity_segment.empty:
        physiology_row.update(
            _activity_features(
                activity_segment,
                resting_activity_segment,
                resting_available=resting_available,
            )
        )
    if not hrv_segment.empty:
        physiology_row.update(
            _hrv_features(
                hrv_segment,
                resting_hrv_segment,
                resting_available=resting_available,
            )
        )
    if not resting_available:
        for change_feature in (
            "bpm_change_from_baseline",
            "respiration_change_from_baseline",
            "rmssd_change_from_baseline",
            "si_change_from_baseline",
        ):
            physiology_row[change_feature] = MISSING_RESTING_STATE

    left_pct, right_pct, combined_pct = _eye_validity_percentages(
        task_folder,
        window_type=window_type,
        warnings=warnings,
    )
    motion_row = _motion_features(
        acc_segment,
        resting_acc_segment,
        resting_available=resting_available,
        warnings=warnings,
    )
    qc_row = {
        "left_valid_percentage": left_pct,
        "right_valid_percentage": right_pct,
        "combined_valid_percentage": combined_pct,
        **motion_row,
        "Reading_comprehension_assessment_score": _reading_comprehension_score(task_folder, warnings),
    }
    if is_resting_state_task(task_folder.name):
        physiology_row.update(physio_baseline_change_zeros())
    return CorsanoTaskLevelResult(
        physiology_row=physiology_row,
        quality_control_row=qc_row,
        warnings=warnings,
    )

