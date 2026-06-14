"""
EEG Phase 10–16 — task-level EEG band-power, ratio, and baseline-normalized features.

Frontal theta (4–8 Hz), occipital alpha (8–12 Hz), ratios, and Resting state deltas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from backend.eeg.inspection import eeg_raw_dir, load_eeg_metadata_json
from backend.eeg.preprocessing import load_eeg_preprocessing_plan
from backend.eeg.preprocessing_exec import load_eeg_preprocessing_audit
from domain.feature_catalog import EEG_TASK_FEATURES
from domain.resting_state import eeg_baseline_change_zeros, is_resting_state_task
from domain.storage_layout import (
    EEG_PREPROCESSED_SEGMENT_FILE,
    TASK_LEVEL_EEG_FEATURES_FILE,
    TASK_LEVEL_EEG_FEATURES_JSON,
)

NOT_AVAILABLE = "Not available"
THETA_BAND_HZ = (4.0, 8.0)
ALPHA_BAND_HZ = (8.0, 12.0)
FEATURE_NAME = "mean_frontal_theta_power"
VARIABILITY_FEATURE_NAME = "theta_power_variability"
ALPHA_FEATURE_NAME = "mean_occipital_alpha_power"
ALPHA_VARIABILITY_FEATURE_NAME = "alpha_power_variability"
RATIO_FEATURE_NAME = "mean_theta_alpha_ratio"
RATIO_VARIABILITY_FEATURE_NAME = "theta_alpha_ratio_variability"
POWER_FEATURE_COLUMNS: tuple[str, ...] = (
    FEATURE_NAME,
    VARIABILITY_FEATURE_NAME,
    ALPHA_FEATURE_NAME,
    ALPHA_VARIABILITY_FEATURE_NAME,
    RATIO_FEATURE_NAME,
    RATIO_VARIABILITY_FEATURE_NAME,
)
BASELINE_CHANGE_FEATURE_COLUMNS: tuple[str, ...] = (
    "theta_power_change_from_baseline",
    "alpha_power_change_from_baseline",
    "theta_alpha_ratio_change_from_baseline",
)
PLV_FEATURE_COLUMNS: tuple[str, ...] = (
    "mean_occipital_temporal_theta_plv",
    "OT_plv_variability",
    "mean_temporal_frontal_theta_plv",
    "TF_plv_variability",
    "mean_occipital_frontal_theta_plv",
    "OF_plv_variability",
)
PLV_BASELINE_CHANGE_FEATURE_COLUMNS: tuple[str, ...] = (
    "OT_plv_change_from_baseline",
    "TF_plv_change_from_baseline",
    "OF_plv_change_from_baseline",
)
TASK_LEVEL_EEG_FEATURE_COLUMNS: tuple[str, ...] = (
    *POWER_FEATURE_COLUMNS,
    *BASELINE_CHANGE_FEATURE_COLUMNS,
    *PLV_FEATURE_COLUMNS,
    *PLV_BASELINE_CHANGE_FEATURE_COLUMNS,
)


def _empty_feature_payload() -> dict[str, Any]:
    return {feature_name: NOT_AVAILABLE for feature_name in POWER_FEATURE_COLUMNS}


def export_columns_for_task(task_name: str | None) -> tuple[str, ...]:
    return TASK_LEVEL_EEG_FEATURE_COLUMNS


def _unavailable_payload(error_message: str) -> dict[str, Any]:
    payload = _empty_feature_payload()
    payload["error_message"] = error_message
    return payload


def _sampling_rate_hz(task_folder: Path, audit: dict[str, Any] | None) -> float | None:
    if audit and audit.get("sampling_rate_hz"):
        return float(audit["sampling_rate_hz"])
    metadata = load_eeg_metadata_json(eeg_raw_dir(task_folder))
    if metadata:
        recordings = metadata.get("recordings") or []
        if recordings and recordings[0].get("sampling_rate_hz"):
            return float(recordings[0]["sampling_rate_hz"])
    return None


def _recording_channel_names(task_folder: Path, audit: dict[str, Any] | None, channel_count: int) -> list[str]:
    audit_channels = audit.get("channels_used") if audit else None
    if isinstance(audit_channels, list) and len(audit_channels) == channel_count:
        return [str(name) for name in audit_channels]

    metadata = load_eeg_metadata_json(eeg_raw_dir(task_folder))
    if metadata:
        recordings = metadata.get("recordings") or []
        if recordings:
            channels = recordings[0].get("channels") or []
            names = [
                str(row["name"])
                for row in channels
                if isinstance(row, dict) and row.get("name")
            ]
            if len(names) == channel_count:
                return names
    return [f"Ch{i + 1}" for i in range(channel_count)]


def _roi_indices(channel_names: list[str], roi_names: list[str]) -> list[int]:
    name_to_idx = {name: idx for idx, name in enumerate(channel_names)}
    return [name_to_idx[name] for name in roi_names if name in name_to_idx]


def _band_power(
    channel_data: np.ndarray,
    sampling_rate_hz: float,
    band_hz: tuple[float, float],
) -> float:
    col = np.asarray(channel_data, dtype=np.float64)
    finite = col[np.isfinite(col)]
    if finite.size < 4:
        return float("nan")

    nperseg = min(len(finite), max(int(sampling_rate_hz), 256))
    freqs, psd = signal.welch(finite, fs=sampling_rate_hz, nperseg=nperseg, axis=0)
    band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(band_mask):
        return float("nan")
    integrate = np.trapz if hasattr(np, "trapz") else np.trapezoid
    return float(integrate(psd[band_mask], freqs[band_mask]))


def _compute_roi_band_features(
    data: np.ndarray,
    channel_names: list[str],
    roi_names: list[str],
    sampling_rate_hz: float,
    band_hz: tuple[float, float],
) -> tuple[float | str, float | str]:
    if not roi_names:
        return NOT_AVAILABLE, NOT_AVAILABLE

    indices = _roi_indices(channel_names, [str(name) for name in roi_names])
    if not indices:
        return NOT_AVAILABLE, NOT_AVAILABLE

    channel_powers = [
        _band_power(data[:, idx], sampling_rate_hz, band_hz)
        for idx in indices
    ]
    finite_powers = [value for value in channel_powers if np.isfinite(value)]
    if not finite_powers:
        return NOT_AVAILABLE, NOT_AVAILABLE

    mean_power = float(np.mean(finite_powers))
    variability = (
        NOT_AVAILABLE
        if len(finite_powers) < 2
        else float(np.std(finite_powers))
    )
    return mean_power, variability


def _compute_mean_theta_alpha_ratio(
    mean_theta: float | str,
    mean_alpha: float | str,
) -> float | str:
    if mean_theta == NOT_AVAILABLE or mean_alpha == NOT_AVAILABLE:
        return NOT_AVAILABLE
    try:
        theta = float(mean_theta)
        alpha = float(mean_alpha)
    except (TypeError, ValueError):
        return NOT_AVAILABLE
    if alpha == 0.0 or not np.isfinite(theta) or not np.isfinite(alpha):
        return NOT_AVAILABLE
    return float(theta / alpha)


def _compute_theta_alpha_ratio_variability(
    theta_variability: float | str,
    alpha_variability: float | str,
) -> float | str:
    if theta_variability == NOT_AVAILABLE or alpha_variability == NOT_AVAILABLE:
        return NOT_AVAILABLE
    try:
        theta_var = float(theta_variability)
        alpha_var = float(alpha_variability)
    except (TypeError, ValueError):
        return NOT_AVAILABLE
    if alpha_var == 0.0 or not np.isfinite(theta_var) or not np.isfinite(alpha_var):
        return NOT_AVAILABLE
    return float(theta_var / alpha_var)


def compute_task_level_eeg_features(task_folder: Path) -> dict[str, Any]:
    """Return implemented task-level EEG band-power features."""
    segment_path = task_folder / EEG_PREPROCESSED_SEGMENT_FILE
    if not segment_path.is_file():
        return _unavailable_payload("eeg_preprocessed_segment.npy not found.")

    plan = load_eeg_preprocessing_plan(task_folder)
    if plan is None:
        return _unavailable_payload("eeg_preprocessing_plan.json not found.")

    audit = load_eeg_preprocessing_audit(task_folder)
    if audit is None or not audit.get("preprocessing_completed"):
        return _unavailable_payload("Preprocessing audit missing or preprocessing not completed.")

    try:
        data = np.load(segment_path)
    except (OSError, ValueError) as exc:
        return _unavailable_payload(str(exc))

    if data.ndim != 2:
        return _unavailable_payload(f"Expected 2-D preprocessed array, got shape {data.shape}.")

    sampling_rate_hz = _sampling_rate_hz(task_folder, audit)
    if sampling_rate_hz is None or sampling_rate_hz <= 0:
        return _unavailable_payload("Sampling rate unavailable.")

    roi_definitions = plan.get("roi_definitions") or {}
    channel_names = _recording_channel_names(task_folder, audit, int(data.shape[1]))

    mean_theta, theta_variability = _compute_roi_band_features(
        data,
        channel_names,
        roi_definitions.get("frontal") or [],
        sampling_rate_hz,
        THETA_BAND_HZ,
    )
    mean_alpha, alpha_variability = _compute_roi_band_features(
        data,
        channel_names,
        roi_definitions.get("occipital") or [],
        sampling_rate_hz,
        ALPHA_BAND_HZ,
    )

    return {
        FEATURE_NAME: mean_theta,
        VARIABILITY_FEATURE_NAME: theta_variability,
        ALPHA_FEATURE_NAME: mean_alpha,
        ALPHA_VARIABILITY_FEATURE_NAME: alpha_variability,
        RATIO_FEATURE_NAME: _compute_mean_theta_alpha_ratio(mean_theta, mean_alpha),
        RATIO_VARIABILITY_FEATURE_NAME: _compute_theta_alpha_ratio_variability(
            theta_variability,
            alpha_variability,
        ),
    }


def compute_mean_frontal_theta_power(task_folder: Path) -> dict[str, Any]:
    """Backward-compatible wrapper."""
    return compute_task_level_eeg_features(task_folder)


def merge_task_level_eeg_exports(
    payload: dict[str, Any],
    *,
    task_name: str | None = None,
) -> dict[str, float | str]:
    columns = export_columns_for_task(task_name)
    return {
        feature_name: payload.get(feature_name, NOT_AVAILABLE)
        for feature_name in columns
    }


def write_task_level_eeg_features(
    task_folder: Path,
    payload: dict[str, Any],
    *,
    task_name: str | None = None,
) -> tuple[Path, Path]:
    export = merge_task_level_eeg_exports(payload, task_name=task_name)
    columns = export_columns_for_task(task_name)
    xlsx_path = task_folder / TASK_LEVEL_EEG_FEATURES_FILE
    json_path = task_folder / TASK_LEVEL_EEG_FEATURES_JSON

    pd.DataFrame([export], columns=list(columns)).to_excel(
        xlsx_path,
        index=False,
        engine="openpyxl",
    )
    json_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    return xlsx_path, json_path


def sync_table_3_eeg_features(
    task_folder: Path,
    payload: dict[str, Any],
    *,
    task_name: str | None = None,
) -> Path:
    """Update table_3_eeg_data.xlsx with implemented EEG features only."""
    from backend.analysis.task_level_tables import TABLE_3_FILE, _empty_row, _save_table

    export = merge_task_level_eeg_exports(payload, task_name=task_name)
    table_path = task_folder / TABLE_3_FILE
    if table_path.is_file():
        table = pd.read_excel(table_path, engine="openpyxl")
        if table.empty:
            table = _empty_row(EEG_TASK_FEATURES)
    else:
        table = _empty_row(EEG_TASK_FEATURES)

    for column in EEG_TASK_FEATURES:
        if column not in table.columns:
            table[column] = None
    table = table.reindex(columns=list(EEG_TASK_FEATURES))
    if table.empty:
        table = _empty_row(EEG_TASK_FEATURES)

    for feature_name in export:
        table[feature_name] = table[feature_name].astype(object)
        table.loc[table.index[0], feature_name] = export[feature_name]
    _save_table(table, table_path)
    return table_path


def save_task_level_eeg_feature_payload(
    task_folder: Path,
    payload: dict[str, Any],
    *,
    task_name: str | None = None,
) -> dict[str, str]:
    resolved_task_name = task_name or task_folder.name
    if is_resting_state_task(resolved_task_name):
        payload = {**payload, **eeg_baseline_change_zeros()}
    xlsx_path, json_path = write_task_level_eeg_features(
        task_folder,
        payload,
        task_name=resolved_task_name,
    )
    table_3_path = sync_table_3_eeg_features(
        task_folder,
        payload,
        task_name=resolved_task_name,
    )
    return {
        "features_path": str(xlsx_path.resolve()),
        "features_json_path": str(json_path.resolve()),
        "table_3_path": str(table_3_path.resolve()),
    }


def run_task_level_eeg_features(
    task_folder: Path,
    *,
    participant_folder: Path | None = None,
    task_name: str | None = None,
) -> dict[str, Any]:
    _ = participant_folder
    resolved_task_name = task_name or task_folder.name
    payload = compute_task_level_eeg_features(task_folder)
    if is_resting_state_task(resolved_task_name):
        payload = {**payload, **eeg_baseline_change_zeros()}
    paths = save_task_level_eeg_feature_payload(
        task_folder,
        payload,
        task_name=resolved_task_name,
    )
    return {
        **paths,
        "features": merge_task_level_eeg_exports(payload, task_name=resolved_task_name),
    }


def load_task_level_eeg_features_json(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / TASK_LEVEL_EEG_FEATURES_JSON
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
