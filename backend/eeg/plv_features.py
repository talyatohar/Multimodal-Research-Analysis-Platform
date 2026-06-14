"""
EEG Phase 17–20 — theta PLV task-level features across ROI pairs.

Occipital–temporal, temporal–frontal, and occipital–frontal mean PLV and variability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from backend.eeg.baseline_linkage import _task_has_eeg
from backend.eeg.preprocessing import load_eeg_preprocessing_plan
from backend.eeg.preprocessing_exec import load_eeg_preprocessing_audit
from backend.eeg.task_level_features import (
    NOT_AVAILABLE,
    THETA_BAND_HZ,
    _recording_channel_names,
    _roi_indices,
    _sampling_rate_hz,
    load_task_level_eeg_features_json,
    merge_task_level_eeg_exports,
    save_task_level_eeg_feature_payload,
)
from domain.storage_layout import EEG_PREPROCESSED_SEGMENT_FILE

OT_PLV_FEATURE_NAME = "mean_occipital_temporal_theta_plv"
OT_PLV_VARIABILITY_FEATURE_NAME = "OT_plv_variability"
TF_PLV_FEATURE_NAME = "mean_temporal_frontal_theta_plv"
TF_PLV_VARIABILITY_FEATURE_NAME = "TF_plv_variability"
OF_PLV_FEATURE_NAME = "mean_occipital_frontal_theta_plv"
OF_PLV_VARIABILITY_FEATURE_NAME = "OF_plv_variability"
PLV_FEATURE_COLUMNS: tuple[str, ...] = (
    OT_PLV_FEATURE_NAME,
    OT_PLV_VARIABILITY_FEATURE_NAME,
    TF_PLV_FEATURE_NAME,
    TF_PLV_VARIABILITY_FEATURE_NAME,
    OF_PLV_FEATURE_NAME,
    OF_PLV_VARIABILITY_FEATURE_NAME,
)


def _unavailable_plv_payload() -> dict[str, str]:
    return {feature_name: NOT_AVAILABLE for feature_name in PLV_FEATURE_COLUMNS}


def _theta_bandpass(channel_data: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    col = np.asarray(channel_data, dtype=np.float64)
    nyquist = sampling_rate_hz / 2.0
    low_hz, high_hz = THETA_BAND_HZ
    if high_hz >= nyquist:
        raise ValueError(f"Theta band high cutoff {high_hz} Hz exceeds Nyquist ({nyquist} Hz).")
    b, a = signal.butter(4, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return signal.filtfilt(b, a, col, axis=0)


def _theta_instantaneous_phase(channel_data: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    theta_filtered = _theta_bandpass(channel_data, sampling_rate_hz)
    return np.angle(signal.hilbert(theta_filtered, axis=0))


def _pair_plv(phase_a: np.ndarray, phase_b: np.ndarray) -> float:
    mask = np.isfinite(phase_a) & np.isfinite(phase_b)
    if int(mask.sum()) < 4:
        return float("nan")
    phase_diff = phase_a[mask] - phase_b[mask]
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))


def _load_preprocessed_plv_context(
    task_folder: Path,
) -> tuple[np.ndarray, float, list[str], dict[str, Any]] | None:
    segment_path = task_folder / EEG_PREPROCESSED_SEGMENT_FILE
    if not segment_path.is_file():
        return None

    plan = load_eeg_preprocessing_plan(task_folder)
    if plan is None:
        return None

    audit = load_eeg_preprocessing_audit(task_folder)
    if audit is None or not audit.get("preprocessing_completed"):
        return None

    try:
        data = np.load(segment_path)
    except (OSError, ValueError):
        return None

    if data.ndim != 2:
        return None

    sampling_rate_hz = _sampling_rate_hz(task_folder, audit)
    if sampling_rate_hz is None or sampling_rate_hz <= 0:
        return None

    channel_names = _recording_channel_names(task_folder, audit, int(data.shape[1]))
    roi_definitions = plan.get("roi_definitions") or {}
    return data, float(sampling_rate_hz), channel_names, roi_definitions


def _roi_pair_plvs(
    data: np.ndarray,
    sampling_rate_hz: float,
    channel_names: list[str],
    roi_a: list[str],
    roi_b: list[str],
) -> list[float] | None:
    indices_a = _roi_indices(channel_names, [str(name) for name in roi_a])
    indices_b = _roi_indices(channel_names, [str(name) for name in roi_b])
    if len(indices_a) < 1 or len(indices_b) < 1:
        return None

    phases_a = [
        _theta_instantaneous_phase(data[:, idx], sampling_rate_hz)
        for idx in indices_a
    ]
    phases_b = [
        _theta_instantaneous_phase(data[:, idx], sampling_rate_hz)
        for idx in indices_b
    ]

    pair_plvs: list[float] = []
    for phase_a in phases_a:
        for phase_b in phases_b:
            plv_value = _pair_plv(phase_a, phase_b)
            if np.isfinite(plv_value):
                pair_plvs.append(plv_value)
    return pair_plvs


def _plv_mean_and_variability(pair_plvs: list[float] | None) -> tuple[float | str, float | str]:
    if pair_plvs is None or not pair_plvs:
        return NOT_AVAILABLE, NOT_AVAILABLE
    variability = (
        NOT_AVAILABLE
        if len(pair_plvs) < 2
        else float(np.std(pair_plvs))
    )
    return float(np.mean(pair_plvs)), variability


def compute_theta_plv_features(task_folder: Path) -> dict[str, float | str]:
    context = _load_preprocessed_plv_context(task_folder)
    if context is None:
        return _unavailable_plv_payload()

    data, sampling_rate_hz, channel_names, roi_definitions = context
    occipital_roi = roi_definitions.get("occipital") or []
    temporal_roi = roi_definitions.get("temporal") or []
    frontal_roi = roi_definitions.get("frontal") or []

    ot_pair_plvs = _roi_pair_plvs(
        data,
        sampling_rate_hz,
        channel_names,
        occipital_roi,
        temporal_roi,
    )
    tf_pair_plvs = _roi_pair_plvs(
        data,
        sampling_rate_hz,
        channel_names,
        temporal_roi,
        frontal_roi,
    )
    of_pair_plvs = _roi_pair_plvs(
        data,
        sampling_rate_hz,
        channel_names,
        occipital_roi,
        frontal_roi,
    )

    ot_mean, ot_variability = _plv_mean_and_variability(ot_pair_plvs)
    tf_mean, tf_variability = _plv_mean_and_variability(tf_pair_plvs)
    of_mean, of_variability = _plv_mean_and_variability(of_pair_plvs)

    return {
        OT_PLV_FEATURE_NAME: ot_mean,
        OT_PLV_VARIABILITY_FEATURE_NAME: ot_variability,
        TF_PLV_FEATURE_NAME: tf_mean,
        TF_PLV_VARIABILITY_FEATURE_NAME: tf_variability,
        OF_PLV_FEATURE_NAME: of_mean,
        OF_PLV_VARIABILITY_FEATURE_NAME: of_variability,
    }


def compute_occipital_temporal_theta_plv_features(task_folder: Path) -> dict[str, float | str]:
    features = compute_theta_plv_features(task_folder)
    return {
        OT_PLV_FEATURE_NAME: features[OT_PLV_FEATURE_NAME],
        OT_PLV_VARIABILITY_FEATURE_NAME: features[OT_PLV_VARIABILITY_FEATURE_NAME],
    }


def compute_mean_occipital_temporal_theta_plv(task_folder: Path) -> dict[str, float | str]:
    return compute_occipital_temporal_theta_plv_features(task_folder)


def run_theta_plv_features(task_folder: Path, task_name: str) -> dict[str, Any] | None:
    """Merge theta PLV features into existing task-level feature exports."""
    if not _task_has_eeg(task_folder):
        return None

    current_features = load_task_level_eeg_features_json(task_folder)
    if current_features is None:
        return None

    plv_features = compute_theta_plv_features(task_folder)
    payload = {**current_features, **plv_features}
    paths = save_task_level_eeg_feature_payload(task_folder, payload, task_name=task_name)
    return {
        "task_name": task_name,
        **paths,
        "features": merge_task_level_eeg_exports(payload, task_name=task_name),
    }


def run_occipital_temporal_theta_plv(task_folder: Path, task_name: str) -> dict[str, Any] | None:
    return run_theta_plv_features(task_folder, task_name)


def refresh_participant_plv_features(participant_folder: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not participant_folder.is_dir():
        return results

    for task_folder in sorted(participant_folder.iterdir()):
        if not task_folder.is_dir():
            continue
        updated = run_theta_plv_features(task_folder, task_folder.name)
        if updated is not None:
            results.append(updated)
    return results
