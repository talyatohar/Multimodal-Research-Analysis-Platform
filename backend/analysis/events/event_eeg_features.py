"""
Event-Level Phase 4 — per-event EEG feature extraction from aligned epochs.

Reuses task-level theta/alpha band-power and theta PLV methods and ROI definitions.
Writes event_level_eeg_features.xlsx only (no aggregation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.analysis.events.eeg_alignment import (
    EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE,
    STATUS_OUTSIDE,
    load_event_database_with_eeg_alignment,
)
from backend.analysis.events.eeg_epoch_extraction import (
    STATUS_EXTRACTED,
    load_event_eeg_epochs_metadata,
)
from backend.eeg.plv_features import _plv_mean_and_variability, _roi_pair_plvs
from backend.eeg.preprocessing import load_eeg_preprocessing_plan
from backend.eeg.preprocessing_exec import load_eeg_preprocessing_audit
from backend.eeg.task_level_features import (
    ALPHA_BAND_HZ,
    NOT_AVAILABLE,
    THETA_BAND_HZ,
    _compute_mean_theta_alpha_ratio,
    _compute_roi_band_features,
    _recording_channel_names,
    _sampling_rate_hz,
)
from domain.feature_catalog import EVENT_EEG_FEATURES
from domain.storage_layout import EEG_PREPROCESSED_SEGMENT_FILE

EVENT_LEVEL_EEG_FEATURES_FILE = "event_level_eeg_features.xlsx"

EVENT_LEVEL_EEG_FEATURE_COLUMNS: tuple[str, ...] = (
    "participant_id",
    "task_name",
    "event_type",
    "event_id",
    *EVENT_EEG_FEATURES,
)


def _extractable_events(
    aligned_database: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    if metadata.empty or "epoch_extraction_status" not in metadata.columns:
        return pd.DataFrame()

    extracted = metadata.loc[
        metadata["epoch_extraction_status"].astype("string") == STATUS_EXTRACTED
    ].copy()
    if extracted.empty:
        return extracted

    if aligned_database.empty:
        return extracted

    join_cols = ["event_id", "event_type"]
    aligned = aligned_database.copy()
    if "eeg_alignment_status" in aligned.columns:
        aligned = aligned.loc[aligned["eeg_alignment_status"].astype("string") != STATUS_OUTSIDE]

    context_cols = [
        column
        for column in ("participant_id", "task_name", "event_id", "event_type")
        if column in aligned.columns
    ]
    if not context_cols:
        return extracted

    return extracted.merge(
        aligned[context_cols].drop_duplicates(),
        on=join_cols,
        how="inner",
    )


def _compute_single_epoch_features(
    epoch: np.ndarray,
    *,
    channel_names: list[str],
    roi_definitions: dict[str, Any],
    sampling_rate_hz: float,
) -> dict[str, float | str]:
    frontal_roi = roi_definitions.get("frontal") or []
    occipital_roi = roi_definitions.get("occipital") or []
    temporal_roi = roi_definitions.get("temporal") or []

    mean_theta, _ = _compute_roi_band_features(
        epoch,
        channel_names,
        frontal_roi,
        sampling_rate_hz,
        THETA_BAND_HZ,
    )
    mean_alpha, _ = _compute_roi_band_features(
        epoch,
        channel_names,
        occipital_roi,
        sampling_rate_hz,
        ALPHA_BAND_HZ,
    )

    ot_mean, _ = _plv_mean_and_variability(
        _roi_pair_plvs(
            epoch,
            sampling_rate_hz,
            channel_names,
            occipital_roi,
            temporal_roi,
        )
    )
    tf_mean, _ = _plv_mean_and_variability(
        _roi_pair_plvs(
            epoch,
            sampling_rate_hz,
            channel_names,
            temporal_roi,
            frontal_roi,
        )
    )
    of_mean, _ = _plv_mean_and_variability(
        _roi_pair_plvs(
            epoch,
            sampling_rate_hz,
            channel_names,
            occipital_roi,
            frontal_roi,
        )
    )

    return {
        "frontal_theta_power": mean_theta,
        "occipital_alpha_power": mean_alpha,
        "frontal_theta_occipital_alpha_ratio": _compute_mean_theta_alpha_ratio(mean_theta, mean_alpha),
        "theta_PLV_occipital_temporal": ot_mean,
        "theta_PLV_temporal_frontal": tf_mean,
        "theta_PLV_occipital_frontal": of_mean,
    }


def build_event_level_eeg_features(
    task_folder: Path,
    participant_id: str,
    task_name: str,
    *,
    aligned_database: pd.DataFrame | None = None,
    epoch_metadata: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []

    if aligned_database is None:
        aligned_database = load_event_database_with_eeg_alignment(task_folder)
    if aligned_database is None or aligned_database.empty:
        warnings.append(f"{EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE} missing or empty — event EEG features skipped.")
        return pd.DataFrame(columns=list(EVENT_LEVEL_EEG_FEATURE_COLUMNS)), warnings

    if epoch_metadata is None:
        epoch_metadata = load_event_eeg_epochs_metadata(task_folder)
    if epoch_metadata is None or epoch_metadata.empty:
        warnings.append("event_eeg_epochs_metadata.xlsx missing or empty — event EEG features skipped.")
        return pd.DataFrame(columns=list(EVENT_LEVEL_EEG_FEATURE_COLUMNS)), warnings

    events = _extractable_events(aligned_database, epoch_metadata)
    if events.empty:
        return pd.DataFrame(columns=list(EVENT_LEVEL_EEG_FEATURE_COLUMNS)), warnings

    plan = load_eeg_preprocessing_plan(task_folder)
    if plan is None:
        warnings.append("eeg_preprocessing_plan.json missing — event EEG features skipped.")
        return pd.DataFrame(columns=list(EVENT_LEVEL_EEG_FEATURE_COLUMNS)), warnings

    audit = load_eeg_preprocessing_audit(task_folder)
    if audit is None or not audit.get("preprocessing_completed"):
        message = audit.get("error_message") if audit else "audit missing"
        warnings.append(f"EEG preprocessing incomplete — event EEG features skipped ({message}).")
        return pd.DataFrame(columns=list(EVENT_LEVEL_EEG_FEATURE_COLUMNS)), warnings

    segment_path = task_folder / EEG_PREPROCESSED_SEGMENT_FILE
    if not segment_path.is_file():
        warnings.append(f"{EEG_PREPROCESSED_SEGMENT_FILE} missing — event EEG features skipped.")
        return pd.DataFrame(columns=list(EVENT_LEVEL_EEG_FEATURE_COLUMNS)), warnings

    try:
        segment = np.load(segment_path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        warnings.append(f"Could not load {EEG_PREPROCESSED_SEGMENT_FILE}: {exc}")
        return pd.DataFrame(columns=list(EVENT_LEVEL_EEG_FEATURE_COLUMNS)), warnings

    if segment.ndim != 2:
        warnings.append(f"Expected 2-D preprocessed array, got shape {segment.shape}.")
        return pd.DataFrame(columns=list(EVENT_LEVEL_EEG_FEATURE_COLUMNS)), warnings

    sampling_rate_hz = _sampling_rate_hz(task_folder, audit)
    if sampling_rate_hz is None or sampling_rate_hz <= 0:
        warnings.append("Sampling rate unavailable — event EEG features skipped.")
        return pd.DataFrame(columns=list(EVENT_LEVEL_EEG_FEATURE_COLUMNS)), warnings

    channel_names = _recording_channel_names(task_folder, audit, int(segment.shape[1]))
    roi_definitions = plan.get("roi_definitions") or {}
    n_samples = int(segment.shape[0])

    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        row: dict[str, Any] = {
            "participant_id": event.get("participant_id") or participant_id,
            "task_name": event.get("task_name") or task_name,
            "event_type": event.get("event_type"),
            "event_id": event.get("event_id"),
        }
        try:
            start = int(event["eeg_window_start_sample"])
            end = int(event["eeg_window_end_sample"])
        except (TypeError, ValueError):
            row.update({feature: NOT_AVAILABLE for feature in EVENT_EEG_FEATURES})
            rows.append(row)
            continue

        if start < 0 or end < start or end >= n_samples:
            row.update({feature: NOT_AVAILABLE for feature in EVENT_EEG_FEATURES})
            rows.append(row)
            continue

        try:
            epoch = np.asarray(segment[start : end + 1])
            row.update(
                _compute_single_epoch_features(
                    epoch,
                    channel_names=channel_names,
                    roi_definitions=roi_definitions,
                    sampling_rate_hz=float(sampling_rate_hz),
                )
            )
        except (ValueError, TypeError):
            row.update({feature: NOT_AVAILABLE for feature in EVENT_EEG_FEATURES})

        rows.append(row)

    return pd.DataFrame(rows, columns=list(EVENT_LEVEL_EEG_FEATURE_COLUMNS)), warnings


def write_event_level_eeg_features(task_folder: Path, features: pd.DataFrame) -> Path:
    out = task_folder / EVENT_LEVEL_EEG_FEATURES_FILE
    features.to_excel(out, index=False, engine="openpyxl")
    return out


def load_event_level_eeg_features(task_folder: Path) -> pd.DataFrame | None:
    path = task_folder / EVENT_LEVEL_EEG_FEATURES_FILE
    if not path.is_file():
        return None
    return pd.read_excel(path, engine="openpyxl")


def run_event_level_eeg_features(
    task_folder: Path,
    participant_id: str,
    task_name: str,
    *,
    aligned_database: pd.DataFrame | None = None,
    epoch_metadata: pd.DataFrame | None = None,
    force_recompute: bool = False,
) -> dict[str, Any]:
    out_path = task_folder / EVENT_LEVEL_EEG_FEATURES_FILE

    if not force_recompute and out_path.is_file() and aligned_database is None and epoch_metadata is None:
        loaded = load_event_level_eeg_features(task_folder)
        if loaded is not None:
            return {
                "features": loaded,
                "features_path": str(out_path.resolve()),
                "warnings": [],
                "loaded_existing": True,
            }

    features, warnings = build_event_level_eeg_features(
        task_folder,
        participant_id,
        task_name,
        aligned_database=aligned_database,
        epoch_metadata=epoch_metadata,
    )
    if features.empty and warnings:
        return {
            "features": features,
            "features_path": None,
            "warnings": warnings,
            "loaded_existing": False,
        }

    out_path = write_event_level_eeg_features(task_folder, features)
    return {
        "features": features,
        "features_path": str(out_path.resolve()),
        "warnings": warnings,
        "loaded_existing": False,
    }
