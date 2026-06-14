"""
EEG Phase 16 — baseline-normalized task-level EEG power features.

Uses existing task and Resting state feature JSON files; does not recompute raw power.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backend.eeg.baseline_linkage import (
    BASELINE_MISSING_MESSAGE,
    RESTING_STATE_BASELINE_TASK_NAME,
    _task_has_eeg,
    load_eeg_baseline_status,
)
from backend.eeg.task_level_features import (
    ALPHA_FEATURE_NAME,
    BASELINE_CHANGE_FEATURE_COLUMNS,
    FEATURE_NAME,
    NOT_AVAILABLE,
    RATIO_FEATURE_NAME,
    load_task_level_eeg_features_json,
    merge_task_level_eeg_exports,
    save_task_level_eeg_feature_payload,
)
from domain.resting_state import eeg_baseline_change_zeros

RESTING_FEATURE_MISSING_MESSAGE = "Not available: Resting state feature missing"
TASK_FEATURE_MISSING_MESSAGE = "Not available: task feature missing"

_BASELINE_SOURCE_FEATURES: dict[str, str] = {
    "theta_power_change_from_baseline": FEATURE_NAME,
    "alpha_power_change_from_baseline": ALPHA_FEATURE_NAME,
    "theta_alpha_ratio_change_from_baseline": RATIO_FEATURE_NAME,
}


def _numeric_feature_value(value: Any) -> float | None:
    if value is None or value == NOT_AVAILABLE:
        return None
    if isinstance(value, str):
        if value.startswith("Not available"):
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _compute_change_from_baseline(
    current_value: Any,
    resting_value: Any,
    *,
    baseline_available: bool,
) -> str | float:
    if not baseline_available:
        return BASELINE_MISSING_MESSAGE
    resting_number = _numeric_feature_value(resting_value)
    if resting_number is None:
        return RESTING_FEATURE_MISSING_MESSAGE
    current_number = _numeric_feature_value(current_value)
    if current_number is None:
        return TASK_FEATURE_MISSING_MESSAGE
    return float(current_number - resting_number)


def compute_baseline_normalized_features(
    task_folder: Path,
    participant_folder: Path,
    task_name: str,
) -> dict[str, str | float] | None:
    """Return change-from-baseline features for a non-resting task."""
    if task_name == RESTING_STATE_BASELINE_TASK_NAME:
        return None
    if not _task_has_eeg(task_folder):
        return None

    baseline_status = load_eeg_baseline_status(task_folder)
    baseline_available = bool(baseline_status.get("baseline_available")) if baseline_status else (
        (participant_folder / RESTING_STATE_BASELINE_TASK_NAME).is_dir()
    )

    current_features = load_task_level_eeg_features_json(task_folder) or {}
    resting_features: dict[str, Any] = {}
    if baseline_available:
        resting_folder = participant_folder / RESTING_STATE_BASELINE_TASK_NAME
        resting_features = load_task_level_eeg_features_json(resting_folder) or {}

    return {
        change_feature: _compute_change_from_baseline(
            current_features.get(source_feature),
            resting_features.get(source_feature),
            baseline_available=baseline_available,
        )
        for change_feature, source_feature in _BASELINE_SOURCE_FEATURES.items()
    }


def run_baseline_normalized_features(
    task_folder: Path,
    participant_folder: Path,
    task_name: str,
) -> dict[str, Any] | None:
    """Merge baseline-normalized features into existing task-level exports."""
    if task_name == RESTING_STATE_BASELINE_TASK_NAME:
        if not _task_has_eeg(task_folder):
            return None
        current_features = load_task_level_eeg_features_json(task_folder)
        if current_features is None:
            return None
        payload = {**current_features, **eeg_baseline_change_zeros()}
        paths = save_task_level_eeg_feature_payload(task_folder, payload, task_name=task_name)
        return {
            "task_name": task_name,
            **paths,
            "features": merge_task_level_eeg_exports(payload, task_name=task_name),
        }
    if not _task_has_eeg(task_folder):
        return None

    current_features = load_task_level_eeg_features_json(task_folder)
    if current_features is None:
        return None

    baseline_changes = compute_baseline_normalized_features(
        task_folder,
        participant_folder,
        task_name,
    )
    if baseline_changes is None:
        return None

    payload = {**current_features, **baseline_changes}
    paths = save_task_level_eeg_feature_payload(task_folder, payload, task_name=task_name)
    return {
        "task_name": task_name,
        **paths,
        "features": merge_task_level_eeg_exports(payload, task_name=task_name),
    }


def refresh_participant_baseline_normalized_features(
    participant_folder: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not participant_folder.is_dir():
        return results

    for task_folder in sorted(participant_folder.iterdir()):
        if not task_folder.is_dir():
            continue
        updated = run_baseline_normalized_features(
            task_folder,
            participant_folder,
            task_folder.name,
        )
        if updated is not None:
            results.append(updated)
    return results
