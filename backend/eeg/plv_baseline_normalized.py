"""
EEG Phase 21 — baseline-normalized PLV task-level features.

Uses existing task and Resting state feature JSON files; does not recompute PLV.
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
from backend.eeg.plv_features import (
    OF_PLV_FEATURE_NAME,
    OT_PLV_FEATURE_NAME,
    TF_PLV_FEATURE_NAME,
)
from backend.eeg.task_level_features import (
    NOT_AVAILABLE,
    load_task_level_eeg_features_json,
    merge_task_level_eeg_exports,
    save_task_level_eeg_feature_payload,
)
from domain.resting_state import eeg_baseline_change_zeros

RESTING_FEATURE_MISSING_MESSAGE = "Not available: Resting state feature missing"
TASK_FEATURE_MISSING_MESSAGE = "Not available: task feature missing"

PLV_BASELINE_CHANGE_FEATURE_COLUMNS: tuple[str, ...] = (
    "OT_plv_change_from_baseline",
    "TF_plv_change_from_baseline",
    "OF_plv_change_from_baseline",
)

_PLV_BASELINE_SOURCE_FEATURES: dict[str, str] = {
    "OT_plv_change_from_baseline": OT_PLV_FEATURE_NAME,
    "TF_plv_change_from_baseline": TF_PLV_FEATURE_NAME,
    "OF_plv_change_from_baseline": OF_PLV_FEATURE_NAME,
}


def _numeric_feature_value(value: Any) -> float | None:
    if value is None or value == NOT_AVAILABLE:
        return None
    if isinstance(value, str) and value.startswith("Not available"):
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


def compute_plv_baseline_normalized_features(
    task_folder: Path,
    participant_folder: Path,
    task_name: str,
) -> dict[str, str | float] | None:
    """Return PLV change-from-baseline features for a non-resting task."""
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
        for change_feature, source_feature in _PLV_BASELINE_SOURCE_FEATURES.items()
    }


def run_plv_baseline_normalized_features(
    task_folder: Path,
    participant_folder: Path,
    task_name: str,
) -> dict[str, Any] | None:
    """Merge PLV baseline-normalized features into existing task-level exports."""
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

    plv_baseline_changes = compute_plv_baseline_normalized_features(
        task_folder,
        participant_folder,
        task_name,
    )
    if plv_baseline_changes is None:
        return None

    payload = {**current_features, **plv_baseline_changes}
    paths = save_task_level_eeg_feature_payload(task_folder, payload, task_name=task_name)
    return {
        "task_name": task_name,
        **paths,
        "features": merge_task_level_eeg_exports(payload, task_name=task_name),
    }


def refresh_participant_plv_baseline_normalized_features(
    participant_folder: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not participant_folder.is_dir():
        return results

    for task_folder in sorted(participant_folder.iterdir()):
        if not task_folder.is_dir():
            continue
        updated = run_plv_baseline_normalized_features(
            task_folder,
            participant_folder,
            task_folder.name,
        )
        if updated is not None:
            results.append(updated)
    return results
