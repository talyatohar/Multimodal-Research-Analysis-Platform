"""
EEG Phase 15 — link non-resting tasks to the Resting state baseline task.

Establishes baseline availability only; does not compute change-from-baseline features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.eeg.inspection import eeg_raw_dir
from backend.eeg.task_level_features import (
    NOT_AVAILABLE,
    POWER_FEATURE_COLUMNS,
    load_task_level_eeg_features_json,
)
from domain.storage_layout import EEG_BASELINE_STATUS_FILE, TASK_LEVEL_EEG_FEATURES_JSON
RESTING_STATE_BASELINE_TASK_NAME = "Resting state"
BASELINE_MISSING_MESSAGE = "Not available: Resting state baseline missing"
BASELINE_LINKAGE_NOTE = (
    "Baseline linkage only. Change-from-baseline features are not computed yet."
)


def _task_has_eeg(task_folder: Path) -> bool:
    raw_folder = eeg_raw_dir(task_folder)
    if raw_folder.is_dir() and any(path.is_file() for path in raw_folder.iterdir()):
        return True
    return (task_folder / "Task.amrk").is_file() or (task_folder / "Task.ahdr").is_file()


def _missing_baseline_features(baseline_features: dict[str, Any] | None) -> list[str]:
    if not baseline_features:
        return list(POWER_FEATURE_COLUMNS)

    missing: list[str] = []
    for feature_name in POWER_FEATURE_COLUMNS:
        value = baseline_features.get(feature_name)
        if value is None or value == NOT_AVAILABLE:
            missing.append(feature_name)
    return missing


def build_eeg_baseline_status(
    task_folder: Path,
    participant_folder: Path,
    task_name: str,
) -> dict[str, Any] | None:
    """Build baseline linkage status for a non-resting EEG task."""
    if task_name == RESTING_STATE_BASELINE_TASK_NAME:
        return None
    if not _task_has_eeg(task_folder):
        return None

    baseline_folder = participant_folder / RESTING_STATE_BASELINE_TASK_NAME
    baseline_available = baseline_folder.is_dir()

    status: dict[str, Any] = {
        "phase": "resting_state_baseline_linkage",
        "baseline_available": baseline_available,
        "baseline_task_name": RESTING_STATE_BASELINE_TASK_NAME,
        "baseline_task_path": str(baseline_folder.resolve()) if baseline_available else None,
        "baseline_features_available": False,
        "missing_baseline_features": [],
        "baseline_features_path": None,
        "baseline_missing_message": BASELINE_MISSING_MESSAGE,
        "note": BASELINE_LINKAGE_NOTE,
    }

    if not baseline_available:
        return status

    baseline_features_path = baseline_folder / TASK_LEVEL_EEG_FEATURES_JSON
    status["baseline_features_path"] = (
        str(baseline_features_path.resolve()) if baseline_features_path.is_file() else None
    )
    baseline_features = load_task_level_eeg_features_json(baseline_folder)
    missing = _missing_baseline_features(baseline_features)
    status["missing_baseline_features"] = missing
    status["baseline_features_available"] = baseline_features is not None and not missing
    return status


def write_eeg_baseline_status(task_folder: Path, status: dict[str, Any]) -> Path:
    out = task_folder / EEG_BASELINE_STATUS_FILE
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return out


def load_eeg_baseline_status(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / EEG_BASELINE_STATUS_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_eeg_baseline_linkage(
    task_folder: Path,
    participant_folder: Path,
    task_name: str,
) -> dict[str, Any] | None:
    status = build_eeg_baseline_status(task_folder, participant_folder, task_name)
    if status is None:
        return None
    path = write_eeg_baseline_status(task_folder, status)
    return {
        "baseline_status_path": str(path.resolve()),
        "status": status,
    }


def refresh_participant_eeg_baseline_linkage(participant_folder: Path) -> list[dict[str, Any]]:
    """Refresh baseline linkage for all non-resting EEG tasks under a participant."""
    results: list[dict[str, Any]] = []
    if not participant_folder.is_dir():
        return results

    for task_folder in sorted(participant_folder.iterdir()):
        if not task_folder.is_dir():
            continue
        linkage = run_eeg_baseline_linkage(task_folder, participant_folder, task_folder.name)
        if linkage is not None:
            results.append(
                {
                    "task_name": task_folder.name,
                    **linkage,
                }
            )
    return results
