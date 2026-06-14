"""
EEG preprocessing — pipeline definition and audit placeholders only.

No filtering, ICA, baseline extraction, or raw EEG modification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from domain.storage_layout import EEG_PREPROCESSING_PLAN_FILE

NOT_IMPLEMENTED_YET = "Not implemented yet"

DEFAULT_ROI_DEFINITIONS: dict[str, list[str]] = {
    "frontal": [
        "Fp1",
        "Fp2",
        "AF3",
        "AF4",
        "AF7",
        "AF8",
        "Fz",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "FCz",
        "FC1",
        "FC2",
        "FC3",
        "FC4",
        "FC5",
        "FC6",
        "FT7",
        "FT8",
        "FT9",
        "FT10",
    ],
    "temporal": [
        "T7",
        "T8",
        "TP7",
        "TP8",
        "TP9",
        "TP10",
        "P7",
        "P9",
        "P10",
    ],
    "occipital": [
        "Oz",
        "O1",
        "O2",
        "POz",
        "PO3",
        "PO4",
        "PO7",
        "PO8",
        "Iz",
    ],
}

DEFAULT_PREPROCESSING_SETTINGS: dict[str, Any] = {
    "reference": "average",
    "bandpass_hz": [0.5, 40],
    "notch_hz": 50,
    "bad_channel_policy": "flag_only",
    "ica_policy": "semi_automatic",
    "baseline_source": "resting_state",
}


def default_preprocessing_status() -> dict[str, str]:
    return {
        "preprocessing_completed": NOT_IMPLEMENTED_YET,
        "bad_channels_flagged": NOT_IMPLEMENTED_YET,
        "ica_completed": NOT_IMPLEMENTED_YET,
        "baseline_available": NOT_IMPLEMENTED_YET,
    }


def build_eeg_preprocessing_plan(task_folder: Path | None = None) -> dict[str, Any]:
    """Build preprocessing plan and audit placeholders (no signal processing)."""
    _ = task_folder
    return {
        "phase": "preprocessing_definition_only",
        "note": "Plan and audit placeholders only. No EEG signals are modified.",
        **DEFAULT_PREPROCESSING_SETTINGS,
        "roi_definitions": {
            region: list(channels)
            for region, channels in DEFAULT_ROI_DEFINITIONS.items()
        },
        "status": default_preprocessing_status(),
    }


def write_eeg_preprocessing_plan(
    task_folder: Path,
    plan: dict[str, Any] | None = None,
) -> Path:
    payload = plan if plan is not None else build_eeg_preprocessing_plan(task_folder)
    out = task_folder / EEG_PREPROCESSING_PLAN_FILE
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_eeg_preprocessing_plan(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / EEG_PREPROCESSING_PLAN_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_eeg_preprocessing_plan_setup(task_folder: Path) -> dict[str, Any]:
    """Write eeg_preprocessing_plan.json for a task (definition only)."""
    plan = build_eeg_preprocessing_plan(task_folder)
    path = write_eeg_preprocessing_plan(task_folder, plan)
    return {
        "preprocessing_plan_path": str(path.resolve()),
        "plan": plan,
    }
