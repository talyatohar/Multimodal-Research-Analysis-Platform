"""
EEG Phase 3 — EEGLAB .set export / compatibility layer (placeholder).

BVRF raw files and audit JSON artifacts are never modified.
No preprocessing, filtering, ICA, ERP, PLV, or feature extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.eeg.inspection import eeg_raw_dir
from domain.storage_layout import EEG_RAW_FOLDER

EEGLAB_SET_EXTENSION = ".set"
MESSAGE_SET_DETECTED = "EEGLAB .set file detected"
MESSAGE_EXPORT_NOT_IMPLEMENTED = "EEGLAB .set export not implemented yet"


def _search_set_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == EEGLAB_SET_EXTENSION
    )


def detect_eeglab_set_files(task_folder: Path) -> list[Path]:
    """Find EEGLAB .set files in the task EEG folders (EEG_raw/ and task root)."""
    found: list[Path] = []
    seen: set[str] = set()
    for folder in (eeg_raw_dir(task_folder), task_folder):
        for path in _search_set_files(folder):
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                found.append(path)
    return found


def get_eeglab_compat_status(task_folder: Path) -> dict[str, Any]:
    """
    Report EEGLAB .set compatibility status for a task.

    Export from BVRF is not implemented; this only detects existing .set uploads.
    """
    set_files = detect_eeglab_set_files(task_folder)
    detected = bool(set_files)
    return {
        "phase": "eeglab_compat_placeholder",
        "set_file_detected": detected,
        "message": MESSAGE_SET_DETECTED if detected else MESSAGE_EXPORT_NOT_IMPLEMENTED,
        "set_files": [str(path.resolve()) for path in set_files],
        "search_paths": [
            str(eeg_raw_dir(task_folder).resolve()),
            str(task_folder.resolve()),
        ],
        "export_implemented": False,
    }
