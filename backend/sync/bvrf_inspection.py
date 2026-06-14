"""Backward-compatible wrappers around backend.eeg.inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.eeg.inspection import (
    BVRF_EXTENSIONS,
    build_eeg_metadata,
    eeg_raw_dir,
    inspect_eeg_raw_folder,
    write_eeg_metadata_json,
)
from domain.storage_layout import EEG_INSPECTION_FILE

__all__ = [
    "BVRF_EXTENSIONS",
    "build_eeg_metadata",
    "eeg_raw_dir",
    "inspect_bvrf_recording",
    "inspect_eeg_raw_folder",
    "write_eeg_file_inspection",
    "write_eeg_metadata_json",
]


def inspect_bvrf_recording(eeg_raw_folder: Path | str) -> dict[str, Any]:
    """Inspect BVRF recording set(s) under EEG_raw/ (legacy wrapper)."""
    metadata = inspect_eeg_raw_folder(eeg_raw_folder)
    return _metadata_to_legacy_inspection(metadata)


def write_eeg_file_inspection(eeg_raw_folder: Path | str, inspection: dict[str, Any] | None = None) -> Path:
    """Write legacy eeg_file_inspection.json (derived from eeg_metadata.json)."""
    folder = Path(eeg_raw_folder)
    folder.mkdir(parents=True, exist_ok=True)
    if inspection is None:
        inspection = inspect_bvrf_recording(folder)
    out = folder / EEG_INSPECTION_FILE
    out.write_text(json.dumps(inspection, indent=2), encoding="utf-8")
    return out


def _metadata_to_legacy_inspection(metadata: dict[str, Any]) -> dict[str, Any]:
    recording_sets = []
    for recording in metadata.get("recordings", []):
        files = recording.get("files") or {}
        recording_sets.append(
            {
                "basename": recording.get("basename"),
                "header_file": _legacy_file_entry(files.get("header")),
                "data_file": _legacy_file_entry(files.get("data")),
                "marker_file": _legacy_file_entry(files.get("marker")),
                "impedance_file": _legacy_file_entry(files.get("impedance")),
                "file_sizes_bytes": {
                    "header": _legacy_size(files.get("header")),
                    "data": recording.get("data_file_size_bytes"),
                    "marker": _legacy_size(files.get("marker")),
                    "impedance": _legacy_size(files.get("impedance")),
                },
                "number_of_channels": recording.get("channel_count"),
                "sampling_rate_hz": recording.get("sampling_rate_hz"),
                "recording_start_time": recording.get("recording_start_time"),
                "missing_required_files": recording.get("missing_required_files", []),
                "missing_optional_files": recording.get("missing_optional_files", []),
            }
        )
    return {
        "format": metadata.get("format", "BVRF"),
        "eeg_raw_folder": metadata.get("eeg_raw_folder"),
        "warnings": metadata.get("warnings", []),
        "recording_sets": recording_sets,
    }


def _legacy_file_entry(path_str: str | None) -> dict[str, Any] | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_file():
        return {"path": path_str, "size_bytes": None}
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size}


def _legacy_size(path_str: str | None) -> int | None:
    if not path_str:
        return None
    path = Path(path_str)
    return path.stat().st_size if path.is_file() else None
