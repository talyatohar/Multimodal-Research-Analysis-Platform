"""
EEG Phase 4 — metadata-based quality control summary.

Uses existing audit/metadata JSON only. No raw signal reads or processing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.eeg.eeglab_compat import detect_eeglab_set_files
from backend.eeg.inspection import eeg_raw_dir, load_eeg_metadata_json
from backend.eeg.segmentation import load_eeg_segment_metadata
from backend.eeg.sync import load_eeg_sync_settings
from domain.storage_layout import EEG_QC_SUMMARY_FILE, EEG_TIME_AUDIT_FILE

NOT_AVAILABLE = "Not available"
NOT_IMPLEMENTED = "Not implemented yet"

# Metadata-only QC threshold (kΩ); not applied to raw signals or exclusion.
HIGH_IMPEDANCE_KOHM_THRESHOLD = 50.0
IMPEDANCE_THRESHOLD_NOTE = (
    "Impedance threshold is used for metadata inspection only and is not an exclusion criterion."
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _primary_recording(eeg_metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not eeg_metadata:
        return None
    recordings = eeg_metadata.get("recordings")
    if not isinstance(recordings, list) or not recordings:
        return None
    first = recordings[0]
    return first if isinstance(first, dict) else None


def _channels_above_metadata_impedance_threshold(
    impedance_qc: list[dict[str, Any]],
) -> list[dict[str, Any]] | str:
    if not impedance_qc:
        return NOT_AVAILABLE
    flagged: list[dict[str, Any]] = []
    for row in impedance_qc:
        value = row.get("impedance_kohm")
        if value is None:
            continue
        try:
            kohm = float(value)
        except (TypeError, ValueError):
            continue
        if kohm > HIGH_IMPEDANCE_KOHM_THRESHOLD:
            flagged.append(
                {
                    "electrode": row.get("electrode"),
                    "impedance_kohm": kohm,
                    "measurement_time": row.get("measurement_time"),
                }
            )
    return flagged


def _field_or_status(value: Any, *, not_implemented: bool = False) -> Any:
    if value is None:
        return NOT_IMPLEMENTED if not_implemented else NOT_AVAILABLE
    return value


def build_eeg_qc_summary(task_folder: Path) -> dict[str, Any]:
    """Assemble metadata-only EEG QC summary for a task."""
    eeg_metadata = load_eeg_metadata_json(eeg_raw_dir(task_folder))
    segment_meta = load_eeg_segment_metadata(task_folder)
    time_audit = _load_json(task_folder / EEG_TIME_AUDIT_FILE)
    recording = _primary_recording(eeg_metadata)

    sampling_rate_hz = None
    channel_count = None
    markers_count = None
    impedance_qc: list[dict[str, Any]] = []

    if recording:
        sampling_rate_hz = recording.get("sampling_rate_hz")
        channel_count = recording.get("channel_count")
        markers_count = recording.get("markers_count")
        raw_impedance = recording.get("impedance_qc")
        if isinstance(raw_impedance, list):
            impedance_qc = raw_impedance

    if sampling_rate_hz is None and segment_meta:
        sampling_rate_hz = segment_meta.get("sampling_rate_hz")

    segment_duration_seconds = segment_meta.get("segment_duration_seconds") if segment_meta else None
    partial_overlap = segment_meta.get("partial_overlap") if segment_meta else None
    missing_task_start_seconds = (
        segment_meta.get("missing_task_start_seconds") if segment_meta else None
    )
    missing_task_end_seconds = (
        segment_meta.get("missing_task_end_seconds") if segment_meta else None
    )

    impedance_available = bool(impedance_qc)
    channels_above_threshold = _channels_above_metadata_impedance_threshold(impedance_qc)
    eeglab_set_detected = bool(detect_eeglab_set_files(task_folder))

    sources_present = {
        "eeg_metadata_json": eeg_metadata is not None,
        "eeg_segment_metadata_json": segment_meta is not None,
        "eeg_time_audit_json": time_audit is not None,
        "eeg_sync_settings_json": load_eeg_sync_settings(task_folder) is not None,
    }

    return {
        "phase": "metadata_qc",
        "task_folder": str(task_folder.resolve()),
        "sources_present": sources_present,
        "sampling_rate_hz": _field_or_status(sampling_rate_hz),
        "channel_count": _field_or_status(channel_count),
        "segment_duration_seconds": _field_or_status(segment_duration_seconds),
        "partial_overlap": _field_or_status(partial_overlap),
        "missing_task_start_seconds": _field_or_status(missing_task_start_seconds),
        "missing_task_end_seconds": _field_or_status(missing_task_end_seconds),
        "markers_count": _field_or_status(markers_count),
        "impedance_available": impedance_available,
        "high_impedance_kohm_threshold": HIGH_IMPEDANCE_KOHM_THRESHOLD,
        "impedance_threshold_note": IMPEDANCE_THRESHOLD_NOTE,
        "channels_above_metadata_impedance_threshold": channels_above_threshold,
        "eeglab_set_detected": eeglab_set_detected,
    }


def write_eeg_qc_summary(
    task_folder: Path,
    summary: dict[str, Any] | None = None,
) -> Path:
    payload = summary if summary is not None else build_eeg_qc_summary(task_folder)
    out = task_folder / EEG_QC_SUMMARY_FILE
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_eeg_qc_summary(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / EEG_QC_SUMMARY_FILE
    return _load_json(path)


def run_eeg_qc(task_folder: Path) -> dict[str, Any]:
    """Build and cache metadata-based EEG QC summary."""
    summary = build_eeg_qc_summary(task_folder)
    path = write_eeg_qc_summary(task_folder, summary)
    return {
        "qc_summary_path": str(path.resolve()),
        "summary": summary,
    }
