"""
EEG segmentation — task-window clip using adjusted EEG timestamps only.

Does not read or modify raw .bvrd / .eeg binary data.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.eeg.inspection import eeg_raw_dir, load_eeg_metadata_json
from backend.eeg.sync import (
    _ensure_utc,
    _parse_iso_datetime,
    _format_datetime,
    apply_clock_offset,
    get_eeg_recording_window_utc_raw,
    load_eeg_sync_settings,
)
from backend.sync.eprime import load_sync_window_json
from domain.storage_layout import EEG_SEGMENT_METADATA_FILE

PARTIAL_OVERLAP_WARNING = (
    "Partial EEG-task overlap: EEG recording does not cover the full task window."
)


def _get_sampling_rate_hz(task_folder: Path) -> float | None:
    metadata = load_eeg_metadata_json(eeg_raw_dir(task_folder))
    if metadata:
        for recording in metadata.get("recordings", []):
            rate = recording.get("sampling_rate_hz")
            if rate is not None:
                try:
                    return float(rate)
                except (TypeError, ValueError):
                    pass

    meta_path = task_folder / "eeg_meta.json"
    if meta_path.is_file():
        try:
            legacy = json.loads(meta_path.read_text(encoding="utf-8"))
            rate = legacy.get("sampling_rate_hz")
            if rate is not None:
                return float(rate)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return None


def _get_sample_count(task_folder: Path) -> int | None:
    metadata = load_eeg_metadata_json(eeg_raw_dir(task_folder))
    if metadata:
        for recording in metadata.get("recordings", []):
            count = recording.get("sample_count")
            if count is not None:
                try:
                    return int(count)
                except (TypeError, ValueError):
                    pass
    return None


def _seconds_to_sample_index(seconds: float, sampling_rate_hz: float) -> int:
    return int(math.floor(seconds * sampling_rate_hz + 1e-9))


def _raw_time_from_adjusted(
    adjusted_time: datetime,
    offset_seconds: float | None,
) -> datetime:
    offset = offset_seconds or 0.0
    return _ensure_utc(adjusted_time) + timedelta(seconds=offset)


def build_eeg_segment_metadata(task_folder: Path) -> dict[str, Any]:
    warnings: list[str] = []

    sync_window = load_sync_window_json(task_folder)
    if sync_window is None:
        warnings.append("sync_window.json missing — EEG segmentation skipped.")
        return {"ok": False, "warnings": warnings}

    task_start = _parse_iso_datetime(sync_window.get("task_start_utc"))
    task_end = _parse_iso_datetime(sync_window.get("task_end_utc"))
    if task_start is None or task_end is None:
        warnings.append("E-Prime task start/end UTC could not be parsed.")
        return {"ok": False, "warnings": warnings}

    task_start = _ensure_utc(task_start)
    task_end = _ensure_utc(task_end)

    settings = load_eeg_sync_settings(task_folder) or {}
    offset_seconds = settings.get("eeg_clock_offset_seconds")
    if offset_seconds is not None:
        try:
            offset_seconds = float(offset_seconds)
        except (TypeError, ValueError):
            warnings.append("Invalid eeg_clock_offset_seconds in eeg_sync_settings.json.")
            offset_seconds = None

    raw_start, raw_end, eeg_warnings = get_eeg_recording_window_utc_raw(task_folder)
    warnings.extend(eeg_warnings)
    if raw_start is None:
        warnings.append("EEG recording start time unavailable — segmentation skipped.")
        return {"ok": False, "warnings": warnings}

    raw_start = _ensure_utc(raw_start)
    raw_end = _ensure_utc(raw_end) if raw_end is not None else None

    adjusted_start, adjusted_end = apply_clock_offset(raw_start, raw_end, offset_seconds)
    if adjusted_start is None:
        warnings.append("Adjusted EEG start time unavailable — segmentation skipped.")
        return {"ok": False, "warnings": warnings}
    adjusted_start = _ensure_utc(adjusted_start)
    adjusted_end = _ensure_utc(adjusted_end) if adjusted_end is not None else None

    if adjusted_end is None or adjusted_end <= adjusted_start:
        warnings.append("Adjusted EEG recording end time unavailable or invalid.")
        return {"ok": False, "warnings": warnings}

    segment_start = max(task_start, adjusted_start)
    segment_end = min(task_end, adjusted_end)

    if segment_end <= segment_start:
        warnings.append("No overlap between adjusted EEG recording and E-Prime task window.")
        return {
            "ok": False,
            "task_start_utc": _format_datetime(task_start),
            "task_end_utc": _format_datetime(task_end),
            "eeg_recording_start_utc_raw": _format_datetime(raw_start),
            "eeg_recording_end_utc_raw": _format_datetime(raw_end),
            "eeg_recording_start_utc_adjusted": _format_datetime(adjusted_start),
            "eeg_recording_end_utc_adjusted": _format_datetime(adjusted_end),
            "eeg_clock_offset_seconds": offset_seconds,
            "segment_start_utc": None,
            "segment_end_utc": None,
            "segment_duration_seconds": 0.0,
            "missing_task_start_seconds": None,
            "missing_task_end_seconds": None,
            "eeg_start_sample_index": None,
            "eeg_end_sample_index": None,
            "sampling_rate_hz": _get_sampling_rate_hz(task_folder),
            "warnings": warnings,
        }

    segment_duration = (segment_end - segment_start).total_seconds()

    missing_task_start = max(0.0, (adjusted_start - task_start).total_seconds())
    missing_task_end = max(0.0, (task_end - adjusted_end).total_seconds())
    partial_overlap = missing_task_start > 0 or missing_task_end > 0
    if partial_overlap:
        warnings.append(PARTIAL_OVERLAP_WARNING)

    sampling_rate_hz = _get_sampling_rate_hz(task_folder)
    sample_count = _get_sample_count(task_folder)

    eeg_start_sample_index: int | None = None
    eeg_end_sample_index: int | None = None

    if sampling_rate_hz and sampling_rate_hz > 0:
        raw_segment_start = _raw_time_from_adjusted(segment_start, offset_seconds)
        raw_segment_end = _raw_time_from_adjusted(segment_end, offset_seconds)
        start_seconds = (raw_segment_start - raw_start).total_seconds()
        end_seconds = (raw_segment_end - raw_start).total_seconds()
        eeg_start_sample_index = max(0, _seconds_to_sample_index(start_seconds, sampling_rate_hz))
        end_candidate = _seconds_to_sample_index(end_seconds, sampling_rate_hz) - 1
        eeg_end_sample_index = max(eeg_start_sample_index, end_candidate)
        if sample_count is not None:
            eeg_end_sample_index = min(eeg_end_sample_index, sample_count - 1)
    else:
        warnings.append("Sampling rate unavailable — sample indices not computed.")

    return {
        "ok": True,
        "task_start_utc": _format_datetime(task_start),
        "task_end_utc": _format_datetime(task_end),
        "eeg_recording_start_utc_raw": _format_datetime(raw_start),
        "eeg_recording_end_utc_raw": _format_datetime(raw_end),
        "eeg_recording_start_utc_adjusted": _format_datetime(adjusted_start),
        "eeg_recording_end_utc_adjusted": _format_datetime(adjusted_end),
        "eeg_clock_offset_seconds": offset_seconds,
        "segment_start_utc": _format_datetime(segment_start),
        "segment_end_utc": _format_datetime(segment_end),
        "segment_duration_seconds": segment_duration,
        "missing_task_start_seconds": missing_task_start,
        "missing_task_end_seconds": missing_task_end,
        "partial_overlap": partial_overlap,
        "eeg_start_sample_index": eeg_start_sample_index,
        "eeg_end_sample_index": eeg_end_sample_index,
        "sampling_rate_hz": sampling_rate_hz,
        "sample_count": sample_count,
        "warnings": warnings,
    }


def write_eeg_segment_metadata(
    task_folder: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    payload = metadata if metadata is not None else build_eeg_segment_metadata(task_folder)
    out = task_folder / EEG_SEGMENT_METADATA_FILE
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_eeg_segment_metadata(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / EEG_SEGMENT_METADATA_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_eeg_segmentation(task_folder: Path) -> dict[str, Any]:
    """Compute task-aligned EEG segment bounds and cache to eeg_segment_metadata.json."""
    metadata = build_eeg_segment_metadata(task_folder)
    path = write_eeg_segment_metadata(task_folder, metadata)
    return {
        "segment_metadata_path": str(path.resolve()),
        "metadata": metadata,
    }
