"""
EEG synchronization — clock offset correction and overlap audit.

Raw EEG timestamps are never modified. Offset is stored separately and applied
only during synchronization / overlap checks.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.eeg.inspection import (
    _discover_bvrf_sets,
    _format_datetime,
    _parse_iso_datetime,
    eeg_raw_dir,
    load_eeg_metadata_json,
    parse_bvrm,
)
from backend.sync.eprime import load_sync_window_json
from domain.storage_layout import EEG_SYNC_SETTINGS_FILE, EEG_TIME_AUDIT_FILE

_LEGACY_MARKER_TS_LEN = 14
_TRIGGER_MATCH_TOLERANCE_SECONDS = 2.0


def _parse_legacy_marker_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    if len(text) < _LEGACY_MARKER_TS_LEN:
        return None
    try:
        base = datetime.strptime(text[:_LEGACY_MARKER_TS_LEN], "%Y%m%d%H%M%S")
        if len(text) > _LEGACY_MARKER_TS_LEN:
            frac = text[_LEGACY_MARKER_TS_LEN :].ljust(6, "0")[:6]
            base = base.replace(microsecond=int(frac))
        return base.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _overlap_seconds(
    start_a: datetime | None,
    end_a: datetime | None,
    start_b: datetime | None,
    end_b: datetime | None,
) -> float | None:
    if start_a is None or end_a is None or start_b is None or end_b is None:
        return None
    overlap_start = max(_ensure_utc(start_a), _ensure_utc(start_b))
    overlap_end = min(_ensure_utc(end_a), _ensure_utc(end_b))
    if overlap_end <= overlap_start:
        return 0.0
    return (overlap_end - overlap_start).total_seconds()


def _legacy_eeg_window(task_folder: Path) -> tuple[datetime | None, datetime | None, list[str]]:
    warnings: list[str] = []
    meta_path = task_folder / "eeg_meta.json"
    if not meta_path.is_file():
        return None, None, warnings

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read eeg_meta.json: {exc}")
        return None, None, warnings

    start_dt = _parse_legacy_marker_timestamp(meta.get("first_marker_timestamp"))
    if start_dt is None:
        warnings.append("Legacy EEG: could not parse first marker timestamp.")
        return None, None, warnings

    rate = meta.get("sampling_rate_hz")
    eeg_path = task_folder / "Task.eeg"
    ahdr_path = task_folder / "Task.ahdr"
    end_dt: datetime | None = None

    if eeg_path.is_file() and ahdr_path.is_file() and rate:
        try:
            from backend.sync.eeg_meta import parse_ahdr

            fields = parse_ahdr(ahdr_path.read_bytes())
            n_channels = 0
            for key, value in fields.items():
                if key.startswith("Ch") and key[2:].isdigit():
                    n_channels += 1
            if n_channels <= 0:
                n_channels = int(fields.get("NumberOfChannels", "0") or "0") or None
            if n_channels and rate > 0:
                nbytes = eeg_path.stat().st_size
                itemsize = 4  # legacy BrainVision float32
                sample_count = nbytes // (n_channels * itemsize)
                end_dt = start_dt + timedelta(seconds=sample_count / rate)
        except (OSError, ValueError, TypeError):
            pass

    return start_dt, end_dt, warnings


def _bvrf_eeg_window(task_folder: Path) -> tuple[datetime | None, datetime | None, list[str]]:
    warnings: list[str] = []
    metadata = load_eeg_metadata_json(eeg_raw_dir(task_folder))
    if not metadata:
        return None, None, warnings

    for recording in metadata.get("recordings", []):
        start_dt = _parse_iso_datetime(recording.get("recording_start_time"))
        end_dt = _parse_iso_datetime(recording.get("recording_end_time"))
        if start_dt is not None:
            return start_dt, end_dt, warnings

    warnings.append("BVRF EEG: no recording start time found in eeg_metadata.json.")
    return None, None, warnings


def get_eeg_recording_window_utc_raw(
    task_folder: Path,
) -> tuple[datetime | None, datetime | None, list[str]]:
    """Return raw EEG recording window (start, end) without clock offset applied."""
    start, end, warnings = _bvrf_eeg_window(task_folder)
    if start is not None:
        return start, end, warnings
    return _legacy_eeg_window(task_folder)


def get_eeg_recording_window_utc_adjusted(
    task_folder: Path,
) -> tuple[datetime | None, datetime | None, list[str]]:
    """Return clock-corrected EEG window for overlap checks and future segmentation."""
    settings = load_eeg_sync_settings(task_folder)
    offset = settings.get("eeg_clock_offset_seconds") if settings else None
    raw_start, raw_end, warnings = get_eeg_recording_window_utc_raw(task_folder)
    adjusted_start, adjusted_end = apply_clock_offset(raw_start, raw_end, offset)
    return adjusted_start, adjusted_end, warnings


def apply_clock_offset(
    start: datetime | None,
    end: datetime | None,
    offset_seconds: float | None,
) -> tuple[datetime | None, datetime | None]:
    if offset_seconds is None or offset_seconds == 0:
        return start, end
    delta = timedelta(seconds=offset_seconds)
    adjusted_start = start - delta if start is not None else None
    adjusted_end = end - delta if end is not None else None
    return adjusted_start, adjusted_end


def _stimulus_marker_datetimes(task_folder: Path) -> list[datetime]:
    """Ordered UTC datetimes of Stimulus-type markers from the task's .bvrm file(s)."""
    grouped = _discover_bvrf_sets(eeg_raw_dir(task_folder))
    for files in grouped.values():
        marker_path = files.get(".bvrm")
        if marker_path is None:
            continue
        markers, _ = parse_bvrm(marker_path)
        stimuli = [m for m in markers if (m.get("type") or "").strip().lower() == "stimulus"]
        stimuli.sort(key=lambda m: int(m.get("sample") or 0))
        datetimes = [
            dt for dt in (_parse_iso_datetime(m.get("datetime")) for m in stimuli) if dt is not None
        ]
        if datetimes:
            return [_ensure_utc(dt) for dt in datetimes]
    return []


def derive_resting_state_offset_seconds(
    task_folder: Path,
    sync_window: dict[str, Any] | None,
) -> tuple[float | None, list[str]]:
    """
    Resting-state recordings carry hardware trigger markers (EyesOpen/EyesClose,
    fired by the same keypress that E-Prime logs). When present and consistent with
    the E-Prime EyesOpen->EyesClose interval, they give an exact per-task clock
    offset — more reliable than the manually-entered PC-to-PC offset, which is a
    single value assumed constant across the whole session and can be invalidated
    mid-session (e.g. an NTP resync on the EEG computer). Reading tasks have no such
    triggers, so this only ever applies to resting-state sync windows.
    """
    warnings: list[str] = []
    if not sync_window:
        return None, warnings

    eyes_open_ms = sync_window.get("eyes_open_onset_ms")
    eyes_close_ms = sync_window.get("eyes_close_onset_ms")
    session_start = _parse_iso_datetime(sync_window.get("session_start_utc"))
    if eyes_open_ms is None or eyes_close_ms is None or session_start is None:
        return None, warnings

    markers = _stimulus_marker_datetimes(task_folder)
    if len(markers) < 2:
        return None, warnings

    marker_open, marker_close = markers[0], markers[1]
    marker_delta = (marker_close - marker_open).total_seconds()
    eprime_delta = (eyes_close_ms - eyes_open_ms) / 1000.0

    if abs(marker_delta - eprime_delta) > _TRIGGER_MATCH_TOLERANCE_SECONDS:
        warnings.append(
            "Resting-state trigger markers found but their spacing "
            f"({marker_delta:.3f}s) does not match the E-Prime EyesOpen-to-EyesClose "
            f"interval ({eprime_delta:.3f}s); keeping the manual clock offset."
        )
        return None, warnings

    eyes_open_utc = _ensure_utc(session_start) + timedelta(milliseconds=eyes_open_ms)
    offset_seconds = (marker_open - eyes_open_utc).total_seconds()
    warnings.append(
        "EEG clock offset derived from resting-state EyesOpen/EyesClose hardware "
        f"trigger markers ({offset_seconds:.3f}s), overriding the manual clock offset."
    )
    return offset_seconds, warnings


def effective_eeg_clock_offset_seconds(
    task_folder: Path,
    sync_window: dict[str, Any] | None = None,
) -> tuple[float | None, str]:
    """
    Offset (EEG_PC_time - E-Prime_PC_time, s) actually used for this task: the
    resting-state trigger-derived value when available, else the manual value from
    eeg_sync_settings.json, else None. Returns (offset, source).
    """
    window = sync_window or load_sync_window_json(task_folder)
    derived, _ = derive_resting_state_offset_seconds(task_folder, window)
    if derived is not None:
        return derived, "resting_state_triggers"
    settings = load_eeg_sync_settings(task_folder) or {}
    manual = settings.get("eeg_clock_offset_seconds")
    try:
        return (float(manual), "manual") if manual is not None else (None, "none")
    except (TypeError, ValueError):
        return None, "none"


def write_eeg_sync_settings(
    task_folder: Path,
    eeg_clock_offset_seconds: float | None,
) -> Path:
    payload = {
        "eeg_clock_offset_seconds": eeg_clock_offset_seconds,
        "definition": "EEG_PC_time - EPRIME_TOBII_PC_time (seconds)",
    }
    out = task_folder / EEG_SYNC_SETTINGS_FILE
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_eeg_sync_settings(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / EEG_SYNC_SETTINGS_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_eeg_time_audit(
    task_folder: Path,
    *,
    eeg_clock_offset_seconds: float | None = None,
    sync_window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = load_eeg_sync_settings(task_folder)
    if eeg_clock_offset_seconds is None and settings is not None:
        eeg_clock_offset_seconds = settings.get("eeg_clock_offset_seconds")

    window = sync_window or load_sync_window_json(task_folder)
    warnings: list[str] = []

    offset_source = "manual"
    derived_offset, derived_warnings = derive_resting_state_offset_seconds(task_folder, window)
    warnings.extend(derived_warnings)
    if derived_offset is not None:
        eeg_clock_offset_seconds = derived_offset
        offset_source = "resting_state_triggers"

    eprime_start = _parse_iso_datetime(window.get("task_start_utc") if window else None)
    eprime_end = _parse_iso_datetime(window.get("task_end_utc") if window else None)
    if window is None:
        warnings.append("sync_window.json missing — E-Prime task times unavailable.")
    elif eprime_start is None or eprime_end is None:
        warnings.append("E-Prime task start/end UTC could not be parsed.")

    raw_start, raw_end, eeg_warnings = get_eeg_recording_window_utc_raw(task_folder)
    warnings.extend(eeg_warnings)
    if raw_start is None:
        warnings.append("EEG recording start time unavailable.")

    adjusted_start, adjusted_end = apply_clock_offset(
        raw_start, raw_end, eeg_clock_offset_seconds
    )

    raw_overlap = _overlap_seconds(raw_start, raw_end, eprime_start, eprime_end)
    adjusted_overlap = _overlap_seconds(adjusted_start, adjusted_end, eprime_start, eprime_end)

    return {
        "eprime_task_start_utc": _format_datetime(eprime_start),
        "eprime_task_end_utc": _format_datetime(eprime_end),
        "eeg_recording_start_utc_raw": _format_datetime(raw_start),
        "eeg_recording_end_utc_raw": _format_datetime(raw_end),
        "eeg_clock_offset_seconds": eeg_clock_offset_seconds,
        "eeg_clock_offset_source": offset_source,
        "eeg_recording_start_utc_adjusted": _format_datetime(adjusted_start),
        "eeg_recording_end_utc_adjusted": _format_datetime(adjusted_end),
        "raw_overlap_seconds": raw_overlap,
        "adjusted_overlap_seconds": adjusted_overlap,
        "warnings": warnings,
    }


def write_eeg_time_audit(
    task_folder: Path,
    audit: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Path:
    payload = audit if audit is not None else build_eeg_time_audit(task_folder, **kwargs)
    out = task_folder / EEG_TIME_AUDIT_FILE
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def run_eeg_synchronization(
    task_folder: Path,
    *,
    eeg_clock_offset_seconds: float | None = None,
    sync_window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Persist sync settings and time audit for a task.

    Does not modify raw EEG files or timestamps in eeg_metadata.json.
    """
    settings_path = write_eeg_sync_settings(task_folder, eeg_clock_offset_seconds)
    audit = build_eeg_time_audit(
        task_folder,
        eeg_clock_offset_seconds=eeg_clock_offset_seconds,
        sync_window=sync_window,
    )
    audit_path = write_eeg_time_audit(task_folder, audit)
    return {
        "settings_path": str(settings_path.resolve()),
        "audit_path": str(audit_path.resolve()),
        "audit": audit,
    }
