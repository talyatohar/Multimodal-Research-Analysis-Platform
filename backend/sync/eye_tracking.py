"""Tobii Pro Lab segmentation using E-Prime sync_window.json per task."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.sync.eprime import load_sync_window_json

TS_COL = "Recording timestamp [ms]"
START_UTC_COL = "Recording start time UTC"
RECORDING_NAME_COL = "Recording name"
DEFAULT_PRE_TASK_BUFFER_MS = 500
DEFAULT_POST_TASK_BUFFER_MS = 500
START_UTC_FALLBACKS = (
    START_UTC_COL,
    "Recording start time",
)


def _parse_task_bounds(
    window: dict[str, Any],
    *,
    start_key: str = "task_start_utc",
    end_key: str = "task_end_utc",
) -> tuple[datetime, datetime]:
    task_start = datetime.fromisoformat(window[start_key].replace("Z", "+00:00"))
    task_end = datetime.fromisoformat(window[end_key].replace("Z", "+00:00"))
    if task_start.tzinfo is None:
        task_start = task_start.replace(tzinfo=timezone.utc)
    if task_end.tzinfo is None:
        task_end = task_end.replace(tzinfo=timezone.utc)
    return task_start, task_end


def _parse_utc_datetime(value, reference_date=None) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if reference_date is not None and value.year <= 1901:
            return datetime.combine(reference_date, value.time(), tzinfo=timezone.utc)
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if isinstance(value, time):
        if reference_date is None:
            return None
        return datetime.combine(reference_date, value, tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None

    try:
        parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        pass

    full_formats = (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
    )
    for fmt in full_formats:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    time_only_formats = ("%H:%M:%S.%f", "%H:%M:%S")
    if reference_date is not None:
        for fmt in time_only_formats:
            try:
                parsed_time = datetime.strptime(s, fmt).time()
                return datetime.combine(reference_date, parsed_time, tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _recording_start_utc(df: pd.DataFrame, task_start: datetime) -> datetime | None:
    ref_date = task_start.astimezone(timezone.utc).date()

    for col in START_UTC_FALLBACKS:
        if col in df.columns:
            for val in df[col].dropna().head(10):
                dt = _parse_utc_datetime(val, reference_date=ref_date)
                if dt:
                    return dt

    if "Recording date" in df.columns and "Recording start time" in df.columns:
        for _, row in df.head(10).iterrows():
            date_s = str(row.get("Recording date", "")).strip()
            time_s = str(row.get("Recording start time", "")).strip()
            if date_s and time_s and date_s.lower() != "nan":
                dt = _parse_utc_datetime(f"{date_s} {time_s}", reference_date=ref_date)
                if dt:
                    return dt
                dt = _parse_utc_datetime(time_s, reference_date=ref_date)
                if dt:
                    return dt
    return None


def _row_utc_series(df: pd.DataFrame, recording_start: datetime) -> pd.Series:
    ms = pd.to_numeric(df[TS_COL], errors="coerce")
    return ms.apply(
        lambda v: recording_start + timedelta(milliseconds=float(v)) if pd.notna(v) else pd.NaT
    )


def _recording_span(row_utc: pd.Series) -> tuple[datetime | None, datetime | None]:
    valid = row_utc.dropna()
    if valid.empty:
        return None, None
    return valid.min().to_pydatetime(), valid.max().to_pydatetime()


def _iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_recording_debug(
    file_name: str,
    recording_name: str | None,
    task_start: datetime,
    task_end: datetime,
    filter_start: datetime,
    filter_end: datetime,
    window: dict[str, Any],
    pre_task_buffer_ms: int,
    post_task_buffer_ms: int,
) -> dict[str, Any]:
    return {
        "file": file_name,
        "recording_name": recording_name,
        "recording_start_utc": None,
        "first_row_utc": None,
        "last_row_utc": None,
        "min_recording_timestamp_ms": None,
        "max_recording_timestamp_ms": None,
        "task_start_utc": window.get("task_start_utc") or _iso_utc(task_start),
        "task_end_utc": window.get("task_end_utc") or _iso_utc(task_end),
        "filter_start_utc": _iso_utc(filter_start),
        "filter_end_utc": _iso_utc(filter_end),
        "pre_task_buffer_ms": pre_task_buffer_ms,
        "post_task_buffer_ms": post_task_buffer_ms,
        "overlap": False,
        "rows_before": 0,
        "rows_in_window": 0,
        "rows_after": 0,
        "first_selected_row_utc": None,
        "last_selected_row_utc": None,
    }


def _intervals_overlap(
    a_start: datetime,
    a_end: datetime,
    b_start: datetime,
    b_end: datetime,
) -> bool:
    return a_start <= b_end and a_end >= b_start


def _fill_timestamp_ms_debug(file_info: dict[str, Any], df: pd.DataFrame) -> None:
    ts = pd.to_numeric(df[TS_COL], errors="coerce").dropna()
    if not ts.empty:
        file_info["min_recording_timestamp_ms"] = float(ts.min())
        file_info["max_recording_timestamp_ms"] = float(ts.max())


def _fill_row_utc_debug(
    file_info: dict[str, Any],
    rec_start: datetime,
    rec_min: datetime | None,
    rec_max: datetime | None,
    filter_start: datetime,
    filter_end: datetime,
) -> None:
    file_info["recording_start_utc"] = _iso_utc(rec_start)
    file_info["first_row_utc"] = _iso_utc(rec_min)
    file_info["last_row_utc"] = _iso_utc(rec_max)
    if rec_min is not None and rec_max is not None:
        file_info["overlap"] = _intervals_overlap(rec_min, rec_max, filter_start, filter_end)


def _recording_groups(df: pd.DataFrame) -> list[tuple[str | None, pd.DataFrame]]:
    if RECORDING_NAME_COL not in df.columns:
        return [(None, df)]

    groups: list[tuple[str | None, pd.DataFrame]] = []
    recording_names = df[RECORDING_NAME_COL]
    for recording_name in pd.unique(recording_names):
        if pd.isna(recording_name):
            name = None
            group = df.loc[recording_names.isna()]
        else:
            name = str(recording_name)
            group = df.loc[recording_names == recording_name]
        groups.append((name, group.copy()))
    return groups


def _selected_row_bounds(row_utc: pd.Series, mask: pd.Series) -> tuple[str | None, str | None]:
    selected = row_utc.loc[mask].dropna()
    if selected.empty:
        return None, None
    return _iso_utc(selected.min().to_pydatetime()), _iso_utc(selected.max().to_pydatetime())


def list_eye_tracking_files(participant_folder: Path) -> list[Path]:
    files = sorted(participant_folder.glob("EyeTracking*.xlsx"))
    if not files:
        files = sorted(participant_folder.glob("EyeTracking*.xls"))
    return files


def segment_eye_tracking_for_task(
    participant_folder: Path,
    task_folder: Path,
    window: dict[str, Any] | None = None,
    pre_task_buffer_ms: int = DEFAULT_PRE_TASK_BUFFER_MS,
    post_task_buffer_ms: int = DEFAULT_POST_TASK_BUFFER_MS,
    output_suffix: str = "eprime",
    end_key: str = "task_end_utc",
) -> tuple[Path | None, Path | None, dict[str, Any], list[str]]:
    """
    Segment all participant eye-tracking recordings against the task sync window.

    Returns (segment_xlsx_path, report_json_path, report_dict, warnings).
    """
    warnings: list[str] = []
    if window is None:
        window = load_sync_window_json(task_folder)
    if window is None:
        warnings.append(f"{task_folder.name}: sync_window.json missing — eye segmentation skipped.")
        return None, None, {}, warnings

    task_start, task_end = _parse_task_bounds(window, end_key=end_key)
    debug_window = dict(window)
    debug_window["task_end_utc"] = window.get(end_key)
    filter_start = task_start - timedelta(milliseconds=pre_task_buffer_ms)
    filter_end = task_end + timedelta(milliseconds=post_task_buffer_ms)
    candidates = list_eye_tracking_files(participant_folder)

    report: dict[str, Any] = {
        "files_checked": [],
        "files_with_overlap": [],
        "recordings_checked": [],
        "recordings_with_overlap": [],
        "rows_before": 0,
        "rows_after": 0,
        "task_start_utc": window.get("task_start_utc"),
        "task_end_utc": window.get(end_key),
        "window_type": output_suffix,
        "window_end_key": end_key,
        "filter_start_utc": _iso_utc(filter_start),
        "filter_end_utc": _iso_utc(filter_end),
        "pre_task_buffer_ms": pre_task_buffer_ms,
        "post_task_buffer_ms": post_task_buffer_ms,
        "first_selected_row_utc": None,
        "last_selected_row_utc": None,
        "per_file": [],
        "per_recording": [],
    }

    if not candidates:
        warnings.append("No eye-tracking files found in participant folder — segmentation skipped.")
        return None, None, report, warnings

    segments: list[pd.DataFrame] = []

    for eye_path in candidates:
        report["files_checked"].append(eye_path.name)

        try:
            df = pd.read_excel(eye_path, engine="openpyxl")
        except Exception as exc:
            warnings.append(f"{eye_path.name}: could not read Excel — {exc}")
            report["per_file"].append({"file": eye_path.name, "error": str(exc), "rows_before": 0})
            continue

        report["rows_before"] += int(len(df))
        file_info: dict[str, Any] = {
            "file": eye_path.name,
            "rows_before": int(len(df)),
            "recordings_checked": [],
            "recordings_with_overlap": [],
        }

        if TS_COL not in df.columns:
            warnings.append(f"{eye_path.name}: missing column '{TS_COL}'.")
            file_info["error"] = f"missing column {TS_COL}"
            report["per_file"].append(file_info)
            continue

        recording_groups = _recording_groups(df)
        if RECORDING_NAME_COL not in df.columns:
            warnings.append(f"{eye_path.name}: missing column '{RECORDING_NAME_COL}'; treating workbook as one recording.")

        for group_index, (recording_name, group) in enumerate(recording_groups, start=1):
            recording_label = recording_name or f"(blank recording {group_index})"
            recording_key = f"{eye_path.name}::{recording_label}"
            report["recordings_checked"].append(recording_key)
            file_info["recordings_checked"].append(recording_key)

            recording_info = _new_recording_debug(
                eye_path.name,
                recording_name,
                task_start,
                task_end,
                filter_start,
                filter_end,
                debug_window,
                pre_task_buffer_ms,
                post_task_buffer_ms,
            )
            recording_info["rows_before"] = int(len(group))
            _fill_timestamp_ms_debug(recording_info, group)

            rec_start = _recording_start_utc(group, task_start)
            if rec_start is None:
                recording_info["error"] = "missing recording start UTC"
                report["per_recording"].append(recording_info)
                continue

            row_utc = _row_utc_series(group, rec_start)
            rec_min, rec_max = _recording_span(row_utc)
            if rec_min is None or rec_max is None:
                recording_info["error"] = "no valid timestamps"
                recording_info["recording_start_utc"] = _iso_utc(rec_start)
                report["per_recording"].append(recording_info)
                continue

            _fill_row_utc_debug(recording_info, rec_start, rec_min, rec_max, filter_start, filter_end)

            in_task_mask = (row_utc >= task_start) & (row_utc <= task_end)
            selection_mask = (row_utc >= filter_start) & (row_utc <= filter_end)
            rows_in = int(in_task_mask.sum())
            rows_selected = int(selection_mask.sum())
            recording_info["rows_in_window"] = rows_in
            recording_info["rows_after"] = rows_selected
            first_selected, last_selected = _selected_row_bounds(row_utc, selection_mask)
            recording_info["first_selected_row_utc"] = first_selected
            recording_info["last_selected_row_utc"] = last_selected

            if recording_info["overlap"]:
                report["recordings_with_overlap"].append(recording_key)
                file_info["recordings_with_overlap"].append(recording_key)

            if rows_selected == 0:
                report["per_recording"].append(recording_info)
                continue

            if eye_path.name not in report["files_with_overlap"]:
                report["files_with_overlap"].append(eye_path.name)
            segment = group.loc[selection_mask].copy()
            segment.insert(0, "_row_utc", row_utc.loc[selection_mask].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").values)
            segment.insert(0, "_eye_tracking_recording_name", recording_name)
            segment.insert(0, "_eye_tracking_source_file", eye_path.name)
            segments.append(segment)
            report["per_recording"].append(recording_info)
        report["per_file"].append(file_info)

    report_path = task_folder / f"segmentation_report_{output_suffix}.json"

    if not segments:
        report["rows_after"] = 0
        warnings.append("No eye-tracking rows found after applying the task window buffer.")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return None, report_path, report, warnings

    combined = pd.concat(segments, ignore_index=True)
    report["rows_after"] = int(len(combined))
    selected_bounds = pd.to_datetime(combined["_row_utc"], errors="coerce", utc=True).dropna()
    if not selected_bounds.empty:
        report["first_selected_row_utc"] = _iso_utc(selected_bounds.min().to_pydatetime())
        report["last_selected_row_utc"] = _iso_utc(selected_bounds.max().to_pydatetime())

    out = task_folder / f"processed_eye_tracking_segment_{output_suffix}.xlsx"
    combined.to_excel(out, index=False, engine="openpyxl")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out, report_path, report, warnings


# Backward-compatible alias used by participant_service
def find_and_segment_eye_tracking(
    participant_folder: Path,
    window: dict[str, Any],
    output_task_folder: Path,
    pre_task_buffer_ms: int = DEFAULT_PRE_TASK_BUFFER_MS,
    post_task_buffer_ms: int = DEFAULT_POST_TASK_BUFFER_MS,
    output_suffix: str = "eprime",
    end_key: str = "task_end_utc",
) -> tuple[Path | None, list[str], dict[str, Any]]:
    seg_path, report_path, report, warnings = segment_eye_tracking_for_task(
        participant_folder,
        output_task_folder,
        window=window,
        pre_task_buffer_ms=pre_task_buffer_ms,
        post_task_buffer_ms=post_task_buffer_ms,
        output_suffix=output_suffix,
        end_key=end_key,
    )
    if report_path:
        report["_report_json_path"] = str(report_path.resolve())
    if seg_path:
        report["_segment_xlsx_path"] = str(seg_path.resolve())
    return seg_path, warnings, report
