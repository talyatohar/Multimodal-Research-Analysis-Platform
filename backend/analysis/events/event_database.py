"""
Event-Level Phase 1 — unified event database and event summary tables.

Builds event_database.xlsx and event_summary.xlsx from processed eye-tracking
segments and existing regression_events.xlsx (read-only import).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.analysis.eye_tracking.task_level import (
    DURATION_COL,
    REGRESSION_EVENTS_FILE,
    ROW_UTC_COL,
    SEGMENT_FILE,
    SEGMENT_FILE_BY_WINDOW,
    TS_COL,
    TYPE_COL,
    TYPE_INDEX_COL,
    _event_sequence,
    _parse_utc,
)
from backend.sync.eprime import load_sync_window_json
from domain.feature_catalog import EVENT_SUMMARY_FEATURES, EVENT_TYPES

EVENT_DATABASE_FILE = "event_database.xlsx"
EVENT_SUMMARY_FILE = "event_summary.xlsx"

PRE_EVENT_WINDOW_MS = 500
POST_EVENT_WINDOW_MS = 500
EYESNOTFOUND_BURST_MIN_MS = 500

EVENT_TYPE_REGRESSION = "Regression Events"
EVENT_TYPE_LONG_FIXATION = "Long Fixation Events"
EVENT_TYPE_EYESNOTFOUND = "EyesNotFound Bursts"

REGRESSION_EVENT_DURATION_DEFINITION = (
    "Regression duration = regression saccade duration only; "
    "neighboring fixations are used for classification, not duration."
)

EVENT_DATABASE_COLUMNS: tuple[str, ...] = (
    "participant_id",
    "task_name",
    "event_type",
    "event_id",
    "event_onset_relative_to_task_seconds",
    "event_duration_ms",
    "event_duration_definition",
    "event_end_relative_to_task_seconds",
    "eeg_window_start_relative_to_task_seconds",
    "eeg_window_end_relative_to_task_seconds",
    "pre_event_window_ms",
    "post_event_window_ms",
    "source_file",
    "detection_method",
)


def _segment_path(task_folder: Path, window_type: str) -> Path:
    named = SEGMENT_FILE_BY_WINDOW.get(window_type)
    if named:
        path = task_folder / named
        if path.is_file():
            return path
    legacy = task_folder / SEGMENT_FILE
    if legacy.is_file():
        return legacy
    return task_folder / (named or SEGMENT_FILE)


def _task_start_timestamp_ms(segment: pd.DataFrame, window: dict[str, Any]) -> float | None:
    task_start = _parse_utc(window.get("task_start_utc"))
    if task_start is None or TS_COL not in segment.columns:
        return None

    timestamps = pd.to_numeric(segment[TS_COL], errors="coerce")
    if ROW_UTC_COL in segment.columns:
        row_utc = pd.to_datetime(segment[ROW_UTC_COL], errors="coerce", utc=True)
        in_task = row_utc >= task_start
        if in_task.any():
            return float(timestamps.loc[in_task].min())
    numeric = timestamps.dropna()
    return float(numeric.min()) if not numeric.empty else None


def _relative_seconds(timestamp_ms: float, task_start_ms: float) -> float:
    return (float(timestamp_ms) - float(task_start_ms)) / 1000.0


def _window_fields(onset_relative_s: float, duration_ms: float) -> dict[str, float | int]:
    duration_s = float(duration_ms) / 1000.0
    end_relative_s = onset_relative_s + duration_s
    return {
        "event_onset_relative_to_task_seconds": float(onset_relative_s),
        "event_duration_ms": float(duration_ms),
        "event_end_relative_to_task_seconds": float(end_relative_s),
        "eeg_window_start_relative_to_task_seconds": float(onset_relative_s - 0.5),
        "eeg_window_end_relative_to_task_seconds": float(end_relative_s + 0.5),
        "pre_event_window_ms": PRE_EVENT_WINDOW_MS,
        "post_event_window_ms": POST_EVENT_WINDOW_MS,
    }


def _base_event_row(
    participant_id: str,
    task_name: str,
    event_type: str,
    event_id: str,
    *,
    source_file: str,
    detection_method: str,
    onset_relative_s: float,
    duration_ms: float,
    event_duration_definition: str | None = None,
) -> dict[str, Any]:
    return {
        "participant_id": participant_id,
        "task_name": task_name,
        "event_type": event_type,
        "event_id": event_id,
        "source_file": source_file,
        "detection_method": detection_method,
        "event_duration_definition": event_duration_definition,
        **_window_fields(onset_relative_s, duration_ms),
    }


def _import_regression_events(
    participant_id: str,
    task_name: str,
    task_folder: Path,
    task_start_ms: float,
) -> list[dict[str, Any]]:
    regression_path = task_folder / REGRESSION_EVENTS_FILE
    if not regression_path.is_file():
        return []

    regression_df = pd.read_excel(regression_path, engine="openpyxl")
    if regression_df.empty:
        return []

    if "is_regression" in regression_df.columns:
        rows = regression_df.loc[regression_df["is_regression"] == True]  # noqa: E712
    else:
        rows = regression_df

    events: list[dict[str, Any]] = []
    source = str(regression_path.resolve())
    for idx, row in rows.iterrows():
        onset_ms = row.get("saccade_timestamp_ms")
        duration_ms = row.get("saccade_duration_ms")
        if pd.isna(onset_ms) or pd.isna(duration_ms):
            continue
        onset_relative = _relative_seconds(float(onset_ms), task_start_ms)
        duration = float(duration_ms)
        if duration <= 0:
            continue
        events.append(
            _base_event_row(
                participant_id,
                task_name,
                EVENT_TYPE_REGRESSION,
                f"regression_{len(events) + 1}",
                source_file=source,
                detection_method="imported_from_regression_events.xlsx",
                onset_relative_s=onset_relative,
                duration_ms=duration,
                event_duration_definition=REGRESSION_EVENT_DURATION_DEFINITION,
            )
        )
    return events


def _derive_long_fixation_events(
    participant_id: str,
    task_name: str,
    segment: pd.DataFrame,
    segment_path: Path,
    task_start_ms: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    sequence = _event_sequence(segment, warnings)
    if sequence.empty:
        return []

    fixations = sequence.loc[sequence["_event_type_normalized"] == "fixation"].copy()
    if fixations.empty:
        return []

    durations = pd.to_numeric(fixations["_event_duration_ms"], errors="coerce").dropna()
    if durations.empty:
        return []

    mean_duration = float(durations.mean())
    source = str(segment_path.resolve())
    events: list[dict[str, Any]] = []
    for _, row in fixations.iterrows():
        duration_ms = row.get("_event_duration_ms")
        onset_ms = row.get("_event_timestamp_ms")
        if pd.isna(duration_ms) or pd.isna(onset_ms):
            continue
        duration = float(duration_ms)
        if duration <= mean_duration:
            continue
        events.append(
            _base_event_row(
                participant_id,
                task_name,
                EVENT_TYPE_LONG_FIXATION,
                f"long_fixation_{len(events) + 1}",
                source_file=source,
                detection_method="fixation_duration_above_task_mean",
                onset_relative_s=_relative_seconds(float(onset_ms), task_start_ms),
                duration_ms=duration,
            )
        )
    return events


def _derive_eyesnotfound_bursts(
    participant_id: str,
    task_name: str,
    segment: pd.DataFrame,
    segment_path: Path,
    task_start_ms: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    sequence = _event_sequence(segment, warnings)
    if sequence.empty:
        return []

    bursts = sequence.loc[sequence["_event_type_normalized"] == "eyesnotfound"].copy()
    if bursts.empty:
        return []

    source = str(segment_path.resolve())
    events: list[dict[str, Any]] = []
    for _, row in bursts.iterrows():
        duration_ms = row.get("_event_duration_ms")
        onset_ms = row.get("_event_timestamp_ms")
        if pd.isna(duration_ms) or pd.isna(onset_ms):
            continue
        duration = float(duration_ms)
        if duration < EYESNOTFOUND_BURST_MIN_MS:
            continue
        events.append(
            _base_event_row(
                participant_id,
                task_name,
                EVENT_TYPE_EYESNOTFOUND,
                f"eyesnotfound_{len(events) + 1}",
                source_file=source,
                detection_method="continuous_eyesnotfound_ge_500ms",
                onset_relative_s=_relative_seconds(float(onset_ms), task_start_ms),
                duration_ms=duration,
            )
        )
    return events


def build_event_database(
    task_folder: Path,
    participant_id: str,
    task_name: str,
    *,
    window_type: str = "eprime",
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    segment_path = _segment_path(task_folder, window_type)
    if not segment_path.is_file():
        warnings.append(f"Missing processed eye-tracking segment: {segment_path.name}")
        return pd.DataFrame(columns=list(EVENT_DATABASE_COLUMNS)), warnings

    window = load_sync_window_json(task_folder)
    if window is None:
        warnings.append("sync_window.json missing; cannot compute task-relative event timing.")
        return pd.DataFrame(columns=list(EVENT_DATABASE_COLUMNS)), warnings

    segment = pd.read_excel(segment_path, engine="openpyxl")
    task_start_ms = _task_start_timestamp_ms(segment, window)
    if task_start_ms is None:
        warnings.append("Could not determine task-start recording timestamp for relative event timing.")
        return pd.DataFrame(columns=list(EVENT_DATABASE_COLUMNS)), warnings

    events: list[dict[str, Any]] = []
    events.extend(_import_regression_events(participant_id, task_name, task_folder, task_start_ms))
    events.extend(
        _derive_long_fixation_events(
            participant_id,
            task_name,
            segment,
            segment_path,
            task_start_ms,
            warnings,
        )
    )
    events.extend(
        _derive_eyesnotfound_bursts(
            participant_id,
            task_name,
            segment,
            segment_path,
            task_start_ms,
            warnings,
        )
    )

    if not events:
        return pd.DataFrame(columns=list(EVENT_DATABASE_COLUMNS)), warnings

    database = pd.DataFrame(events, columns=list(EVENT_DATABASE_COLUMNS))
    return database, warnings


def build_event_summary(database: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_type in EVENT_TYPES:
        subset = database.loc[database["event_type"] == event_type, "event_duration_ms"]
        durations = pd.to_numeric(subset, errors="coerce").dropna()
        if durations.empty:
            rows.append(
                {
                    "event_type": event_type,
                    "number_of_events": 0,
                    "mean_event_duration": 0.0,
                    "event_duration_variability": 0.0,
                }
            )
            continue
        rows.append(
            {
                "event_type": event_type,
                "number_of_events": int(len(durations)),
                "mean_event_duration": float(durations.mean()),
                "event_duration_variability": (
                    float(durations.std()) if len(durations) > 1 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows, columns=["event_type", *EVENT_SUMMARY_FEATURES])


def write_event_database(task_folder: Path, database: pd.DataFrame) -> Path:
    out = task_folder / EVENT_DATABASE_FILE
    database.to_excel(out, index=False, engine="openpyxl")
    return out


def write_event_summary(task_folder: Path, summary: pd.DataFrame) -> Path:
    out = task_folder / EVENT_SUMMARY_FILE
    summary.to_excel(out, index=False, engine="openpyxl")
    return out


def load_event_database(task_folder: Path) -> pd.DataFrame | None:
    path = task_folder / EVENT_DATABASE_FILE
    if not path.is_file():
        return None
    return pd.read_excel(path, engine="openpyxl")


def load_event_summary(task_folder: Path) -> pd.DataFrame | None:
    path = task_folder / EVENT_SUMMARY_FILE
    if not path.is_file():
        return None
    return pd.read_excel(path, engine="openpyxl")


def run_event_level_tables(
    task_folder: Path,
    participant_id: str,
    task_name: str,
    *,
    window_type: str = "eprime",
    force_recompute: bool = False,
) -> dict[str, Any]:
    database_path = task_folder / EVENT_DATABASE_FILE
    summary_path = task_folder / EVENT_SUMMARY_FILE

    if not force_recompute and database_path.is_file() and summary_path.is_file():
        database = load_event_database(task_folder)
        summary = load_event_summary(task_folder)
        return {
            "database": database if database is not None else pd.DataFrame(),
            "summary": summary if summary is not None else pd.DataFrame(),
            "database_path": str(database_path.resolve()),
            "summary_path": str(summary_path.resolve()),
            "warnings": [],
            "loaded_existing": True,
        }

    database, warnings = build_event_database(
        task_folder,
        participant_id,
        task_name,
        window_type=window_type,
    )
    summary = build_event_summary(database)
    database_path = write_event_database(task_folder, database)
    summary_path = write_event_summary(task_folder, summary)
    return {
        "database": database,
        "summary": summary,
        "database_path": str(database_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "warnings": warnings,
        "loaded_existing": False,
    }
