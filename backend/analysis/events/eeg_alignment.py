"""
Event-Level Phase 2 — map event EEG windows to preprocessed segment sample indices.

Reads event_database.xlsx (read-only) and writes event_database_with_eeg_alignment.xlsx.
Does not extract epochs or compute event-level EEG features.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.analysis.events.event_database import EVENT_DATABASE_FILE, load_event_database
from backend.eeg.preprocessing_exec import load_eeg_preprocessing_audit
from backend.eeg.segmentation import load_eeg_segment_metadata
from domain.storage_layout import (
    EEG_PREPROCESSED_SEGMENT_FILE,
    EEG_PREPROCESSING_AUDIT_FILE,
    EEG_SEGMENT_METADATA_FILE,
)

EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE = "event_database_with_eeg_alignment.xlsx"

STATUS_ALIGNED = "Aligned"
STATUS_OUTSIDE = "Outside EEG segment"
STATUS_PARTIAL_START = "Partial window: starts before EEG segment"
STATUS_PARTIAL_END = "Partial window: ends after EEG segment"
STATUS_PARTIAL_BOTH = (
    "Partial window: starts before EEG segment; ends after EEG segment"
)

ALIGNMENT_COLUMNS: tuple[str, ...] = (
    "eeg_window_start_sample",
    "eeg_window_end_sample",
    "eeg_window_sample_count",
    "eeg_alignment_status",
)

DISPLAY_ALIGNMENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "event_type",
    "eeg_window_start_relative_to_task_seconds",
    "eeg_window_end_relative_to_task_seconds",
    "eeg_window_start_sample",
    "eeg_window_end_sample",
    "eeg_window_sample_count",
    "eeg_alignment_status",
)


def _seconds_to_sample_index(seconds: float, sampling_rate_hz: float) -> int:
    return int(math.floor(float(seconds) * float(sampling_rate_hz) + 1e-9))


def _segment_task_bounds(segment_meta: dict[str, Any]) -> tuple[float, float]:
    offset_s = float(segment_meta.get("missing_task_start_seconds") or 0.0)
    duration_s = float(segment_meta.get("segment_duration_seconds") or 0.0)
    return offset_s, offset_s + duration_s


def _load_preprocessed_sample_count(task_folder: Path, audit: dict[str, Any]) -> int:
    shape = audit.get("preprocessed_shape")
    if isinstance(shape, list) and len(shape) >= 1:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            pass

    segment_path = task_folder / EEG_PREPROCESSED_SEGMENT_FILE
    array = np.load(segment_path, mmap_mode="r")
    return int(array.shape[0])


def align_event_database_to_eeg(
    event_database: pd.DataFrame,
    *,
    segment_meta: dict[str, Any],
    sampling_rate_hz: float,
    segment_sample_count: int,
) -> pd.DataFrame:
    if event_database.empty:
        aligned = event_database.copy()
        for column in ALIGNMENT_COLUMNS:
            aligned[column] = pd.Series(dtype="object")
        return aligned

    segment_start_task_s, segment_end_task_s = _segment_task_bounds(segment_meta)
    last_sample = max(0, int(segment_sample_count) - 1)
    fs = float(sampling_rate_hz)

    rows: list[dict[str, Any]] = []
    for _, event in event_database.iterrows():
        row = event.to_dict()
        start_rel = row.get("eeg_window_start_relative_to_task_seconds")
        end_rel = row.get("eeg_window_end_relative_to_task_seconds")
        if pd.isna(start_rel) or pd.isna(end_rel):
            row.update(
                {
                    "eeg_window_start_sample": None,
                    "eeg_window_end_sample": None,
                    "eeg_window_sample_count": 0,
                    "eeg_alignment_status": STATUS_OUTSIDE,
                }
            )
            rows.append(row)
            continue

        start_rel_f = float(start_rel)
        end_rel_f = float(end_rel)
        if end_rel_f < segment_start_task_s or start_rel_f > segment_end_task_s:
            row.update(
                {
                    "eeg_window_start_sample": None,
                    "eeg_window_end_sample": None,
                    "eeg_window_sample_count": 0,
                    "eeg_alignment_status": STATUS_OUTSIDE,
                }
            )
            rows.append(row)
            continue

        partial_start = start_rel_f < segment_start_task_s
        partial_end = end_rel_f > segment_end_task_s

        start_sample = _seconds_to_sample_index(start_rel_f - segment_start_task_s, fs)
        end_sample = _seconds_to_sample_index(end_rel_f - segment_start_task_s, fs) - 1

        if partial_start:
            start_sample = 0
        if partial_end:
            end_sample = last_sample

        start_sample = max(0, min(start_sample, last_sample))
        end_sample = max(0, min(end_sample, last_sample))

        if start_sample > end_sample:
            row.update(
                {
                    "eeg_window_start_sample": None,
                    "eeg_window_end_sample": None,
                    "eeg_window_sample_count": 0,
                    "eeg_alignment_status": STATUS_OUTSIDE,
                }
            )
            rows.append(row)
            continue

        if partial_start and partial_end:
            status = STATUS_PARTIAL_BOTH
        elif partial_start:
            status = STATUS_PARTIAL_START
        elif partial_end:
            status = STATUS_PARTIAL_END
        else:
            status = STATUS_ALIGNED

        row.update(
            {
                "eeg_window_start_sample": int(start_sample),
                "eeg_window_end_sample": int(end_sample),
                "eeg_window_sample_count": int(end_sample - start_sample + 1),
                "eeg_alignment_status": status,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_eeg_alignment(aligned_database: pd.DataFrame) -> dict[str, int]:
    if aligned_database.empty or "eeg_alignment_status" not in aligned_database.columns:
        return {
            "total_events": 0,
            "aligned_events": 0,
            "partial_window_events": 0,
            "outside_eeg_events": 0,
        }

    status = aligned_database["eeg_alignment_status"].astype("string")
    total = int(len(aligned_database))
    aligned = int((status == STATUS_ALIGNED).sum())
    outside = int((status == STATUS_OUTSIDE).sum())
    partial = int(status.str.startswith("Partial window", na=False).sum())
    return {
        "total_events": total,
        "aligned_events": aligned,
        "partial_window_events": partial,
        "outside_eeg_events": outside,
    }


def aligned_event_window_table(aligned_database: pd.DataFrame) -> pd.DataFrame:
    if aligned_database.empty or "eeg_alignment_status" not in aligned_database.columns:
        return pd.DataFrame(columns=list(DISPLAY_ALIGNMENT_COLUMNS))

    usable = aligned_database.loc[
        aligned_database["eeg_alignment_status"].astype("string") != STATUS_OUTSIDE
    ].copy()
    columns = [column for column in DISPLAY_ALIGNMENT_COLUMNS if column in usable.columns]
    return usable[columns].reset_index(drop=True)


def build_event_database_with_eeg_alignment(
    task_folder: Path,
    *,
    event_database: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []

    if event_database is None:
        event_database = load_event_database(task_folder)
    if event_database is None or event_database.empty:
        warnings.append(f"{EVENT_DATABASE_FILE} missing or empty — EEG alignment skipped.")
        empty = pd.DataFrame()
        return empty, warnings

    def _unaligned_copy() -> pd.DataFrame:
        copy = event_database.copy()
        for column in ALIGNMENT_COLUMNS:
            copy[column] = None
        if not copy.empty:
            copy["eeg_window_sample_count"] = 0
            copy["eeg_alignment_status"] = STATUS_OUTSIDE
        return copy

    segment_meta = load_eeg_segment_metadata(task_folder)
    if segment_meta is None:
        warnings.append(f"{EEG_SEGMENT_METADATA_FILE} missing — EEG alignment skipped.")
        return _unaligned_copy(), warnings
    if not segment_meta.get("ok"):
        warnings.append("EEG segmentation metadata is not OK — EEG alignment skipped.")
        return _unaligned_copy(), warnings

    audit = load_eeg_preprocessing_audit(task_folder)
    if audit is None:
        warnings.append(f"{EEG_PREPROCESSING_AUDIT_FILE} missing — EEG alignment skipped.")
        return _unaligned_copy(), warnings
    if not audit.get("preprocessing_completed"):
        message = audit.get("error_message") or "preprocessing not completed"
        warnings.append(f"EEG preprocessing incomplete — EEG alignment skipped ({message}).")
        return _unaligned_copy(), warnings

    preprocessed_path = task_folder / EEG_PREPROCESSED_SEGMENT_FILE
    if not preprocessed_path.is_file():
        warnings.append(f"{EEG_PREPROCESSED_SEGMENT_FILE} missing — EEG alignment skipped.")
        return _unaligned_copy(), warnings

    sampling_rate_hz = audit.get("sampling_rate_hz") or segment_meta.get("sampling_rate_hz")
    if sampling_rate_hz is None or float(sampling_rate_hz) <= 0:
        warnings.append("Sampling rate unavailable — EEG alignment skipped.")
        return _unaligned_copy(), warnings

    segment_sample_count = _load_preprocessed_sample_count(task_folder, audit)
    aligned = align_event_database_to_eeg(
        event_database,
        segment_meta=segment_meta,
        sampling_rate_hz=float(sampling_rate_hz),
        segment_sample_count=segment_sample_count,
    )
    return aligned, warnings


def write_event_database_with_eeg_alignment(task_folder: Path, aligned_database: pd.DataFrame) -> Path:
    out = task_folder / EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE
    aligned_database.to_excel(out, index=False, engine="openpyxl")
    return out


def load_event_database_with_eeg_alignment(task_folder: Path) -> pd.DataFrame | None:
    path = task_folder / EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE
    if not path.is_file():
        return None
    return pd.read_excel(path, engine="openpyxl")


def summarize_event_eeg_alignment_counts(
    task_folder: Path,
    event_type: str,
) -> dict[str, int]:
    """Read-only counts for UI messaging; does not modify alignment or features."""
    database = load_event_database(task_folder)
    aligned_database = load_event_database_with_eeg_alignment(task_folder)

    detected = 0
    if database is not None and not database.empty and "event_type" in database.columns:
        detected = int((database["event_type"].astype("string") == event_type).sum())

    aligned = 0
    outside = 0
    if (
        aligned_database is not None
        and not aligned_database.empty
        and "event_type" in aligned_database.columns
    ):
        subset = aligned_database.loc[aligned_database["event_type"].astype("string") == event_type]
        if not subset.empty and "eeg_alignment_status" in subset.columns:
            status = subset["eeg_alignment_status"].astype("string")
            aligned = int((status == STATUS_ALIGNED).sum())
            outside = int((status == STATUS_OUTSIDE).sum())

    return {
        "detected": detected,
        "aligned": aligned,
        "outside": outside,
    }


def run_eeg_event_alignment(
    task_folder: Path,
    *,
    event_database: pd.DataFrame | None = None,
    force_recompute: bool = False,
) -> dict[str, Any]:
    out_path = task_folder / EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE

    if not force_recompute and out_path.is_file() and event_database is None:
        loaded = load_event_database_with_eeg_alignment(task_folder)
        if loaded is not None:
            summary = summarize_eeg_alignment(loaded)
            return {
                "aligned_database": loaded,
                "alignment_path": str(out_path.resolve()),
                "summary": summary,
                "alignment_table": aligned_event_window_table(loaded),
                "warnings": [],
                "loaded_existing": True,
            }

    aligned, warnings = build_event_database_with_eeg_alignment(
        task_folder,
        event_database=event_database,
    )
    if aligned.empty and warnings:
        return {
            "aligned_database": aligned,
            "alignment_path": None,
            "summary": summarize_eeg_alignment(aligned),
            "alignment_table": aligned_event_window_table(aligned),
            "warnings": warnings,
            "loaded_existing": False,
        }

    out_path = write_event_database_with_eeg_alignment(task_folder, aligned)
    summary = summarize_eeg_alignment(aligned)
    return {
        "aligned_database": aligned,
        "alignment_path": str(out_path.resolve()),
        "summary": summary,
        "alignment_table": aligned_event_window_table(aligned),
        "warnings": warnings,
        "loaded_existing": False,
    }
