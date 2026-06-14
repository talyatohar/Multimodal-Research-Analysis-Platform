"""
Event-Level Phase 3 — extract preprocessed EEG epochs for aligned events.

Reads event_database_with_eeg_alignment.xlsx (read-only) and writes
event_eeg_epochs_metadata.xlsx / .json. Epoch arrays are not persisted.
Does not compute event-level EEG features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.analysis.events.eeg_alignment import (
    EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE,
    STATUS_OUTSIDE,
    load_event_database_with_eeg_alignment,
)
from backend.eeg.preprocessing_exec import load_eeg_preprocessing_audit
from domain.feature_catalog import EVENT_EEG_DISTRIBUTION_COLUMNS, EVENT_EEG_FEATURES
from domain.storage_layout import EEG_PREPROCESSED_SEGMENT_FILE, EEG_PREPROCESSING_AUDIT_FILE

EVENT_EEG_EPOCHS_METADATA_FILE = "event_eeg_epochs_metadata.xlsx"
EVENT_EEG_EPOCHS_METADATA_JSON = "event_eeg_epochs_metadata.json"

STATUS_EXTRACTED = "Extracted"
STATUS_EXTRACTION_FAILED = "Extraction failed"

METADATA_COLUMNS: tuple[str, ...] = (
    "event_id",
    "event_type",
    "eeg_window_start_sample",
    "eeg_window_end_sample",
    "eeg_window_sample_count",
    "epoch_shape",
    "eeg_alignment_status",
    "epoch_extraction_status",
)

NOT_AVAILABLE = "Not available"


def _usable_events(aligned_database: pd.DataFrame) -> pd.DataFrame:
    if aligned_database.empty:
        return aligned_database.copy()
    if "eeg_alignment_status" not in aligned_database.columns:
        return aligned_database.iloc[0:0].copy()

    status = aligned_database["eeg_alignment_status"].astype("string")
    sample_count = pd.to_numeric(aligned_database.get("eeg_window_sample_count"), errors="coerce").fillna(0)
    mask = (status != STATUS_OUTSIDE) & (sample_count > 0)
    return aligned_database.loc[mask].copy()


def _format_epoch_shape(samples: int, channels: int) -> str:
    return f"{int(samples)}×{int(channels)}"


def _load_preprocessed_segment(task_folder: Path) -> tuple[np.ndarray | None, list[str]]:
    path = task_folder / EEG_PREPROCESSED_SEGMENT_FILE
    if not path.is_file():
        return None, [f"{EEG_PREPROCESSED_SEGMENT_FILE} missing — epoch extraction skipped."]
    try:
        array = np.load(path, mmap_mode="r")
        return array, []
    except (OSError, ValueError) as exc:
        return None, [f"Could not load {EEG_PREPROCESSED_SEGMENT_FILE}: {exc}"]


def extract_event_eeg_epochs(
    aligned_database: pd.DataFrame,
    preprocessed_segment: np.ndarray,
) -> pd.DataFrame:
    usable = _usable_events(aligned_database)
    if usable.empty:
        return pd.DataFrame(columns=list(METADATA_COLUMNS))

    n_samples = int(preprocessed_segment.shape[0])
    n_channels = int(preprocessed_segment.shape[1]) if preprocessed_segment.ndim > 1 else 1
    rows: list[dict[str, Any]] = []

    for _, event in usable.iterrows():
        row = {
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "eeg_window_start_sample": event.get("eeg_window_start_sample"),
            "eeg_window_end_sample": event.get("eeg_window_end_sample"),
            "eeg_window_sample_count": event.get("eeg_window_sample_count"),
            "eeg_alignment_status": event.get("eeg_alignment_status"),
            "epoch_shape": None,
            "epoch_extraction_status": STATUS_EXTRACTION_FAILED,
        }

        try:
            start = int(event["eeg_window_start_sample"])
            end = int(event["eeg_window_end_sample"])
        except (TypeError, ValueError):
            rows.append(row)
            continue

        if start < 0 or end < start or end >= n_samples:
            rows.append(row)
            continue

        try:
            epoch = np.asarray(preprocessed_segment[start : end + 1])
            if epoch.shape[0] != (end - start + 1):
                rows.append(row)
                continue
            channels = int(epoch.shape[1]) if epoch.ndim > 1 else 1
            row["epoch_shape"] = _format_epoch_shape(epoch.shape[0], channels)
            row["epoch_extraction_status"] = STATUS_EXTRACTED
        except (IndexError, ValueError, TypeError):
            pass

        rows.append(row)

    return pd.DataFrame(rows, columns=list(METADATA_COLUMNS))


def build_event_eeg_epochs_metadata(
    task_folder: Path,
    *,
    aligned_database: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []

    if aligned_database is None:
        aligned_database = load_event_database_with_eeg_alignment(task_folder)
    if aligned_database is None or aligned_database.empty:
        warnings.append(f"{EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE} missing or empty — epoch extraction skipped.")
        return pd.DataFrame(columns=list(METADATA_COLUMNS)), warnings

    audit = load_eeg_preprocessing_audit(task_folder)
    if audit is None:
        warnings.append(f"{EEG_PREPROCESSING_AUDIT_FILE} missing — epoch extraction skipped.")
        return pd.DataFrame(columns=list(METADATA_COLUMNS)), warnings
    if not audit.get("preprocessing_completed"):
        message = audit.get("error_message") or "preprocessing not completed"
        warnings.append(f"EEG preprocessing incomplete — epoch extraction skipped ({message}).")
        return pd.DataFrame(columns=list(METADATA_COLUMNS)), warnings

    preprocessed_segment, load_warnings = _load_preprocessed_segment(task_folder)
    warnings.extend(load_warnings)
    if preprocessed_segment is None:
        return pd.DataFrame(columns=list(METADATA_COLUMNS)), warnings

    metadata = extract_event_eeg_epochs(aligned_database, preprocessed_segment)
    return metadata, warnings


def write_event_eeg_epochs_metadata(task_folder: Path, metadata: pd.DataFrame) -> tuple[Path, Path]:
    xlsx_path = task_folder / EVENT_EEG_EPOCHS_METADATA_FILE
    json_path = task_folder / EVENT_EEG_EPOCHS_METADATA_JSON
    metadata.to_excel(xlsx_path, index=False, engine="openpyxl")
    payload = metadata.where(pd.notna(metadata), None).to_dict(orient="records")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return xlsx_path, json_path


def load_event_eeg_epochs_metadata(task_folder: Path) -> pd.DataFrame | None:
    path = task_folder / EVENT_EEG_EPOCHS_METADATA_FILE
    if not path.is_file():
        return None
    return pd.read_excel(path, engine="openpyxl")


def summarize_epoch_extraction(metadata: pd.DataFrame) -> dict[str, int]:
    if metadata.empty:
        return {
            "usable_events": 0,
            "extracted_epochs": 0,
            "failed_extractions": 0,
        }
    status = metadata["epoch_extraction_status"].astype("string")
    extracted = int((status == STATUS_EXTRACTED).sum())
    failed = int((status == STATUS_EXTRACTION_FAILED).sum())
    return {
        "usable_events": int(len(metadata)),
        "extracted_epochs": extracted,
        "failed_extractions": failed,
    }


def build_event_eeg_distribution_table(
    metadata: pd.DataFrame,
    *,
    event_type: str,
) -> pd.DataFrame:
    """
    Populate the Event-Level EEG Distribution table for the selected event type.

    One row per EEG feature with mean / standard_deviation / variance columns.
    Values remain Not available until a later phase computes event-level features.
    """
    _ = metadata.loc[metadata["event_type"] == event_type].copy() if not metadata.empty else metadata

    rows = [
        {
            "feature": feature,
            "mean": NOT_AVAILABLE,
            "standard_deviation": NOT_AVAILABLE,
            "variance": NOT_AVAILABLE,
        }
        for feature in EVENT_EEG_FEATURES
    ]
    return pd.DataFrame(rows, columns=list(EVENT_EEG_DISTRIBUTION_COLUMNS))


def run_eeg_epoch_extraction(
    task_folder: Path,
    *,
    aligned_database: pd.DataFrame | None = None,
    event_type: str | None = None,
    force_recompute: bool = False,
) -> dict[str, Any]:
    xlsx_path = task_folder / EVENT_EEG_EPOCHS_METADATA_FILE
    json_path = task_folder / EVENT_EEG_EPOCHS_METADATA_JSON

    if not force_recompute and xlsx_path.is_file() and json_path.is_file() and aligned_database is None:
        metadata = load_event_eeg_epochs_metadata(task_folder)
        if metadata is not None:
            summary = summarize_epoch_extraction(metadata)
            distribution = (
                build_event_eeg_distribution_table(metadata, event_type=event_type)
                if event_type
                else pd.DataFrame()
            )
            return {
                "metadata": metadata,
                "metadata_xlsx_path": str(xlsx_path.resolve()),
                "metadata_json_path": str(json_path.resolve()),
                "summary": summary,
                "distribution_table": distribution,
                "warnings": [],
                "loaded_existing": True,
            }

    metadata, warnings = build_event_eeg_epochs_metadata(
        task_folder,
        aligned_database=aligned_database,
    )
    if metadata.empty and warnings:
        return {
            "metadata": metadata,
            "metadata_xlsx_path": None,
            "metadata_json_path": None,
            "summary": summarize_epoch_extraction(metadata),
            "distribution_table": (
                build_event_eeg_distribution_table(metadata, event_type=event_type)
                if event_type
                else pd.DataFrame()
            ),
            "warnings": warnings,
            "loaded_existing": False,
        }

    xlsx_path, json_path = write_event_eeg_epochs_metadata(task_folder, metadata)
    summary = summarize_epoch_extraction(metadata)
    distribution = (
        build_event_eeg_distribution_table(metadata, event_type=event_type)
        if event_type
        else pd.DataFrame()
    )
    return {
        "metadata": metadata,
        "metadata_xlsx_path": str(xlsx_path.resolve()),
        "metadata_json_path": str(json_path.resolve()),
        "summary": summary,
        "distribution_table": distribution,
        "warnings": warnings,
        "loaded_existing": False,
    }
