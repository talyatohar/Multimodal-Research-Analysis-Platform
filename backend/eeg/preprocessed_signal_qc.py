"""
EEG Phase 9 — descriptive QC on the preprocessed segment array.

No channel exclusion, ICA, ERP, PLV, or power features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.eeg.inspection import eeg_raw_dir, load_eeg_metadata_json
from backend.eeg.preprocessing_exec import load_eeg_preprocessing_audit
from domain.storage_layout import (
    EEG_PREPROCESSED_SEGMENT_FILE,
    EEG_PREPROCESSED_QC_FILE,
)

PREPROCESSED_QC_NOTE = (
    "Preprocessed signal QC is descriptive only. No exclusion decision is made."
)

_FLAT_STD_TOLERANCE = 0.0


def _channel_names_from_metadata(metadata: dict[str, Any] | None, channel_count: int) -> list[str]:
    if metadata:
        channels = metadata.get("channels") or []
        names: list[str] = []
        for row in channels:
            if isinstance(row, dict) and row.get("name"):
                names.append(str(row["name"]))
        if len(names) == channel_count:
            return names
    return [f"Ch{i + 1}" for i in range(channel_count)]


def _per_channel_stats(values: np.ndarray) -> dict[str, Any]:
    col = np.asarray(values, dtype=np.float64)
    finite = col[np.isfinite(col)]

    if finite.size == 0:
        return {
            "mean": None,
            "standard_deviation": None,
            "min": None,
            "max": None,
            "flat_channel_flag": True,
        }

    std = float(np.std(finite))
    flat_channel_flag = finite.size == 1 or std <= _FLAT_STD_TOLERANCE
    return {
        "mean": float(np.mean(finite)),
        "standard_deviation": std,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "flat_channel_flag": flat_channel_flag,
    }


def build_eeg_preprocessed_qc(task_folder: Path) -> dict[str, Any]:
    audit = load_eeg_preprocessing_audit(task_folder)
    preprocessing_completed = bool(audit.get("preprocessing_completed")) if audit else False

    base: dict[str, Any] = {
        "phase": "preprocessed_signal_qc",
        "qc_note": PREPROCESSED_QC_NOTE,
        "preprocessing_completed": preprocessing_completed,
        "read_success": False,
        "error_message": None,
        "shape": None,
        "sampling_rate_hz": audit.get("sampling_rate_hz") if audit else None,
        "total_nan_count": None,
        "total_inf_count": None,
        "flat_channel_count": None,
        "per_channel": [],
    }

    segment_path = task_folder / EEG_PREPROCESSED_SEGMENT_FILE
    if not segment_path.is_file():
        base["error_message"] = "eeg_preprocessed_segment.npy not found."
        return base

    if audit is None:
        base["error_message"] = "eeg_preprocessing_audit.json not found."
        return base

    try:
        data = np.load(segment_path)
    except (OSError, ValueError) as exc:
        base["error_message"] = f"Failed to load preprocessed segment: {exc}"
        return base

    if data.ndim != 2:
        base["error_message"] = f"Expected 2-D array, got shape {data.shape}."
        return base

    channel_count = int(data.shape[1])
    metadata = load_eeg_metadata_json(eeg_raw_dir(task_folder))
    metadata_channels = metadata.get("channels") if metadata else None
    if isinstance(metadata_channels, list):
        meta_names = [
            str(row["name"])
            for row in metadata_channels
            if isinstance(row, dict) and row.get("name")
        ]
        if len(meta_names) == channel_count:
            names = meta_names
        else:
            names = _channel_names_from_metadata(metadata, channel_count)
    else:
        audit_channels = audit.get("channels_used") or []
        if isinstance(audit_channels, list) and len(audit_channels) == channel_count:
            names = [str(name) for name in audit_channels]
        else:
            names = _channel_names_from_metadata(metadata, channel_count)

    total_nan = int(np.isnan(data).sum())
    total_inf = int(np.isinf(data).sum())

    per_channel: list[dict[str, Any]] = []
    flat_channel_count = 0
    for idx, name in enumerate(names):
        stats = _per_channel_stats(data[:, idx])
        if stats["flat_channel_flag"]:
            flat_channel_count += 1
        per_channel.append({"channel": name, **stats})

    base.update(
        {
            "read_success": True,
            "shape": [int(data.shape[0]), int(data.shape[1])],
            "total_nan_count": total_nan,
            "total_inf_count": total_inf,
            "flat_channel_count": flat_channel_count,
            "per_channel": per_channel,
        }
    )
    return base


def write_eeg_preprocessed_qc(
    task_folder: Path,
    payload: dict[str, Any] | None = None,
) -> Path:
    out = task_folder / EEG_PREPROCESSED_QC_FILE
    data = payload if payload is not None else build_eeg_preprocessed_qc(task_folder)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def load_eeg_preprocessed_qc(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / EEG_PREPROCESSED_QC_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_eeg_preprocessed_signal_qc(task_folder: Path) -> dict[str, Any]:
    payload = build_eeg_preprocessed_qc(task_folder)
    path = write_eeg_preprocessed_qc(task_folder, payload)
    return {
        "preprocessed_qc_path": str(path.resolve()),
        "result": payload,
    }
