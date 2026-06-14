"""
EEG Phase 6 — descriptive raw-signal QC (no exclusion decisions).

Loads the segmented BVRF slice and reports per-channel summary statistics only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.eeg.inspection import eeg_raw_dir, parse_bvrh
from backend.eeg.raw_access import _find_primary_bvrf_set, read_bvrd_segment
from backend.eeg.segmentation import load_eeg_segment_metadata
from domain.storage_layout import EEG_RAW_SIGNAL_QC_FILE

DESCRIPTIVE_QC_NOTE = "Descriptive QC only. No exclusion decision is made."

_FLAT_STD_TOLERANCE = 0.0


def _channel_names(header_meta: dict[str, Any], channel_count: int) -> list[str]:
    channels = header_meta.get("channels") or []
    names: list[str] = []
    for row in channels:
        if isinstance(row, dict) and row.get("name"):
            names.append(str(row["name"]))
    if len(names) == channel_count:
        return names
    return [f"Ch{i + 1}" for i in range(channel_count)]


def load_eeg_segment_array(task_folder: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Load segmented EEG array (samples × channels) or raise."""
    segment_meta = load_eeg_segment_metadata(task_folder)
    if segment_meta is None:
        raise ValueError("eeg_segment_metadata.json not found.")
    if not segment_meta.get("ok"):
        raise ValueError("EEG segmentation was not successful.")

    start_idx = segment_meta.get("eeg_start_sample_index")
    end_idx = segment_meta.get("eeg_end_sample_index")
    if start_idx is None or end_idx is None:
        raise ValueError("Segment sample indices unavailable.")

    raw_folder = eeg_raw_dir(task_folder)
    _basename, files = _find_primary_bvrf_set(raw_folder)
    bvrh_path = files.get(".bvrh")
    bvrd_path = files.get(".bvrd")
    if bvrh_path is None or bvrd_path is None:
        raise ValueError("BVRF .bvrh / .bvrd pair not found in EEG_raw/.")

    header_meta, _ = parse_bvrh(bvrh_path)
    channel_count = header_meta.get("channel_count")
    numeric_data_type = header_meta.get("numeric_data_type")
    if channel_count is None or not numeric_data_type:
        raise ValueError("BVRF header missing channel_count or numeric data type.")

    channel_count = int(channel_count)
    data = read_bvrd_segment(
        bvrd_path,
        channel_count=channel_count,
        numeric_data_type=str(numeric_data_type),
        start_sample_index=int(start_idx),
        end_sample_index=int(end_idx),
    )
    names = _channel_names(header_meta, channel_count)
    context = {
        "sampling_rate_hz": header_meta.get("sampling_rate_hz") or segment_meta.get("sampling_rate_hz"),
        "start_sample_index": int(start_idx),
        "end_sample_index": int(end_idx),
        "data_dtype": numeric_data_type,
    }
    return data, names, context


def _per_channel_stats(values: np.ndarray) -> dict[str, Any]:
    col = np.asarray(values, dtype=np.float64)
    nan_count = int(np.isnan(col).sum())
    inf_count = int(np.isinf(col).sum())
    finite = col[np.isfinite(col)]

    if finite.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "standard_deviation": None,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "flat_signal": True,
        }

    std = float(np.std(finite))
    flat_signal = finite.size == 1 or std <= _FLAT_STD_TOLERANCE
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "standard_deviation": std,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "flat_signal": flat_signal,
    }


def build_eeg_raw_signal_qc(task_folder: Path) -> dict[str, Any]:
    base: dict[str, Any] = {
        "phase": "descriptive_raw_signal_qc",
        "qc_note": DESCRIPTIVE_QC_NOTE,
        "read_success": False,
        "error_message": None,
        "shape": None,
        "sampling_rate_hz": None,
        "total_nan_count": None,
        "total_inf_count": None,
        "flat_channel_count": None,
        "flat_channels": [],
        "per_channel": [],
    }

    try:
        data, names, context = load_eeg_segment_array(task_folder)
    except (OSError, ValueError) as exc:
        base["error_message"] = str(exc)
        return base

    per_channel: list[dict[str, Any]] = []
    flat_channels: list[str] = []
    total_nan = 0
    total_inf = 0

    for idx, name in enumerate(names):
        stats = _per_channel_stats(data[:, idx])
        total_nan += stats["nan_count"]
        total_inf += stats["inf_count"]
        if stats["flat_signal"]:
            flat_channels.append(name)
        per_channel.append({"channel": name, **stats})

    shape = [int(data.shape[0]), int(data.shape[1])]
    del data

    base.update(
        {
            "read_success": True,
            "shape": shape,
            "sampling_rate_hz": context.get("sampling_rate_hz"),
            "data_dtype": context.get("data_dtype"),
            "start_sample_index": context.get("start_sample_index"),
            "end_sample_index": context.get("end_sample_index"),
            "total_nan_count": total_nan,
            "total_inf_count": total_inf,
            "flat_channel_count": len(flat_channels),
            "flat_channels": flat_channels,
            "per_channel": per_channel,
        }
    )
    return base


def write_eeg_raw_signal_qc(
    task_folder: Path,
    payload: dict[str, Any] | None = None,
) -> Path:
    out = task_folder / EEG_RAW_SIGNAL_QC_FILE
    data = payload if payload is not None else build_eeg_raw_signal_qc(task_folder)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def load_eeg_raw_signal_qc(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / EEG_RAW_SIGNAL_QC_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_eeg_raw_signal_qc(task_folder: Path) -> dict[str, Any]:
    payload = build_eeg_raw_signal_qc(task_folder)
    path = write_eeg_raw_signal_qc(task_folder, payload)
    return {
        "raw_signal_qc_path": str(path.resolve()),
        "result": payload,
    }
