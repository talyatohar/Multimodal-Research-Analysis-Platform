"""
EEG Phase 2 — raw BVRF signal access for segmented sample ranges.

Reads only the requested sample slice from .bvrd; does not modify raw files
or persist EEG matrices to disk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from backend.eeg.inspection import _discover_bvrf_sets, parse_bvrh
from backend.eeg.inspection import eeg_raw_dir
from backend.eeg.segmentation import load_eeg_segment_metadata
from domain.storage_layout import EEG_SEGMENT_INFO_FILE

_NUMPY_DTYPES = {
    "Int16": np.int16,
    "Int32": np.int32,
    "Single": np.float32,
    "Double": np.float64,
}

_DURATION_SAMPLE_TOLERANCE = 1.0  # samples


def _find_primary_bvrf_set(eeg_raw_folder: Path) -> tuple[str | None, dict[str, Path | None]]:
    grouped = _discover_bvrf_sets(eeg_raw_folder)
    for basename, files in sorted(grouped.items()):
        if files.get(".bvrh") is not None and files.get(".bvrd") is not None:
            return basename, files
    return None, {}


def read_bvrd_segment(
    bvrd_path: Path,
    *,
    channel_count: int,
    numeric_data_type: str,
    start_sample_index: int,
    end_sample_index: int,
) -> np.ndarray:
    """
    Read inclusive sample range from .bvrd.

    Returns array with shape (n_samples, n_channels) — samples along axis 0.
    """
    dtype = _NUMPY_DTYPES.get(numeric_data_type)
    if dtype is None:
        raise ValueError(f"Unsupported BVRF numeric data type: {numeric_data_type!r}")
    if channel_count <= 0:
        raise ValueError("channel_count must be positive")
    if end_sample_index < start_sample_index:
        raise ValueError("end_sample_index must be >= start_sample_index")

    itemsize = int(np.dtype(dtype).itemsize)
    frame_bytes = channel_count * itemsize
    n_samples = end_sample_index - start_sample_index + 1
    offset = start_sample_index * frame_bytes
    count = n_samples * channel_count

    with bvrd_path.open("rb") as handle:
        handle.seek(offset)
        raw = np.frombuffer(handle.read(count * itemsize), dtype=dtype)

    if raw.size != count:
        raise ValueError(
            f"Expected {count} values from .bvrd but read {raw.size} "
            f"(sample indices {start_sample_index}–{end_sample_index})."
        )

    return raw.reshape((channel_count, n_samples), order="F").T


def build_eeg_segment_info(task_folder: Path) -> dict[str, Any]:
    warnings: list[str] = []
    segment_meta = load_eeg_segment_metadata(task_folder)
    if segment_meta is None:
        warnings.append("eeg_segment_metadata.json missing — raw access verification skipped.")
        return {"ok": False, "read_successful": False, "warnings": warnings}

    if not segment_meta.get("ok"):
        warnings.append("EEG segmentation was not successful — raw access verification skipped.")
        return {"ok": False, "read_successful": False, "warnings": warnings}

    start_idx = segment_meta.get("eeg_start_sample_index")
    end_idx = segment_meta.get("eeg_end_sample_index")
    if start_idx is None or end_idx is None:
        warnings.append("Segment sample indices unavailable — raw access verification skipped.")
        return {"ok": False, "read_successful": False, "warnings": warnings}

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    raw_folder = eeg_raw_dir(task_folder)
    basename, files = _find_primary_bvrf_set(raw_folder)
    bvrh_path = files.get(".bvrh")
    bvrd_path = files.get(".bvrd")
    if bvrh_path is None or bvrd_path is None:
        warnings.append("BVRF .bvrh / .bvrd pair not found in EEG_raw/.")
        return {"ok": False, "read_successful": False, "warnings": warnings}

    header_meta, header_warnings = parse_bvrh(bvrh_path)
    warnings.extend(header_warnings)

    channel_count = header_meta.get("channel_count")
    numeric_data_type = header_meta.get("numeric_data_type")
    sampling_rate_hz = header_meta.get("sampling_rate_hz") or segment_meta.get("sampling_rate_hz")
    segment_duration_seconds = segment_meta.get("segment_duration_seconds")

    if channel_count is None or not numeric_data_type:
        warnings.append("BVRF header missing channel count or numeric data type.")
        return {"ok": False, "read_successful": False, "warnings": warnings}

    raw_data_file_size_bytes = bvrd_path.stat().st_size
    read_successful = False
    shape: list[int] | None = None
    sample_count: int | None = None

    try:
        segment_data = read_bvrd_segment(
            bvrd_path,
            channel_count=int(channel_count),
            numeric_data_type=str(numeric_data_type),
            start_sample_index=start_idx,
            end_sample_index=end_idx,
        )
        shape = [int(segment_data.shape[0]), int(segment_data.shape[1])]
        sample_count = shape[0]
        read_successful = True
        del segment_data
    except (OSError, ValueError) as exc:
        warnings.append(f"Failed to read BVRF segment: {exc}")

    expected_samples = end_idx - start_idx + 1
    sample_count_matches_indices = sample_count == expected_samples if sample_count is not None else False
    channel_count_matches_header = (
        shape is not None and shape[1] == int(channel_count)
    )

    sample_count_matches_duration = False
    if (
        sample_count is not None
        and sampling_rate_hz
        and segment_duration_seconds is not None
    ):
        expected_from_duration = segment_duration_seconds * float(sampling_rate_hz)
        sample_count_matches_duration = (
            abs(sample_count - expected_from_duration) <= _DURATION_SAMPLE_TOLERANCE
        )
    elif sample_count is not None and segment_duration_seconds is None:
        warnings.append("segment_duration_seconds unavailable — duration check skipped.")

    if read_successful and not sample_count_matches_indices:
        warnings.append(
            f"Sample count {sample_count} does not match index range "
            f"({start_idx}–{end_idx}, expected {expected_samples})."
        )
    if read_successful and not channel_count_matches_header:
        warnings.append(
            f"Read channel dimension {shape[1] if shape else '—'} "
            f"does not match .bvrh channel count {channel_count}."
        )
    if read_successful and sampling_rate_hz and not sample_count_matches_duration:
        warnings.append(
            "sample_count does not match expected duration × sampling rate."
        )

    ok = (
        read_successful
        and sample_count_matches_indices
        and channel_count_matches_header
        and (not sampling_rate_hz or sample_count_matches_duration)
    )

    return {
        "ok": ok,
        "read_successful": read_successful,
        "phase": "raw_signal_access",
        "bvrf_basename": basename,
        "files": {
            "header": str(bvrh_path.resolve()),
            "data": str(bvrd_path.resolve()),
            "segment_metadata": str((task_folder / "eeg_segment_metadata.json").resolve()),
        },
        "eeg_start_sample_index": start_idx,
        "eeg_end_sample_index": end_idx,
        "channel_count": int(channel_count),
        "header_channel_count": int(channel_count),
        "sample_count": sample_count,
        "sampling_rate_hz": sampling_rate_hz,
        "segment_duration_seconds": segment_duration_seconds,
        "shape": shape,
        "raw_data_file_size_bytes": raw_data_file_size_bytes,
        "channel_count_matches_header": channel_count_matches_header,
        "sample_count_matches_indices": sample_count_matches_indices,
        "sample_count_matches_duration": sample_count_matches_duration,
        "warnings": warnings,
    }


def write_eeg_segment_info(
    task_folder: Path,
    info: dict[str, Any] | None = None,
) -> Path:
    payload = info if info is not None else build_eeg_segment_info(task_folder)
    out = task_folder / EEG_SEGMENT_INFO_FILE
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_eeg_segment_info(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / EEG_SEGMENT_INFO_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_eeg_raw_access_verification(task_folder: Path) -> dict[str, Any]:
    """Verify segmented BVRF samples can be read; cache summary to eeg_segment_info.json."""
    info = build_eeg_segment_info(task_folder)
    path = write_eeg_segment_info(task_folder, info)
    return {
        "segment_info_path": str(path.resolve()),
        "info": info,
    }
