"""
EEG Phase 5 — raw BVRF read test using eeg_segment_metadata.json.

Verifies .bvrd samples can be read for the segmented index range.
Does not save EEG matrices or modify raw/audit files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.eeg.inspection import eeg_raw_dir, parse_bvrh
from backend.eeg.raw_access import _find_primary_bvrf_set, read_bvrd_segment
from backend.eeg.segmentation import load_eeg_segment_metadata
from domain.storage_layout import EEG_RAW_READ_TEST_FILE


def build_eeg_raw_read_test(task_folder: Path) -> dict[str, Any]:
    segment_meta = load_eeg_segment_metadata(task_folder)
    if segment_meta is None:
        return {
            "phase": "raw_read_test",
            "read_success": False,
            "error_message": "eeg_segment_metadata.json not found.",
            "sampling_rate_hz": None,
            "channel_count": None,
            "start_sample_index": None,
            "end_sample_index": None,
            "expected_sample_count": None,
            "actual_sample_count": None,
            "expected_shape": None,
            "actual_shape": None,
            "data_dtype": None,
            "raw_file_size_bytes": None,
        }

    if not segment_meta.get("ok"):
        return {
            "phase": "raw_read_test",
            "read_success": False,
            "error_message": "EEG segmentation was not successful.",
            "sampling_rate_hz": segment_meta.get("sampling_rate_hz"),
            "channel_count": None,
            "start_sample_index": segment_meta.get("eeg_start_sample_index"),
            "end_sample_index": segment_meta.get("eeg_end_sample_index"),
            "expected_sample_count": None,
            "actual_sample_count": None,
            "expected_shape": None,
            "actual_shape": None,
            "data_dtype": None,
            "raw_file_size_bytes": None,
        }

    start_idx = segment_meta.get("eeg_start_sample_index")
    end_idx = segment_meta.get("eeg_end_sample_index")
    if start_idx is None or end_idx is None:
        return {
            "phase": "raw_read_test",
            "read_success": False,
            "error_message": "eeg_start_sample_index or eeg_end_sample_index missing in segment metadata.",
            "sampling_rate_hz": segment_meta.get("sampling_rate_hz"),
            "channel_count": None,
            "start_sample_index": start_idx,
            "end_sample_index": end_idx,
            "expected_sample_count": None,
            "actual_sample_count": None,
            "expected_shape": None,
            "actual_shape": None,
            "data_dtype": None,
            "raw_file_size_bytes": None,
        }

    start_idx = int(start_idx)
    end_idx = int(end_idx)
    expected_sample_count = end_idx - start_idx + 1

    raw_folder = eeg_raw_dir(task_folder)
    _basename, files = _find_primary_bvrf_set(raw_folder)
    bvrh_path = files.get(".bvrh")
    bvrd_path = files.get(".bvrd")
    if bvrh_path is None or bvrd_path is None:
        return {
            "phase": "raw_read_test",
            "read_success": False,
            "error_message": "BVRF .bvrh / .bvrd pair not found in EEG_raw/.",
            "sampling_rate_hz": segment_meta.get("sampling_rate_hz"),
            "channel_count": None,
            "start_sample_index": start_idx,
            "end_sample_index": end_idx,
            "expected_sample_count": expected_sample_count,
            "actual_sample_count": None,
            "expected_shape": None,
            "actual_shape": None,
            "data_dtype": None,
            "raw_file_size_bytes": None,
        }

    header_meta, _header_warnings = parse_bvrh(bvrh_path)
    channel_count = header_meta.get("channel_count")
    numeric_data_type = header_meta.get("numeric_data_type")
    sampling_rate_hz = header_meta.get("sampling_rate_hz") or segment_meta.get("sampling_rate_hz")
    raw_file_size_bytes = bvrd_path.stat().st_size

    if channel_count is None or not numeric_data_type:
        return {
            "phase": "raw_read_test",
            "read_success": False,
            "error_message": "BVRF header missing channel_count or numeric data type.",
            "sampling_rate_hz": sampling_rate_hz,
            "channel_count": channel_count,
            "start_sample_index": start_idx,
            "end_sample_index": end_idx,
            "expected_sample_count": expected_sample_count,
            "actual_sample_count": None,
            "expected_shape": [expected_sample_count, int(channel_count)] if channel_count else None,
            "actual_shape": None,
            "data_dtype": numeric_data_type,
            "raw_file_size_bytes": raw_file_size_bytes,
        }

    channel_count = int(channel_count)
    expected_shape = [expected_sample_count, channel_count]

    try:
        segment_data = read_bvrd_segment(
            bvrd_path,
            channel_count=channel_count,
            numeric_data_type=str(numeric_data_type),
            start_sample_index=start_idx,
            end_sample_index=end_idx,
        )
        actual_shape = [int(segment_data.shape[0]), int(segment_data.shape[1])]
        actual_sample_count = actual_shape[0]
        del segment_data
        return {
            "phase": "raw_read_test",
            "read_success": True,
            "error_message": None,
            "sampling_rate_hz": sampling_rate_hz,
            "channel_count": channel_count,
            "start_sample_index": start_idx,
            "end_sample_index": end_idx,
            "expected_sample_count": expected_sample_count,
            "actual_sample_count": actual_sample_count,
            "expected_shape": expected_shape,
            "actual_shape": actual_shape,
            "data_dtype": numeric_data_type,
            "raw_file_size_bytes": raw_file_size_bytes,
        }
    except (OSError, ValueError) as exc:
        return {
            "phase": "raw_read_test",
            "read_success": False,
            "error_message": str(exc),
            "sampling_rate_hz": sampling_rate_hz,
            "channel_count": channel_count,
            "start_sample_index": start_idx,
            "end_sample_index": end_idx,
            "expected_sample_count": expected_sample_count,
            "actual_sample_count": None,
            "expected_shape": expected_shape,
            "actual_shape": None,
            "data_dtype": numeric_data_type,
            "raw_file_size_bytes": raw_file_size_bytes,
        }


def write_eeg_raw_read_test(
    task_folder: Path,
    payload: dict[str, Any] | None = None,
) -> Path:
    out = task_folder / EEG_RAW_READ_TEST_FILE
    data = payload if payload is not None else build_eeg_raw_read_test(task_folder)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def load_eeg_raw_read_test(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / EEG_RAW_READ_TEST_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_eeg_raw_read_test(task_folder: Path) -> dict[str, Any]:
    """Run raw read test and cache results to eeg_raw_read_test.json."""
    payload = build_eeg_raw_read_test(task_folder)
    path = write_eeg_raw_read_test(task_folder, payload)
    return {
        "raw_read_test_path": str(path.resolve()),
        "result": payload,
    }
