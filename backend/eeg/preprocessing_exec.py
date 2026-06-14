"""
EEG Phase 8 — basic deterministic preprocessing on the segmented BVRF slice.

Applies band-pass, notch, and average reference only.
Raw BVRF files are never modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from backend.eeg.preprocessing import load_eeg_preprocessing_plan
from backend.eeg.raw_signal_qc import load_eeg_segment_array
from domain.storage_layout import (
    EEG_PREPROCESSED_SEGMENT_FILE,
    EEG_PREPROCESSING_AUDIT_FILE,
)

PREPROCESSING_NOTES = (
    "Basic preprocessing only. No ICA or artifact rejection applied."
)


def _bandpass_filter(
    data: np.ndarray,
    sampling_rate_hz: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    nyquist = sampling_rate_hz / 2.0
    if high_hz >= nyquist:
        raise ValueError(
            f"Band-pass high cutoff {high_hz} Hz must be below Nyquist ({nyquist} Hz)."
        )
    b, a = signal.butter(4, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return signal.filtfilt(b, a, data, axis=0)


def _notch_filter(data: np.ndarray, sampling_rate_hz: float, notch_hz: float) -> np.ndarray:
    if notch_hz <= 0 or notch_hz >= sampling_rate_hz / 2.0:
        raise ValueError(f"Notch frequency {notch_hz} Hz is invalid for fs={sampling_rate_hz}.")
    b, a = signal.iirnotch(w0=notch_hz, Q=30.0, fs=sampling_rate_hz)
    return signal.filtfilt(b, a, data, axis=0)


def _average_reference(data: np.ndarray) -> np.ndarray:
    return data - np.mean(data, axis=1, keepdims=True)


def apply_basic_preprocessing(
    data: np.ndarray,
    sampling_rate_hz: float,
    *,
    bandpass_hz: list[float] | tuple[float, float],
    notch_hz: float,
) -> tuple[np.ndarray, list[str], str]:
    """Return preprocessed array (samples × channels), filter labels, reference label."""
    working = np.asarray(data, dtype=np.float64)
    filters_applied: list[str] = []

    low_hz, high_hz = float(bandpass_hz[0]), float(bandpass_hz[1])
    working = _bandpass_filter(working, sampling_rate_hz, low_hz, high_hz)
    filters_applied.append(f"bandpass_{low_hz}_{high_hz}_hz")

    working = _notch_filter(working, sampling_rate_hz, float(notch_hz))
    filters_applied.append(f"notch_{int(notch_hz) if notch_hz.is_integer() else notch_hz}_hz")

    working = _average_reference(working)
    reference_applied = "average"

    return working, filters_applied, reference_applied


def build_preprocessing_audit(
    *,
    success: bool,
    filters_applied: list[str] | None = None,
    reference_applied: str | None = None,
    original_shape: list[int] | None = None,
    preprocessed_shape: list[int] | None = None,
    sampling_rate_hz: float | None = None,
    channels_used: list[str] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "phase": "basic_preprocessing",
        "preprocessing_completed": success,
        "filters_applied": filters_applied or [],
        "reference_applied": reference_applied,
        "original_shape": original_shape,
        "preprocessed_shape": preprocessed_shape,
        "sampling_rate_hz": sampling_rate_hz,
        "channels_used": channels_used or [],
        "channels_removed": [],
        "notes": PREPROCESSING_NOTES,
        "error_message": error_message,
    }


def run_basic_preprocessing(task_folder: Path) -> dict[str, Any]:
    """
    Load segment, apply basic preprocessing, save .npy and audit JSON.

    Does not modify raw BVRF files or other existing audit artifacts.
    """
    plan = load_eeg_preprocessing_plan(task_folder)
    if plan is None:
        audit = build_preprocessing_audit(
            success=False,
            error_message="eeg_preprocessing_plan.json not found.",
        )
        audit_path = write_eeg_preprocessing_audit(task_folder, audit)
        return {
            "preprocessing_completed": False,
            "preprocessed_segment_path": None,
            "audit_path": str(audit_path.resolve()),
            "audit": audit,
        }

    try:
        data, channel_names, context = load_eeg_segment_array(task_folder)
        sampling_rate_hz = context.get("sampling_rate_hz")
        if sampling_rate_hz is None or float(sampling_rate_hz) <= 0:
            raise ValueError("Sampling rate unavailable for preprocessing.")

        fs = float(sampling_rate_hz)
        bandpass_hz = plan.get("bandpass_hz", [0.5, 40])
        notch_hz = float(plan.get("notch_hz", 50))
        reference = plan.get("reference", "average")
        if reference != "average":
            raise ValueError(f"Unsupported reference in plan: {reference!r}")

        original_shape = [int(data.shape[0]), int(data.shape[1])]
        preprocessed, filters_applied, reference_applied = apply_basic_preprocessing(
            data,
            fs,
            bandpass_hz=bandpass_hz,
            notch_hz=notch_hz,
        )
        preprocessed_shape = [int(preprocessed.shape[0]), int(preprocessed.shape[1])]

        segment_path = task_folder / EEG_PREPROCESSED_SEGMENT_FILE
        np.save(segment_path, preprocessed)

        audit = build_preprocessing_audit(
            success=True,
            filters_applied=filters_applied,
            reference_applied=reference_applied,
            original_shape=original_shape,
            preprocessed_shape=preprocessed_shape,
            sampling_rate_hz=fs,
            channels_used=channel_names,
        )
        audit_path = write_eeg_preprocessing_audit(task_folder, audit)
        return {
            "preprocessing_completed": True,
            "preprocessed_segment_path": str(segment_path.resolve()),
            "audit_path": str(audit_path.resolve()),
            "audit": audit,
        }
    except (OSError, ValueError, TypeError) as exc:
        audit = build_preprocessing_audit(
            success=False,
            error_message=str(exc),
        )
        audit_path = write_eeg_preprocessing_audit(task_folder, audit)
        return {
            "preprocessing_completed": False,
            "preprocessed_segment_path": None,
            "audit_path": str(audit_path.resolve()),
            "audit": audit,
        }


def write_eeg_preprocessing_audit(
    task_folder: Path,
    audit: dict[str, Any],
) -> Path:
    out = task_folder / EEG_PREPROCESSING_AUDIT_FILE
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return out


def load_eeg_preprocessing_audit(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / EEG_PREPROCESSING_AUDIT_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
