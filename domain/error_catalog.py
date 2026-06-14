"""
User-facing error templates (spec).

Missing inputs should block only the dependent module, not the whole record.
"""

from __future__ import annotations


def missing_eeg_bundle() -> str:
    return "EEG analysis cannot be generated because one or more BrainVision files (Task.ahdr / Task.amrk / Task.eeg) are missing."


def missing_eye_tracking() -> str:
    return "Eye-tracking analysis cannot be generated because EyeTracking.xlsx is missing."


def missing_eprime() -> str:
    return "Synchronization cannot proceed because the E-Prime log (Eprime.txt) is missing for this task."


def missing_resting_for_baseline() -> str:
    return "Baseline comparison cannot be generated because Resting state data is missing."


def missing_corsano_timestamp() -> str:
    return "Corsano analysis cannot proceed because required timestamp fields are missing in the physiological export."


def no_tobii_window_match() -> str:
    return "No eye-tracking recording contains the selected task time window."


def generic_missing_file(label: str) -> str:
    return f"Required file missing: {label}. This analysis path is skipped until uploads are complete."
