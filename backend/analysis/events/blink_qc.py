"""
Blink contamination QC for event-locked EEG (no ocular correction / ICA is applied).

An event EEG window is flagged ``window_contains_blink`` when it overlaps
  * an EyesNotFound run of >= 80 ms in the eye-tracking segment (same eye selection as
    Table 1), or
  * a blink peak on Fp1/Fp2 in the preprocessed EEG (0.5-10 Hz, |x| > 5 x MAD, >= 300 ms apart).
EyesNotFound-burst EEG is always labelled blink-contaminated. For the other event types
the flagged windows can be excluded from the EEG features (participant setting
``event_eeg_settings.json``: {"exclude_blink_windows": true}).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVENT_EEG_SETTINGS_FILE = "event_eeg_settings.json"
EVENT_BLINK_QC_FILE = "event_blink_qc.json"
BLINK_CONTAMINATED_LABEL = "blink-contaminated, not for interpretation"
BLINK_EVENT_TYPE = "EyesNotFound Bursts"
EYESNOTFOUND_BLINK_MIN_MS = 80.0
FRONTAL_BLINK_CHANNELS = ("Fp1", "Fp2")


def load_event_eeg_settings(participant_folder: Path) -> dict[str, Any]:
    path = participant_folder / EVENT_EEG_SETTINGS_FILE
    settings = {"exclude_blink_windows": False}
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            settings["exclude_blink_windows"] = bool(raw.get("exclude_blink_windows", False))
        except (OSError, json.JSONDecodeError):
            pass
    return settings


def write_event_eeg_settings(participant_folder: Path, exclude_blink_windows: bool) -> Path:
    path = participant_folder / EVENT_EEG_SETTINGS_FILE
    path.write_text(json.dumps({"exclude_blink_windows": bool(exclude_blink_windows)}, indent=2), encoding="utf-8")
    return path


def eyesnotfound_intervals_s(segment: pd.DataFrame, task_start_ms: float, *, type_col: str, ts_col: str) -> list[tuple[float, float]]:
    """Task-relative (start, end) seconds of contiguous EyesNotFound runs >= 80 ms."""
    if segment is None or segment.empty or type_col not in segment.columns or ts_col not in segment.columns:
        return []
    frame = segment[[type_col, ts_col]].copy()
    frame[ts_col] = pd.to_numeric(frame[ts_col], errors="coerce")
    frame = frame.dropna(subset=[ts_col]).sort_values(ts_col, kind="mergesort")
    enf = frame[type_col].astype(str).str.strip().str.casefold().eq("eyesnotfound").to_numpy()
    ts = frame[ts_col].to_numpy(float)
    out: list[tuple[float, float]] = []
    i, n = 0, len(enf)
    while i < n:
        if enf[i]:
            j = i
            while j + 1 < n and enf[j + 1]:
                j += 1
            if ts[j] - ts[i] >= EYESNOTFOUND_BLINK_MIN_MS:
                out.append(((ts[i] - task_start_ms) / 1000.0, (ts[j] - task_start_ms) / 1000.0))
            i = j + 1
        else:
            i += 1
    return out


def frontal_blink_peak_samples(segment: np.ndarray, channel_names: list[str], sampling_rate_hz: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt, find_peaks

    idx = [channel_names.index(c) for c in FRONTAL_BLINK_CHANNELS if c in channel_names]
    if not idx or segment.shape[0] < int(sampling_rate_hz):
        return np.array([], dtype=int)
    x = np.asarray(segment[:, idx], dtype=float).mean(axis=1)
    nyq = sampling_rate_hz / 2.0
    b, a = butter(2, [0.5 / nyq, 10.0 / nyq], btype="band")
    f = filtfilt(b, a, x - x.mean())
    mad = float(np.median(np.abs(f - np.median(f)))) * 1.4826
    if mad <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(np.abs(f), height=5 * mad, distance=int(0.3 * sampling_rate_hz))
    return peaks


def flag_blink_windows(
    events: pd.DataFrame,
    *,
    enf_intervals_s: list[tuple[float, float]],
    peak_samples: np.ndarray,
) -> pd.DataFrame:
    """Adds window_contains_blink / blink_source columns to an aligned event table."""
    out = events.copy()
    contains, source = [], []
    for _, ev in out.iterrows():
        reasons = []
        s_rel = ev.get("eeg_window_start_relative_to_task_seconds")
        e_rel = ev.get("eeg_window_end_relative_to_task_seconds")
        if pd.notna(s_rel) and pd.notna(e_rel):
            if any(a <= float(e_rel) and b >= float(s_rel) for a, b in enf_intervals_s):
                reasons.append("EyesNotFound>=80ms")
        s_smp, e_smp = ev.get("eeg_window_start_sample"), ev.get("eeg_window_end_sample")
        if pd.notna(s_smp) and pd.notna(e_smp) and len(peak_samples):
            if np.any((peak_samples >= int(s_smp)) & (peak_samples <= int(e_smp))):
                reasons.append("Fp1/Fp2 blink peak")
        contains.append(bool(reasons))
        source.append("; ".join(reasons))
    out["window_contains_blink"] = contains
    out["blink_source"] = source
    return out


def summarize_blink_qc(flagged: pd.DataFrame, *, excluded: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {"exclude_blink_windows": excluded, "by_event_type": {}}
    if flagged.empty or "event_type" not in flagged.columns:
        return summary
    for event_type, group in flagged.groupby("event_type"):
        n = int(len(group))
        k = int(group["window_contains_blink"].sum())
        summary["by_event_type"][str(event_type)] = {
            "event_windows": n,
            "windows_with_blink": k,
            "share_with_blink": round(k / n, 3) if n else None,
            "excluded_from_eeg_features": (k if excluded and event_type != BLINK_EVENT_TYPE else 0),
            "interpretation": BLINK_CONTAMINATED_LABEL if event_type == BLINK_EVENT_TYPE else "",
        }
    return summary
