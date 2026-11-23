"""
Eye selection and per-eye I-VT fixation classification for Tobii Pro Lab exports.

The Tobii export classifies eye movements once, on the eye selection chosen at export
time (usually "Average"). That classification cannot be split per eye. When a single
eye is selected here, fixations are therefore re-computed from that eye's raw gaze with
an I-VT filter (Tobii I-VT structure: gap fill-in, moving-median noise reduction,
velocity over a ~20 ms window, velocity threshold, merge adjacent fixations, discard
short fixations), and the export's classification columns are replaced in memory.
Raw exports and processed segment files on disk are never modified.

Settings live per participant in ``eye_tracking_settings.json`` (participant folder):
    {"eye_selection": "Left", "ivt": {"merge_max_angle_deg": 1.0, ...}}
Without that file the selection is "Average": the export's own classification and the
original both-eye validity rule are used unchanged (previous behaviour).

On both paths the saccade labels are then consolidated (``consolidate_saccades``): one
saccade is the whole run of non-fixation samples between two consecutive fixations, provided
the run has no tracking loss and lasts at most ``SACCADE_MAX_DURATION_MS``. Fixation and
EyesNotFound samples are never relabelled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EYE_SELECTION_FILE = "eye_tracking_settings.json"
EYE_SELECTIONS = ("Average", "Left", "Right")
DEFAULT_EYE_SELECTION = "Average"

# Tobii I-VT (Fixation) defaults.
DEFAULT_IVT_PARAMETERS: dict[str, float] = {
    "gap_fill_max_ms": 75.0,
    "noise_median_samples": 3,
    "velocity_window_ms": 20.0,
    "velocity_threshold_deg_s": 30.0,
    "merge_max_gap_ms": 75.0,
    "merge_max_angle_deg": 0.5,
    "discard_min_duration_ms": 60.0,
}

TYPE_COL = "Eye movement type"
TYPE_INDEX_COL = "Eye movement type index"
DURATION_COL = "Gaze event duration [ms]"
TS_COL = "Recording timestamp [ms]"
SENSOR_COL = "Sensor"
FIX_X_COL = "Fixation point X [DACS px]"
FIX_Y_COL = "Fixation point Y [DACS px]"
VALID_COLS = {"Left": "Validity left", "Right": "Validity right"}
PUPIL_COLS = {"Left": "Pupil diameter left [mm]", "Right": "Pupil diameter right [mm]"}

# A run of non-fixation samples between two fixations longer than this is not one saccade.
SACCADE_MAX_DURATION_MS = 100.0
FIXATION_LABEL = "Fixation"
SACCADE_LABEL = "Saccade"
EYES_NOT_FOUND_LABEL = "EyesNotFound"
UNCLASSIFIED_LABEL = "Unclassified"


def load_eye_selection_settings(participant_folder: Path) -> dict[str, Any]:
    path = participant_folder / EYE_SELECTION_FILE
    settings: dict[str, Any] = {"eye_selection": DEFAULT_EYE_SELECTION, "ivt": dict(DEFAULT_IVT_PARAMETERS)}
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        selection = str(raw.get("eye_selection", DEFAULT_EYE_SELECTION)).strip().capitalize()
        settings["eye_selection"] = selection if selection in EYE_SELECTIONS else DEFAULT_EYE_SELECTION
        settings["ivt"].update({k: float(v) for k, v in (raw.get("ivt") or {}).items() if k in DEFAULT_IVT_PARAMETERS})
        if raw.get("note"):
            settings["note"] = raw["note"]
    return settings


def participant_settings_mtime(participant_folder: Path) -> float:
    """Latest modification time of the participant's analysis settings files (0 if none)."""
    names = (EYE_SELECTION_FILE, "event_eeg_settings.json")
    times = [(participant_folder / n).stat().st_mtime for n in names if (participant_folder / n).is_file()]
    return max(times) if times else 0.0


def write_eye_selection_settings(participant_folder: Path, eye_selection: str, ivt: dict[str, float] | None = None, note: str | None = None) -> Path:
    if eye_selection not in EYE_SELECTIONS:
        raise ValueError(f"eye_selection must be one of {EYE_SELECTIONS}")
    payload: dict[str, Any] = {"eye_selection": eye_selection}
    if ivt:
        payload["ivt"] = ivt
    if note:
        payload["note"] = note
    path = participant_folder / EYE_SELECTION_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _gaze_unit_vectors(df: pd.DataFrame, eye: str) -> tuple[np.ndarray, np.ndarray]:
    side = eye.lower()
    num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy(float)  # noqa: E731
    gx, gy = num(f"Gaze point {side} X [DACS mm]"), num(f"Gaze point {side} Y [DACS mm]")
    ex, ey, ez = (num(f"Eye position {side} {a} [DACS mm]") for a in ("X", "Y", "Z"))
    valid = (df[VALID_COLS[eye]].astype(str).str.strip() == "Valid").to_numpy()
    v = np.stack([gx - ex, gy - ey, -ez], axis=1)
    v[~valid] = np.nan
    n = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = v / n
    ok = valid & np.isfinite(u).all(axis=1)
    u[~ok] = np.nan
    return u, ok


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def classify_ivt(t_ms: np.ndarray, u: np.ndarray, valid: np.ndarray, p: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample labels (Fixation/Saccade/EyesNotFound/Unclassified), event ids, and the
    usable-sample mask (valid or gap-filled)."""
    n = len(t_ms)
    dt = float(np.median(np.diff(t_ms))) if n > 1 else 8.33
    frame = pd.DataFrame(u)
    # gap fill-in (linear interpolation) for invalid runs shorter than gap_fill_max_ms
    filled = valid.copy()
    for i0, i1 in _runs(~valid):
        if i0 > 0 and i1 < n - 1 and (t_ms[i1 + 1] - t_ms[i0 - 1]) < p["gap_fill_max_ms"]:
            filled[i0:i1 + 1] = True
    frame = frame.interpolate(limit_area="inside")
    frame[~filled] = np.nan
    k = int(p["noise_median_samples"])
    if k > 1:
        frame = frame.rolling(k, center=True, min_periods=1).median()
    U = frame.to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
    half = max(1, int(round(p["velocity_window_ms"] / 2 / dt)))
    vel = np.full(n, np.nan)
    if n > 2 * half:
        a, b = U[: n - 2 * half], U[2 * half:]
        ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1, 1)))
        span = (t_ms[2 * half:] - t_ms[: n - 2 * half]) / 1000.0
        vel[half: n - half] = ang / span
    labels = np.full(n, "Unclassified", dtype=object)
    labels[~filled] = "EyesNotFound"
    labels[np.isfinite(vel) & (vel >= p["velocity_threshold_deg_s"])] = "Saccade"
    is_fix = np.isfinite(vel) & (vel < p["velocity_threshold_deg_s"]) & filled
    fix_runs = _runs(is_fix)
    # merge adjacent fixations
    merged: list[list[Any]] = []
    for i0, i1 in fix_runs:
        centre = np.nanmean(U[i0:i1 + 1], axis=0)
        if merged:
            prev = merged[-1]
            gap = t_ms[i0] - t_ms[prev[1]]
            c1 = prev[2] / np.linalg.norm(prev[2])
            c2 = centre / np.linalg.norm(centre)
            angle = np.degrees(np.arccos(np.clip(float(np.dot(c1, c2)), -1, 1)))
            if gap < p["merge_max_gap_ms"] and angle < p["merge_max_angle_deg"]:
                prev[1] = i1
                prev[2] = np.nanmean(U[prev[0]:i1 + 1], axis=0)
                continue
        merged.append([i0, i1, centre])
    for i0, i1, _c in merged:
        duration = t_ms[i1] - t_ms[i0] + dt
        if duration >= p["discard_min_duration_ms"]:
            labels[i0:i1 + 1] = "Fixation"
        else:
            labels[i0:i1 + 1] = np.where(labels[i0:i1 + 1] == "Saccade", "Saccade", "Unclassified")
    # event ids: consecutive runs of the same label
    change = np.r_[True, labels[1:] != labels[:-1]]
    event_id = np.cumsum(change)
    return labels, event_id, filled


def _norm_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in ("nan", "none", "<na>") else text


def consolidate_saccades(
    df: pd.DataFrame,
    max_saccade_ms: float = SACCADE_MAX_DURATION_MS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Relabel the samples between consecutive fixations so that each fixation pair has at most
    one saccade (in memory only).

    For every run of non-fixation samples between two fixations:
      * the run contains EyesNotFound -> not a saccade: EyesNotFound stays, the rest becomes
        Unclassified;
      * otherwise, if its duration (first run sample to first sample of the next fixation) is
        <= max_saccade_ms -> the whole run is one Saccade event (Unclassified fragments
        absorbed) with that duration;
      * otherwise (longer) -> the whole run is Unclassified.
    Non-fixation runs before the first / after the last fixation are not between two
    fixations: their samples become Unclassified (EyesNotFound and blank samples unchanged).
    Fixation and EyesNotFound samples, their event indices and durations are never changed.
    """
    info: dict[str, Any] = {"saccade_max_duration_ms": float(max_saccade_ms)}
    if df is None or df.empty or TYPE_COL not in df.columns or TS_COL not in df.columns:
        return df, info
    out = df.copy()
    t_all = pd.to_numeric(out[TS_COL], errors="coerce").to_numpy(float)
    pos = np.flatnonzero(np.isfinite(t_all))
    pos = pos[np.argsort(t_all[pos], kind="mergesort")]
    n = len(pos)
    if n == 0:
        return out, info
    t = t_all[pos]
    labels = np.array([_norm_label(v) for v in out[TYPE_COL].to_numpy(dtype=object)[pos]], dtype=object)
    new = labels.copy()
    is_fix = labels == FIXATION_LABEL
    counts = {"runs_between_fixations": 0, "saccades": 0, "tracking_loss_runs": 0, "over_cap_runs": 0, "edge_runs": 0}
    over_cap_durations: list[float] = []
    for i0, i1 in _runs(~is_fix):
        seg = labels[i0:i1 + 1]
        keep = (seg == EYES_NOT_FOUND_LABEL) | (seg == "")
        if i0 == 0 or i1 == n - 1:
            counts["edge_runs"] += 1
            new[i0:i1 + 1] = np.where(keep, seg, UNCLASSIFIED_LABEL)
            continue
        counts["runs_between_fixations"] += 1
        if (seg == EYES_NOT_FOUND_LABEL).any():
            counts["tracking_loss_runs"] += 1
            new[i0:i1 + 1] = np.where(keep, seg, UNCLASSIFIED_LABEL)
            continue
        duration = float(t[i1 + 1] - t[i0])
        if duration > max_saccade_ms:
            counts["over_cap_runs"] += 1
            over_cap_durations.append(duration)
            new[i0:i1 + 1] = UNCLASSIFIED_LABEL
        else:
            counts["saccades"] += 1
            new[i0:i1 + 1] = SACCADE_LABEL

    # Re-index Saccade / Unclassified events (one event per contiguous run); Fixation and
    # EyesNotFound keep their exported/classified index and duration.
    dt = float(np.median(np.diff(t))) if n > 1 else 8.33
    index_full = (
        pd.to_numeric(out[TYPE_INDEX_COL], errors="coerce").to_numpy(float)
        if TYPE_INDEX_COL in out.columns else np.full(len(out), np.nan)
    )
    duration_full = (
        pd.to_numeric(out[DURATION_COL], errors="coerce").to_numpy(float)
        if DURATION_COL in out.columns else np.full(len(out), np.nan)
    )
    type_index = index_full[pos]
    durations = duration_full[pos]
    for label in (SACCADE_LABEL, UNCLASSIFIED_LABEL):
        for k, (i0, i1) in enumerate(_runs(new == label), start=1):
            type_index[i0:i1 + 1] = k
            end = t[i1 + 1] if i1 + 1 < n else t[i1] + dt
            durations[i0:i1 + 1] = round(float(end - t[i0]))
    index_full[pos] = type_index
    duration_full[pos] = durations
    type_values = out[TYPE_COL].to_numpy(dtype=object).copy()
    type_values[pos] = np.where(new == "", type_values[pos], new)
    out[TYPE_COL] = type_values
    out[TYPE_INDEX_COL] = index_full
    out[DURATION_COL] = duration_full
    info.update(counts)
    info["over_cap_durations_ms"] = [round(d, 1) for d in over_cap_durations]
    info["relabelled_samples"] = int((new != labels).sum())
    return out, info


def apply_eye_selection(
    df: pd.DataFrame,
    settings: dict[str, Any],
    *,
    consolidate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the segment with eye-movement columns for the selected eye (in memory only).

    Saccades are then consolidated between consecutive fixations (``consolidate_saccades``);
    ``consolidate=False`` skips that step (diagnostics only)."""
    out, info = _apply_eye_selection_labels(df, settings)
    if consolidate:
        out, sacc_info = consolidate_saccades(out)
        info["saccade_consolidation"] = sacc_info
    return out, info


def _apply_eye_selection_labels(df: pd.DataFrame, settings: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    selection = settings.get("eye_selection", DEFAULT_EYE_SELECTION)
    info: dict[str, Any] = {"eye_selection": selection}
    if selection == "Average":
        info["fixation_classification"] = "Tobii export (Eye movement type), export-time eye selection"
        info["validity_rule"] = "both eyes valid (>=50% of event samples)"
        return df, info
    params = dict(DEFAULT_IVT_PARAMETERS)
    params.update(settings.get("ivt") or {})
    needed = [VALID_COLS[selection], TS_COL] + [f"Gaze point {selection.lower()} {a} [DACS mm]" for a in ("X", "Y")]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        info["error"] = f"missing columns for {selection} eye: {missing}; export classification kept"
        return df, info

    out = df.copy()
    if SENSOR_COL in out.columns:
        sensor = out[SENSOR_COL].astype(str).str.strip()
        out = out.loc[(sensor == "Eye Tracker") | (sensor == "")].copy()
        out = out.loc[out[VALID_COLS[selection]].notna()].copy()
    out = out.sort_values(TS_COL, kind="mergesort")
    t_ms = pd.to_numeric(out[TS_COL], errors="coerce").to_numpy(float)
    u, valid = _gaze_unit_vectors(out, selection)
    labels, event_id, usable = classify_ivt(t_ms, u, valid, params)

    out[TYPE_COL] = labels
    # per-type dense counters, like the Tobii export
    ev = pd.DataFrame({"label": labels, "eid": event_id})
    first = ev.drop_duplicates("eid")
    first = first.assign(type_index=first.groupby("label").cumcount() + 1)
    out[TYPE_INDEX_COL] = ev["eid"].map(first.set_index("eid")["type_index"]).to_numpy()
    dt = float(np.median(np.diff(t_ms))) if len(t_ms) > 1 else 8.33
    ts = pd.Series(t_ms)
    dur = ts.groupby(event_id).transform(lambda s: s.max() - s.min() + dt).to_numpy()
    out[DURATION_COL] = np.round(dur).astype(int)
    gx = pd.to_numeric(out.get(f"Gaze point {selection.lower()} X [DACS px]"), errors="coerce")
    gy = pd.to_numeric(out.get(f"Gaze point {selection.lower()} Y [DACS px]"), errors="coerce")
    fx = gx.groupby(event_id).transform("mean").to_numpy()
    fy = gy.groupby(event_id).transform("mean").to_numpy()
    is_fix = labels == "Fixation"
    out[FIX_X_COL] = np.where(is_fix, np.round(fx), np.nan)
    out[FIX_Y_COL] = np.where(is_fix, np.round(fy), np.nan)
    # validity rule and pupil apply to the selected eye only
    # Gap-filled samples count as valid, as in Tobii's I-VT (they are interpolated, not lost).
    selected_validity = np.where(usable, "Valid", "Invalid")
    for col in VALID_COLS.values():
        out[col] = selected_validity
    if PUPIL_COLS[selection] in out.columns:
        for col in PUPIL_COLS.values():
            out[col] = out[PUPIL_COLS[selection]]

    info.update({
        "fixation_classification": f"I-VT recomputed on the {selection.lower()} eye (export classification replaced in memory)",
        "ivt_parameters": params,
        "validity_rule": f"{selection.lower()} eye valid or gap-filled (>=50% of event samples)",
        "samples_used": int(len(out)),
        "sample_label_pct": {k: round(float(v) * 100, 1) for k, v in pd.Series(labels).value_counts(normalize=True).items()},
    })
    if "note" in settings:
        info["note"] = settings["note"]
    return out, info
