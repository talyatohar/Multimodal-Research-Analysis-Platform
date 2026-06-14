"""Task-level eye-tracking feature extraction from processed Tobii segments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.sync.eprime import load_sync_window_json
from domain.resting_state import (
    RESTING_STATE_REGRESSION_SKIP_MESSAGE,
    is_resting_state_task,
    resting_state_regression_metric_zeros,
)

SEGMENT_FILE = "processed_eye_tracking_segment.xlsx"
SEGMENT_FILE_BY_WINDOW = {
    "eprime": "processed_eye_tracking_segment_eprime.xlsx",
    "manual": "processed_eye_tracking_segment_manual.xlsx",
}
SYNC_FILE = "sync_window.json"
FEATURES_FILE = "task_level_eye_tracking_features.xlsx"
QUALITY_CONTROL_FILE = "quality_control_report.json"
REGRESSION_EVENTS_FILE = "regression_events.xlsx"
REGRESSION_REPORT_FILE = "regression_analysis_report.json"

TYPE_COL = "Eye movement type"
TYPE_INDEX_COL = "Eye movement type index"
DURATION_COL = "Gaze event duration [ms]"
TS_COL = "Recording timestamp [ms]"
VALID_LEFT_COL = "Validity left"
VALID_RIGHT_COL = "Validity right"
PUPIL_LEFT_COL = "Pupil diameter left [mm]"
PUPIL_RIGHT_COL = "Pupil diameter right [mm]"
FIXATION_X_COL = "Fixation point X [DACS px]"
FIXATION_Y_COL = "Fixation point Y [DACS px]"
ROW_UTC_COL = "_row_utc"

FEATURE_COLUMNS = [
    "mean_fixation_duration",
    "fixation_time_percentage",
    "fixation_duration_variability",
    "mean_saccade_duration",
    "saccade_time_percentage",
    "saccade_duration_variability",
    "EyesNotFound_percentage",
    "mean_EyesNotFound_duration",
    "EyesNotFound_duration_variability",
    "mean_pupil_diameter_left",
    "mean_pupil_diameter_right",
    "mean_pupil_diameter",
    "pupil_diameter_variability",
    "left_valid_percentage",
    "right_valid_percentage",
    "combined_valid_percentage",
    "regression_count",
    "regression_percentage",
    "mean_regression_distance",
    "mean_regression_duration",
    "regression_duration_variability",
]


@dataclass
class EyeTrackingTaskAnalysisResult:
    features: pd.DataFrame
    quality_control: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    loaded_existing: bool = False
    features_path: Path | None = None
    quality_control_path: Path | None = None
    message: str | None = None


def _empty_features() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_COLUMNS)


def _parse_utc(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _task_duration_ms(window: dict[str, Any], warnings: list[str], *, end_key: str = "task_end_utc") -> float | None:
    if end_key != "task_end_utc":
        task_start = _parse_utc(window.get("task_start_utc"))
        task_end = _parse_utc(window.get(end_key))
        if task_start is None or task_end is None or task_end <= task_start:
            warnings.append(f"Cannot determine total task duration using {end_key} from sync_window.json.")
            return None
        return (task_end - task_start).total_seconds() * 1000

    raw_duration = window.get("task_duration_ms")
    duration = pd.to_numeric(pd.Series([raw_duration]), errors="coerce").iloc[0]
    if pd.notna(duration) and float(duration) > 0:
        return float(duration)

    task_start = _parse_utc(window.get("task_start_utc"))
    task_end = _parse_utc(window.get("task_end_utc"))
    if task_start is None or task_end is None or task_end <= task_start:
        warnings.append("Cannot determine total task duration from sync_window.json.")
        return None
    return (task_end - task_start).total_seconds() * 1000


def _filter_to_task_window(
    df: pd.DataFrame,
    window: dict[str, Any],
    warnings: list[str],
    *,
    end_key: str = "task_end_utc",
) -> pd.DataFrame:
    if ROW_UTC_COL not in df.columns:
        warnings.append(f"Missing '{ROW_UTC_COL}'; using all processed rows as the task segment.")
        return df.copy()

    task_start = _parse_utc(window.get("task_start_utc"))
    task_end = _parse_utc(window.get(end_key))
    if task_start is None or task_end is None:
        warnings.append(f"Could not parse task_start_utc/{end_key}; using all processed rows.")
        return df.copy()

    row_utc = pd.to_datetime(df[ROW_UTC_COL], errors="coerce", utc=True)
    mask = (row_utc >= task_start) & (row_utc <= task_end)
    filtered = df.loc[mask].copy()
    if filtered.empty and not df.empty:
        warnings.append("No processed rows fall inside the exact task window; analysis output may be empty.")
    return filtered


def _normalise_type(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def _is_numeric_like(series: pd.Series) -> bool:
    converted = pd.to_numeric(series.dropna(), errors="coerce")
    return not converted.empty and converted.notna().all()


def _validity_mask(df: pd.DataFrame, col: str, warnings: list[str]) -> pd.Series | None:
    if col not in df.columns:
        warnings.append(f"Missing '{col}'; related validity filtering/percentage is unavailable.")
        return None

    series = df[col]
    if _is_numeric_like(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric == 0

    text = series.astype("string").str.strip().str.casefold()
    valid_tokens = {"valid", "true", "yes", "ok", "0"}
    invalid_tokens = {"invalid", "false", "no", "1", "2", "3", "4"}
    recognised = text.isin(valid_tokens | invalid_tokens)
    if not bool(recognised.any()):
        warnings.append(f"Could not recognise validity coding in '{col}'.")
        return None
    return text.isin(valid_tokens)


def _percentage(mask: pd.Series | None, total_rows: int) -> float | None:
    if mask is None or total_rows == 0:
        return None
    return float(mask.sum() / total_rows * 100)


def _combined_valid_mask(left: pd.Series | None, right: pd.Series | None) -> pd.Series | None:
    if left is not None and right is not None:
        return left & right
    if left is not None:
        return left
    if right is not None:
        return right
    return None


def _event_validity_by_index(
    rows: pd.DataFrame,
    left_valid: pd.Series | None,
    right_valid: pd.Series | None,
) -> pd.Series | None:
    combined = _combined_valid_mask(left_valid, right_valid)
    if combined is None or TYPE_INDEX_COL not in rows.columns:
        return None
    aligned = combined.loc[rows.index]
    return aligned.groupby(rows[TYPE_INDEX_COL]).all()


def _event_durations(
    df: pd.DataFrame,
    event_type: str,
    left_valid: pd.Series | None,
    right_valid: pd.Series | None,
    warnings: list[str],
    *,
    require_valid: bool,
) -> pd.Series:
    required = [TYPE_COL, TYPE_INDEX_COL, DURATION_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        warnings.append(f"Cannot compute {event_type} duration features; missing columns: {', '.join(missing)}.")
        return pd.Series(dtype="float64")

    type_mask = _normalise_type(df[TYPE_COL]) == event_type.casefold()
    rows = df.loc[type_mask, [TYPE_INDEX_COL, DURATION_COL]].copy()
    if rows.empty:
        return pd.Series(dtype="float64")

    durations = pd.to_numeric(rows[DURATION_COL], errors="coerce")
    rows = rows.loc[durations.notna()].copy()
    rows[DURATION_COL] = durations.loc[durations.notna()]
    if rows.empty:
        warnings.append(f"No numeric '{DURATION_COL}' values found for {event_type} events.")
        return pd.Series(dtype="float64")

    grouped = rows.groupby(TYPE_INDEX_COL, sort=False)[DURATION_COL].max()
    if require_valid:
        event_valid = _event_validity_by_index(rows, left_valid, right_valid)
        if event_valid is None:
            warnings.append(f"{event_type} events were not validity-filtered because validity data is incomplete.")
        else:
            grouped = grouped.loc[event_valid.reindex(grouped.index, fill_value=False)]
    return grouped.astype(float)


def _duration_features(durations: pd.Series, total_task_duration_ms: float | None) -> tuple[float | None, float | None, float | None]:
    if durations.empty:
        return None, 0.0 if total_task_duration_ms else None, None
    mean_duration = float(durations.mean())
    total_percentage = None
    if total_task_duration_ms:
        total_percentage = float(durations.sum() / total_task_duration_ms * 100)
    variability = float(durations.std()) if len(durations) > 1 else 0.0
    return mean_duration, total_percentage, variability


def _numeric_column(df: pd.DataFrame, col: str, warnings: list[str]) -> pd.Series | None:
    if col not in df.columns:
        warnings.append(f"Missing '{col}'; related pupil feature is unavailable.")
        return None
    numeric = pd.to_numeric(df[col], errors="coerce")
    return numeric.dropna()


def _pupil_features(
    df: pd.DataFrame,
    left_valid: pd.Series | None,
    right_valid: pd.Series | None,
    warnings: list[str],
) -> tuple[float | None, float | None, float | None]:
    non_loss_mask = pd.Series(True, index=df.index)
    if TYPE_COL in df.columns:
        non_loss_mask = _normalise_type(df[TYPE_COL]) != "eyesnotfound"
    else:
        warnings.append(f"Missing '{TYPE_COL}'; pupil analysis cannot explicitly exclude EyesNotFound rows.")

    left = _numeric_column(df, PUPIL_LEFT_COL, warnings)
    right = _numeric_column(df, PUPIL_RIGHT_COL, warnings)

    left_values = pd.Series(dtype="float64")
    right_values = pd.Series(dtype="float64")
    if left is not None:
        left_mask = non_loss_mask.loc[left.index]
        if left_valid is not None:
            left_mask = left_mask & left_valid.loc[left.index]
        left_values = left.loc[left_mask]
    if right is not None:
        right_mask = non_loss_mask.loc[right.index]
        if right_valid is not None:
            right_mask = right_mask & right_valid.loc[right.index]
        right_values = right.loc[right_mask]

    mean_left = float(left_values.mean()) if not left_values.empty else None
    mean_right = float(right_values.mean()) if not right_values.empty else None

    pupil_frame = pd.DataFrame(index=df.index)
    if left is not None:
        pupil_frame["left"] = left.reindex(df.index)
    if right is not None:
        pupil_frame["right"] = right.reindex(df.index)
    if pupil_frame.empty:
        return mean_left, mean_right, None

    combined_valid = _combined_valid_mask(left_valid, right_valid)
    combined_mask = non_loss_mask.copy()
    if combined_valid is not None:
        combined_mask = combined_mask & combined_valid
    synchronized = pupil_frame.loc[combined_mask].mean(axis=1, skipna=True).dropna()
    variability = float(synchronized.std()) if len(synchronized) > 1 else (0.0 if len(synchronized) == 1 else None)
    return mean_left, mean_right, variability


def _first_numeric(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.iloc[0])


def _event_sequence(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    required = [TYPE_COL, TYPE_INDEX_COL, DURATION_COL, TS_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        warnings.append(f"Cannot reconstruct event sequence; missing columns: {', '.join(missing)}.")
        return pd.DataFrame()

    rows = df.copy()
    if rows.empty:
        return pd.DataFrame()

    rows["_event_type_normalized"] = _normalise_type(rows[TYPE_COL])
    rows["_event_timestamp_ms"] = pd.to_numeric(rows[TS_COL], errors="coerce")
    rows["_event_duration_ms"] = pd.to_numeric(rows[DURATION_COL], errors="coerce")
    rows["_event_row_count"] = 1
    rows = rows.loc[rows["_event_timestamp_ms"].notna()]
    if rows.empty:
        warnings.append(f"No numeric '{TS_COL}' values available for event sequence reconstruction.")
        return pd.DataFrame()

    aggregations: dict[str, Any] = {
        "_event_type_normalized": "first",
        "_event_timestamp_ms": "min",
        "_event_duration_ms": "max",
    }
    if FIXATION_X_COL in rows.columns:
        aggregations[FIXATION_X_COL] = _first_numeric
    if FIXATION_Y_COL in rows.columns:
        aggregations[FIXATION_Y_COL] = _first_numeric

    aggregations["_event_row_count"] = "sum"
    sequence = rows.groupby(TYPE_INDEX_COL, sort=False).agg(aggregations).reset_index()
    sequence = sequence.sort_values(["_event_timestamp_ms", TYPE_INDEX_COL], kind="mergesort").reset_index(drop=True)
    sequence.attrs["raw_rows_used_for_sequence"] = int(len(rows))
    sequence.attrs["event_sequence_count"] = int(len(sequence))
    sequence.attrs["duplicate_rows_collapsed_count"] = int(len(rows) - len(sequence))
    sequence.attrs["ignored_event_type_count"] = int(
        (~sequence["_event_type_normalized"].isin(["fixation", "saccade"])).sum()
    )
    return sequence


def _largest_gap_threshold(values: pd.Series) -> tuple[float | None, dict[str, Any]]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    clean = clean.loc[clean > 0].sort_values().reset_index(drop=True)
    details: dict[str, Any] = {"sample_count": int(len(clean))}
    if len(clean) < 6:
        details["reason"] = "not enough positive samples for gap estimation"
        return None, details

    gaps = clean.diff().iloc[1:]
    if gaps.empty:
        details["reason"] = "no gaps available"
        return None, details
    max_gap_idx = int(gaps.idxmax())
    max_gap = float(gaps.loc[max_gap_idx])
    lower = float(clean.iloc[max_gap_idx - 1])
    upper = float(clean.iloc[max_gap_idx])
    median = float(clean.median())
    details.update(
        {
            "method": "largest_gap_midpoint",
            "lower_cluster_max": lower,
            "upper_cluster_min": upper,
            "max_gap": max_gap,
            "median": median,
        }
    )
    if max_gap <= max(median * 0.5, 1.0):
        details["reason"] = "largest gap was not distinct enough"
        return None, details
    return (lower + upper) / 2, details


def _lower_distribution_threshold(values: pd.Series, multiplier: float) -> tuple[float | None, dict[str, Any]]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    clean = clean.loc[clean > 0]
    details: dict[str, Any] = {"sample_count": int(len(clean)), "method": "lower_distribution_percentile"}
    if len(clean) < 4:
        details["reason"] = "not enough positive samples for lower-distribution estimation"
        return None, details

    lower_part = clean.loc[clean <= clean.quantile(0.60)]
    if lower_part.empty:
        details["reason"] = "lower distribution was empty"
        return None, details
    base = float(lower_part.quantile(0.75))
    threshold = base * multiplier
    details.update(
        {
            "lower_distribution_p75": base,
            "multiplier": multiplier,
            "raw_threshold": threshold,
        }
    )
    return threshold, details


def _estimate_line_transition_threshold(sequence: pd.DataFrame) -> tuple[float, dict[str, Any], str | None]:
    fallback = 30.0
    fixations = sequence.loc[sequence["_event_type_normalized"] == "fixation"].copy()
    if FIXATION_Y_COL not in fixations.columns:
        return fallback, {
            "method": "fallback",
            "fallback_threshold_px": fallback,
            "reason": f"missing {FIXATION_Y_COL}",
        }, f"Auto line-transition threshold estimation failed; using documented fallback {fallback}px."

    y = pd.to_numeric(fixations[FIXATION_Y_COL], errors="coerce").dropna()
    deltas = y.diff().abs().dropna()
    threshold, details = _largest_gap_threshold(deltas)
    if threshold is not None:
        details["selected_threshold_px"] = float(threshold)
        return float(threshold), details, None

    threshold, lower_details = _lower_distribution_threshold(deltas, multiplier=3.0)
    if threshold is not None:
        lower_details["selected_threshold_px"] = float(threshold)
        lower_details["fallback_from_gap_details"] = details
        return float(threshold), lower_details, None

    return fallback, {
        "method": "fallback",
        "fallback_threshold_px": fallback,
        "gap_details": details,
        "lower_distribution_details": lower_details,
    }, f"Auto line-transition threshold estimation failed; using documented fallback {fallback}px."


def _saccade_transition_deltas(sequence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    saccades = sequence.loc[sequence["_event_type_normalized"] == "saccade"]
    for pos, event in saccades.iterrows():
        previous_fixations = sequence.loc[: pos - 1]
        previous_fixations = previous_fixations.loc[previous_fixations["_event_type_normalized"] == "fixation"]
        next_fixations = sequence.loc[pos + 1 :]
        next_fixations = next_fixations.loc[next_fixations["_event_type_normalized"] == "fixation"]
        if previous_fixations.empty or next_fixations.empty:
            continue
        previous = previous_fixations.iloc[-1]
        next_fixation = next_fixations.iloc[0]
        previous_x = previous.get(FIXATION_X_COL)
        next_x = next_fixation.get(FIXATION_X_COL)
        if pd.isna(previous_x) or pd.isna(next_x):
            continue
        rows.append(
            {
                "saccade_event_index": event[TYPE_INDEX_COL],
                "delta_x": float(next_x) - float(previous_x),
            }
        )
    return pd.DataFrame(rows)


def _nearest_fixation_context(sequence: pd.DataFrame, saccade_position: int) -> tuple[pd.Series | None, pd.Series | None, int, int]:
    previous_events = sequence.loc[: saccade_position - 1]
    previous_fixations = previous_events.loc[previous_events["_event_type_normalized"] == "fixation"]
    next_events = sequence.loc[saccade_position + 1 :]
    next_fixations = next_events.loc[next_events["_event_type_normalized"] == "fixation"]

    previous = previous_fixations.iloc[-1] if not previous_fixations.empty else None
    next_fixation = next_fixations.iloc[0] if not next_fixations.empty else None
    ignored_before = 0 if previous is None else int(saccade_position - int(previous.name) - 1)
    ignored_after = 0 if next_fixation is None else int(int(next_fixation.name) - saccade_position - 1)
    return previous, next_fixation, ignored_before, ignored_after


def _estimate_regression_threshold(sequence: pd.DataFrame) -> tuple[float, dict[str, Any], str | None]:
    fallback = 25.0
    if FIXATION_X_COL not in sequence.columns:
        return fallback, {
            "method": "fallback",
            "fallback_threshold_px": fallback,
            "reason": f"missing {FIXATION_X_COL}",
        }, f"Auto regression threshold estimation failed; using documented fallback {fallback}px."

    transitions = _saccade_transition_deltas(sequence)
    if transitions.empty:
        return fallback, {
            "method": "fallback",
            "fallback_threshold_px": fallback,
            "reason": "no fixation-saccade-fixation transitions",
        }, f"Auto regression threshold estimation failed; using documented fallback {fallback}px."

    threshold, details = _lower_distribution_threshold(transitions["delta_x"].abs(), multiplier=3.0)
    if threshold is not None:
        details["selected_threshold_px"] = float(threshold)
        details["transition_count"] = int(len(transitions))
        return float(threshold), details, None

    return fallback, {
        "method": "fallback",
        "fallback_threshold_px": fallback,
        "transition_count": int(len(transitions)),
        "lower_distribution_details": details,
    }, f"Auto regression threshold estimation failed; using documented fallback {fallback}px."


def _regression_analysis(
    df: pd.DataFrame,
    task_folder: Path,
    warnings: list[str],
    *,
    line_transition_threshold_px: float | None,
    regression_threshold_px: float | None,
    regression_events_file: str = REGRESSION_EVENTS_FILE,
) -> tuple[dict[str, Any], pd.DataFrame]:
    report: dict[str, Any] = {
        "analysis": "task_level_regression",
        "definition": {
            "reading_direction": "Hebrew right-to-left",
            "forward_rule": "delta_x < 0",
            "regression_candidate_rule": "delta_x > 0",
            "same_line_rule": "abs(delta_y) < line_transition_threshold_px",
            "classification_rule": "same_line == True and delta_x > regression_threshold_px",
            "percentage_rule": "regression_count / total_saccade_count * 100",
        },
        "parameters": {},
        "threshold_estimation": {},
        "warnings": [],
        "validation_summary": {},
    }

    required = [TYPE_COL, TYPE_INDEX_COL, FIXATION_X_COL, FIXATION_Y_COL, DURATION_COL, TS_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        msg = f"Regression analysis skipped; missing columns: {', '.join(missing)}."
        warnings.append(msg)
        report["warnings"].append(msg)
        return report, pd.DataFrame()

    if (line_transition_threshold_px is not None and line_transition_threshold_px <= 0) or (
        regression_threshold_px is not None and regression_threshold_px <= 0
    ):
        msg = "Regression analysis skipped; thresholds must be greater than 0 px to avoid tiny noisy movements."
        warnings.append(msg)
        report["warnings"].append(msg)
        return report, pd.DataFrame()

    sequence = _event_sequence(df, warnings)
    if sequence.empty:
        report["validation_summary"] = {
            "total_saccade_count": 0,
            "classifiable_saccade_count": 0,
            "regression_count": 0,
        }
        return report, pd.DataFrame()

    threshold_warnings: list[str] = []
    if line_transition_threshold_px is None:
        line_transition_threshold_px, line_details, line_warning = _estimate_line_transition_threshold(sequence)
        line_source = "auto"
        if line_warning:
            threshold_warnings.append(line_warning)
    else:
        line_details = {"method": "manual_override"}
        line_source = "manual_override"

    if regression_threshold_px is None:
        regression_threshold_px, regression_details, regression_warning = _estimate_regression_threshold(sequence)
        regression_source = "auto"
        if regression_warning:
            threshold_warnings.append(regression_warning)
    else:
        regression_details = {"method": "manual_override"}
        regression_source = "manual_override"

    for warning in threshold_warnings:
        warnings.append(warning)
        report["warnings"].append(warning)

    report["parameters"] = {
        "line_transition_threshold_px": float(line_transition_threshold_px),
        "regression_threshold_px": float(regression_threshold_px),
        "line_transition_threshold_source": line_source,
        "regression_threshold_source": regression_source,
    }
    report["threshold_estimation"] = {
        "line_transition_threshold": line_details,
        "regression_threshold": regression_details,
    }

    saccades = sequence.loc[sequence["_event_type_normalized"] == "saccade"]
    total_saccade_count = int(len(saccades))
    events: list[dict[str, Any]] = []
    candidate_count = 0
    same_line_count = 0
    no_previous_fixation_count = 0
    no_next_fixation_count = 0
    missing_neighbor_count = 0
    missing_coordinate_count = 0
    ignored_intermediate_event_count = 0

    for pos, event in saccades.iterrows():
        previous, next_fixation, ignored_before, ignored_after = _nearest_fixation_context(sequence, int(pos))
        ignored_intermediate_event_count += ignored_before + ignored_after

        if previous is None:
            no_previous_fixation_count += 1
        if next_fixation is None:
            no_next_fixation_count += 1
        if previous is None or next_fixation is None:
            missing_neighbor_count += 1
            continue

        previous_x = previous.get(FIXATION_X_COL)
        previous_y = previous.get(FIXATION_Y_COL)
        next_x = next_fixation.get(FIXATION_X_COL)
        next_y = next_fixation.get(FIXATION_Y_COL)
        if pd.isna(previous_x) or pd.isna(previous_y) or pd.isna(next_x) or pd.isna(next_y):
            missing_coordinate_count += 1
            continue

        delta_x = float(next_x) - float(previous_x)
        delta_y = float(next_y) - float(previous_y)
        same_line = abs(delta_y) < line_transition_threshold_px
        regression_candidate = delta_x > 0
        is_regression = same_line and delta_x > regression_threshold_px
        if regression_candidate:
            candidate_count += 1
        if same_line:
            same_line_count += 1

        events.append(
            {
                "saccade_event_index": event[TYPE_INDEX_COL],
                "saccade_timestamp_ms": float(event["_event_timestamp_ms"]),
                "saccade_duration_ms": float(event["_event_duration_ms"]) if pd.notna(event["_event_duration_ms"]) else None,
                "previous_fixation_event_index": previous[TYPE_INDEX_COL],
                "next_fixation_event_index": next_fixation[TYPE_INDEX_COL],
                "previous_fixation_x": float(previous_x),
                "previous_fixation_y": float(previous_y),
                "next_fixation_x": float(next_x),
                "next_fixation_y": float(next_y),
                "delta_x": delta_x,
                "delta_y": delta_y,
                "same_line": bool(same_line),
                "regression_candidate": bool(regression_candidate),
                "is_regression": bool(is_regression),
                "ignored_intermediate_events_before": ignored_before,
                "ignored_intermediate_events_after": ignored_after,
                "line_transition_threshold_px": float(line_transition_threshold_px),
                "regression_threshold_px": float(regression_threshold_px),
            }
        )

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df.to_excel(task_folder / regression_events_file, index=False, engine="openpyxl")

    regression_rows = events_df.loc[events_df["is_regression"]] if not events_df.empty else pd.DataFrame()
    if "saccade_duration_ms" in regression_rows.columns:
        regression_durations = pd.to_numeric(regression_rows["saccade_duration_ms"], errors="coerce").dropna()
    else:
        regression_durations = pd.Series(dtype="float64")
    report["validation_summary"] = {
        "total_saccade_count": total_saccade_count,
        "classifiable_saccade_count": int(len(events_df)),
        "regression_candidate_count": candidate_count,
        "same_line_saccade_count": same_line_count,
        "regression_count": int(len(regression_rows)),
        "missing_neighbor_count": missing_neighbor_count,
        "no_previous_fixation_count": no_previous_fixation_count,
        "no_next_fixation_count": no_next_fixation_count,
        "missing_coordinate_count": missing_coordinate_count,
        "ignored_intermediate_event_count": ignored_intermediate_event_count,
        "raw_rows_used_for_sequence": sequence.attrs.get("raw_rows_used_for_sequence"),
        "event_sequence_count": sequence.attrs.get("event_sequence_count"),
        "duplicate_rows_collapsed_count": sequence.attrs.get("duplicate_rows_collapsed_count"),
        "ignored_event_type_count": sequence.attrs.get("ignored_event_type_count"),
        "line_transition_excluded_count": int(candidate_count - len(regression_rows)),
        "regression_events_path": str((task_folder / regression_events_file).resolve()) if not events_df.empty else None,
    }
    report["metrics"] = {
        "regression_count": int(len(regression_rows)),
        "regression_percentage": float(len(regression_rows) / total_saccade_count * 100) if total_saccade_count else None,
        "mean_regression_distance": (
            float(pd.to_numeric(regression_rows["delta_x"], errors="coerce").dropna().mean())
            if "delta_x" in regression_rows.columns and not regression_rows.empty
            else None
        ),
        "mean_regression_duration": float(regression_durations.mean()) if not regression_durations.empty else None,
        "regression_duration_variability": (
            float(regression_durations.std()) if len(regression_durations) > 1 else (0.0 if len(regression_durations) == 1 else None)
        ),
    }
    return report, events_df


def _resting_state_regression_skip_result(
    task_folder: Path,
    *,
    regression_events_file: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    regression_zeros = resting_state_regression_metric_zeros()
    events_df = pd.DataFrame()
    events_path = task_folder / regression_events_file
    events_df.to_excel(events_path, index=False, engine="openpyxl")
    report: dict[str, Any] = {
        "analysis": "task_level_regression",
        "skipped": True,
        "skip_reason": RESTING_STATE_REGRESSION_SKIP_MESSAGE,
        "definition": {
            "reading_direction": "Not applicable for Resting state",
            "note": RESTING_STATE_REGRESSION_SKIP_MESSAGE,
        },
        "parameters": {},
        "threshold_estimation": {},
        "warnings": [],
        "validation_summary": {
            "total_saccade_count": 0,
            "classifiable_saccade_count": 0,
            "regression_count": 0,
        },
        "metrics": {
            "regression_count": 0,
            "regression_percentage": regression_zeros["regression_percentage"],
            "mean_regression_distance": regression_zeros["mean_regression_distance"],
            "mean_regression_duration": 0.0,
            "regression_duration_variability": regression_zeros["regression_duration_variability"],
        },
        "regression_events_path": str(events_path.resolve()),
    }
    return report, events_df


def _write_quality_control(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_task_level_eye_tracking_analysis(
    task_folder: Path,
    *,
    force_recompute: bool = False,
    line_transition_threshold_px: float | None = None,
    regression_threshold_px: float | None = None,
    window_type: str = "eprime",
) -> EyeTrackingTaskAnalysisResult:
    suffix = "" if window_type == "eprime" else f"_{window_type}"
    features_path = task_folder / (FEATURES_FILE if window_type == "eprime" else f"task_level_eye_tracking_features{suffix}.xlsx")
    qc_path = task_folder / (QUALITY_CONTROL_FILE if window_type == "eprime" else f"quality_control_report{suffix}.json")
    regression_report_path = task_folder / (
        REGRESSION_REPORT_FILE if window_type == "eprime" else f"regression_analysis_report{suffix}.json"
    )
    regression_events_file = REGRESSION_EVENTS_FILE if window_type == "eprime" else f"regression_events{suffix}.xlsx"
    end_key = "manual_task_end_utc" if window_type == "manual" else "task_end_utc"
    warnings: list[str] = []

    if features_path.is_file() and not force_recompute:
        features = pd.read_excel(features_path, engine="openpyxl")
        if qc_path.is_file():
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
        else:
            qc = {"warning": f"{QUALITY_CONTROL_FILE} is missing; loaded cached feature table only."}
            warnings.append(qc["warning"])
        return EyeTrackingTaskAnalysisResult(
            features=features,
            quality_control=qc,
            warnings=warnings,
            loaded_existing=True,
            features_path=features_path,
            quality_control_path=qc_path if qc_path.is_file() else None,
            message="Loaded existing task-level eye-tracking analysis.",
        )

    segment_path = task_folder / SEGMENT_FILE_BY_WINDOW.get(window_type, SEGMENT_FILE_BY_WINDOW["eprime"])
    sync_path = task_folder / SYNC_FILE
    qc: dict[str, Any] = {
        "analysis": "task_level_eye_tracking",
        "segment_path": str(segment_path.resolve()),
        "sync_window_path": str(sync_path.resolve()),
        "features_path": str(features_path.resolve()),
        "quality_control_path": str(qc_path.resolve()),
        "regression_report_path": str(regression_report_path.resolve()),
        "regression_events_path": str((task_folder / REGRESSION_EVENTS_FILE).resolve()),
        "loaded_existing": False,
        "window_type": window_type,
        "force_recompute": force_recompute,
        "warnings": warnings,
        "missing_columns": [],
        "interpretation_boundaries": [
            "Fixation duration alone cannot distinguish visual from cognitive difficulty.",
            "Saccade timing interpretation depends on reading context.",
            "EyesNotFound can reflect technical tracking failure, blinks, head movement, or disengagement.",
            "Pupil diameter is sensitive to lighting, blinks, tracking loss, and validity.",
            "Validity metrics are quality indicators, not direct behavioral outcomes.",
        ],
        "implemented_preprocessing": [
            "Event durations are aggregated by Eye movement type index before task-level metrics.",
            "Fixation and saccade events require available valid samples across the event.",
            "EyesNotFound is treated as tracking loss and is not validity-filtered away.",
            "Pupil means exclude invalid samples and EyesNotFound rows when those columns are available.",
            "Pupil diameter variability is the standard deviation of the synchronized per-sample left/right pupil average after validity filtering.",
            "Regression detection uses Hebrew right-to-left reading direction, neighboring fixation transitions, same-line filtering by delta_y, and positive delta_x above the configured threshold.",
        ],
        "not_implemented_without_defined_parameters": [
            "Extreme pupil outlier removal",
            "Pupil smoothing",
            "Participant-level normalization",
            "Empirical exclusion thresholds",
        ],
    }

    if not segment_path.is_file():
        warnings.append(f"Missing {SEGMENT_FILE}; task-level eye-tracking analysis was not run.")
        qc["warnings"] = warnings
        _write_quality_control(qc_path, qc)
        return EyeTrackingTaskAnalysisResult(_empty_features(), qc, warnings, features_path=None, quality_control_path=qc_path)

    if not sync_path.is_file():
        warnings.append(f"Missing {SYNC_FILE}; task-level eye-tracking analysis was not run.")
        qc["warnings"] = warnings
        _write_quality_control(qc_path, qc)
        return EyeTrackingTaskAnalysisResult(_empty_features(), qc, warnings, features_path=None, quality_control_path=qc_path)

    window = load_sync_window_json(task_folder) or {}
    raw_df = pd.read_excel(segment_path, engine="openpyxl")
    df = _filter_to_task_window(raw_df, window, warnings, end_key=end_key)
    total_task_duration_ms = _task_duration_ms(window, warnings, end_key=end_key)

    expected_cols = [
        TYPE_COL,
        TYPE_INDEX_COL,
        DURATION_COL,
        VALID_LEFT_COL,
        VALID_RIGHT_COL,
        PUPIL_LEFT_COL,
        PUPIL_RIGHT_COL,
        FIXATION_X_COL,
        FIXATION_Y_COL,
        TS_COL,
    ]
    qc["missing_columns"] = [col for col in expected_cols if col not in raw_df.columns]
    qc["rows_in_processed_segment"] = int(len(raw_df))
    qc["rows_used_for_task_analysis"] = int(len(df))
    qc["task_duration_ms"] = total_task_duration_ms

    left_valid = _validity_mask(df, VALID_LEFT_COL, warnings)
    right_valid = _validity_mask(df, VALID_RIGHT_COL, warnings)
    combined_valid = _combined_valid_mask(left_valid, right_valid)

    fixation_durations = _event_durations(df, "Fixation", left_valid, right_valid, warnings, require_valid=True)
    saccade_durations = _event_durations(df, "Saccade", left_valid, right_valid, warnings, require_valid=True)
    eyes_not_found_durations = _event_durations(
        df,
        "EyesNotFound",
        left_valid,
        right_valid,
        warnings,
        require_valid=False,
    )

    fixation_mean, fixation_pct, fixation_var = _duration_features(fixation_durations, total_task_duration_ms)
    saccade_mean, saccade_pct, saccade_var = _duration_features(saccade_durations, total_task_duration_ms)
    enf_pct_mean, enf_pct, enf_var = _duration_features(eyes_not_found_durations, total_task_duration_ms)
    pupil_left, pupil_right, pupil_var = _pupil_features(df, left_valid, right_valid, warnings)
    pupil_means = [v for v in (pupil_left, pupil_right) if v is not None]
    mean_pupil = float(sum(pupil_means) / len(pupil_means)) if pupil_means else None
    if is_resting_state_task(task_folder.name):
        regression_report, regression_events = _resting_state_regression_skip_result(
            task_folder,
            regression_events_file=regression_events_file,
        )
        qc["regression_detection_info"] = RESTING_STATE_REGRESSION_SKIP_MESSAGE
    else:
        regression_report, regression_events = _regression_analysis(
            df,
            task_folder,
            warnings,
            line_transition_threshold_px=line_transition_threshold_px,
            regression_threshold_px=regression_threshold_px,
            regression_events_file=regression_events_file,
        )
    regression_metrics = regression_report.get("metrics") or {}

    row = {
        "mean_fixation_duration": fixation_mean,
        "fixation_time_percentage": fixation_pct,
        "fixation_duration_variability": fixation_var,
        "mean_saccade_duration": saccade_mean,
        "saccade_time_percentage": saccade_pct,
        "saccade_duration_variability": saccade_var,
        "EyesNotFound_percentage": enf_pct,
        "mean_EyesNotFound_duration": enf_pct_mean,
        "EyesNotFound_duration_variability": enf_var,
        "mean_pupil_diameter_left": pupil_left,
        "mean_pupil_diameter_right": pupil_right,
        "mean_pupil_diameter": mean_pupil,
        "pupil_diameter_variability": pupil_var,
        "left_valid_percentage": _percentage(left_valid, len(df)),
        "right_valid_percentage": _percentage(right_valid, len(df)),
        "combined_valid_percentage": _percentage(combined_valid, len(df)),
        "regression_count": regression_metrics.get("regression_count"),
        "regression_percentage": regression_metrics.get("regression_percentage"),
        "mean_regression_distance": regression_metrics.get("mean_regression_distance"),
        "mean_regression_duration": regression_metrics.get("mean_regression_duration"),
        "regression_duration_variability": regression_metrics.get("regression_duration_variability"),
    }
    features = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    qc["event_counts_after_preprocessing"] = {
        "Fixation": int(len(fixation_durations)),
        "Saccade": int(len(saccade_durations)),
        "EyesNotFound": int(len(eyes_not_found_durations)),
    }
    qc["event_duration_totals_ms"] = {
        "Fixation": float(fixation_durations.sum()) if not fixation_durations.empty else 0.0,
        "Saccade": float(saccade_durations.sum()) if not saccade_durations.empty else 0.0,
        "EyesNotFound": float(eyes_not_found_durations.sum()) if not eyes_not_found_durations.empty else 0.0,
    }
    qc["validity_counts"] = {
        "left_valid_samples": int(left_valid.sum()) if left_valid is not None else None,
        "right_valid_samples": int(right_valid.sum()) if right_valid is not None else None,
        "combined_valid_samples": int(combined_valid.sum()) if combined_valid is not None else None,
        "total_samples": int(len(df)),
    }
    qc["regression_validation"] = regression_report.get("validation_summary", {})
    qc["regression_parameters"] = regression_report.get("parameters", {})
    qc["regression_metrics"] = regression_metrics
    qc["regression_events_rows"] = int(len(regression_events))
    qc["warnings"] = warnings
    regression_report["warnings"] = list(dict.fromkeys(regression_report.get("warnings", []) + warnings))

    features.to_excel(features_path, index=False, engine="openpyxl")
    _write_quality_control(qc_path, qc)
    regression_report_path.write_text(json.dumps(regression_report, indent=2), encoding="utf-8")
    return EyeTrackingTaskAnalysisResult(
        features=features,
        quality_control=qc,
        warnings=warnings,
        loaded_existing=False,
        features_path=features_path,
        quality_control_path=qc_path,
        message="Generated task-level eye-tracking analysis.",
    )
