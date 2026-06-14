"""Build and cache the documented task-level analysis tables."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from backend.analysis.corsano_task_level import CorsanoTaskLevelResult, build_corsano_task_level_features
from backend.analysis.eye_tracking import run_task_level_eye_tracking_analysis
from backend.eeg.task_level_features import load_task_level_eeg_features_json
from domain.feature_catalog import EEG_TASK_FEATURES as TASK_LEVEL_EEG_FEATURE_COLUMNS
from domain.feature_catalog import EEG_TASK_FEATURES, EYE_TASK_FEATURES, PHYSIO_TASK_FEATURES, QC_TASK_FEATURES
from domain.resting_state import (
    eeg_baseline_change_zeros,
    is_resting_state_task,
    physio_baseline_change_zeros,
    resting_state_regression_metric_zeros,
)
TABLE_1_FILE = "table_1_eye_tracking_data.xlsx"
TABLE_2_FILE = "table_2_physiological_data.xlsx"
TABLE_3_FILE = "table_3_eeg_data.xlsx"
TABLE_4_FILE = "table_4_quality_control.xlsx"


@dataclass
class TaskLevelTablesResult:
    tables: dict[str, pd.DataFrame]
    table_paths: dict[str, Path]
    warnings: dict[str, list[str]] = field(default_factory=dict)
    loaded_existing: dict[str, bool] = field(default_factory=dict)
    quality_control: dict[str, Any] = field(default_factory=dict)


def _empty_row(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame([{col: None for col in columns}], columns=list(columns))


def _normalize_cached_row(cached: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if cached.empty:
        return _empty_row(columns)
    source = cached.iloc[0].to_dict()
    return pd.DataFrame([{column: source.get(column) for column in columns}], columns=list(columns))


def _apply_resting_state_table_defaults(
    table: pd.DataFrame,
    task_folder: Path,
    *,
    table_kind: str,
) -> pd.DataFrame:
    if not is_resting_state_task(task_folder.name) or table.empty:
        return table
    updated = table.copy()
    if table_kind == "eye":
        for feature, value in resting_state_regression_metric_zeros().items():
            if feature in updated.columns:
                updated.loc[updated.index[0], feature] = value
    elif table_kind == "physio":
        for feature, value in physio_baseline_change_zeros().items():
            if feature in updated.columns:
                updated.loc[updated.index[0], feature] = value
    elif table_kind == "eeg":
        for feature, value in eeg_baseline_change_zeros().items():
            if feature in updated.columns:
                updated.loc[updated.index[0], feature] = value
    return updated


def _split_corsano_warnings(warnings: list[str]) -> tuple[list[str], list[str]]:
    shared_prefixes = (
        "Corsano file has sparse timestamps",
        "Baseline comparison unavailable:",
    )
    physio_warnings: list[str] = []
    qc_warnings: list[str] = []
    for warning in warnings:
        if warning.startswith("Quality Control:"):
            qc_warnings.append(warning)
            continue
        physio_warnings.append(warning)
        if warning.startswith(shared_prefixes):
            qc_warnings.append(warning)
    return physio_warnings, qc_warnings


def _load_if_cached(path: Path, force_recompute: bool) -> pd.DataFrame | None:
    if path.is_file() and not force_recompute:
        return pd.read_excel(path, engine="openpyxl")
    return None


def _save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_excel(path, index=False, engine="openpyxl")


def _build_eye_table(
    task_folder: Path,
    force_recompute: bool,
    warnings: dict[str, list[str]],
    quality_control: dict[str, Any],
    *,
    line_transition_threshold_px: float | None,
    regression_threshold_px: float | None,
    window_type: str,
) -> tuple[pd.DataFrame, bool]:
    suffix = "" if window_type == "eprime" else f"_{window_type}"
    table_path = task_folder / (TABLE_1_FILE if window_type == "eprime" else f"table_1_eye_tracking_data{suffix}.xlsx")
    cached = _load_if_cached(table_path, force_recompute)
    if cached is not None:
        qc_path = task_folder / ("quality_control_report.json" if window_type == "eprime" else f"quality_control_report{suffix}.json")
        if qc_path.is_file():
            quality_control["eye_tracking"] = json.loads(qc_path.read_text(encoding="utf-8"))
        return _apply_resting_state_table_defaults(
            _normalize_cached_row(cached, EYE_TASK_FEATURES),
            task_folder,
            table_kind="eye",
        ), True

    analysis = run_task_level_eye_tracking_analysis(
        task_folder,
        force_recompute=force_recompute,
        line_transition_threshold_px=line_transition_threshold_px,
        regression_threshold_px=regression_threshold_px,
        window_type=window_type,
    )
    warnings["Table 1 · Eye Tracking Data"] = list(analysis.warnings)
    quality_control["eye_tracking"] = analysis.quality_control

    if analysis.features.empty:
        table = _empty_row(EYE_TASK_FEATURES)
    else:
        source = analysis.features.iloc[0].to_dict()
        table = pd.DataFrame(
            [
                {
                    "mean_fixation_duration": source.get("mean_fixation_duration"),
                    "fixation_time_percentage": source.get("fixation_time_percentage"),
                    "fixation_duration_variability": source.get("fixation_duration_variability"),
                    "mean_saccade_duration": source.get("mean_saccade_duration"),
                    "saccade_time_percentage": source.get("saccade_time_percentage"),
                    "saccade_duration_variability": source.get("saccade_duration_variability"),
                    "EyesNotFound_percentage": source.get("EyesNotFound_percentage"),
                    "mean_EyesNotFound_duration": source.get("mean_EyesNotFound_duration"),
                    "EyesNotFound_duration_variability": source.get("EyesNotFound_duration_variability"),
                    "mean_pupil_diameter": source.get("mean_pupil_diameter"),
                    "pupil_diameter_variability": source.get("pupil_diameter_variability"),
                    "regression_percentage": source.get("regression_percentage"),
                    "mean_regression_distance": source.get("mean_regression_distance"),
                    "regression_duration_variability": source.get("regression_duration_variability"),
                }
            ],
            columns=list(EYE_TASK_FEATURES),
        )
    table = _apply_resting_state_table_defaults(table, task_folder, table_kind="eye")
    _save_table(table, table_path)
    if window_type == "eprime":
        _save_table(table, task_folder / "table_1_eye_tracking_data_eprime.xlsx")
    return table, False


def _build_physiology_table(
    task_folder: Path,
    force_recompute: bool,
    warnings: dict[str, list[str]],
    *,
    window_type: str,
    corsano: CorsanoTaskLevelResult | None,
) -> tuple[pd.DataFrame, bool]:
    suffix = "" if window_type == "eprime" else f"_{window_type}"
    table_path = task_folder / (TABLE_2_FILE if window_type == "eprime" else f"table_2_physiological_data{suffix}.xlsx")
    cached = _load_if_cached(table_path, force_recompute)
    if cached is not None:
        return _apply_resting_state_table_defaults(
            _normalize_cached_row(cached, PHYSIO_TASK_FEATURES),
            task_folder,
            table_kind="physio",
        ), True

    if corsano is None:
        corsano = build_corsano_task_level_features(task_folder, window_type=window_type)
    table = pd.DataFrame([corsano.physiology_row], columns=list(PHYSIO_TASK_FEATURES))
    table = _apply_resting_state_table_defaults(table, task_folder, table_kind="physio")
    physio_warnings, _ = _split_corsano_warnings(corsano.warnings)
    if physio_warnings:
        warnings["Table 2 · Physiological Data"] = physio_warnings
    _save_table(table, table_path)
    return table, False


def _implemented_eeg_features_from_json(task_folder: Path) -> dict[str, float | str] | None:
    features = load_task_level_eeg_features_json(task_folder)
    if not features:
        return None
    return {
        feature_name: features[feature_name]
        for feature_name in TASK_LEVEL_EEG_FEATURE_COLUMNS
        if feature_name in features
    }


def _merge_implemented_eeg_features_into_table(
    table: pd.DataFrame,
    feature_values: dict[str, float | str],
) -> pd.DataFrame:
    merged = table.copy()
    if merged.empty:
        merged = _empty_row(EEG_TASK_FEATURES)
    for column in EEG_TASK_FEATURES:
        if column not in merged.columns:
            merged[column] = None
    merged = merged.reindex(columns=list(EEG_TASK_FEATURES))
    for feature_name, value in feature_values.items():
        merged.loc[merged.index[0], feature_name] = value
    return merged


def _build_eeg_table(
    task_folder: Path,
    force_recompute: bool,
    warnings: dict[str, list[str]],
    *,
    window_type: str,
) -> tuple[pd.DataFrame, bool]:
    suffix = "" if window_type == "eprime" else f"_{window_type}"
    table_path = task_folder / (TABLE_3_FILE if window_type == "eprime" else f"table_3_eeg_data{suffix}.xlsx")
    feature_values = (
        _implemented_eeg_features_from_json(task_folder) if window_type == "eprime" else None
    )

    cached = _load_if_cached(table_path, force_recompute)
    if cached is not None:
        normalized = _normalize_cached_row(cached, EEG_TASK_FEATURES)
        normalized = _apply_resting_state_table_defaults(normalized, task_folder, table_kind="eeg")
        if feature_values is not None:
            stale = False
            if normalized.empty:
                stale = True
            else:
                for feature_name, value in feature_values.items():
                    if normalized.iloc[0].get(feature_name) != value:
                        stale = True
                        break
            if stale:
                merged = _merge_implemented_eeg_features_into_table(normalized, feature_values)
                merged = _apply_resting_state_table_defaults(merged, task_folder, table_kind="eeg")
                _save_table(merged, table_path)
                return merged, False
        return normalized, True

    table = _empty_row(EEG_TASK_FEATURES)
    table_warnings: list[str] = []
    if feature_values is not None:
        for feature_name, value in feature_values.items():
            table.loc[table.index[0], feature_name] = value
    else:
        table_warnings.append(
            "EEG task-level features are partially implemented; band-power and theta/alpha ratio features are available when task_level_eeg_features.json exists.",
        )
        for filename in ("Task.ahdr", "Task.amrk", "Task.eeg"):
            if not (task_folder / filename).is_file():
                table_warnings.append(f"EEG Data: missing {filename}.")
    if table_warnings:
        warnings["Table 3 · EEG Data"] = table_warnings
    table = _apply_resting_state_table_defaults(table, task_folder, table_kind="eeg")
    _save_table(table, table_path)
    return table, False


def _build_quality_control_table(
    task_folder: Path,
    force_recompute: bool,
    warnings: dict[str, list[str]],
    table_1: pd.DataFrame,
    window_type: str,
    corsano: CorsanoTaskLevelResult | None,
) -> tuple[pd.DataFrame, bool]:
    suffix = "" if window_type == "eprime" else f"_{window_type}"
    table_path = task_folder / (TABLE_4_FILE if window_type == "eprime" else f"table_4_quality_control{suffix}.xlsx")
    cached = _load_if_cached(table_path, force_recompute)
    if cached is not None:
        return _normalize_cached_row(cached, QC_TASK_FEATURES), True

    if corsano is None:
        corsano = build_corsano_task_level_features(task_folder, window_type=window_type)
    table = pd.DataFrame([corsano.quality_control_row], columns=list(QC_TASK_FEATURES))
    _, qc_warnings = _split_corsano_warnings(corsano.warnings)
    if qc_warnings:
        warnings["Table 4 · Quality Control"] = qc_warnings
    _save_table(table, table_path)
    return table, False


def build_task_level_tables(
    task_folder: Path,
    *,
    force_recompute: bool = False,
    line_transition_threshold_px: float | None = None,
    regression_threshold_px: float | None = None,
    window_type: str = "eprime",
) -> TaskLevelTablesResult:
    warnings: dict[str, list[str]] = {}
    quality_control: dict[str, Any] = {}
    suffix = "" if window_type == "eprime" else f"_{window_type}"
    paths = {
        "t1": task_folder / (TABLE_1_FILE if window_type == "eprime" else f"table_1_eye_tracking_data{suffix}.xlsx"),
        "t2": task_folder / (TABLE_2_FILE if window_type == "eprime" else f"table_2_physiological_data{suffix}.xlsx"),
        "t3": task_folder / (TABLE_3_FILE if window_type == "eprime" else f"table_3_eeg_data{suffix}.xlsx"),
        "t4": task_folder / (TABLE_4_FILE if window_type == "eprime" else f"table_4_quality_control{suffix}.xlsx"),
    }

    t1, t1_loaded = _build_eye_table(
        task_folder,
        force_recompute,
        warnings,
        quality_control,
        line_transition_threshold_px=line_transition_threshold_px,
        regression_threshold_px=regression_threshold_px,
        window_type=window_type,
    )
    t2_path = paths["t2"]
    t4_path = paths["t4"]
    corsano: CorsanoTaskLevelResult | None = None
    if force_recompute or not t2_path.is_file() or not t4_path.is_file():
        corsano = build_corsano_task_level_features(task_folder, window_type=window_type)

    t2, t2_loaded = _build_physiology_table(
        task_folder,
        force_recompute,
        warnings,
        window_type=window_type,
        corsano=corsano,
    )
    t3, t3_loaded = _build_eeg_table(task_folder, force_recompute, warnings, window_type=window_type)
    t4, t4_loaded = _build_quality_control_table(
        task_folder,
        force_recompute,
        warnings,
        t1,
        window_type,
        corsano=corsano,
    )

    return TaskLevelTablesResult(
        tables={"t1": t1, "t2": t2, "t3": t3, "t4": t4},
        table_paths=paths,
        warnings=warnings,
        loaded_existing={"t1": t1_loaded, "t2": t2_loaded, "t3": t3_loaded, "t4": t4_loaded},
        quality_control=quality_control,
    )
