"""
Event-Level Phase 5 — aggregate event summaries and EEG feature distributions.

Reads event_database.xlsx and event_level_eeg_features.xlsx (read-only).
Writes event_summary.xlsx and event_level_eeg_distribution.xlsx.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.analysis.events.event_database import (
    EVENT_DATABASE_FILE,
    EVENT_SUMMARY_FILE,
    build_event_summary,
    load_event_database,
    load_event_summary,
    write_event_summary,
)
from backend.analysis.events.event_eeg_features import (
    EVENT_LEVEL_EEG_FEATURES_FILE,
    load_event_level_eeg_features,
)
from backend.eeg.task_level_features import NOT_AVAILABLE
from domain.feature_catalog import (
    EVENT_EEG_DISTRIBUTION_COLUMNS,
    EVENT_EEG_FEATURES,
    EVENT_SUMMARY_FEATURES,
    EVENT_TYPES,
)

EVENT_LEVEL_EEG_DISTRIBUTION_FILE = "event_level_eeg_distribution.xlsx"

EVENT_LEVEL_EEG_DISTRIBUTION_STORE_COLUMNS: tuple[str, ...] = (
    "event_type",
    *EVENT_EEG_DISTRIBUTION_COLUMNS,
)


def _is_valid_feature_value(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        if value.strip().casefold() == NOT_AVAILABLE.casefold():
            return False
        try:
            float(value)
            return True
        except ValueError:
            return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(numeric)


def _numeric_feature_values(series: pd.Series) -> pd.Series:
    clean = series.loc[series.map(_is_valid_feature_value)].copy()
    return pd.to_numeric(clean, errors="coerce").dropna()


def _distribution_stats(values: pd.Series) -> tuple[float | str, float | str, float | str]:
    if values.empty:
        return NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE
    if len(values) == 1:
        mean_val = float(values.iloc[0])
        return mean_val, 0.0, 0.0
    return (
        float(values.mean()),
        float(values.std()),
        float(values.var()),
    )


def build_event_level_eeg_distribution(
    features: pd.DataFrame,
    *,
    database: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_type in EVENT_TYPES:
        subset = (
            features.loc[features["event_type"] == event_type].copy()
            if not features.empty and "event_type" in features.columns
            else pd.DataFrame()
        )
        detected = 0
        if database is not None and not database.empty and "event_type" in database.columns:
            detected = int((database["event_type"] == event_type).sum())
        for feature in EVENT_EEG_FEATURES:
            if subset.empty or feature not in subset.columns:
                if detected == 0:
                    mean_val, std_val, var_val = 0.0, 0.0, 0.0
                else:
                    mean_val, std_val, var_val = NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE
            else:
                values = _numeric_feature_values(subset[feature])
                mean_val, std_val, var_val = _distribution_stats(values)
            rows.append(
                {
                    "event_type": event_type,
                    "feature": feature,
                    "mean": mean_val,
                    "standard_deviation": std_val,
                    "variance": var_val,
                }
            )
    return pd.DataFrame(rows, columns=list(EVENT_LEVEL_EEG_DISTRIBUTION_STORE_COLUMNS))


def distribution_table_for_event_type(
    distribution: pd.DataFrame,
    event_type: str,
) -> pd.DataFrame:
    if distribution.empty or "event_type" not in distribution.columns:
        return _empty_distribution_table()

    subset = distribution.loc[distribution["event_type"] == event_type].copy()
    if subset.empty:
        return _empty_distribution_table()

    columns = [column for column in EVENT_EEG_DISTRIBUTION_COLUMNS if column in subset.columns]
    return subset[columns].reset_index(drop=True)


def summary_table_for_event_type(summary: pd.DataFrame, event_type: str) -> pd.DataFrame:
    if summary.empty or "event_type" not in summary.columns:
        return _empty_summary_row()

    subset = summary.loc[summary["event_type"] == event_type].copy()
    if subset.empty:
        return _empty_summary_row()

    return subset[list(EVENT_SUMMARY_FEATURES)].reset_index(drop=True)


def _empty_summary_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "number_of_events": 0,
                "mean_event_duration": None,
                "event_duration_variability": None,
            }
        ],
        columns=list(EVENT_SUMMARY_FEATURES),
    )


def _empty_distribution_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "mean": NOT_AVAILABLE,
                "standard_deviation": NOT_AVAILABLE,
                "variance": NOT_AVAILABLE,
            }
            for feature in EVENT_EEG_FEATURES
        ],
        columns=list(EVENT_EEG_DISTRIBUTION_COLUMNS),
    )


def write_event_level_eeg_distribution(task_folder: Path, distribution: pd.DataFrame) -> Path:
    out = task_folder / EVENT_LEVEL_EEG_DISTRIBUTION_FILE
    distribution.to_excel(out, index=False, engine="openpyxl")
    return out


def load_event_level_eeg_distribution(task_folder: Path) -> pd.DataFrame | None:
    path = task_folder / EVENT_LEVEL_EEG_DISTRIBUTION_FILE
    if not path.is_file():
        return None
    return pd.read_excel(path, engine="openpyxl")


def build_event_aggregation_tables(
    task_folder: Path,
    *,
    database: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []

    if database is None:
        database = load_event_database(task_folder)
    if database is None or database.empty:
        warnings.append(f"{EVENT_DATABASE_FILE} missing or empty — event summary unavailable.")
        summary = pd.DataFrame(
            [
                {
                    "event_type": event_type,
                    "number_of_events": 0,
                    "mean_event_duration": None,
                    "event_duration_variability": None,
                }
                for event_type in EVENT_TYPES
            ],
            columns=["event_type", *EVENT_SUMMARY_FEATURES],
        )
    else:
        summary = build_event_summary(database)

    if features is None:
        features = load_event_level_eeg_features(task_folder)
    if features is None or features.empty:
        warnings.append(f"{EVENT_LEVEL_EEG_FEATURES_FILE} missing or empty — EEG distribution uses Not available.")
        features = pd.DataFrame()

    distribution = build_event_level_eeg_distribution(features, database=database)
    return summary, distribution, warnings


def run_event_aggregation(
    task_folder: Path,
    *,
    database: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    event_type: str | None = None,
    force_recompute: bool = False,
) -> dict[str, Any]:
    summary_path = task_folder / EVENT_SUMMARY_FILE
    distribution_path = task_folder / EVENT_LEVEL_EEG_DISTRIBUTION_FILE

    if (
        not force_recompute
        and summary_path.is_file()
        and distribution_path.is_file()
        and database is None
        and features is None
    ):
        summary = load_event_summary(task_folder)
        distribution = load_event_level_eeg_distribution(task_folder)
        if summary is not None and distribution is not None:
            return {
                "summary": summary,
                "distribution": distribution,
                "summary_path": str(summary_path.resolve()),
                "distribution_path": str(distribution_path.resolve()),
                "selected_summary": (
                    summary_table_for_event_type(summary, event_type) if event_type else pd.DataFrame()
                ),
                "selected_distribution": (
                    distribution_table_for_event_type(distribution, event_type)
                    if event_type
                    else pd.DataFrame()
                ),
                "warnings": [],
                "loaded_existing": True,
            }

    summary, distribution, warnings = build_event_aggregation_tables(
        task_folder,
        database=database,
        features=features,
    )

    if summary.empty and distribution.empty and warnings:
        return {
            "summary": summary,
            "distribution": distribution,
            "summary_path": None,
            "distribution_path": None,
            "selected_summary": _empty_summary_row() if event_type else pd.DataFrame(),
            "selected_distribution": (
                distribution_table_for_event_type(distribution, event_type)
                if event_type
                else _empty_distribution_table()
            ),
            "warnings": warnings,
            "loaded_existing": False,
        }

    summary_path = write_event_summary(task_folder, summary)
    distribution_path = write_event_level_eeg_distribution(task_folder, distribution)
    return {
        "summary": summary,
        "distribution": distribution,
        "summary_path": str(summary_path.resolve()),
        "distribution_path": str(distribution_path.resolve()),
        "selected_summary": summary_table_for_event_type(summary, event_type) if event_type else pd.DataFrame(),
        "selected_distribution": (
            distribution_table_for_event_type(distribution, event_type)
            if event_type
            else _empty_distribution_table()
        ),
        "warnings": warnings,
        "loaded_existing": False,
    }

