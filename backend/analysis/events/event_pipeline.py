"""
Orchestrate event-level analysis with disk caching.

GENERATE loads saved outputs when all required files exist.
Regenerate forces recomputation of the full pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.analysis.events.eeg_alignment import (
    EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE,
    load_event_database_with_eeg_alignment,
    run_eeg_event_alignment,
)
from backend.analysis.events.eeg_epoch_extraction import run_eeg_epoch_extraction
from backend.analysis.events.event_aggregation import (
    EVENT_LEVEL_EEG_DISTRIBUTION_FILE,
    run_event_aggregation,
)
from backend.analysis.events.event_database import (
    EVENT_DATABASE_FILE,
    EVENT_SUMMARY_FILE,
    load_event_database,
    run_event_level_tables,
)
from backend.analysis.events.event_eeg_features import (
    EVENT_LEVEL_EEG_FEATURES_FILE,
    load_event_level_eeg_features,
    run_event_level_eeg_features,
)

EVENT_LEVEL_CACHE_FILES: tuple[str, ...] = (
    EVENT_DATABASE_FILE,
    EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE,
    EVENT_LEVEL_EEG_FEATURES_FILE,
    EVENT_SUMMARY_FILE,
    EVENT_LEVEL_EEG_DISTRIBUTION_FILE,
)


def event_level_cache_complete(task_folder: Path) -> bool:
    return all((task_folder / name).is_file() for name in EVENT_LEVEL_CACHE_FILES)


def run_event_level_analysis(
    task_folder: Path,
    participant_id: str,
    task_name: str,
    event_type: str,
    *,
    window_type: str = "eprime",
    force_recompute: bool = False,
) -> dict[str, Any]:
    if event_level_cache_complete(task_folder) and not force_recompute:
        aggregation_result = run_event_aggregation(
            task_folder,
            event_type=event_type,
            force_recompute=False,
        )
        database = load_event_database(task_folder)
        aligned_database = load_event_database_with_eeg_alignment(task_folder)
        features = load_event_level_eeg_features(task_folder)
        return {
            "event_result": {
                "database": database if database is not None else pd.DataFrame(),
                "database_path": str((task_folder / EVENT_DATABASE_FILE).resolve()),
                "summary_path": str((task_folder / EVENT_SUMMARY_FILE).resolve()),
                "warnings": [],
                "loaded_existing": True,
            },
            "alignment_result": {
                "aligned_database": aligned_database if aligned_database is not None else pd.DataFrame(),
                "alignment_path": str((task_folder / EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE).resolve()),
                "warnings": [],
                "loaded_existing": True,
            },
            "epoch_result": {
                "warnings": [],
                "loaded_existing": True,
            },
            "feature_result": {
                "features": features if features is not None else pd.DataFrame(),
                "features_path": str((task_folder / EVENT_LEVEL_EEG_FEATURES_FILE).resolve()),
                "warnings": [],
                "loaded_existing": True,
            },
            "aggregation_result": aggregation_result,
            "loaded_from_cache": True,
        }

    event_result = run_event_level_tables(
        task_folder,
        participant_id,
        task_name,
        window_type=window_type,
        force_recompute=force_recompute,
    )
    alignment_result = run_eeg_event_alignment(
        task_folder,
        event_database=event_result.get("database"),
        force_recompute=force_recompute,
    )
    epoch_result = run_eeg_epoch_extraction(
        task_folder,
        aligned_database=alignment_result.get("aligned_database"),
        event_type=event_type,
        force_recompute=force_recompute,
    )
    feature_result = run_event_level_eeg_features(
        task_folder,
        participant_id,
        task_name,
        aligned_database=alignment_result.get("aligned_database"),
        epoch_metadata=epoch_result.get("metadata"),
        force_recompute=force_recompute,
    )
    aggregation_result = run_event_aggregation(
        task_folder,
        database=event_result.get("database"),
        features=feature_result.get("features"),
        event_type=event_type,
        force_recompute=force_recompute,
    )
    return {
        "event_result": event_result,
        "alignment_result": alignment_result,
        "epoch_result": epoch_result,
        "feature_result": feature_result,
        "aggregation_result": aggregation_result,
        "loaded_from_cache": False,
    }
