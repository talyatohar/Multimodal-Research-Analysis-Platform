"""Event-level analysis helpers."""

from backend.analysis.events.eeg_alignment import (
    EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE,
    aligned_event_window_table,
    run_eeg_event_alignment,
    summarize_eeg_alignment,
    summarize_event_eeg_alignment_counts,
)
from backend.analysis.events.eeg_epoch_extraction import (
    EVENT_EEG_EPOCHS_METADATA_FILE,
    EVENT_EEG_EPOCHS_METADATA_JSON,
    build_event_eeg_distribution_table,
    run_eeg_epoch_extraction,
    summarize_epoch_extraction,
)
from backend.analysis.events.event_aggregation import (
    EVENT_LEVEL_EEG_DISTRIBUTION_FILE,
    distribution_table_for_event_type,
    load_event_level_eeg_distribution,
    run_event_aggregation,
    summary_table_for_event_type,
)
from backend.analysis.events.event_eeg_features import (
    EVENT_LEVEL_EEG_FEATURES_FILE,
    load_event_level_eeg_features,
    run_event_level_eeg_features,
)
from backend.analysis.events.event_pipeline import (
    event_level_cache_complete,
    run_event_level_analysis,
)
from backend.analysis.events.event_database import (
    EVENT_DATABASE_FILE,
    EVENT_SUMMARY_FILE,
    REGRESSION_EVENT_DURATION_DEFINITION,
    build_event_database,
    build_event_summary,
    load_event_database,
    load_event_summary,
    run_event_level_tables,
)

__all__ = [
    "EVENT_DATABASE_FILE",
    "EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE",
    "EVENT_EEG_EPOCHS_METADATA_FILE",
    "EVENT_EEG_EPOCHS_METADATA_JSON",
    "EVENT_LEVEL_EEG_DISTRIBUTION_FILE",
    "EVENT_LEVEL_EEG_FEATURES_FILE",
    "EVENT_SUMMARY_FILE",
    "REGRESSION_EVENT_DURATION_DEFINITION",
    "aligned_event_window_table",
    "build_event_database",
    "build_event_eeg_distribution_table",
    "build_event_summary",
    "load_event_database",
    "load_event_level_eeg_distribution",
    "load_event_level_eeg_features",
    "load_event_summary",
    "distribution_table_for_event_type",
    "event_level_cache_complete",
    "run_eeg_epoch_extraction",
    "run_event_aggregation",
    "run_event_level_analysis",
    "run_event_level_eeg_features",
    "summary_table_for_event_type",
    "run_eeg_event_alignment",
    "run_event_level_tables",
    "summarize_eeg_alignment",
    "summarize_event_eeg_alignment_counts",
    "summarize_epoch_extraction",
]
