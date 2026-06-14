"""
On-disk layout (specification).

Central registry plus one folder per participant (named by participant ID).
Raw uploads are never overwritten; derived tables live under each task subfolder.
"""

from __future__ import annotations

CENTRAL_PARTICIPANTS_TABLE = "database/participants_table.xlsx"
PARTICIPANTS_ROOT = "database/participants"

# Participant-level Tobii / Corsano exports (example filenames from spec)
PARTICIPANT_LEVEL_EYE_FILES = ("EyeTracking.xlsx",)  # one or more .xlsx allowed
PARTICIPANT_LEVEL_CORSONO = (
    "acc.xlsx",
    "activity.xlsx",
    "heart_rate_variability.xlsx",
)

# Per-task BrainVision bundle + E-Prime + comprehension
TASK_RAW_FILES = (
    "Task.ahdr",
    "Task.eeg",
    "Task.amrk",
    "Eprime.txt",
)

# BrainVision Recorder 2 (BVRF) — saved unchanged under each task's EEG_raw/
EEG_RAW_FOLDER = "EEG_raw"
BVRF_FILE_EXTENSIONS = (".bvrh", ".bvrd", ".bvrm", ".bvri")
EEG_METADATA_FILE = "eeg_metadata.json"
EEG_INSPECTION_FILE = "eeg_file_inspection.json"
EEG_SYNC_SETTINGS_FILE = "eeg_sync_settings.json"
EEG_TIME_AUDIT_FILE = "eeg_time_audit.json"
EEG_SEGMENT_METADATA_FILE = "eeg_segment_metadata.json"
EEG_SEGMENT_INFO_FILE = "eeg_segment_info.json"
EEG_QC_SUMMARY_FILE = "eeg_qc_summary.json"
EEG_RAW_READ_TEST_FILE = "eeg_raw_read_test.json"
EEG_RAW_SIGNAL_QC_FILE = "eeg_raw_signal_qc.json"
EEG_PREPROCESSING_PLAN_FILE = "eeg_preprocessing_plan.json"
EEG_PREPROCESSED_SEGMENT_FILE = "eeg_preprocessed_segment.npy"
EEG_PREPROCESSING_AUDIT_FILE = "eeg_preprocessing_audit.json"
EEG_PREPROCESSED_QC_FILE = "eeg_preprocessed_qc.json"
TASK_LEVEL_EEG_FEATURES_FILE = "task_level_eeg_features.xlsx"
TASK_LEVEL_EEG_FEATURES_JSON = "task_level_eeg_features.json"
EEG_BASELINE_STATUS_FILE = "eeg_baseline_status.json"
EEG_TASK_LEVEL_FEATURE_AUDIT_FILE = "eeg_task_level_feature_audit.json"
# Future Phase 3: optional EEGLAB export target (not generated yet)
EEGLAB_SET_EXTENSION = ".set"
TASK_COMPREHENSION_FILE = "reading_comprehension_score.txt"

# Generated analysis artifacts (saved once; numbered tables)
TASK_LEVEL_TABLES = (
    "task_level_eye_tracking_features.xlsx",
    "task_level_eye_tracking_features_manual.xlsx",
    "regression_events.xlsx",
    "regression_events_manual.xlsx",
    "regression_analysis_report.json",
    "regression_analysis_report_manual.json",
    "processed_eye_tracking_segment_eprime.xlsx",
    "processed_eye_tracking_segment_manual.xlsx",
    "table_1_eye_tracking_data.xlsx",
    "table_1_eye_tracking_data_manual.xlsx",
    "table_2_physiological_data.xlsx",
    "table_2_physiological_data_manual.xlsx",
    "table_3_eeg_data.xlsx",
    "table_3_eeg_data_manual.xlsx",
    "table_4_quality_control.xlsx",
    "table_4_quality_control_manual.xlsx",
    "quality_control_report.json",
    "quality_control_report_manual.json",
)

EVENT_DATABASE_FILE = "event_database.xlsx"
EVENT_SUMMARY_FILE = "event_summary.xlsx"
EVENT_DATABASE_WITH_EEG_ALIGNMENT_FILE = "event_database_with_eeg_alignment.xlsx"
EVENT_EEG_EPOCHS_METADATA_FILE = "event_eeg_epochs_metadata.xlsx"
EVENT_EEG_EPOCHS_METADATA_JSON = "event_eeg_epochs_metadata.json"
EVENT_LEVEL_EEG_FEATURES_FILE = "event_level_eeg_features.xlsx"
EVENT_LEVEL_EEG_DISTRIBUTION_FILE = "event_level_eeg_distribution.xlsx"

EVENT_TABLE_BY_TYPE = {
    "Long Fixation Events": ("table_5_Long_Fixation_Events.xlsx", "table_8_EEG_Long_Fixation.xlsx"),
    "Regression Events": ("table_6_Regression_Events.xlsx", "table_9_EEG_Regression.xlsx"),
    "EyesNotFound Bursts": ("table_7_EyesNotFound_Bursts.xlsx", "table_10_EEG_EyesNotFound.xlsx"),
}


def example_tree(participant_id: str = "123456789", task: str = "Oral Reading - Erased Text") -> str:
    return f"""
database/
├── participants_table.xlsx
└── participants/
    └── participant_{participant_id}/
        ├── EyeTracking.xlsx
        ├── acc.xlsx
        ├── activity.xlsx
        ├── heart_rate_variability.xlsx
        └── {task}/
            ├── EEG_raw/
            │   ├── Task.bvrh
            │   ├── Task.bvrd
            │   ├── Task.bvrm
            │   ├── Task.bvri
            │   ├── eeg_metadata.json
            │   └── eeg_file_inspection.json
            ├── eeg_sync_settings.json
            ├── eeg_time_audit.json
            ├── eeg_segment_metadata.json
            ├── eeg_segment_info.json
            ├── eeg_qc_summary.json
            ├── eeg_raw_read_test.json
            ├── eeg_raw_signal_qc.json
            ├── eeg_preprocessing_plan.json
            ├── eeg_preprocessed_segment.npy
            ├── eeg_preprocessing_audit.json
            ├── eeg_preprocessed_qc.json
            ├── task_level_eeg_features.xlsx
            ├── task_level_eeg_features.json
            ├── eeg_baseline_status.json
            ├── eeg_task_level_feature_audit.json
            ├── Task.ahdr
            ├── Task.eeg
            ├── Task.amrk
            ├── Eprime.txt
            ├── reading_comprehension_score.txt
            ├── table_3_EEG_Data.xlsx
            └── table_5_Long_Fixation_Events.xlsx
""".strip()
