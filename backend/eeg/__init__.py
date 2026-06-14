"""EEG module — inspection only (Phase 1)."""

from backend.eeg.inspection import (
    build_eeg_metadata,
    inspect_eeg_raw_folder,
    load_eeg_metadata_json,
    write_eeg_metadata_json,
)
from backend.eeg.eeglab_compat import (
    detect_eeglab_set_files,
    get_eeglab_compat_status,
    MESSAGE_EXPORT_NOT_IMPLEMENTED,
    MESSAGE_SET_DETECTED,
)
from backend.eeg.preprocessing import (
    build_eeg_preprocessing_plan,
    load_eeg_preprocessing_plan,
    run_eeg_preprocessing_plan_setup,
    write_eeg_preprocessing_plan,
)
from backend.eeg.preprocessing_exec import (
    PREPROCESSING_NOTES,
    apply_basic_preprocessing,
    load_eeg_preprocessing_audit,
    run_basic_preprocessing,
    write_eeg_preprocessing_audit,
)
from backend.eeg.task_level_feature_audit import (
    detect_table_3_eeg_columns,
    load_eeg_task_level_feature_audit,
    refresh_participant_eeg_task_level_feature_audits,
    run_eeg_task_level_feature_audit,
)
from backend.eeg.plv_baseline_normalized import (
    PLV_BASELINE_CHANGE_FEATURE_COLUMNS,
    compute_plv_baseline_normalized_features,
    refresh_participant_plv_baseline_normalized_features,
    run_plv_baseline_normalized_features,
)
from backend.eeg.plv_features import (
    OT_PLV_FEATURE_NAME,
    OT_PLV_VARIABILITY_FEATURE_NAME,
    PLV_FEATURE_COLUMNS,
    TF_PLV_FEATURE_NAME,
    TF_PLV_VARIABILITY_FEATURE_NAME,
    OF_PLV_FEATURE_NAME,
    OF_PLV_VARIABILITY_FEATURE_NAME,
    compute_mean_occipital_temporal_theta_plv,
    compute_theta_plv_features,
    refresh_participant_plv_features,
    run_occipital_temporal_theta_plv,
    run_theta_plv_features,
)
from backend.eeg.baseline_normalized import (
    RESTING_FEATURE_MISSING_MESSAGE,
    TASK_FEATURE_MISSING_MESSAGE,
    compute_baseline_normalized_features,
    refresh_participant_baseline_normalized_features,
    run_baseline_normalized_features,
)
from backend.eeg.baseline_linkage import (
    BASELINE_LINKAGE_NOTE,
    BASELINE_MISSING_MESSAGE,
    RESTING_STATE_BASELINE_TASK_NAME,
    load_eeg_baseline_status,
    refresh_participant_eeg_baseline_linkage,
    run_eeg_baseline_linkage,
    write_eeg_baseline_status,
)
from backend.eeg.task_level_features import (
    ALPHA_FEATURE_NAME,
    ALPHA_VARIABILITY_FEATURE_NAME,
    BASELINE_CHANGE_FEATURE_COLUMNS,
    POWER_FEATURE_COLUMNS,
    RATIO_FEATURE_NAME,
    RATIO_VARIABILITY_FEATURE_NAME,
    FEATURE_NAME,
    NOT_AVAILABLE,
    TASK_LEVEL_EEG_FEATURE_COLUMNS,
    VARIABILITY_FEATURE_NAME,
    load_task_level_eeg_features_json,
    run_task_level_eeg_features,
)
from backend.eeg.preprocessed_signal_qc import (
    PREPROCESSED_QC_NOTE,
    build_eeg_preprocessed_qc,
    load_eeg_preprocessed_qc,
    run_eeg_preprocessed_signal_qc,
    write_eeg_preprocessed_qc,
)
from backend.eeg.qc import (
    build_eeg_qc_summary,
    load_eeg_qc_summary,
    run_eeg_qc,
    write_eeg_qc_summary,
)
from backend.eeg.raw_signal_qc import (
    DESCRIPTIVE_QC_NOTE,
    build_eeg_raw_signal_qc,
    load_eeg_raw_signal_qc,
    load_eeg_segment_array,
    run_eeg_raw_signal_qc,
    write_eeg_raw_signal_qc,
)
from backend.eeg.raw_read_test import (
    build_eeg_raw_read_test,
    load_eeg_raw_read_test,
    run_eeg_raw_read_test,
    write_eeg_raw_read_test,
)
from backend.eeg.raw_access import (
    build_eeg_segment_info,
    load_eeg_segment_info,
    read_bvrd_segment,
    run_eeg_raw_access_verification,
    write_eeg_segment_info,
)
from backend.eeg.segmentation import (
    build_eeg_segment_metadata,
    load_eeg_segment_metadata,
    run_eeg_segmentation,
    write_eeg_segment_metadata,
)
from backend.eeg.sync import (
    build_eeg_time_audit,
    get_eeg_recording_window_utc_adjusted,
    load_eeg_sync_settings,
    run_eeg_synchronization,
    write_eeg_sync_settings,
    write_eeg_time_audit,
)

__all__ = [
    "MESSAGE_EXPORT_NOT_IMPLEMENTED",
    "MESSAGE_SET_DETECTED",
    "build_eeg_metadata",
    "build_eeg_raw_read_test",
    "build_eeg_raw_signal_qc",
    "build_eeg_segment_info",
    "DESCRIPTIVE_QC_NOTE",
    "build_eeg_preprocessed_qc",
    "build_eeg_preprocessing_plan",
    "build_eeg_qc_summary",
    "build_eeg_segment_metadata",
    "build_eeg_time_audit",
    "detect_eeglab_set_files",
    "get_eeg_recording_window_utc_adjusted",
    "get_eeglab_compat_status",
    "inspect_eeg_raw_folder",
    "load_eeg_metadata_json",
    "load_eeg_segment_info",
    "load_eeg_preprocessed_qc",
    "load_eeg_preprocessing_audit",
    "load_eeg_preprocessing_plan",
    "load_eeg_segment_array",
    "load_eeg_qc_summary",
    "load_eeg_raw_read_test",
    "load_eeg_raw_signal_qc",
    "load_eeg_segment_metadata",
    "load_eeg_sync_settings",
    "read_bvrd_segment",
    "PREPROCESSED_QC_NOTE",
    "PREPROCESSING_NOTES",
    "apply_basic_preprocessing",
    "run_basic_preprocessing",
    "run_eeg_preprocessed_signal_qc",
    "run_eeg_preprocessing_plan_setup",
    "run_eeg_qc",
    "run_eeg_raw_access_verification",
    "run_eeg_raw_read_test",
    "run_eeg_raw_signal_qc",
    "run_eeg_segmentation",
    "run_eeg_synchronization",
    "write_eeg_metadata_json",
    "write_eeg_preprocessed_qc",
    "write_eeg_preprocessing_audit",
    "write_eeg_preprocessing_plan",
    "write_eeg_qc_summary",
    "write_eeg_raw_read_test",
    "write_eeg_raw_signal_qc",
    "write_eeg_segment_info",
    "write_eeg_segment_metadata",
    "write_eeg_sync_settings",
    "write_eeg_time_audit",
]
