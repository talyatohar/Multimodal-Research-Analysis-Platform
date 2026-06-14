"""
Feature names for UI dropdowns and table headers (spec).

`FEATURE_HELP` gives short, practice-oriented blurbs for the glossary
(placeholder prose until wired to full documentation strings).
"""

from __future__ import annotations

from backend.analysis.feature_definitions import feature_short_help

EYE_TASK_FEATURES: tuple[str, ...] = (
    "mean_fixation_duration",
    "fixation_time_percentage",
    "fixation_duration_variability",
    "mean_saccade_duration",
    "saccade_time_percentage",
    "saccade_duration_variability",
    "EyesNotFound_percentage",
    "mean_EyesNotFound_duration",
    "EyesNotFound_duration_variability",
    "mean_pupil_diameter",
    "pupil_diameter_variability",
    "regression_percentage",
    "mean_regression_distance",
    "regression_duration_variability",
)

PHYSIO_TASK_FEATURES: tuple[str, ...] = (
    "mean_bpm",
    "bpm_change_from_baseline",
    "mean_respiration_rate",
    "respiration_variability",
    "respiration_change_from_baseline",
    "mean_rmssd",
    "rmssd_variability",
    "rmssd_change_from_baseline",
    "mean_si",
    "si_variability",
    "si_change_from_baseline",
)

EEG_TASK_FEATURES: tuple[str, ...] = (
    "mean_frontal_theta_power",
    "theta_power_change_from_baseline",
    "theta_power_variability",
    "mean_occipital_alpha_power",
    "alpha_power_change_from_baseline",
    "alpha_power_variability",
    "mean_theta_alpha_ratio",
    "theta_alpha_ratio_change_from_baseline",
    "theta_alpha_ratio_variability",
    "mean_occipital_temporal_theta_plv",
    "OT_plv_change_from_baseline",
    "OT_plv_variability",
    "mean_temporal_frontal_theta_plv",
    "TF_plv_change_from_baseline",
    "TF_plv_variability",
    "mean_occipital_frontal_theta_plv",
    "OF_plv_change_from_baseline",
    "OF_plv_variability",
)

QC_TASK_FEATURES: tuple[str, ...] = (
    "left_valid_percentage",
    "right_valid_percentage",
    "combined_valid_percentage",
    "mean_motion_magnitude",
    "motion_variability",
    "high_motion_percentage",
    "Reading_comprehension_assessment_score",
)

EVENT_SUMMARY_FEATURES: tuple[str, ...] = (
    "number_of_events",
    "mean_event_duration",
    "event_duration_variability",
)

EVENT_EEG_FEATURES: tuple[str, ...] = (
    "frontal_theta_power",
    "occipital_alpha_power",
    "frontal_theta_occipital_alpha_ratio",
    "theta_PLV_occipital_temporal",
    "theta_PLV_temporal_frontal",
    "theta_PLV_occipital_frontal",
)

EVENT_STAT_COLUMNS: tuple[str, ...] = ("mean", "standart_deviation", "variance")

EVENT_EEG_DISTRIBUTION_COLUMNS: tuple[str, ...] = (
    "feature",
    "mean",
    "standard_deviation",
    "variance",
)

EVENT_TYPES: tuple[str, ...] = ("Long Fixation Events", "Regression Events", "EyesNotFound Bursts")

_GLOSSARY_FEATURES: tuple[str, ...] = (
    *EYE_TASK_FEATURES,
    *PHYSIO_TASK_FEATURES,
    *EEG_TASK_FEATURES,
    *QC_TASK_FEATURES,
    *EVENT_SUMMARY_FEATURES,
    *EVENT_EEG_FEATURES,
    "feature",
    "mean",
    "standard_deviation",
    "variance",
    "standart_deviation",
)

FEATURE_HELP: dict[str, str] = {
    name: feature_short_help(name) for name in _GLOSSARY_FEATURES
}
