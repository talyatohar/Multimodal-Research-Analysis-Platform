"""Central feature definitions used by analysis tables and UI tooltips."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    feature_name: str
    short_definition: str
    unit: str | None = None
    practical_note: str = ""

    @property
    def tooltip(self) -> str:
        parts = [self.short_definition]
        if self.unit:
            parts.append(f"Unit: {self.unit}")
        if self.practical_note:
            parts.append(self.practical_note)
        return "\n\n".join(parts)


_TABLE2_OUTLIER_QC_NOTE = (
    "Quality Control:\n"
    "- Low-quality samples are removed before calculation (when applicable).\n"
    "- Extreme values outside the 1st–99th percentile range are excluded only when "
    "at least 10 valid samples are available.\n"
    "- For tasks containing fewer than 10 valid samples, outlier removal is skipped "
    "to prevent loss of all available data."
)


FEATURE_DEFINITIONS: dict[str, FeatureDefinition] = {
    # Table 1 — Eye tracking
    "mean_fixation_duration": FeatureDefinition(
        "mean_fixation_duration",
        "Average duration of fixation events during the task window.",
        "ms",
        "Computed from Tobii fixation events after segmentation to the selected analysis window.",
    ),
    "fixation_time_percentage": FeatureDefinition(
        "fixation_time_percentage",
        "Share of total task time spent in fixation events.",
        "%",
        "Sum of fixation durations divided by total task duration, multiplied by 100.",
    ),
    "fixation_duration_variability": FeatureDefinition(
        "fixation_duration_variability",
        "Spread of fixation-event durations in the task window.",
        "ms",
        "Standard deviation of fixation durations after event aggregation.",
    ),
    "mean_saccade_duration": FeatureDefinition(
        "mean_saccade_duration",
        "Average duration of saccadic eye movements in the task window.",
        "ms",
        "Computed from Tobii saccade events after segmentation to the selected analysis window.",
    ),
    "saccade_time_percentage": FeatureDefinition(
        "saccade_time_percentage",
        "Share of total task time spent in saccade events.",
        "%",
        "Sum of saccade durations divided by total task duration, multiplied by 100.",
    ),
    "saccade_duration_variability": FeatureDefinition(
        "saccade_duration_variability",
        "Spread of saccade-event durations in the task window.",
        "ms",
        "Standard deviation of saccade durations after event aggregation.",
    ),
    "EyesNotFound_percentage": FeatureDefinition(
        "EyesNotFound_percentage",
        "Share of task time where the eye tracker did not detect the eyes.",
        "%",
        "Sum of EyesNotFound event durations divided by total task duration, multiplied by 100.",
    ),
    "mean_EyesNotFound_duration": FeatureDefinition(
        "mean_EyesNotFound_duration",
        "Average duration of EyesNotFound tracking-loss events.",
        "ms",
        "Mean duration of EyesNotFound events after aggregation.",
    ),
    "EyesNotFound_duration_variability": FeatureDefinition(
        "EyesNotFound_duration_variability",
        "Spread of EyesNotFound event durations.",
        "ms",
        "Standard deviation of EyesNotFound durations after event aggregation.",
    ),
    "mean_pupil_diameter_left": FeatureDefinition(
        "mean_pupil_diameter_left",
        "Average left pupil diameter during valid task samples.",
        "mm",
        "Mean left-eye pupil diameter after validity filtering.",
    ),
    "mean_pupil_diameter_right": FeatureDefinition(
        "mean_pupil_diameter_right",
        "Average right pupil diameter during valid task samples.",
        "mm",
        "Mean right-eye pupil diameter after validity filtering.",
    ),
    "mean_pupil_diameter": FeatureDefinition(
        "mean_pupil_diameter",
        "Average pupil diameter during valid task samples.",
        "mm",
        "Mean of available left/right pupil values after validity filtering.",
    ),
    "pupil_diameter_variability": FeatureDefinition(
        "pupil_diameter_variability",
        "Fluctuation of synchronized pupil diameter across the task window.",
        "mm",
        "Standard deviation of the per-sample pupil average after validity filtering.",
    ),
    "left_valid_percentage": FeatureDefinition(
        "left_valid_percentage",
        "Percentage of eye-tracking samples where the left eye was tracked as valid.",
        "%",
        "Computed from the processed eye-tracking segment for the selected task window.",
    ),
    "right_valid_percentage": FeatureDefinition(
        "right_valid_percentage",
        "Percentage of eye-tracking samples where the right eye was tracked as valid.",
        "%",
        "Computed from the processed eye-tracking segment for the selected task window.",
    ),
    "combined_valid_percentage": FeatureDefinition(
        "combined_valid_percentage",
        "Percentage of samples where at least one eye was tracked as valid.",
        "%",
        "Computed using left OR right validity because participants may have asymmetric eye-tracking reliability due to amblyopia.",
    ),
    "regression_count": FeatureDefinition(
        "regression_count",
        "Number of backward same-line saccadic transitions classified as regressions.",
        "count",
        "Detected when abs(delta_y) is below the line threshold and delta_x exceeds the regression threshold.",
    ),
    "regression_percentage": FeatureDefinition(
        "regression_percentage",
        "Share of saccades classified as regressions.",
        "%",
        "Regression count divided by total saccade count, multiplied by 100.",
    ),
    "mean_regression_distance": FeatureDefinition(
        "mean_regression_distance",
        "Average horizontal distance of regression-classified saccades.",
        "px",
        "Mean positive delta_x for same-line saccades classified as regressions.",
    ),
    "mean_regression_duration": FeatureDefinition(
        "mean_regression_duration",
        "Average duration of saccades classified as regressions.",
        "ms",
        "Mean saccade duration for regression-classified events only.",
    ),
    "regression_duration_variability": FeatureDefinition(
        "regression_duration_variability",
        "Spread of durations for regression-classified saccades.",
        "ms",
        "Standard deviation of regression saccade durations.",
    ),
    # Table 2 — Physiological
    "mean_bpm": FeatureDefinition(
        "mean_bpm",
        "Average heart rate during the selected task.",
        "beats per minute (BPM)",
        f"Calculated after keeping only samples with bpm_q >= 3.\n\n{_TABLE2_OUTLIER_QC_NOTE}",
    ),
    "bpm_change_from_baseline": FeatureDefinition(
        "bpm_change_from_baseline",
        "Difference between task mean BPM and Resting state mean BPM.",
        "BPM",
        f"Formula: task mean - resting-state mean. Task and resting means use bpm_q >= 3 filtering.\n\n{_TABLE2_OUTLIER_QC_NOTE}",
    ),
    "mean_respiration_rate": FeatureDefinition(
        "mean_respiration_rate",
        "Average respiration rate during the selected task.",
        "breaths per minute, or Corsano respiration-rate units if the file does not specify units",
        f"Calculated after keeping only samples with resp_q >= 3.\n\n{_TABLE2_OUTLIER_QC_NOTE}",
    ),
    "respiration_variability": FeatureDefinition(
        "respiration_variability",
        "Standard deviation of respiration_rate during the selected task.",
        "same as respiration_rate",
        f"Calculated after resp_q >= 3 filtering.\n\n{_TABLE2_OUTLIER_QC_NOTE}",
    ),
    "respiration_change_from_baseline": FeatureDefinition(
        "respiration_change_from_baseline",
        "Difference between task mean respiration rate and Resting state mean respiration rate.",
        "same as respiration_rate",
        f"Formula: task mean - resting-state mean. Task and resting means use resp_q >= 3 filtering.\n\n{_TABLE2_OUTLIER_QC_NOTE}",
    ),
    "mean_rmssd": FeatureDefinition(
        "mean_rmssd",
        "Average RMSSD during the selected task.",
        "ms",
        _TABLE2_OUTLIER_QC_NOTE,
    ),
    "rmssd_variability": FeatureDefinition(
        "rmssd_variability",
        "Standard deviation of RMSSD during the selected task.",
        "ms",
        _TABLE2_OUTLIER_QC_NOTE,
    ),
    "rmssd_change_from_baseline": FeatureDefinition(
        "rmssd_change_from_baseline",
        "Difference between task mean RMSSD and Resting state mean RMSSD.",
        "ms",
        f"Formula: task mean - resting-state mean.\n\n{_TABLE2_OUTLIER_QC_NOTE}",
    ),
    "mean_si": FeatureDefinition(
        "mean_si",
        "Average Corsano stress index during the selected task.",
        "Corsano SI units",
        _TABLE2_OUTLIER_QC_NOTE,
    ),
    "si_variability": FeatureDefinition(
        "si_variability",
        "Standard deviation of SI during the selected task.",
        "Corsano SI units",
        _TABLE2_OUTLIER_QC_NOTE,
    ),
    "si_change_from_baseline": FeatureDefinition(
        "si_change_from_baseline",
        "Difference between task mean SI and Resting state mean SI.",
        "Corsano SI units",
        f"Formula: task mean - resting-state mean.\n\n{_TABLE2_OUTLIER_QC_NOTE}",
    ),
    # Table 3 — EEG
    "mean_frontal_theta_power": FeatureDefinition(
        "mean_frontal_theta_power",
        "Average frontal theta-band (4–8 Hz) power during the task.",
        "µV²",
        "Welch PSD integrated over theta band, averaged across frontal ROI channels.",
    ),
    "theta_power_change_from_baseline": FeatureDefinition(
        "theta_power_change_from_baseline",
        "Task frontal theta power minus resting-state baseline theta power.",
        "µV²",
        "Uses the same frontal theta power method as the task-level feature.",
    ),
    "theta_power_variability": FeatureDefinition(
        "theta_power_variability",
        "Spread of frontal theta power across frontal ROI channels.",
        "µV²",
        "Standard deviation of per-channel theta power estimates in the frontal ROI.",
    ),
    "mean_occipital_alpha_power": FeatureDefinition(
        "mean_occipital_alpha_power",
        "Average occipital alpha-band (8–12 Hz) power during the task.",
        "µV²",
        "Welch PSD integrated over alpha band, averaged across occipital ROI channels.",
    ),
    "alpha_power_change_from_baseline": FeatureDefinition(
        "alpha_power_change_from_baseline",
        "Task occipital alpha power minus resting-state baseline alpha power.",
        "µV²",
        "Uses the same occipital alpha power method as the task-level feature.",
    ),
    "alpha_power_variability": FeatureDefinition(
        "alpha_power_variability",
        "Spread of occipital alpha power across occipital ROI channels.",
        "µV²",
        "Standard deviation of per-channel alpha power estimates in the occipital ROI.",
    ),
    "mean_theta_alpha_ratio": FeatureDefinition(
        "mean_theta_alpha_ratio",
        "Ratio of mean frontal theta power to mean occipital alpha power.",
        "unitless",
        "Computed as frontal theta power divided by occipital alpha power.",
    ),
    "theta_alpha_ratio_change_from_baseline": FeatureDefinition(
        "theta_alpha_ratio_change_from_baseline",
        "Task theta/alpha ratio minus resting-state baseline theta/alpha ratio.",
        "unitless",
        "Uses the same ratio definition as the task-level feature.",
    ),
    "theta_alpha_ratio_variability": FeatureDefinition(
        "theta_alpha_ratio_variability",
        "Spread of the theta/alpha ratio estimate across ROI channels.",
        "unitless",
        "Derived from theta and alpha variability across ROI channels.",
    ),
    "mean_occipital_temporal_theta_plv": FeatureDefinition(
        "mean_occipital_temporal_theta_plv",
        "Average theta-band phase-locking value between occipital and temporal ROIs.",
        "unitless, range 0–1",
        "Mean PLV from Hilbert theta phase differences across occipital–temporal channel pairs.",
    ),
    "OT_plv_change_from_baseline": FeatureDefinition(
        "OT_plv_change_from_baseline",
        "Task occipital–temporal theta PLV minus resting-state baseline PLV.",
        "unitless",
        "Uses the same occipital–temporal PLV method as the task-level feature.",
    ),
    "OT_plv_variability": FeatureDefinition(
        "OT_plv_variability",
        "Spread of occipital–temporal theta PLV across ROI channel pairs.",
        "unitless",
        "Standard deviation of per-pair occipital–temporal theta PLV estimates.",
    ),
    "mean_temporal_frontal_theta_plv": FeatureDefinition(
        "mean_temporal_frontal_theta_plv",
        "Average theta-band phase-locking value between temporal and frontal ROIs.",
        "unitless, range 0–1",
        "Mean PLV from Hilbert theta phase differences across temporal–frontal channel pairs.",
    ),
    "TF_plv_change_from_baseline": FeatureDefinition(
        "TF_plv_change_from_baseline",
        "Task temporal–frontal theta PLV minus resting-state baseline PLV.",
        "unitless",
        "Uses the same temporal–frontal PLV method as the task-level feature.",
    ),
    "TF_plv_variability": FeatureDefinition(
        "TF_plv_variability",
        "Spread of temporal–frontal theta PLV across ROI channel pairs.",
        "unitless",
        "Standard deviation of per-pair temporal–frontal theta PLV estimates.",
    ),
    "mean_occipital_frontal_theta_plv": FeatureDefinition(
        "mean_occipital_frontal_theta_plv",
        "Average theta-band phase-locking value between occipital and frontal ROIs.",
        "unitless, range 0–1",
        "Mean PLV from Hilbert theta phase differences across occipital–frontal channel pairs.",
    ),
    "OF_plv_change_from_baseline": FeatureDefinition(
        "OF_plv_change_from_baseline",
        "Task occipital–frontal theta PLV minus resting-state baseline PLV.",
        "unitless",
        "Uses the same occipital–frontal PLV method as the task-level feature.",
    ),
    "OF_plv_variability": FeatureDefinition(
        "OF_plv_variability",
        "Spread of occipital–frontal theta PLV across ROI channel pairs.",
        "unitless",
        "Standard deviation of per-pair occipital–frontal theta PLV estimates.",
    ),
    # Table 4 — Quality control
    "mean_motion_magnitude": FeatureDefinition(
        "mean_motion_magnitude",
        "Average accelerometer motion magnitude during the selected task.",
        "ACC magnitude units",
        "Formula: sqrt(accX^2 + accY^2 + accZ^2). No outlier removal is applied because this is a quality-control measure.",
    ),
    "motion_variability": FeatureDefinition(
        "motion_variability",
        "Standard deviation of accelerometer motion magnitude during the selected task.",
        "ACC magnitude units",
        "No outlier removal is applied.",
    ),
    "high_motion_percentage": FeatureDefinition(
        "high_motion_percentage",
        "Percentage of ACC samples exceeding the participant's Resting state motion threshold.",
        "%",
        "High motion is defined as motion_magnitude > resting mean + 2×STD. Used as physiological quality control because Corsano HR/HRV measurements may be less reliable during movement.",
    ),
    "Reading_comprehension_assessment_score": FeatureDefinition(
        "Reading_comprehension_assessment_score",
        "Reading comprehension score entered for the selected task.",
        "%",
        "Loaded from reading_comprehension_score.txt when available for the task.",
    ),
    # Event summary
    "number_of_events": FeatureDefinition(
        "number_of_events",
        "Count of detected events of the selected event type in the task.",
        "count",
        "Summed across the event database for the chosen event type.",
    ),
    "mean_event_duration": FeatureDefinition(
        "mean_event_duration",
        "Average behavioral duration of events of the selected type.",
        "ms",
        "Behavioral event duration only; excludes the 500 ms pre-event and 500 ms post-event EEG alignment window.",
    ),
    "event_duration_variability": FeatureDefinition(
        "event_duration_variability",
        "Spread of behavioral event durations for the selected event type.",
        "ms",
        "Standard deviation of behavioral event durations; excludes EEG alignment padding.",
    ),
    # Event-level EEG features (per-event rows)
    "frontal_theta_power": FeatureDefinition(
        "frontal_theta_power",
        "Frontal theta-band power in the event-centered EEG epoch.",
        "µV²",
        "Same Welch theta-band method and frontal ROI as task-level EEG features.",
    ),
    "occipital_alpha_power": FeatureDefinition(
        "occipital_alpha_power",
        "Occipital alpha-band power in the event-centered EEG epoch.",
        "µV²",
        "Same Welch alpha-band method and occipital ROI as task-level EEG features.",
    ),
    "frontal_theta_occipital_alpha_ratio": FeatureDefinition(
        "frontal_theta_occipital_alpha_ratio",
        "Ratio of frontal theta power to occipital alpha power in the event epoch.",
        "unitless",
        "Computed per event from the event-level theta and alpha power values.",
    ),
    "theta_PLV_occipital_temporal": FeatureDefinition(
        "theta_PLV_occipital_temporal",
        "Theta-band PLV between occipital and temporal ROIs in the event epoch.",
        "unitless, range 0–1",
        "Same theta PLV method as task-level occipital–temporal PLV.",
    ),
    "theta_PLV_temporal_frontal": FeatureDefinition(
        "theta_PLV_temporal_frontal",
        "Theta-band PLV between temporal and frontal ROIs in the event epoch.",
        "unitless, range 0–1",
        "Same theta PLV method as task-level temporal–frontal PLV.",
    ),
    "theta_PLV_occipital_frontal": FeatureDefinition(
        "theta_PLV_occipital_frontal",
        "Theta-band PLV between occipital and frontal ROIs in the event epoch.",
        "unitless, range 0–1",
        "Same theta PLV method as task-level occipital–frontal PLV.",
    ),
    # Event-level EEG distribution columns (feature row help finalized below)
    "feature": FeatureDefinition(
        "feature",
        "EEG feature name for each row in the distribution table.",
    ),
    "mean": FeatureDefinition(
        "mean",
        "Average value of the EEG feature across events of the selected event type.",
        None,
        "Units match the EEG feature named in the same row (e.g. µV² for power, unitless for PLV).",
    ),
    "standard_deviation": FeatureDefinition(
        "standard_deviation",
        "Sample standard deviation of the EEG feature across events of the selected event type.",
        None,
        "Units match the EEG feature named in the same row; computed with ddof=1 across events.",
    ),
    "variance": FeatureDefinition(
        "variance",
        "Sample variance of the EEG feature across events of the selected event type.",
        None,
        "Units match the squared scale of the EEG feature in the same row; computed with ddof=1 across events.",
    ),
    "standart_deviation": FeatureDefinition(
        "standart_deviation",
        "Sample standard deviation of the EEG feature across events of the selected event type.",
        None,
        "Alternate spelling used in some spec labels; same meaning as standard_deviation.",
    ),
}


_EVENT_EEG_ROW_FEATURES: tuple[str, ...] = (
    "frontal_theta_power",
    "occipital_alpha_power",
    "frontal_theta_occipital_alpha_ratio",
    "theta_PLV_occipital_temporal",
    "theta_PLV_temporal_frontal",
    "theta_PLV_occipital_frontal",
)


def event_distribution_feature_column_help() -> str:
    lines = ["Each row is one event-level EEG feature:"]
    for name in _EVENT_EEG_ROW_FEATURES:
        definition = FEATURE_DEFINITIONS.get(name)
        if definition is None:
            continue
        unit_suffix = f" ({definition.unit})" if definition.unit else ""
        lines.append(f"• {name}{unit_suffix}: {definition.short_definition}")
    return "\n".join(lines)


FEATURE_DEFINITIONS["feature"] = FeatureDefinition(
    "feature",
    "EEG feature name for each row in the distribution table.",
    None,
    event_distribution_feature_column_help(),
)


def feature_tooltip(feature_name: str) -> str | None:
    definition = FEATURE_DEFINITIONS.get(feature_name)
    return definition.tooltip if definition else None


def feature_short_help(feature_name: str) -> str:
    definition = FEATURE_DEFINITIONS.get(feature_name)
    if definition is None:
        return "No definition available."
    if definition.unit:
        return f"{definition.short_definition} Unit: {definition.unit}."
    return definition.short_definition
