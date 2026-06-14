"""Resting state task conventions shared across analysis modules."""

from __future__ import annotations

RESTING_STATE_TASK = "Resting state"

RESTING_STATE_REGRESSION_SKIP_MESSAGE = (
    "Regression detection skipped for Resting state because no reading direction exists "
    "during baseline recording."
)

PHYSIO_BASELINE_CHANGE_FEATURES: tuple[str, ...] = (
    "bpm_change_from_baseline",
    "respiration_change_from_baseline",
    "rmssd_change_from_baseline",
    "si_change_from_baseline",
)

EEG_BASELINE_CHANGE_FEATURES: tuple[str, ...] = (
    "theta_power_change_from_baseline",
    "alpha_power_change_from_baseline",
    "theta_alpha_ratio_change_from_baseline",
    "OT_plv_change_from_baseline",
    "TF_plv_change_from_baseline",
    "OF_plv_change_from_baseline",
)

BASELINE_CHANGE_FEATURES: tuple[str, ...] = (
    *PHYSIO_BASELINE_CHANGE_FEATURES,
    *EEG_BASELINE_CHANGE_FEATURES,
)

RESTING_STATE_ZERO_REGRESSION_FEATURES: tuple[str, ...] = (
    "regression_percentage",
    "mean_regression_distance",
    "regression_duration_variability",
)


def is_resting_state_task(task_name: str | None) -> bool:
    return (task_name or "").strip() == RESTING_STATE_TASK


def baseline_change_zeros() -> dict[str, float]:
    return {feature: 0.0 for feature in BASELINE_CHANGE_FEATURES}


def eeg_baseline_change_zeros() -> dict[str, float]:
    return {feature: 0.0 for feature in EEG_BASELINE_CHANGE_FEATURES}


def physio_baseline_change_zeros() -> dict[str, float]:
    return {feature: 0.0 for feature in PHYSIO_BASELINE_CHANGE_FEATURES}


def resting_state_regression_metric_zeros() -> dict[str, float]:
    return {feature: 0.0 for feature in RESTING_STATE_ZERO_REGRESSION_FEATURES}
