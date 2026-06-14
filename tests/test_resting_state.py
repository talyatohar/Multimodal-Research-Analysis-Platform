"""Tests for Resting state task handling."""

from __future__ import annotations

from backend.analysis.events.event_aggregation import build_event_level_eeg_distribution
from backend.analysis.events.event_database import build_event_summary
from domain.resting_state import (
    RESTING_STATE_REGRESSION_SKIP_MESSAGE,
    baseline_change_zeros,
    is_resting_state_task,
)


def test_is_resting_state_task() -> None:
    assert is_resting_state_task("Resting state")
    assert not is_resting_state_task("Oral Reading - Erased Text")


def test_event_summary_empty_regression_uses_zeros() -> None:
    import pandas as pd

    summary = build_event_summary(pd.DataFrame(columns=["event_type", "event_duration_ms"]))
    regression = summary.loc[summary["event_type"] == "Regression Events"].iloc[0]
    assert regression["number_of_events"] == 0
    assert regression["mean_event_duration"] == 0.0
    assert regression["event_duration_variability"] == 0.0


def test_event_distribution_zero_detected_events_uses_zeros() -> None:
    import pandas as pd

    distribution = build_event_level_eeg_distribution(
        pd.DataFrame(),
        database=pd.DataFrame(columns=["event_type"]),
    )
    regression = distribution.loc[distribution["event_type"] == "Regression Events"]
    assert (regression["mean"] == 0.0).all()
    assert (regression["standard_deviation"] == 0.0).all()
    assert (regression["variance"] == 0.0).all()


def test_baseline_change_zeros_cover_all_modalities() -> None:
    zeros = baseline_change_zeros()
    assert zeros["bpm_change_from_baseline"] == 0.0
    assert zeros["theta_power_change_from_baseline"] == 0.0
    assert zeros["OT_plv_change_from_baseline"] == 0.0
    assert RESTING_STATE_REGRESSION_SKIP_MESSAGE.startswith("Regression detection skipped")
