"""Saccade consolidation between fixations and de-duplicated regression detection."""

from __future__ import annotations

import pandas as pd

from backend.analysis.eye_tracking.eye_selection import consolidate_saccades
from backend.analysis.eye_tracking.task_level import (
    DURATION_COL,
    FIXATION_X_COL,
    FIXATION_Y_COL,
    TRANSITION_OVER_CAP,
    TRANSITION_SACCADE,
    TRANSITION_TRACKING_LOSS,
    TS_COL,
    TYPE_COL,
    TYPE_INDEX_COL,
    _event_durations,
    _event_sequence,
    _fixation_transitions,
    _regression_analysis,
)

DT = 10.0


def _segment(spec: list[tuple]) -> pd.DataFrame:
    """spec: (label, n_samples[, x, y]) blocks; per-type dense indices, as in Tobii exports."""
    rows = []
    counters: dict[str, int] = {}
    t = 0.0
    for block in spec:
        label, n = block[0], block[1]
        counters[label] = counters.get(label, 0) + 1
        x, y = (block[2], block[3]) if len(block) > 2 else (None, None)
        for _ in range(n):
            rows.append(
                {
                    TS_COL: t,
                    TYPE_COL: label,
                    TYPE_INDEX_COL: counters[label],
                    DURATION_COL: n * DT,
                    FIXATION_X_COL: x if label == "Fixation" else None,
                    FIXATION_Y_COL: y if label == "Fixation" else None,
                    "Validity left": "Valid",
                    "Validity right": "Valid",
                }
            )
            t += DT
    return pd.DataFrame(rows)


def _saccade_events(df: pd.DataFrame) -> pd.Series:
    return _event_durations(df, "Saccade", None, None, [], require_valid=False)


def test_unclassified_fragment_is_absorbed_into_one_saccade():
    df = _segment([("Fixation", 20, 500, 300), ("Saccade", 2), ("Unclassified", 2), ("Saccade", 2), ("Fixation", 20, 400, 300)])
    out, info = consolidate_saccades(df)
    saccades = _saccade_events(out)
    assert len(saccades) == 1
    assert saccades.iloc[0] == 60.0  # full interval: 6 samples x 10 ms
    transitions = _fixation_transitions(_event_sequence(out, []))
    assert len(transitions) == 1
    assert transitions["transition_status"].tolist() == [TRANSITION_SACCADE]
    assert info["saccades"] == 1


def test_tracking_loss_between_fixations_is_not_a_saccade(tmp_path):
    df = _segment([("Fixation", 20, 400, 300), ("Saccade", 2), ("EyesNotFound", 3), ("Saccade", 2), ("Fixation", 20, 500, 300)])
    out, _ = consolidate_saccades(df)
    assert _saccade_events(out).empty
    assert (out[TYPE_COL] == "EyesNotFound").sum() == 3  # EyesNotFound unchanged
    assert set(out[TYPE_COL]) == {"Fixation", "EyesNotFound", "Unclassified"}
    report, events = _regression_analysis(
        out, tmp_path, [], line_transition_threshold_px=30.0, regression_threshold_px=25.0
    )
    assert events["transition_status"].tolist() == [TRANSITION_TRACKING_LOSS]
    assert report["metrics"]["regression_count"] == 0


def test_fragmented_rightward_same_line_transition_is_one_regression(tmp_path):
    df = _segment(
        [
            ("Fixation", 20, 400, 300),
            ("Saccade", 2),
            ("Unclassified", 2),
            ("Saccade", 2),
            ("Unclassified", 2),
            ("Saccade", 2),
            ("Fixation", 20, 500, 305),  # 100 px to the right, same line: regression (RTL reading)
            ("Saccade", 3),
            ("Fixation", 20, 450, 305),  # forward saccade
        ]
    )
    assert df[TYPE_COL].eq("Saccade").groupby(df[TYPE_INDEX_COL].where(df[TYPE_COL] == "Saccade")).size().size == 4
    out, _ = consolidate_saccades(df)
    report, events = _regression_analysis(
        out, tmp_path, [], line_transition_threshold_px=30.0, regression_threshold_px=25.0
    )
    assert len(events) == 2  # one row per fixation pair
    assert report["metrics"]["regression_count"] == 1
    assert report["validation_summary"]["total_saccade_count"] == 2
    assert report["metrics"]["regression_percentage"] == 50.0
    assert report["metrics"]["mean_regression_duration"] == 100.0  # whole 10-sample run
    assert report["metrics"]["mean_regression_distance"] == 100.0


def test_120_ms_same_line_rightward_run_is_no_saccade_but_one_regression(tmp_path):
    # 120 ms run without tracking loss, same line, 100 px to the right (backward in RTL reading)
    df = _segment([("Fixation", 20, 400, 300), ("Saccade", 6), ("Unclassified", 6), ("Fixation", 20, 500, 300)])
    out, info = consolidate_saccades(df)
    assert _saccade_events(out).empty  # not a saccade in the saccade metrics
    assert (out[TYPE_COL] == "Unclassified").sum() == 12
    assert info["over_cap_runs"] == 1
    report, events = _regression_analysis(
        out, tmp_path, [], line_transition_threshold_px=30.0, regression_threshold_px=25.0
    )
    assert events["transition_status"].tolist() == [TRANSITION_OVER_CAP]
    assert report["metrics"]["regression_count"] == 1
    assert report["metrics"]["regression_percentage"] == 100.0
    assert report["metrics"]["mean_regression_duration"] == 120.0  # uncapped run duration
    assert report["saccade_cap"]["over_cap_same_line"]["count"] == 1


def test_fixations_are_unchanged():
    df = _segment([("Saccade", 2), ("Fixation", 20, 400, 300), ("Saccade", 2), ("Unclassified", 1), ("Fixation", 25, 300, 300), ("Saccade", 2)])
    out, _ = consolidate_saccades(df)
    fix_in = df.loc[df[TYPE_COL] == "Fixation", [TYPE_INDEX_COL, DURATION_COL]].astype(float)
    fix_out = out.loc[out[TYPE_COL] == "Fixation", [TYPE_INDEX_COL, DURATION_COL]].astype(float)
    pd.testing.assert_frame_equal(fix_in, fix_out)
    # leading/trailing non-fixation runs are not between two fixations
    assert len(_saccade_events(out)) == 1
