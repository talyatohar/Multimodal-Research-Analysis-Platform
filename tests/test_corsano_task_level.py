"""Unit tests for Corsano Table 2 / Table 4 task-level features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.analysis.corsano_task_level import (
    INSUFFICIENT_DATA,
    MISSING_RESTING_STATE,
    SPARSE_TIMESTAMP_WARNING,
    _mean_and_std,
    _mean_only,
    _segment_corsano_sparse,
    _segment_corsano_strict,
    _trim_percentile_outliers,
    build_corsano_task_level_features,
)
from backend.sync.eprime import write_sync_window_json


def _write_corsano_files(participant_folder: Path, start_ms: int, n: int = 120) -> None:
    timestamps = [start_ms + i * 1000 for i in range(n)]
    activity = pd.DataFrame(
        {
            "timestamp": timestamps,
            "bpm": [70 + (i % 5) for i in range(n)],
            "bpm_q": [4] * n,
            "respiration_rate": [15 + (i % 3) for i in range(n)],
            "resp_q": [4] * n,
        }
    )
    hrv = pd.DataFrame(
        {
            "timestamp": timestamps,
            "rmssd": [40 + (i % 4) for i in range(n)],
            "si": [1.0 + (i % 2) * 0.1 for i in range(n)],
        }
    )
    acc = pd.DataFrame(
        {
            "timestamp": timestamps,
            "accX": [0.1] * n,
            "accY": [0.2] * n,
            "accZ": [0.3] * n,
        }
    )
    activity.to_excel(participant_folder / "activity.xlsx", index=False)
    hrv.to_excel(participant_folder / "heart_rate_variability.xlsx", index=False)
    acc.to_excel(participant_folder / "acc.xlsx", index=False)


def _write_sync(task_folder: Path, start: str, end: str) -> None:
    write_sync_window_json(
        task_folder,
        {
            "task_start_utc": start,
            "task_end_utc": end,
            "task_duration_ms": 120000,
        },
    )


def test_corsano_features_with_resting_baseline(tmp_path: Path) -> None:
    participant = tmp_path / "participant_1"
    task = participant / "Oral Reading - Erased Text"
    resting = participant / "Resting state"
    task.mkdir(parents=True)
    resting.mkdir(parents=True)

    start_ms = int(pd.Timestamp("2026-06-09T07:57:00", tz="UTC").timestamp() * 1000)
    _write_corsano_files(participant, start_ms, n=300)
    _write_sync(task, "2026-06-09T08:00:10", "2026-06-09T08:01:50")
    _write_sync(resting, "2026-06-09T07:58:00", "2026-06-09T07:59:30")

    result = build_corsano_task_level_features(task)
    assert result.physiology_row["mean_bpm"] != INSUFFICIENT_DATA
    assert isinstance(result.physiology_row["bpm_change_from_baseline"], float)
    assert result.quality_control_row["mean_motion_magnitude"] != INSUFFICIENT_DATA
    assert result.quality_control_row["high_motion_percentage"] != MISSING_RESTING_STATE
    assert "motion_burst_count" not in result.quality_control_row


def test_sparse_activity_uses_sixty_second_boundary_expansion() -> None:
    task_start = pd.Timestamp("2026-06-09T08:00:10", tz="UTC")
    task_end = pd.Timestamp("2026-06-09T08:01:50", tz="UTC")
    ts_before = int(pd.Timestamp("2026-06-09T08:00:00", tz="UTC").timestamp() * 1000)
    ts_after = int(pd.Timestamp("2026-06-09T08:02:20", tz="UTC").timestamp() * 1000)
    df = pd.DataFrame({"timestamp": [ts_before, ts_after], "bpm": [70, 72]})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    warnings: list[str] = []
    segmented = _segment_corsano_sparse(df, task_start, task_end, warnings, label="activity.xlsx")
    assert len(segmented) == 2
    assert SPARSE_TIMESTAMP_WARNING in warnings


def test_sparse_activity_rejects_boundaries_beyond_sixty_seconds() -> None:
    task_start = pd.Timestamp("2026-06-09T08:00:10", tz="UTC")
    task_end = pd.Timestamp("2026-06-09T08:01:50", tz="UTC")
    ts_before = int(pd.Timestamp("2026-06-09T07:58:00", tz="UTC").timestamp() * 1000)
    ts_after = int(pd.Timestamp("2026-06-09T08:05:00", tz="UTC").timestamp() * 1000)
    df = pd.DataFrame({"timestamp": [ts_before, ts_after], "bpm": [70, 72]})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    warnings: list[str] = []
    segmented = _segment_corsano_sparse(df, task_start, task_end, warnings, label="activity.xlsx")
    assert segmented.empty


def test_acc_uses_strict_task_window_only() -> None:
    task_start = pd.Timestamp("2026-06-09T08:00:10", tz="UTC")
    task_end = pd.Timestamp("2026-06-09T08:01:50", tz="UTC")
    ts_before = int(pd.Timestamp("2026-06-09T08:00:00", tz="UTC").timestamp() * 1000)
    ts_inside = int(pd.Timestamp("2026-06-09T08:01:00", tz="UTC").timestamp() * 1000)
    ts_after = int(pd.Timestamp("2026-06-09T08:02:20", tz="UTC").timestamp() * 1000)
    df = pd.DataFrame(
        {
            "timestamp": [ts_before, ts_inside, ts_after],
            "accX": [0.1, 0.2, 0.3],
            "accY": [0.1, 0.2, 0.3],
            "accZ": [0.1, 0.2, 0.3],
        }
    )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    warnings: list[str] = []
    segmented = _segment_corsano_strict(df, task_start, task_end, warnings, label="acc.xlsx")
    assert len(segmented) == 1
    assert segmented.iloc[0]["timestamp_utc"] == pd.Timestamp("2026-06-09T08:01:00", tz="UTC")


def test_small_sample_outlier_removal_is_skipped() -> None:
    samples = pd.Series([60.0, 58.0])
    trimmed = _trim_percentile_outliers(samples)
    assert trimmed.tolist() == [60.0, 58.0]
    mean_val, std_val = _mean_and_std(samples)
    assert mean_val == 59.0
    assert std_val != INSUFFICIENT_DATA
    assert abs(float(std_val) - 1.4142135623730951) < 1e-9


def test_single_sample_mean_without_variability() -> None:
    samples = pd.Series([63.0])
    assert _mean_only(samples) == 63.0
    mean_val, std_val = _mean_and_std(samples)
    assert mean_val == 63.0
    assert std_val == INSUFFICIENT_DATA


def test_outlier_removal_applies_with_ten_or_more_samples() -> None:
    samples = pd.Series([70.0] * 14 + [200.0])
    trimmed = _trim_percentile_outliers(samples)
    assert len(trimmed) == 14
    assert 200.0 not in trimmed.tolist()

    skipped = pd.Series([70.0] * 8 + [200.0])
    assert _trim_percentile_outliers(skipped).tolist() == skipped.tolist()


def test_missing_resting_state_baseline_labels(tmp_path: Path) -> None:
    participant = tmp_path / "participant_2"
    task = participant / "Oral Reading - Erased Text"
    task.mkdir(parents=True)
    start_ms = int(pd.Timestamp("2026-06-09T08:00:00", tz="UTC").timestamp() * 1000)
    _write_corsano_files(participant, start_ms)
    _write_sync(task, "2026-06-09T08:00:10", "2026-06-09T08:01:50")

    result = build_corsano_task_level_features(task)
    assert result.physiology_row["bpm_change_from_baseline"] == MISSING_RESTING_STATE
    assert result.quality_control_row["high_motion_percentage"] == MISSING_RESTING_STATE
