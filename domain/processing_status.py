"""
Internal processing-status fields (spec).

Not intended as a primary UI surface; used by orchestration to decide
whether to generate vs load cached tables and which modules may run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskProcessingFlags:
    """Per-task modality and artifact flags (placeholder dataclass — no I/O)."""

    eye_tracking_uploaded: bool = False
    acc_uploaded: bool = False
    activity_uploaded: bool = False
    hrv_uploaded: bool = False
    eeg_uploaded: bool = False
    eprime_uploaded: bool = False
    task_level_features_generated: bool = False
    event_level_features_generated: bool = False
    quality_control_generated: bool = False
    event_tables: dict[str, bool] = field(default_factory=dict)


# Example keys the orchestrator will mirror for event caches
EVENT_TYPES = ("Long Fixation Events", "Regression Events", "EyesNotFound Bursts")
