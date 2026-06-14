"""EEG preprocessing stages (specification ordering — no DSP implementation)."""

from __future__ import annotations

EEG_PREPROCESSING_STAGES: tuple[str, ...] = (
    "raw EEG import",
    "channel selection",
    "band-pass filtering",
    "bad channel detection",
    "artifact removal",
    "movement/blink artifact handling",
    "task segmentation",
    "event-centered epoch extraction",
    "baseline normalization",
    "frequency analysis",
    "connectivity analysis",
)
