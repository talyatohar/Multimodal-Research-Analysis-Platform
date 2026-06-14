"""Participant ID normalization for registry, UI, and on-disk paths."""

from __future__ import annotations


def normalize_participant_id(value: object | None) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    return s
