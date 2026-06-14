"""In-memory cohort synced with database/participants_table.xlsx."""

from __future__ import annotations

import streamlit as st

from backend.registry import list_participants
from domain.participant_id import normalize_participant_id
from shared.session_keys import COHORT_ROWS


def reload_cohort_from_disk() -> list[dict]:
    rows = list_participants()
    st.session_state[COHORT_ROWS] = rows
    return rows


def ensure_cohort() -> list[dict]:
    if COHORT_ROWS not in st.session_state:
        return reload_cohort_from_disk()
    return st.session_state[COHORT_ROWS]


def participant_options() -> list[str]:
    rows = ensure_cohort()
    ids = [r["participant_id"] for r in rows if r.get("participant_id")]
    if not ids:
        return ["— No participants registered —"]
    return ids


def find_participant(pid: str) -> dict | None:
    normalized = normalize_participant_id(pid)
    if not normalized:
        return None
    for row in ensure_cohort():
        if row.get("participant_id") == normalized:
            return row
    return None


def summarize_tasks(row: dict) -> str:
    tasks = row.get("tasks") or []
    if not tasks:
        return "—"
    names = [t.get("task_name") for t in tasks if t.get("task_name")]
    return ", ".join(names) if names else "—"
