"""Central participants_table.xlsx registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.files import ensure_dir
from backend.paths import PARTICIPANTS_TABLE_PATH, REGISTRY_COLUMNS
from domain.participant_id import normalize_participant_id


def _empty_registry() -> pd.DataFrame:
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def _save_registry(df: pd.DataFrame) -> None:
    out = df[REGISTRY_COLUMNS].copy()
    out["participant_id"] = out["participant_id"].apply(normalize_participant_id).astype(str)
    out.to_excel(PARTICIPANTS_TABLE_PATH, index=False, engine="openpyxl")


def ensure_registry_file() -> Path:
    ensure_dir(PARTICIPANTS_TABLE_PATH.parent)
    if not PARTICIPANTS_TABLE_PATH.exists():
        _empty_registry().to_excel(PARTICIPANTS_TABLE_PATH, index=False, engine="openpyxl")
    return PARTICIPANTS_TABLE_PATH


def load_registry() -> pd.DataFrame:
    ensure_registry_file()
    try:
        df = pd.read_excel(
            PARTICIPANTS_TABLE_PATH,
            engine="openpyxl",
            dtype={"participant_id": str},
        )
    except (ValueError, KeyError):
        df = pd.read_excel(PARTICIPANTS_TABLE_PATH, engine="openpyxl")
    for col in REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["participant_id"] = df["participant_id"].apply(normalize_participant_id)
    return df[REGISTRY_COLUMNS]


def participant_exists(participant_id: str) -> bool:
    pid = normalize_participant_id(participant_id)
    if not pid:
        return False
    df = load_registry()
    return pid in df["participant_id"].values


def remove_participant_row(participant_id: str) -> bool:
    """Remove one participant row from participants_table.xlsx. Returns True if a row was removed."""
    pid = normalize_participant_id(participant_id)
    if not pid:
        return False
    df = load_registry()
    mask = df["participant_id"] == pid
    if not mask.any():
        return False
    df = df.loc[~mask].reset_index(drop=True)
    _save_registry(df)
    return True


def append_participant_row(
    participant_id: str,
    participant_name: str | None,
    participant_age: int | None,
    participant_group: str | None,
    notes: str | None,
) -> Path:
    ensure_registry_file()
    df = load_registry()
    pid = normalize_participant_id(participant_id)
    row = {
        "participant_id": pid,
        "participant_name": participant_name or "",
        "participant_age": participant_age if participant_age is not None else "",
        "participant_group": participant_group or "",
        "notes": notes or "",
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_registry(df)
    return PARTICIPANTS_TABLE_PATH


def list_participants() -> list[dict]:
    df = load_registry()
    if df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        age = r["participant_age"]
        try:
            age_val = int(age) if age != "" and pd.notna(age) else None
        except (TypeError, ValueError):
            age_val = None
        pid = normalize_participant_id(r["participant_id"])
        if not pid:
            continue
        out.append(
            {
                "participant_id": pid,
                "participant_name": str(r["participant_name"]) if pd.notna(r["participant_name"]) else None,
                "participant_age": age_val,
                "participant_group": str(r["participant_group"]) if pd.notna(r["participant_group"]) else None,
                "notes": str(r["notes"]) if pd.notna(r["notes"]) else None,
                "tasks": _tasks_from_disk(pid),
            }
        )
    return out


def _tasks_from_disk(participant_id: str) -> list[dict]:
    from backend.paths import participant_dir
    from domain.tasks import TASK_NAMES

    root = participant_dir(participant_id)
    if not root.is_dir():
        return []
    tasks = []
    for name in TASK_NAMES:
        if (root / name).is_dir():
            tasks.append({"task_name": name})
    return tasks
