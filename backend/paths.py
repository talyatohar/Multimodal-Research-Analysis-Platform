"""Database root and participant path resolution."""

from __future__ import annotations

from pathlib import Path

from domain.participant_id import normalize_participant_id
from domain.storage_layout import CENTRAL_PARTICIPANTS_TABLE, PARTICIPANTS_ROOT

# Project root = parent of backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "database"
PARTICIPANTS_TABLE_PATH = PROJECT_ROOT / CENTRAL_PARTICIPANTS_TABLE
PARTICIPANTS_DIR = PROJECT_ROOT / PARTICIPANTS_ROOT

REGISTRY_COLUMNS = [
    "participant_id",
    "participant_name",
    "participant_age",
    "participant_group",
    "notes",
]


def participant_dir(participant_id: str) -> Path:
    pid = normalize_participant_id(participant_id)
    return PARTICIPANTS_DIR / f"participant_{pid}"


def task_dir(participant_id: str, task_name: str) -> Path:
    return participant_dir(participant_id) / task_name


def init_database() -> list[Path]:
    """Create database layout; returns paths created or ensured."""
    ensured = []
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    ensured.append(DATABASE_DIR)
    PARTICIPANTS_DIR.mkdir(parents=True, exist_ok=True)
    ensured.append(PARTICIPANTS_DIR)
    return ensured
