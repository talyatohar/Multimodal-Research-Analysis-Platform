"""Tests for participant ID normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.paths import participant_dir
from backend.registry import append_participant_row, list_participants, load_registry
from domain.participant_id import normalize_participant_id


def test_normalize_participant_id_strips_numeric_excel_artifacts() -> None:
    assert normalize_participant_id(None) == ""
    assert normalize_participant_id(1) == "1"
    assert normalize_participant_id(1.0) == "1"
    assert normalize_participant_id("1.0") == "1"
    assert normalize_participant_id(" 11 ") == "11"
    assert normalize_participant_id("10.0") == "10"


def test_participant_dir_uses_normalized_id() -> None:
    assert participant_dir("1.0").name == "participant_1"
    assert participant_dir(1).name == "participant_1"


def test_registry_round_trip_stores_string_id() -> None:
    import tempfile

    import backend.paths as paths

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths.PROJECT_ROOT = root
        paths.DATABASE_DIR = root / "database"
        paths.PARTICIPANTS_TABLE_PATH = root / "database" / "participants_table.xlsx"
        paths.PARTICIPANTS_DIR = root / "database" / "participants"
        paths.PARTICIPANTS_DIR.mkdir(parents=True)

        append_participant_row("1", "Alice", 8, "Control", "note")
        df = load_registry()
        assert df.iloc[0]["participant_id"] == "1"
        assert isinstance(df.iloc[0]["participant_id"], str)

        pd.DataFrame(
            [
                {
                    "participant_id": 1.0,
                    "participant_name": "Alice",
                    "participant_age": 8,
                    "participant_group": "Control",
                    "notes": "note",
                }
            ]
        ).to_excel(paths.PARTICIPANTS_TABLE_PATH, index=False, engine="openpyxl")

        reloaded = load_registry()
        assert reloaded.iloc[0]["participant_id"] == "1"

        participants = list_participants()
        assert participants[0]["participant_id"] == "1"
