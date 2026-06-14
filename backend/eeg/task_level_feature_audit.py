"""
EEG Phase 22 — completeness audit for Table 3 EEG task-level features.

Descriptive only; does not modify feature values or compute new features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.analysis.task_level_tables import TABLE_3_FILE
from backend.eeg.baseline_linkage import _task_has_eeg
from domain.feature_catalog import EEG_TASK_FEATURES
from domain.storage_layout import (
    EEG_TASK_LEVEL_FEATURE_AUDIT_FILE,
    TASK_LEVEL_EEG_FEATURES_JSON,
)

NOT_AVAILABLE_PREFIX = "Not available"
AUDIT_NOTE = (
    "Completeness audit only. Feature values are not modified and no new features are computed."
)


def detect_table_3_eeg_columns(table_path: Path) -> list[str]:
    """Return expected EEG feature columns present in a Table 3 file header."""
    if not table_path.is_file():
        return []
    header = pd.read_excel(table_path, engine="openpyxl", nrows=0)
    present = set(header.columns)
    return [feature_name for feature_name in EEG_TASK_FEATURES if feature_name in present]


def _classify_feature_value(value: Any) -> tuple[str, Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "missing", None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "missing", None
        if stripped.startswith(NOT_AVAILABLE_PREFIX):
            return "not_available", stripped
        try:
            number = float(stripped)
        except ValueError:
            return "missing", stripped
        return "numeric", number if pd.notna(number) else None
    try:
        if pd.isna(value):
            return "missing", None
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "missing", value
    if not pd.notna(number):
        return "missing", None
    return "numeric", number


def build_eeg_task_level_feature_audit(task_folder: Path) -> dict[str, Any] | None:
    if not _task_has_eeg(task_folder) and not (task_folder / TABLE_3_FILE).is_file():
        return None

    table_path = task_folder / TABLE_3_FILE
    features_json_path = task_folder / TASK_LEVEL_EEG_FEATURES_JSON
    source_file = str(table_path.resolve()) if table_path.is_file() else None
    features_json_source = (
        str(features_json_path.resolve()) if features_json_path.is_file() else None
    )

    row: dict[str, Any] = {}
    if table_path.is_file():
        table = pd.read_excel(table_path, engine="openpyxl")
        if not table.empty:
            row = table.iloc[0].to_dict()

    per_feature: list[dict[str, Any]] = []
    numeric_count = 0
    not_available_count = 0
    missing_columns: list[str] = []

    for feature_name in EEG_TASK_FEATURES:
        present = feature_name in row
        status, normalized_value = _classify_feature_value(row.get(feature_name)) if present else ("missing", None)
        if not present:
            missing_columns.append(feature_name)
        elif status == "numeric":
            numeric_count += 1
        elif status == "not_available":
            not_available_count += 1

        per_feature.append(
            {
                "feature": feature_name,
                "present": present,
                "status": status,
                "value": normalized_value,
                "source_file": source_file or features_json_source,
            }
        )

    return {
        "phase": "task_level_feature_completeness_audit",
        "audit_note": AUDIT_NOTE,
        "expected_feature_count": len(EEG_TASK_FEATURES),
        "implemented_feature_count": len(EEG_TASK_FEATURES) - len(missing_columns),
        "numeric_feature_count": numeric_count,
        "not_available_feature_count": not_available_count,
        "missing_feature_columns": missing_columns,
        "source_file": source_file,
        "task_level_eeg_features_json": features_json_source,
        "per_feature": per_feature,
    }


def write_eeg_task_level_feature_audit(task_folder: Path, audit: dict[str, Any]) -> Path:
    out = task_folder / EEG_TASK_LEVEL_FEATURE_AUDIT_FILE
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return out


def load_eeg_task_level_feature_audit(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / EEG_TASK_LEVEL_FEATURE_AUDIT_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_eeg_task_level_feature_audit(task_folder: Path) -> dict[str, Any] | None:
    audit = build_eeg_task_level_feature_audit(task_folder)
    if audit is None:
        return None
    path = write_eeg_task_level_feature_audit(task_folder, audit)
    return {
        "audit_path": str(path.resolve()),
        "audit": audit,
    }


def refresh_participant_eeg_task_level_feature_audits(
    participant_folder: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not participant_folder.is_dir():
        return results

    for task_folder in sorted(participant_folder.iterdir()):
        if not task_folder.is_dir():
            continue
        audited = run_eeg_task_level_feature_audit(task_folder)
        if audited is not None:
            results.append(
                {
                    "task_name": task_folder.name,
                    **audited,
                }
            )
    return results
