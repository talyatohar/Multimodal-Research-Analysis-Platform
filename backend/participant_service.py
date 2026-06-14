"""Register participant, persist files, run sync pipelines."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Optional

from backend.files import copy_upload_preserve_name, ensure_dir
from backend.paths import init_database, participant_dir, task_dir
from backend.registry import append_participant_row, participant_exists, remove_participant_row
from backend.eeg.inspection import (
    BVRF_EXTENSIONS,
    eeg_raw_dir,
    inspect_eeg_raw_folder,
    write_eeg_metadata_json,
)
from backend.eeg.eeglab_compat import get_eeglab_compat_status
from backend.eeg.preprocessing import run_eeg_preprocessing_plan_setup
from backend.eeg.preprocessing_exec import run_basic_preprocessing
from backend.eeg.preprocessed_signal_qc import run_eeg_preprocessed_signal_qc
from backend.eeg.baseline_linkage import refresh_participant_eeg_baseline_linkage
from backend.eeg.baseline_normalized import refresh_participant_baseline_normalized_features
from backend.eeg.plv_baseline_normalized import refresh_participant_plv_baseline_normalized_features
from backend.eeg.plv_features import refresh_participant_plv_features
from backend.eeg.task_level_feature_audit import refresh_participant_eeg_task_level_feature_audits
from backend.eeg.task_level_features import run_task_level_eeg_features
from backend.eeg.qc import run_eeg_qc
from backend.eeg.raw_access import run_eeg_raw_access_verification
from backend.eeg.raw_read_test import run_eeg_raw_read_test
from backend.eeg.raw_signal_qc import run_eeg_raw_signal_qc
from backend.eeg.segmentation import run_eeg_segmentation
from backend.eeg.sync import run_eeg_synchronization
from backend.sync.bvrf_inspection import inspect_bvrf_recording, write_eeg_file_inspection
from backend.sync.eeg_meta import build_eeg_meta, write_eeg_meta_json
from backend.sync.eprime import (
    add_manual_task_end,
    eprime_file_preview,
    load_sync_window_json,
    parse_eprime_log,
    write_sync_window_json,
)
from backend.sync.eye_tracking import find_and_segment_eye_tracking
from domain.participant_id import normalize_participant_id
from domain.storage_layout import TASK_COMPREHENSION_FILE
from domain.tasks import TASK_NAME_PLACEHOLDER

FileInput = Optional[BinaryIO]


@dataclass
class TaskUpload:
    task_name: str
    ahdr: FileInput = None
    eeg: FileInput = None
    amrk: FileInput = None
    bvrf_files: list[FileInput] = field(default_factory=list)
    eprime: FileInput = None
    comprehension_score: float | None = None
    manual_reading_end_time: str | None = None
    manual_time_interpretation: str | None = None
    eeg_clock_offset_seconds: float | None = None


@dataclass
class ParticipantUpload:
    participant_id: str
    participant_name: str | None = None
    participant_age: int | None = None
    participant_group: str | None = None
    notes: str | None = None
    eye_tracking_files: list[FileInput] = field(default_factory=list)
    acc: FileInput = None
    activity: FileInput = None
    hrv: FileInput = None
    tasks: list[TaskUpload] = field(default_factory=list)


@dataclass
class SaveResult:
    ok: bool
    message: str
    saved_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sync_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_inspection_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_sync_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_segment_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_raw_access_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_eeglab_compat_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_qc_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_preprocessing_plan_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_preprocessing_exec_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_preprocessed_signal_qc_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_task_level_features_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_baseline_linkage_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_task_level_feature_audit_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_raw_read_test_results: list[dict[str, Any]] = field(default_factory=list)
    eeg_raw_signal_qc_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DeleteParticipantResult:
    ok: bool
    message: str
    folder_deleted: bool = False
    registry_updated: bool = False


def delete_participant(participant_id: str) -> DeleteParticipantResult:
    """Delete one participant folder and registry row. Folder is removed before the registry row."""
    pid = normalize_participant_id(participant_id)
    if not pid:
        return DeleteParticipantResult(False, "No participant ID provided.")
    if not participant_exists(pid):
        return DeleteParticipantResult(False, f"Participant `{pid}` was not found in participants_table.xlsx.")

    folder = participant_dir(pid)
    try:
        if folder.is_dir():
            shutil.rmtree(folder)
        elif folder.exists():
            return DeleteParticipantResult(
                False,
                f"Cannot delete participant data: `{folder}` exists but is not a directory.",
            )
    except OSError as exc:
        return DeleteParticipantResult(
            False,
            f"Failed to delete participant folder `{folder}`: {exc}",
        )

    try:
        if not remove_participant_row(pid):
            return DeleteParticipantResult(
                False,
                f"Participant folder was deleted, but `{pid}` was not found in participants_table.xlsx.",
                folder_deleted=True,
            )
    except OSError as exc:
        return DeleteParticipantResult(
            False,
            f"Participant folder was deleted, but updating participants_table.xlsx failed: {exc}",
            folder_deleted=True,
        )

    return DeleteParticipantResult(
        True,
        f"Participant `{pid}` was deleted from participants_table.xlsx and `{folder.name}/` was removed.",
        folder_deleted=True,
        registry_updated=True,
    )


def _save_optional(src: FileInput, folder: Path, filename: str, result: SaveResult) -> None:
    if src is None:
        return
    path, warn = copy_upload_preserve_name(src, folder, filename)
    if path:
        result.saved_paths.append(str(path.resolve()))
    if warn:
        result.warnings.append(warn)


def _save_bvrf_files(files: list[FileInput], task_folder: Path, result: SaveResult) -> Path | None:
    uploads = [f for f in files if f is not None]
    if not uploads:
        return None

    raw_folder = ensure_dir(eeg_raw_dir(task_folder))
    for upload in uploads:
        filename = getattr(upload, "name", None) or "recording.bvrh"
        suffix = Path(filename).suffix.lower()
        if suffix not in BVRF_EXTENSIONS:
            result.warnings.append(
                f"Skipped non-BVRF EEG file `{filename}` (expected one of {', '.join(BVRF_EXTENSIONS)})."
            )
            continue
        _save_optional(upload, raw_folder, filename, result)
    return raw_folder


def _process_bvrf_inspection(
    participant_id: str,
    task: TaskUpload,
    task_folder: Path,
    result: SaveResult,
) -> None:
    raw_folder = eeg_raw_dir(task_folder)
    if not raw_folder.is_dir():
        return

    has_bvrf = any(
        path.is_file() and path.suffix.lower() in BVRF_EXTENSIONS
        for path in raw_folder.iterdir()
    )
    if not has_bvrf:
        return

    try:
        metadata = inspect_eeg_raw_folder(raw_folder)
        metadata_path = write_eeg_metadata_json(raw_folder, metadata)
        legacy_inspection = inspect_bvrf_recording(raw_folder)
        legacy_path = write_eeg_file_inspection(raw_folder, legacy_inspection)
        result.saved_paths.append(str(metadata_path.resolve()))
        result.saved_paths.append(str(legacy_path.resolve()))
        result.eeg_inspection_results.append(
            {
                "task_name": task.task_name,
                "participant_id": participant_id,
                "metadata": metadata,
                "metadata_json_path": str(metadata_path.resolve()),
                "inspection": legacy_inspection,
                "inspection_json_path": str(legacy_path.resolve()),
            }
        )
        result.warnings.extend(
            f"{task.task_name} (BVRF): {warning}" for warning in metadata.get("warnings", [])
        )
    except OSError as exc:
        result.warnings.append(f"{task.task_name}: BVRF inspection failed — {exc}")


def _process_eeg_sync(
    task: TaskUpload,
    task_folder: Path,
    result: SaveResult,
    sync_window: dict[str, Any] | None,
) -> None:
    has_eeg = eeg_raw_dir(task_folder).is_dir() and any(
        path.suffix.lower() in BVRF_EXTENSIONS
        for path in eeg_raw_dir(task_folder).iterdir()
        if path.is_file()
    )
    has_legacy = (task_folder / "Task.amrk").is_file() or (task_folder / "Task.ahdr").is_file()
    if not has_eeg and not has_legacy:
        return

    try:
        sync_result = run_eeg_synchronization(
            task_folder,
            eeg_clock_offset_seconds=task.eeg_clock_offset_seconds,
            sync_window=sync_window,
        )
        audit = sync_result.get("audit") or {}
        result.saved_paths.append(sync_result["settings_path"])
        result.saved_paths.append(sync_result["audit_path"])
        result.eeg_sync_results.append(
            {
                "task_name": task.task_name,
                "settings_path": sync_result["settings_path"],
                "audit_path": sync_result["audit_path"],
                "audit": audit,
            }
        )
        result.warnings.extend(
            f"{task.task_name} (EEG sync): {warning}" for warning in audit.get("warnings", [])
        )
    except OSError as exc:
        result.warnings.append(f"{task.task_name}: EEG synchronization failed — {exc}")


def _process_eeg_segmentation(
    task: TaskUpload,
    task_folder: Path,
    result: SaveResult,
) -> None:
    has_eeg = eeg_raw_dir(task_folder).is_dir() and any(
        path.suffix.lower() in BVRF_EXTENSIONS
        for path in eeg_raw_dir(task_folder).iterdir()
        if path.is_file()
    )
    has_legacy = (task_folder / "Task.amrk").is_file() or (task_folder / "Task.ahdr").is_file()
    if not has_eeg and not has_legacy:
        return

    if not (task_folder / "sync_window.json").is_file():
        return

    try:
        segment_result = run_eeg_segmentation(task_folder)
        metadata = segment_result.get("metadata") or {}
        result.saved_paths.append(segment_result["segment_metadata_path"])
        result.eeg_segment_results.append(
            {
                "task_name": task.task_name,
                "segment_metadata_path": segment_result["segment_metadata_path"],
                "metadata": metadata,
            }
        )
        result.warnings.extend(
            f"{task.task_name} (EEG segment): {warning}"
            for warning in metadata.get("warnings", [])
        )
    except OSError as exc:
        result.warnings.append(f"{task.task_name}: EEG segmentation failed — {exc}")


def _process_eeg_raw_access(
    task: TaskUpload,
    task_folder: Path,
    result: SaveResult,
) -> None:
    raw_folder = eeg_raw_dir(task_folder)
    if not raw_folder.is_dir():
        return
    has_bvrd = any(
        path.is_file() and path.suffix.lower() == ".bvrd"
        for path in raw_folder.iterdir()
    )
    if not has_bvrd:
        return
    if not (task_folder / "eeg_segment_metadata.json").is_file():
        return

    try:
        access_result = run_eeg_raw_access_verification(task_folder)
        info = access_result.get("info") or {}
        result.saved_paths.append(access_result["segment_info_path"])
        result.eeg_raw_access_results.append(
            {
                "task_name": task.task_name,
                "segment_info_path": access_result["segment_info_path"],
                "info": info,
            }
        )
        result.warnings.extend(
            f"{task.task_name} (EEG raw access): {warning}"
            for warning in info.get("warnings", [])
        )
    except OSError as exc:
        result.warnings.append(f"{task.task_name}: EEG raw access verification failed — {exc}")


def _process_eeg_raw_read_test(
    task: TaskUpload,
    task_folder: Path,
    result: SaveResult,
) -> None:
    raw_folder = eeg_raw_dir(task_folder)
    if not raw_folder.is_dir():
        return
    has_bvrd = any(
        path.is_file() and path.suffix.lower() == ".bvrd"
        for path in raw_folder.iterdir()
    )
    if not has_bvrd:
        return
    if not (task_folder / "eeg_segment_metadata.json").is_file():
        return

    try:
        test_result = run_eeg_raw_read_test(task_folder)
        payload = test_result.get("result") or {}
        result.saved_paths.append(test_result["raw_read_test_path"])
        result.eeg_raw_read_test_results.append(
            {
                "task_name": task.task_name,
                "raw_read_test_path": test_result["raw_read_test_path"],
                "result": payload,
            }
        )
        if not payload.get("read_success"):
            error = payload.get("error_message") or "Raw EEG read test failed."
            result.warnings.append(f"{task.task_name} (EEG raw read test): {error}")
    except OSError as exc:
        result.warnings.append(f"{task.task_name}: EEG raw read test failed — {exc}")


def _process_eeg_raw_signal_qc(
    task: TaskUpload,
    task_folder: Path,
    result: SaveResult,
) -> None:
    raw_folder = eeg_raw_dir(task_folder)
    if not raw_folder.is_dir():
        return
    has_bvrd = any(
        path.is_file() and path.suffix.lower() == ".bvrd"
        for path in raw_folder.iterdir()
    )
    if not has_bvrd:
        return
    if not (task_folder / "eeg_segment_metadata.json").is_file():
        return

    try:
        qc_result = run_eeg_raw_signal_qc(task_folder)
        payload = qc_result.get("result") or {}
        result.saved_paths.append(qc_result["raw_signal_qc_path"])
        result.eeg_raw_signal_qc_results.append(
            {
                "task_name": task.task_name,
                "raw_signal_qc_path": qc_result["raw_signal_qc_path"],
                "result": payload,
            }
        )
        if not payload.get("read_success"):
            error = payload.get("error_message") or "EEG segment could not be read for descriptive QC."
            result.warnings.append(f"{task.task_name} (EEG raw signal QC): {error}")
    except OSError as exc:
        result.warnings.append(f"{task.task_name}: EEG raw signal QC failed — {exc}")


def _process_eeglab_compat(
    task: TaskUpload,
    task_folder: Path,
    result: SaveResult,
) -> None:
    has_eeg = eeg_raw_dir(task_folder).is_dir() and any(
        path.is_file()
        for path in eeg_raw_dir(task_folder).iterdir()
    )
    has_legacy = (task_folder / "Task.amrk").is_file() or (task_folder / "Task.ahdr").is_file()
    if not has_eeg and not has_legacy:
        return

    status = get_eeglab_compat_status(task_folder)
    result.eeg_eeglab_compat_results.append(
        {
            "task_name": task.task_name,
            "status": status,
        }
    )


def _process_eeg_qc(
    task: TaskUpload,
    task_folder: Path,
    result: SaveResult,
) -> None:
    has_eeg = eeg_raw_dir(task_folder).is_dir() and any(
        path.is_file()
        for path in eeg_raw_dir(task_folder).iterdir()
    )
    has_legacy = (task_folder / "Task.amrk").is_file() or (task_folder / "Task.ahdr").is_file()
    if not has_eeg and not has_legacy:
        return

    try:
        qc_result = run_eeg_qc(task_folder)
        result.saved_paths.append(qc_result["qc_summary_path"])
        result.eeg_qc_results.append(
            {
                "task_name": task.task_name,
                "qc_summary_path": qc_result["qc_summary_path"],
                "summary": qc_result["summary"],
            }
        )
    except OSError as exc:
        result.warnings.append(f"{task.task_name}: EEG QC summary failed — {exc}")


def _process_eeg_preprocessing_plan(
    task: TaskUpload,
    task_folder: Path,
    result: SaveResult,
) -> None:
    has_eeg = eeg_raw_dir(task_folder).is_dir() and any(
        path.is_file()
        for path in eeg_raw_dir(task_folder).iterdir()
    )
    has_legacy = (task_folder / "Task.amrk").is_file() or (task_folder / "Task.ahdr").is_file()
    if not has_eeg and not has_legacy:
        return

    try:
        plan_result = run_eeg_preprocessing_plan_setup(task_folder)
        result.saved_paths.append(plan_result["preprocessing_plan_path"])
        result.eeg_preprocessing_plan_results.append(
            {
                "task_name": task.task_name,
                "preprocessing_plan_path": plan_result["preprocessing_plan_path"],
                "plan": plan_result["plan"],
            }
        )
    except OSError as exc:
        result.warnings.append(f"{task.task_name}: EEG preprocessing plan setup failed — {exc}")


def _process_eeg_preprocessing_exec(
    task: TaskUpload,
    task_folder: Path,
    result: SaveResult,
) -> None:
    has_eeg = eeg_raw_dir(task_folder).is_dir() and any(
        path.is_file()
        for path in eeg_raw_dir(task_folder).iterdir()
    )
    has_legacy = (task_folder / "Task.amrk").is_file() or (task_folder / "Task.ahdr").is_file()
    if not has_eeg and not has_legacy:
        return

    try:
        exec_result = run_basic_preprocessing(task_folder)
        if exec_result.get("preprocessed_segment_path"):
            result.saved_paths.append(exec_result["preprocessed_segment_path"])
        if exec_result.get("audit_path"):
            result.saved_paths.append(exec_result["audit_path"])
        result.eeg_preprocessing_exec_results.append(
            {
                "task_name": task.task_name,
                "preprocessed_segment_path": exec_result.get("preprocessed_segment_path"),
                "audit_path": exec_result.get("audit_path"),
                "audit": exec_result.get("audit"),
            }
        )
        if not exec_result.get("preprocessing_completed"):
            audit = exec_result.get("audit") or {}
            err = audit.get("error_message") or "Basic preprocessing failed."
            result.warnings.append(f"{task.task_name}: EEG basic preprocessing — {err}")
    except OSError as exc:
        result.warnings.append(f"{task.task_name}: EEG basic preprocessing failed — {exc}")


def _process_eeg_preprocessed_signal_qc(
    task: TaskUpload,
    task_folder: Path,
    result: SaveResult,
) -> None:
    has_eeg = eeg_raw_dir(task_folder).is_dir() and any(
        path.is_file()
        for path in eeg_raw_dir(task_folder).iterdir()
    )
    has_legacy = (task_folder / "Task.amrk").is_file() or (task_folder / "Task.ahdr").is_file()
    if not has_eeg and not has_legacy:
        return

    try:
        qc_result = run_eeg_preprocessed_signal_qc(task_folder)
        payload = qc_result.get("result") or {}
        result.saved_paths.append(qc_result["preprocessed_qc_path"])
        result.eeg_preprocessed_signal_qc_results.append(
            {
                "task_name": task.task_name,
                "preprocessed_qc_path": qc_result["preprocessed_qc_path"],
                "result": payload,
            }
        )
        if not payload.get("read_success"):
            error = payload.get("error_message") or "Preprocessed EEG QC could not be completed."
            result.warnings.append(f"{task.task_name} (EEG preprocessed signal QC): {error}")
    except OSError as exc:
        result.warnings.append(f"{task.task_name}: EEG preprocessed signal QC failed — {exc}")


def _process_eeg_task_level_features(
    task: TaskUpload,
    task_folder: Path,
    participant_id: str,
    result: SaveResult,
) -> None:
    has_eeg = eeg_raw_dir(task_folder).is_dir() and any(
        path.is_file()
        for path in eeg_raw_dir(task_folder).iterdir()
    )
    has_legacy = (task_folder / "Task.amrk").is_file() or (task_folder / "Task.ahdr").is_file()
    if not has_eeg and not has_legacy:
        return

    try:
        feature_result = run_task_level_eeg_features(
            task_folder,
            participant_folder=participant_dir(participant_id),
            task_name=task.task_name,
        )
        result.saved_paths.append(feature_result["features_path"])
        result.saved_paths.append(feature_result["features_json_path"])
        result.saved_paths.append(feature_result["table_3_path"])
        result.eeg_task_level_features_results.append(
            {
                "task_name": task.task_name,
                "features_path": feature_result["features_path"],
                "features_json_path": feature_result["features_json_path"],
                "table_3_path": feature_result["table_3_path"],
                "features": feature_result["features"],
            }
        )
        features = feature_result.get("features") or {}
        if features.get("mean_frontal_theta_power") == "Not available":
            err = features.get("error_message") or "mean_frontal_theta_power not available."
            result.warnings.append(f"{task.task_name} (EEG task-level features): {err}")
    except OSError as exc:
        result.warnings.append(f"{task.task_name}: EEG task-level feature extraction failed — {exc}")


def _process_task_sync(participant_id: str, task: TaskUpload, result: SaveResult) -> None:
    folder = task_dir(participant_id, task.task_name)
    eprime_path = folder / "Eprime.txt"
    if not eprime_path.is_file():
        result.warnings.append(
            f"{task.task_name}: Eprime.txt missing — sync_window.json and eye segmentation skipped."
        )
        return

    try:
        window = parse_eprime_log(eprime_path)
        window = add_manual_task_end(
            window,
            task.manual_reading_end_time,
            task.manual_time_interpretation,
        )
        sync_path = write_sync_window_json(folder, window)
        result.saved_paths.append(str(sync_path.resolve()))
        window = load_sync_window_json(folder) or window
        result.sync_results.append(
            {
                "task_name": task.task_name,
                "success": True,
                "message": "E-Prime parsing succeeded.",
                "session_start_utc": window["session_start_utc"],
                "task_start_utc": window["task_start_utc"],
                "task_end_utc": window["task_end_utc"],
                "manual_task_end_local_time": window.get("manual_task_end_local_time"),
                "manual_task_end_utc": window.get("manual_task_end_utc"),
                "manual_end_source": window.get("manual_end_source"),
                "task_duration_ms": window["task_duration_ms"],
                "sync_json_path": str(sync_path.resolve()),
                "eprime_debug_preview": None,
            }
        )
    except (ValueError, TypeError) as exc:
        preview = eprime_file_preview(eprime_path, max_lines=30)
        result.warnings.append(f"{task.task_name}: E-Prime sync failed — {exc}")
        result.sync_results.append(
            {
                "task_name": task.task_name,
                "success": False,
                "message": str(exc),
                "session_start_utc": None,
                "task_start_utc": None,
                "task_end_utc": None,
                "task_duration_ms": None,
                "sync_json_path": None,
                "eprime_debug_preview": preview,
            }
        )
        return

    seg_path, seg_warns, seg_report = find_and_segment_eye_tracking(
        participant_dir(participant_id),
        window,
        folder,
        output_suffix="eprime",
        end_key="task_end_utc",
    )
    result.warnings.extend(seg_warns)
    if seg_path:
        result.saved_paths.append(str(seg_path.resolve()))
    report_json = seg_report.get("_report_json_path")
    if report_json:
        result.saved_paths.append(report_json)

    manual_seg_path = None
    manual_report_json = None
    manual_seg_report: dict[str, Any] = {}
    if window.get("manual_task_end_utc"):
        manual_seg_path, manual_seg_warns, manual_seg_report = find_and_segment_eye_tracking(
            participant_dir(participant_id),
            window,
            folder,
            output_suffix="manual",
            end_key="manual_task_end_utc",
        )
        result.warnings.extend(manual_seg_warns)
        if manual_seg_path:
            result.saved_paths.append(str(manual_seg_path.resolve()))
        manual_report_json = manual_seg_report.get("_report_json_path")
        if manual_report_json:
            result.saved_paths.append(manual_report_json)

    sync_entry = result.sync_results[-1]
    sync_entry["eye_segmentation"] = {
        "segment_path": str(seg_path.resolve()) if seg_path else None,
        "report_path": report_json,
        "manual_segment_path": str(manual_seg_path.resolve()) if manual_seg_path else None,
        "manual_report_path": manual_report_json,
        "files_checked": seg_report.get("files_checked", []),
        "files_with_overlap": seg_report.get("files_with_overlap", []),
        "recordings_checked": seg_report.get("recordings_checked", []),
        "recordings_with_overlap": seg_report.get("recordings_with_overlap", []),
        "rows_before": seg_report.get("rows_before", 0),
        "rows_after": seg_report.get("rows_after", 0),
        "task_start_utc": seg_report.get("task_start_utc"),
        "task_end_utc": seg_report.get("task_end_utc"),
        "pre_task_buffer_ms": seg_report.get("pre_task_buffer_ms"),
        "post_task_buffer_ms": seg_report.get("post_task_buffer_ms"),
        "first_selected_row_utc": seg_report.get("first_selected_row_utc"),
        "last_selected_row_utc": seg_report.get("last_selected_row_utc"),
        "per_file": seg_report.get("per_file", []),
        "per_recording": seg_report.get("per_recording", []),
        "manual_rows_after": manual_seg_report.get("rows_after", 0),
        "manual_per_recording": manual_seg_report.get("per_recording", []),
    }

    ahdr_path = folder / "Task.ahdr"
    amrk_path = folder / "Task.amrk"
    if ahdr_path.is_file() or amrk_path.is_file():
        meta = build_eeg_meta(
            ahdr_path.read_bytes() if ahdr_path.is_file() else None,
            amrk_path.read_bytes() if amrk_path.is_file() else None,
            [p.name for p in (ahdr_path, amrk_path, folder / "Task.eeg") if p.is_file()],
        )
        meta_path = write_eeg_meta_json(folder, meta)
        result.saved_paths.append(str(meta_path.resolve()))


def register_participant(upload: ParticipantUpload) -> SaveResult:
    pid = normalize_participant_id(upload.participant_id)
    result = SaveResult(ok=False, message="")

    if not pid:
        result.message = "Participant ID is required."
        result.errors.append(result.message)
        return result

    init_database()

    if participant_exists(pid):
        result.message = f"Participant `{pid}` already exists in participants_table.xlsx."
        result.errors.append(result.message)
        return result

    p_folder = ensure_dir(participant_dir(pid))
    result.saved_paths.append(str(p_folder.resolve()))

    for i, eye in enumerate(upload.eye_tracking_files):
        if eye is None:
            continue
        name = "EyeTracking.xlsx" if i == 0 else f"EyeTracking_{i + 1}.xlsx"
        _save_optional(eye, p_folder, name, result)

    _save_optional(upload.acc, p_folder, "acc.xlsx", result)
    _save_optional(upload.activity, p_folder, "activity.xlsx", result)
    _save_optional(upload.hrv, p_folder, "heart_rate_variability.xlsx", result)

    valid_tasks = [
        t for t in upload.tasks
        if t.task_name and t.task_name != TASK_NAME_PLACEHOLDER
    ]

    for task in valid_tasks:
        t_folder = ensure_dir(task_dir(pid, task.task_name))
        result.saved_paths.append(str(t_folder.resolve()))

        _save_optional(task.ahdr, t_folder, "Task.ahdr", result)
        _save_optional(task.eeg, t_folder, "Task.eeg", result)
        _save_optional(task.amrk, t_folder, "Task.amrk", result)
        _save_bvrf_files(task.bvrf_files, t_folder, result)
        _save_optional(task.eprime, t_folder, "Eprime.txt", result)

        if task.comprehension_score is not None and task.comprehension_score > 0:
            comp_path = t_folder / TASK_COMPREHENSION_FILE
            if not comp_path.exists():
                comp_path.write_text(f"{task.comprehension_score}\n", encoding="utf-8")
                result.saved_paths.append(str(comp_path.resolve()))
            else:
                result.warnings.append(f"Skipped (already exists): {comp_path}")

        _process_bvrf_inspection(pid, task, t_folder, result)
        _process_task_sync(pid, task, result)
        sync_window = load_sync_window_json(t_folder)
        _process_eeg_sync(task, t_folder, result, sync_window)
        _process_eeg_segmentation(task, t_folder, result)
        _process_eeg_raw_access(task, t_folder, result)
        _process_eeg_raw_read_test(task, t_folder, result)
        _process_eeg_raw_signal_qc(task, t_folder, result)
        _process_eeglab_compat(task, t_folder, result)
        _process_eeg_qc(task, t_folder, result)
        _process_eeg_preprocessing_plan(task, t_folder, result)
        _process_eeg_preprocessing_exec(task, t_folder, result)
        _process_eeg_preprocessed_signal_qc(task, t_folder, result)
        _process_eeg_task_level_features(task, t_folder, pid, result)

    p_dir = participant_dir(pid)
    baseline_results = refresh_participant_eeg_baseline_linkage(p_dir)
    for linkage in baseline_results:
        if linkage.get("baseline_status_path"):
            result.saved_paths.append(linkage["baseline_status_path"])
        result.eeg_baseline_linkage_results.append(linkage)

    normalized_results = refresh_participant_baseline_normalized_features(p_dir)
    for normalized in normalized_results:
        for path_key in ("features_path", "features_json_path", "table_3_path"):
            if normalized.get(path_key):
                result.saved_paths.append(normalized[path_key])
        task_name = normalized.get("task_name")
        existing = next(
            (
                entry
                for entry in result.eeg_task_level_features_results
                if entry.get("task_name") == task_name
            ),
            None,
        )
        if existing is not None:
            existing.update(
                {
                    "features_path": normalized.get("features_path"),
                    "features_json_path": normalized.get("features_json_path"),
                    "table_3_path": normalized.get("table_3_path"),
                    "features": normalized.get("features"),
                }
            )
        else:
            result.eeg_task_level_features_results.append(normalized)

    plv_results = refresh_participant_plv_features(p_dir)
    for plv_entry in plv_results:
        for path_key in ("features_path", "features_json_path", "table_3_path"):
            if plv_entry.get(path_key):
                result.saved_paths.append(plv_entry[path_key])
        task_name = plv_entry.get("task_name")
        existing = next(
            (
                entry
                for entry in result.eeg_task_level_features_results
                if entry.get("task_name") == task_name
            ),
            None,
        )
        if existing is not None:
            existing.update(
                {
                    "features_path": plv_entry.get("features_path"),
                    "features_json_path": plv_entry.get("features_json_path"),
                    "table_3_path": plv_entry.get("table_3_path"),
                    "features": plv_entry.get("features"),
                }
            )
        else:
            result.eeg_task_level_features_results.append(plv_entry)

    plv_baseline_results = refresh_participant_plv_baseline_normalized_features(p_dir)
    for plv_baseline_entry in plv_baseline_results:
        for path_key in ("features_path", "features_json_path", "table_3_path"):
            if plv_baseline_entry.get(path_key):
                result.saved_paths.append(plv_baseline_entry[path_key])
        task_name = plv_baseline_entry.get("task_name")
        existing = next(
            (
                entry
                for entry in result.eeg_task_level_features_results
                if entry.get("task_name") == task_name
            ),
            None,
        )
        if existing is not None:
            existing.update(
                {
                    "features_path": plv_baseline_entry.get("features_path"),
                    "features_json_path": plv_baseline_entry.get("features_json_path"),
                    "table_3_path": plv_baseline_entry.get("table_3_path"),
                    "features": plv_baseline_entry.get("features"),
                }
            )
        else:
            result.eeg_task_level_features_results.append(plv_baseline_entry)

    audit_results = refresh_participant_eeg_task_level_feature_audits(p_dir)
    for audit_entry in audit_results:
        if audit_entry.get("audit_path"):
            result.saved_paths.append(audit_entry["audit_path"])
        result.eeg_task_level_feature_audit_results.append(audit_entry)

    table_path = append_participant_row(
        pid,
        upload.participant_name,
        upload.participant_age,
        upload.participant_group,
        upload.notes,
    )
    result.saved_paths.insert(0, str(table_path.resolve()))

    result.ok = True
    result.message = f"Participant `{pid}` saved to database."
    return result
