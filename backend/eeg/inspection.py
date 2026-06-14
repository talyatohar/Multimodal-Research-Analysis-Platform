"""
EEG Inspection — Phase 1 (BVRF only).

Parses BrainVision Recorder BVRF files, extracts metadata, and caches results.
No signal processing, segmentation, or feature extraction.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from domain.storage_layout import EEG_METADATA_FILE, EEG_RAW_FOLDER

BVRF_EXTENSIONS = (".bvrh", ".bvrd", ".bvrm", ".bvri")
BVRF_REQUIRED_SUFFIXES = (".bvrh", ".bvrd", ".bvrm")
BVRF_OPTIONAL_SUFFIXES = (".bvri",)

_MARKER_COLUMNS = ("Sample", "Type", "Code", "DateTime", "Comment")
_SEGMENT_TYPES = ("new segment", "recording start", "recordingstart")

_DTYPE_BYTES = {
    "Int16": 2,
    "Int32": 4,
    "Single": 4,
    "Double": 8,
}

_ISO_DATETIME_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)


def eeg_raw_dir(task_folder: Path) -> Path:
    return task_folder / EEG_RAW_FOLDER


def _discover_bvrf_sets(eeg_raw_folder: Path) -> dict[str, dict[str, Path | None]]:
    sets: dict[str, dict[str, Path | None]] = {}
    if not eeg_raw_folder.is_dir():
        return sets

    for path in sorted(eeg_raw_folder.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in BVRF_EXTENSIONS:
            continue
        stem = path.stem
        entry = sets.setdefault(stem, {ext: None for ext in BVRF_EXTENSIONS})
        if entry.get(suffix) is None:
            entry[suffix] = path
    return sets


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        match = _ISO_DATETIME_RE.search(value)
        if not match:
            return None
        candidate = match.group(1)
        if candidate.endswith("Z"):
            candidate = f"{candidate[:-1]}+00:00"
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            return None


def _format_datetime(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.isoformat()
    return dt.astimezone(timezone.utc).isoformat()


def parse_bvrh(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Parse .bvrh JSON header."""
    warnings: list[str] = []
    payload: dict[str, Any] = {
        "sampling_rate_hz": None,
        "channel_count": None,
        "channels": [],
        "amplifier": None,
        "numeric_data_type": None,
    }

    try:
        with path.open(encoding="utf-8-sig") as handle:
            header = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not parse {path.name}: {exc}")
        return payload, warnings

    eeg_modality = header.get("EEGModality")
    if not isinstance(eeg_modality, dict):
        warnings.append(f"No EEGModality section found in {path.name}")
        return payload, warnings

    data_specific = eeg_modality.get("DataSpecific")
    if isinstance(data_specific, dict):
        raw_rate = data_specific.get("SamplingFrequencyInHertz")
        if raw_rate is not None:
            try:
                payload["sampling_rate_hz"] = float(raw_rate)
            except (TypeError, ValueError):
                warnings.append(f"Invalid SamplingFrequencyInHertz in {path.name}: {raw_rate!r}")

    channels = eeg_modality.get("Channels")
    channel_rows: list[dict[str, Any]] = []
    if isinstance(channels, list):
        for channel in channels:
            if not isinstance(channel, dict):
                continue
            name = channel.get("Name")
            if not isinstance(name, str):
                continue
            participant_id = channel.get("ParticipantId")
            display_name = f"{name} ({participant_id})" if participant_id else name
            channel_rows.append(
                {
                    "name": display_name,
                    "type": channel.get("Type"),
                    "unit": channel.get("Unit"),
                    "participant_id": participant_id,
                    "amplifier_id": channel.get("AmplifierId"),
                }
            )
        payload["channels"] = channel_rows
        payload["channel_count"] = len(channel_rows)

    amplifiers = eeg_modality.get("Amplifiers")
    if isinstance(amplifiers, list) and amplifiers:
        payload["amplifier"] = amplifiers[0] if len(amplifiers) == 1 else amplifiers
    elif isinstance(amplifiers, dict):
        payload["amplifier"] = amplifiers

    bvrf_files = eeg_modality.get("BVRFFiles")
    if isinstance(bvrf_files, dict):
        data_file = bvrf_files.get("DataFile")
        if isinstance(data_file, dict):
            payload["numeric_data_type"] = data_file.get("NumericDataType")

    recording_software = header.get("RecordingSoftware")
    if isinstance(recording_software, dict):
        payload["recording_software"] = recording_software

    return payload, warnings


def parse_bvrm(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse .bvrm marker TSV."""
    warnings: list[str] = []
    markers: list[dict[str, Any]] = []

    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return markers, warnings

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        warnings.append(f"{path.name}: no marker rows found")
        return markers, warnings

    header_cols = [col.strip() for col in lines[0].split("\t")]
    col_index = {name: idx for idx, name in enumerate(header_cols)}

    def _cell(parts: list[str], column: str) -> str | None:
        idx = col_index.get(column)
        if idx is None or idx >= len(parts):
            return None
        value = parts[idx].strip()
        return value or None

    for line in lines[1:]:
        parts = line.split("\t")
        marker: dict[str, Any] = {}
        for column in _MARKER_COLUMNS:
            marker[column.lower()] = _cell(parts, column)
        if any(marker.values()):
            markers.append(marker)

    return markers, warnings


def parse_bvri(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse .bvri impedance TSV into QC rows."""
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return rows, warnings

    electrode_line = next((line for line in lines if line.startswith("Electrode")), None)
    participant_line = next((line for line in lines if line.startswith("ParticipantId")), None)
    measurement_lines = [line for line in lines if re.match(r"^\d{4}-\d{2}-\d{2}", line)]

    if electrode_line is None:
        warnings.append(f"{path.name}: no Electrode header row found")
        return rows, warnings
    if not measurement_lines:
        warnings.append(f"{path.name}: no impedance measurement rows found")
        return rows, warnings

    electrodes = electrode_line.split("\t")[1:]
    participant_ids = (
        participant_line.split("\t")[1:]
        if participant_line is not None
        else [None] * len(electrodes)
    )

    for measurement_line in measurement_lines:
        parts = measurement_line.split("\t")
        measurement_time = parts[0].strip()
        values = parts[1:]
        for electrode, participant_id, raw_value in zip(electrodes, participant_ids, values):
            electrode = electrode.strip()
            if not electrode:
                continue
            impedance_kohm: float | None
            try:
                impedance_kohm = float(raw_value.strip())
            except (TypeError, ValueError):
                impedance_kohm = None
            rows.append(
                {
                    "electrode": electrode,
                    "participant_id": participant_id.strip() if participant_id else None,
                    "impedance_kohm": impedance_kohm,
                    "measurement_time": measurement_time,
                }
            )

    return rows, warnings


def _marker_datetimes(markers: list[dict[str, Any]]) -> list[datetime]:
    parsed: list[datetime] = []
    for marker in markers:
        dt = _parse_iso_datetime(marker.get("datetime"))
        if dt is not None:
            parsed.append(dt)
    return parsed


def _recording_start_from_markers(markers: list[dict[str, Any]]) -> str | None:
    for marker in markers:
        marker_type = (marker.get("type") or "").lower()
        if any(token in marker_type for token in _SEGMENT_TYPES):
            dt = _parse_iso_datetime(marker.get("datetime"))
            if dt is not None:
                return _format_datetime(dt)
    datetimes = _marker_datetimes(markers)
    if datetimes:
        return _format_datetime(min(datetimes))
    return None


def _recording_end_from_markers(markers: list[dict[str, Any]]) -> str | None:
    datetimes = _marker_datetimes(markers)
    if datetimes:
        return _format_datetime(max(datetimes))
    return None


def _estimate_samples_from_bvrd(
    data_path: Path | None,
    channel_count: int | None,
    numeric_data_type: str | None,
) -> int | None:
    if data_path is None or not data_path.is_file():
        return None
    if channel_count is None or channel_count <= 0:
        return None
    itemsize = _DTYPE_BYTES.get(numeric_data_type or "")
    if itemsize is None:
        return None
    nbytes = data_path.stat().st_size
    if nbytes == 0:
        return 0
    frame_bytes = channel_count * itemsize
    if nbytes % frame_bytes != 0:
        return nbytes // frame_bytes
    return nbytes // frame_bytes


def _duration_seconds(
    start_time: str | None,
    end_time: str | None,
    sample_count: int | None,
    sampling_rate_hz: float | None,
) -> float | None:
    start_dt = _parse_iso_datetime(start_time)
    end_dt = _parse_iso_datetime(end_time)
    if start_dt is not None and end_dt is not None and end_dt >= start_dt:
        return (end_dt - start_dt).total_seconds()

    if sample_count is not None and sampling_rate_hz and sampling_rate_hz > 0:
        return sample_count / sampling_rate_hz
    return None


def _recording_end_from_duration(start_time: str | None, duration_seconds: float | None) -> str | None:
    start_dt = _parse_iso_datetime(start_time)
    if start_dt is None or duration_seconds is None:
        return None
    return _format_datetime(start_dt + timedelta(seconds=duration_seconds))


def _channels_for_metadata_export(channels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose parsed channels in eeg_metadata.json with explicit order."""
    return [
        {
            "order": index,
            "name": channel.get("name"),
            "type": channel.get("type"),
            "unit": channel.get("unit"),
        }
        for index, channel in enumerate(channels, start=1)
    ]


def build_eeg_metadata(files: dict[str, Path | None], basename: str) -> dict[str, Any]:
    """Build inspection metadata for one BVRF recording set."""
    warnings: list[str] = []
    missing_required = [suffix for suffix in BVRF_REQUIRED_SUFFIXES if files.get(suffix) is None]
    missing_optional = [suffix for suffix in BVRF_OPTIONAL_SUFFIXES if files.get(suffix) is None]

    if missing_required:
        warnings.append(f"{basename}: missing required BVRF file(s): {', '.join(missing_required)}")
    if missing_optional:
        warnings.append(f"{basename}: optional BVRF file(s) not present: {', '.join(missing_optional)}")

    header_path = files.get(".bvrh")
    marker_path = files.get(".bvrm")
    data_path = files.get(".bvrd")
    impedance_path = files.get(".bvri")

    header_meta: dict[str, Any] = {}
    if header_path is not None:
        header_meta, header_warnings = parse_bvrh(header_path)
        warnings.extend(header_warnings)

    markers: list[dict[str, Any]] = []
    if marker_path is not None:
        markers, marker_warnings = parse_bvrm(marker_path)
        warnings.extend(marker_warnings)

    impedance_qc: list[dict[str, Any]] = []
    if impedance_path is not None:
        impedance_qc, impedance_warnings = parse_bvri(impedance_path)
        warnings.extend(impedance_warnings)

    recording_start_time = _recording_start_from_markers(markers)
    recording_end_time = _recording_end_from_markers(markers)

    sample_count = _estimate_samples_from_bvrd(
        data_path,
        header_meta.get("channel_count"),
        header_meta.get("numeric_data_type"),
    )
    sampling_rate_hz = header_meta.get("sampling_rate_hz")

    duration_seconds = _duration_seconds(
        recording_start_time,
        recording_end_time,
        sample_count,
        sampling_rate_hz,
    )
    if recording_end_time is None:
        recording_end_time = _recording_end_from_duration(recording_start_time, duration_seconds)

    data_file_present = data_path is not None and data_path.is_file()
    if not data_file_present:
        warnings.append(f"{basename}: .bvrd data file is missing")

    return {
        "basename": basename,
        "sampling_rate_hz": sampling_rate_hz,
        "channel_count": header_meta.get("channel_count"),
        "channels": _channels_for_metadata_export(header_meta.get("channels", [])),
        "amplifier": header_meta.get("amplifier"),
        "recording_software": header_meta.get("recording_software"),
        "recording_start_time": recording_start_time,
        "recording_end_time": recording_end_time,
        "duration_seconds": duration_seconds,
        "sample_count": sample_count,
        "markers_count": len(markers),
        "markers": markers,
        "impedance_qc": impedance_qc,
        "data_file_present": data_file_present,
        "data_file_size_bytes": data_path.stat().st_size if data_file_present and data_path else None,
        "files": {
            "header": str(header_path.resolve()) if header_path else None,
            "data": str(data_path.resolve()) if data_file_present and data_path else None,
            "marker": str(marker_path.resolve()) if marker_path else None,
            "impedance": str(impedance_path.resolve()) if impedance_path else None,
        },
        "missing_required_files": missing_required,
        "missing_optional_files": missing_optional,
        "warnings": warnings,
    }


def inspect_eeg_raw_folder(eeg_raw_folder: Path | str) -> dict[str, Any]:
    """Inspect all BVRF recording sets under EEG_raw/."""
    folder = Path(eeg_raw_folder)
    result: dict[str, Any] = {
        "format": "BVRF",
        "phase": "inspection_only",
        "eeg_raw_folder": str(folder.resolve()) if folder.exists() else str(folder),
        "warnings": [],
        "recordings": [],
    }

    if not folder.is_dir():
        result["warnings"].append(f"EEG_raw folder not found: {folder}")
        return result

    grouped = _discover_bvrf_sets(folder)
    if not grouped:
        result["warnings"].append(f"No BVRF files found in {folder}")
        return result

    for basename, files in sorted(grouped.items()):
        recording = build_eeg_metadata(files, basename)
        result["recordings"].append(recording)
        result["warnings"].extend(recording.pop("warnings", []))

    return result


def write_eeg_metadata_json(
    eeg_raw_folder: Path | str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Cache extracted metadata to EEG_raw/eeg_metadata.json."""
    folder = Path(eeg_raw_folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = metadata if metadata is not None else inspect_eeg_raw_folder(folder)
    out = folder / EEG_METADATA_FILE
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_eeg_metadata_json(eeg_raw_folder: Path | str) -> dict[str, Any] | None:
    path = Path(eeg_raw_folder) / EEG_METADATA_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
