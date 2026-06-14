"""Basic BrainVision .ahdr / .amrk metadata (no feature extraction)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EEGRecordingMeta:
    sampling_interval_us: int | None
    sampling_rate_hz: float | None
    first_marker_timestamp: str | None
    source_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_ahdr(content: str | bytes) -> dict[str, str]:
    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="replace")
    else:
        text = content
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            out[key.strip()] = val.strip()
    return out


def parse_amrk_first_marker(content: str | bytes) -> str | None:
    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="replace")
    else:
        text = content
    for line in text.splitlines():
        if line.startswith("Mk1="):
            parts = line.split(",")
            if len(parts) >= 6:
                return parts[-1].strip()
    m = re.search(r"Mk1=.*?,\s*(\d{14,})", text)
    return m.group(1) if m else None


def build_eeg_meta(ahdr_bytes: bytes | None, amrk_bytes: bytes | None, sources: list[str]) -> EEGRecordingMeta:
    interval = None
    rate = None
    marker = None
    if ahdr_bytes:
        fields = parse_ahdr(ahdr_bytes)
        raw = fields.get("SamplingInterval")
        if raw and raw.isdigit():
            interval = int(raw)
            rate = 1_000_000.0 / interval
    if amrk_bytes:
        marker = parse_amrk_first_marker(amrk_bytes)
    return EEGRecordingMeta(
        sampling_interval_us=interval,
        sampling_rate_hz=rate,
        first_marker_timestamp=marker,
        source_files=sources,
    )


def write_eeg_meta_json(task_folder: Path, meta: EEGRecordingMeta) -> Path:
    out = task_folder / "eeg_meta.json"
    out.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return out
