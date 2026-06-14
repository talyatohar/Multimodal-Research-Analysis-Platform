"""E-Prime log parser and sync_window.json I/O."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _israel_timezone_for_date(local_date: date) -> timezone:
    """Israel timezone by date; uses current DST rules when zoneinfo is unavailable."""
    year = local_date.year
    last_sunday_march = _last_weekday(year, 3, 6)
    dst_start = last_sunday_march - timedelta(days=2)  # Friday before the last Sunday in March
    dst_end = _last_weekday(year, 10, 6)  # Last Sunday in October
    offset = 3 if dst_start <= local_date < dst_end else 2
    return timezone(timedelta(hours=offset), name=f"UTC+{offset}")


def _utc_to_israel_local(utc_dt: datetime) -> datetime:
    provisional = utc_dt.astimezone(timezone(timedelta(hours=3)))
    return utc_dt.astimezone(_israel_timezone_for_date(provisional.date()))


def read_eprime_text(file_path: str | Path) -> str:
    """
    Read E-Prime .txt exports.

    Tobii/E-Prime logs are often UTF-16 LE (BOM ``\\xff\\xfe``); plain UTF-8 is also supported.
    Uploaded files are stored as raw bytes — this must match how they are parsed on disk.
    """
    raw = Path(file_path).read_bytes()
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    if len(raw) >= 4 and raw[1:2] == b"\x00":
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("utf-8", errors="ignore")


def parse_eprime_log(file_path: str | Path) -> dict[str, Any]:
    text = read_eprime_text(file_path)

    def get_value(key: str) -> str | None:
        pattern = rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$"
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            return None
        return match.group(1).strip()

    session_start_raw = get_value("SessionStartDateTimeUtc")
    fixation_onset = get_value("FixationStart.OnsetTime")
    fixation_duration = get_value("FixationStart.OnsetToOnsetTime")
    story_duration = get_value("SixationOpenS.OnsetToOnsetTime")

    if not session_start_raw:
        raise ValueError("Could not find SessionStartDateTimeUtc")

    session_start = datetime.strptime(session_start_raw, "%d/%m/%Y %H:%M:%S")

    task_start_utc = session_start + timedelta(milliseconds=int(fixation_onset))
    task_duration_ms = int(fixation_duration) + int(story_duration)
    task_end_utc = task_start_utc + timedelta(milliseconds=task_duration_ms)

    return {
        "session_start_utc": session_start.isoformat(),
        "task_start_utc": task_start_utc.isoformat(),
        "task_end_utc": task_end_utc.isoformat(),
        "task_duration_ms": task_duration_ms,
        "fixation_onset_ms": int(fixation_onset),
        "fixation_duration_ms": int(fixation_duration),
        "story_duration_ms": int(story_duration),
    }


def _parse_manual_datetime(value: str) -> datetime:
    text = value.strip()
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            parsed_time = datetime.strptime(text, fmt).time()
            return datetime.combine(datetime.min.date(), parsed_time)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Manual reading end time must be HH:MM:SS.sss, HH:MM:SS, or a full datetime.") from exc


def add_manual_task_end(
    window: dict[str, Any],
    manual_value: str | None,
    interpretation: str | None,
) -> dict[str, Any]:
    """
    Add user-reviewed reading end time while preserving the original E-Prime task window.

    Time-only values are combined with the experiment date. Local Israel Time uses the
    Asia/Jerusalem timezone database so UTC+2/UTC+3 is selected by date when available.
    """
    if not manual_value or not manual_value.strip():
        return window

    task_start = datetime.fromisoformat(window["task_start_utc"])
    if task_start.tzinfo is None:
        task_start = task_start.replace(tzinfo=timezone.utc)
    else:
        task_start = task_start.astimezone(timezone.utc)

    parsed = _parse_manual_datetime(manual_value)
    mode = interpretation or "Local Israel Time"
    if mode.startswith("Local Israel"):
        experiment_local_date = _utc_to_israel_local(task_start).date()
        israel_tz = _israel_timezone_for_date(experiment_local_date)
        if parsed.date() == datetime.min.date():
            local_dt = datetime.combine(experiment_local_date, parsed.time(), tzinfo=israel_tz)
        else:
            parsed_tz = _israel_timezone_for_date(parsed.date())
            local_dt = parsed.replace(tzinfo=parsed_tz) if parsed.tzinfo is None else parsed.astimezone(parsed_tz)
        utc_dt = local_dt.astimezone(timezone.utc)
    else:
        if parsed.date() == datetime.min.date():
            utc_dt = datetime.combine(task_start.date(), parsed.time(), tzinfo=timezone.utc)
        else:
            utc_dt = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        local_dt = _utc_to_israel_local(utc_dt)

    updated = dict(window)
    updated["manual_task_end_local_time"] = local_dt.isoformat()
    updated["manual_task_end_utc"] = utc_dt.isoformat()
    updated["manual_end_source"] = "user_screen_recording_review"
    updated["manual_time_interpretation"] = mode
    updated["manual_task_end_entered"] = manual_value.strip()
    return updated


def eprime_file_preview(file_path: str | Path, max_lines: int = 30) -> str:
    """First N lines of the E-Prime log for UI debug when parsing fails."""
    text = read_eprime_text(file_path)
    lines = text.splitlines()[:max_lines]
    return "\n".join(line.rstrip("\r") for line in lines)


def write_sync_window_json(task_folder: Path, window: dict[str, Any]) -> Path:
    out = task_folder / "sync_window.json"
    out.write_text(json.dumps(window, indent=2), encoding="utf-8")
    return out


def load_sync_window_json(task_folder: Path) -> dict[str, Any] | None:
    path = task_folder / "sync_window.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
