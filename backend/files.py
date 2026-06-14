"""Filesystem helpers — safe copy, never overwrite raw uploads."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO, Union

FileLike = Union[BinaryIO, bytes]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_upload(
    src: FileLike,
    dest: Path,
    *,
    original_name: str | None = None,
) -> tuple[Path | None, str | None]:
    """
    Copy upload to dest. If dest exists, do not overwrite.

    Returns (saved_path, warning_message).
    """
    ensure_dir(dest.parent)
    if dest.exists():
        return None, f"Skipped (already exists, not overwritten): {dest}"

    if hasattr(src, "read"):
        src.seek(0)
        data = src.read()
    else:
        data = src

    dest.write_bytes(data)
    return dest, None


def unique_dest(base: Path, stem: str, suffix: str) -> Path:
    """Return base/stem+suffix or base/stem_2+suffix if taken."""
    candidate = base / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = base / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def copy_upload_preserve_name(
    src: FileLike,
    folder: Path,
    filename: str,
) -> tuple[Path | None, str | None]:
    dest = folder / filename
    if dest.exists():
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        dest = unique_dest(folder, stem, suffix)
    return copy_upload(src, dest)
