"""Configuration helpers for the backend service."""
from __future__ import annotations

import os
from pathlib import Path


def get_upload_dir() -> Path:
    """Return the resolved upload directory path and ensure it exists."""
    path_str = os.environ.get("GATEKEEPER_UPLOAD_DIR", "temp_uploads")
    upload_path = Path(path_str).resolve()
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path
DEFAULT_MAX_UPLOAD_MB = 100


def get_max_upload_bytes() -> int:
    """Return the maximum accepted upload size in bytes.

    Configurable with GATEKEEPER_MAX_UPLOAD_MB; invalid values fall back
    to the default.
    """
    raw = os.environ.get("GATEKEEPER_MAX_UPLOAD_MB")
    try:
        megabytes = int(raw) if raw else DEFAULT_MAX_UPLOAD_MB
    except ValueError:
        megabytes = DEFAULT_MAX_UPLOAD_MB
    if megabytes <= 0:
        megabytes = DEFAULT_MAX_UPLOAD_MB
    return megabytes * 1024 * 1024
