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
