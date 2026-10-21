"""Upload-time DICOM validation: structure only, no pixel decoding."""
from __future__ import annotations

from pathlib import Path

import pydicom


class UploadValidationError(Exception):
    """A validation failure with an HTTP status and a stable error code."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def validate_dicom_file(path: Path) -> None:
    """Raise UploadValidationError unless the file parses as a DICOM image.

    Only the header is parsed, so compressed files pass without any decoder
    package. Corruption inside the pixel data is not detected here; it
    surfaces later, at analysis time.
    """
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    except Exception as exc:
        raise UploadValidationError(
            422, "INVALID_DICOM", "The uploaded file is not a valid DICOM file."
        ) from exc

    if "Rows" not in ds or "Columns" not in ds:
        raise UploadValidationError(
            422, "NO_IMAGE_DATA", "The DICOM file does not contain image data."
        )