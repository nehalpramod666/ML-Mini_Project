"""
DICOM loading and normalization.

FR1: normalize DICOM pixel data to 8-bit uint8 for OpenCV.
"""

import numpy as np
import pydicom


class DicomLoadError(Exception):
    """Raised when a DICOM file cannot be loaded or processed."""
InvalidDICOMError = DicomLoadError


def load_dicom(path: str) -> np.ndarray:
    """
    Load a DICOM file and return a normalized uint8 grayscale array.

    Parameters
    ----------
    path : str
        Path to the DICOM file.

    Returns
    -------
    np.ndarray
        8-bit normalized grayscale image.

    Raises
    ------
    DicomLoadError
        If the DICOM cannot be read or contains no usable pixel data.
    """
    try:
        ds = pydicom.dcmread(path)

        if not hasattr(ds, "PixelData"):
            raise DicomLoadError(
                f"DICOM file has no pixel data: {path}"
            )

        arr = ds.pixel_array.astype(np.float64)

    except DicomLoadError:
        raise

    except Exception as exc:
        raise DicomLoadError(
            f"Failed to load DICOM file '{path}': {exc}"
        ) from exc

    # Apply DICOM rescale transformation when present.
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    arr = arr * slope + intercept

    # Handle dimensionality.
    #
    # 4D: multi-frame color
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        arr = arr[0, ..., :3].mean(axis=-1)

    # 3D color: single-frame color
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(axis=-1)

    # 3D grayscale: multi-frame
    elif arr.ndim == 3:
        arr = arr[0]

    # MONOCHROME1 means lower values represent brighter pixels.
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = np.max(arr) - arr

    # Normalize to [0, 255].
    min_value = np.min(arr)
    max_value = np.max(arr)

    # Constant image.
    if max_value == min_value:
        return np.zeros(arr.shape, dtype=np.uint8)

    normalized = (
        (arr - min_value)
        / (max_value - min_value)
        * 255.0
    )

    return np.clip(normalized, 0, 255).astype(np.uint8)