import numpy as np
import pydicom


class DicomLoadError(Exception):
    """Raised when a DICOM file cannot be loaded or processed."""


def load_dicom(path: str) -> np.ndarray:
    """
    Load a DICOM image and return its pixel data normalized to uint8.

    Parameters
    ----------
    path : str
        Path to the DICOM file.

    Returns
    -------
    np.ndarray
        8-bit normalized pixel array.
    """
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array

        arr = np.asarray(arr, dtype=np.float32)

        min_value = arr.min()
        max_value = arr.max()

        # Handle constant images such as all-black/all-white images.
        if max_value == min_value:
            return np.zeros(arr.shape, dtype=np.uint8)

        normalized = (
            (arr - min_value)
            / (max_value - min_value)
            * 255.0
        )

        return normalized.astype(np.uint8)

    except Exception as exc:
        raise DicomLoadError(
            f"Failed to load DICOM file '{path}': {exc}"
        ) from exc