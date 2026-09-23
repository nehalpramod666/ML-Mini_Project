from pathlib import Path

import numpy as np
import pytest

from cv_fuzzy.dicom_loader import DicomLoadError, load_dicom


SAMPLE_FILE = (
    Path("cv_fuzzy/tests/sample_data/CT_small.dcm").resolve().parent.parent
    / "sample_data"
    / "CT_small.dcm"
)


def test_load_returns_nonempty_uint8_array():
    """Loader should return a non-empty uint8 NumPy array."""
    arr = load_dicom(str(SAMPLE_FILE))

    assert isinstance(arr, np.ndarray)
    assert arr.size > 0
    assert arr.dtype == np.uint8


def test_load_returns_2d_array():
    """Loader should return a 2D grayscale image."""
    arr = load_dicom(str(SAMPLE_FILE))

    assert arr.ndim == 2
    assert all(size > 0 for size in arr.shape)


def test_load_value_range_not_degenerate():
    """Normalized image should have a meaningful pixel range."""
    arr = load_dicom(str(SAMPLE_FILE))

    assert arr.min() >= 0
    assert arr.max() <= 255
    assert arr.max() > arr.min()


def test_load_raises_on_corrupt_file(tmp_path):
    """Loader should raise DicomLoadError for invalid DICOM data."""
    bad_file = tmp_path / "corrupt.dcm"
    bad_file.write_bytes(b"not a real dicom")

    with pytest.raises(DicomLoadError):
        load_dicom(str(bad_file))