from pathlib import Path

import numpy as np
import pytest

from cv_fuzzy.dicom_loader import DicomLoadError, load_dicom


SAMPLE_DIR = Path(__file__).parent / "sample_data"


def get_sample_dicom():
    files = list(SAMPLE_DIR.glob("*.dcm"))

    if not files:
        pytest.fail("No sample DICOM files found in sample_data/")

    return files[0]


def test_load_dicom_returns_ndarray():
    image = load_dicom(str(get_sample_dicom()))

    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8


def test_load_dicom_nonempty_shape():
    image = load_dicom(str(get_sample_dicom()))

    assert image.ndim >= 2
    assert all(size > 0 for size in image.shape)


def test_load_dicom_value_range():
    image = load_dicom(str(get_sample_dicom()))

    assert image.min() >= 0
    assert image.max() <= 255


def test_load_dicom_missing_file_raises():
    missing_file = SAMPLE_DIR / "does_not_exist.dcm"

    with pytest.raises(DicomLoadError):
        load_dicom(str(missing_file))