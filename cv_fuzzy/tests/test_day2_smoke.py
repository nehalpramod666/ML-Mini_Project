from pathlib import Path

import pydicom


SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def test_sample_dicom_loads_and_has_pixel_data():
    dcm_files = list(SAMPLE_DIR.glob("*.dcm"))

    assert len(dcm_files) > 0, "No sample DICOM files found"

    ds = pydicom.dcmread(dcm_files[0])
    arr = ds.pixel_array

    assert arr.size > 0