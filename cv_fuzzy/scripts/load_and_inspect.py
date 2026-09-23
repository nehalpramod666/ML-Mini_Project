from pathlib import Path

import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DICOM_PATH = PROJECT_ROOT / "cv_fuzzy" / "sample_data" / "CT_small.dcm"


def main():
    ds = pydicom.dcmread(DICOM_PATH)
    arr = ds.pixel_array

    print(f"File: {DICOM_PATH.name}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  BitsAllocated: {ds.BitsAllocated}")
    print(f"  PhotometricInterpretation: {ds.PhotometricInterpretation}")


if __name__ == "__main__":
    main()