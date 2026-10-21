import io

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


@pytest.fixture
def make_dicom_bytes():
    """Return a function that builds a tiny valid DICOM file in memory."""

    def _make(rows: int = 16, cols: int = 16, with_image: bool = True) -> bytes:
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(
            None,
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
            is_implicit_VR=False,
            is_little_endian=True,
        )
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        if with_image:
            ds.Rows = rows
            ds.Columns = cols
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelData = np.zeros((rows, cols), dtype=np.uint16).tobytes()

        buffer = io.BytesIO()
        ds.save_as(buffer)
        return buffer.getvalue()

    return _make