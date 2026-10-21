import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_upload_dir(tmp_path, monkeypatch):
    """Keep test uploads out of the real temp_uploads folder."""
    monkeypatch.setenv("GATEKEEPER_UPLOAD_DIR", str(tmp_path))
    return tmp_path


def _post(name: str, data: bytes):
    return client.post(
        "/upload", files={"file": (name, data, "application/octet-stream")}
    )


def test_accepts_valid_dicom(make_dicom_bytes, isolated_upload_dir):
    response = _post("scan.dcm", make_dicom_bytes())
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "scan.dcm"
    assert (isolated_upload_dir / f"{body['file_id']}.dcm").is_file()


def test_rejects_wrong_extension(isolated_upload_dir):
    response = _post("photo.jpeg", b"not dicom")
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_FILE_TYPE"
    assert list(isolated_upload_dir.iterdir()) == []


def test_rejects_empty_file(isolated_upload_dir):
    response = _post("empty.dcm", b"")
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "EMPTY_FILE"
    assert list(isolated_upload_dir.iterdir()) == []


def test_rejects_non_dicom_bytes(isolated_upload_dir):
    response = _post("fake.dcm", b"hello world, this is plain text")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_DICOM"
    assert list(isolated_upload_dir.iterdir()) == []


def test_rejects_truncated_dicom(make_dicom_bytes, isolated_upload_dir):
    response = _post("cut.dcm", make_dicom_bytes()[:100])
    assert response.status_code == 422
    assert response.json()["detail"]["code"] in {"INVALID_DICOM", "NO_IMAGE_DATA"}
    assert list(isolated_upload_dir.iterdir()) == []


def test_rejects_dicom_without_image_data(make_dicom_bytes, isolated_upload_dir):
    response = _post("noimage.dcm", make_dicom_bytes(with_image=False))
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "NO_IMAGE_DATA"
    assert list(isolated_upload_dir.iterdir()) == []


def test_rejects_oversized_file(monkeypatch, isolated_upload_dir):
    monkeypatch.setenv("GATEKEEPER_MAX_UPLOAD_MB", "1")
    response = _post("big.dcm", b"\0" * (2 * 1024 * 1024))
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "FILE_TOO_LARGE"
    assert list(isolated_upload_dir.iterdir()) == []