from fastapi.testclient import TestClient
import sys
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


def test_upload_rejects_non_dcm_file():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_FILE_TYPE"
    assert detail["message"] == "Only .dcm files are supported"


def test_upload_accepts_dcm_file(make_dicom_bytes):
    response = client.post(
        "/upload",
        files={"file": ("sample.dcm", make_dicom_bytes(), "application/octet-stream")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["filename"] == "sample.dcm"