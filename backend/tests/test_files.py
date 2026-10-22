import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_upload_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEKEEPER_UPLOAD_DIR", str(tmp_path))
    return tmp_path


def test_serves_uploaded_file_unchanged(make_dicom_bytes):
    original = make_dicom_bytes()
    upload = client.post(
        "/upload",
        files={"file": ("scan.dcm", original, "application/octet-stream")},
    )
    file_id = upload.json()["file_id"]

    response = client.get(f"/files/{file_id}")
    assert response.status_code == 200
    assert response.content == original
    assert response.headers["content-type"] == "application/dicom"
    assert response.headers["cache-control"] == "no-store"


def test_unknown_file_returns_404():
    response = client.get("/files/does-not-exist-123")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "FILE_NOT_FOUND"


def test_unsafe_file_id_returns_404():
    response = client.get("/files/bad.id")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "FILE_NOT_FOUND"