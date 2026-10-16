from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_upload_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEKEEPER_UPLOAD_DIR", str(tmp_path))
    return tmp_path


def test_upload_rejects_non_dcm_file():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only .dcm files are supported"


def test_upload_accepts_dcm_file():
    response = client.post(
        "/upload",
        files={"file": ("sample.dcm", b"fake dicom bytes", "application/octet-stream")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["filename"] == "sample.dcm"